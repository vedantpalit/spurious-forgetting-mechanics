"""Knowledge injection and interference: pretrain on A + ballast, then inject B.

Plain next-token prediction throughout. No teacher, no distillation, no KL, no divergence
family, no sharpness probes — this is a dynamics measurement, not a distillation study.

Phases
------
pretrain  NTP over A ∪ ballast (sampled uniformly over the union), evaluated on A and
          ballast separately. Gated on A's first-token accuracy: nothing downstream is
          meaningful if the model never learned A's facts.
inject    NTP on B under the set relatedness condition, continuing from the pretrained
          params. Dense checkpoints across the collapse.
sweep     Pretraining peak-LR grid, selected on final training loss.

Why the optimizer differs from `TrainConfig`
--------------------------------------------
The repo's `create_optimizer` hardcodes a warmup leg, inherits optax's default β₂ = 0.999,
and decays to exactly zero. All three distort a dynamics measurement, so the optimizer is
built here instead: AdamW β₁ 0.9 / β₂ 0.95, weight decay 0.1, no warmup, cosine to a
nonzero floor for pretraining, and a **constant** LR for injection (a decaying schedule
during a collapse that happens in the first few hundred steps would make the learning rate
a time-varying confound).

Value partition
---------------
Each attribute pool is split in half (X / Y) by a seeded permutation written to disk, since
the measured-overlap regression consumes it and must reproduce across sessions. A draws
from X, ballast from Y — ballast exists so that half-Y value tokens are trained just as
hard as half-X ones, which is why |ballast| = |A| and why it is never oversampled. B draws
from whichever half the condition names.

Run: uv run python -m src.experiments.knowledge_injection --phase pretrain
"""
import json
import os
from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb

from src.config import Config, TrainConfig, parse_config, vars_nested
from src.data.biography import NUM_ATTRIBUTES, BiographyDataset, BiographyPopulation
from src.data.config import DataConfig
from src.experiments.ckpt import ckpt_path, experiment_key, file_hash, load_state, save_params, save_state
from src.model.config import ModelConfig
from src.model.factory import create_model
from src.train import TrainState, create_train_state, eval_forward, train_step

# Pretraining peak-LR grid. Brackets the reference paper's 4e-4 over five values.
PRETRAIN_LR_GRID = (2e-4, 5e-4, 1e-3, 2e-3, 5e-3)

# The reference protocol pairs a 4e-4 pretraining peak with a constant 3e-5 during
# injection. Injection LR is derived from whichever pretrain peak the sweep selected, at
# this ratio, rather than left as a default someone has to remember to update.
REFERENCE_PRETRAIN_LR, REFERENCE_INJECT_LR = 4e-4, 3e-5
INJECT_LR_RATIO = REFERENCE_PRETRAIN_LR / REFERENCE_INJECT_LR

CONDITIONS = ("identical", "high_overlap", "partial", "disjoint", "all_values")
# "all_values": B draws from the WHOLE pool rather than one half -- the ordinary-forgetting
# contrast, in which the new facts' answers do not occupy a region of their own and there is
# no half for a common shift to point at. A and the ballast keep their halves, so their
# own-half metrics are unchanged; B has no own half and is skipped by the rank metrics.
B_HALF = {"high_overlap": "X", "partial": "mixed", "disjoint": "Y", "identical": "X",
          "all_values": "all"}


@dataclass
class OptConfig:
    """Supplied by the reference protocol; not the repo's TrainConfig defaults."""
    peak_lr: float = 4e-4
    final_lr: float = 1e-7   # cosine floor; ignored when schedule == "constant"
    b1: float = 0.9
    b2: float = 0.95         # optax defaults to 0.999
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    schedule: str = "cosine"  # "cosine" (no warmup) or "constant"


@dataclass
class PretrainConfig:
    total_steps: int = 8000
    batch_size: int = 256
    eval_interval: int = 250
    # Gate: A's first_token_accuracy must reach this before injection is meaningful.
    target_acc: float = 0.90
    # Pretraining evals only rank configurations (the LR sweep), so a subsample is fine.
    max_eval_people: int = 500
    opt: OptConfig = field(default_factory=OptConfig)


@dataclass
class InjectConfig:
    total_steps: int = 400
    batch_size: int = 256
    eval_interval: int = 10
    condition: str = "high_overlap"  # identical | high_overlap | partial | disjoint
    # Control arm. False inherits Adam's μ and ν from pretraining but resets step and
    # schedule — inheriting the decayed LR too would conflate the control with an LR change.
    fresh_optimizer: bool = True
    # Replicate seed for injection's OWN stochasticity only (offset added to reset_stream's
    # seed in phase_inject) -- NOT cfg.seed, which is baked into pretrain_key and would force
    # a full repretrain per replicate. get_batch ignores the JAX rng it's handed and reads
    # from the dataset's internal numpy _np_rng (see BiographyDataset.get_batch's docstring),
    # so this offset is the only thing that actually varies the batch stream across replicates
    # of the same pretrained checkpoint. 0 reproduces today's behaviour exactly.
    seed: int = 0
    checkpoint_steps: str = "0,10,25,50,100,200,400"
    save_opt_state: bool = False  # dense checkpoints are params-only by default
    # Injection reports retention as a *result*, so it evaluates on all of A. The eval set
    # is pre-generated and deterministic, making this a fixed cost; subsampling would only
    # add variance to the headline metric. 0 = every person.
    max_eval_people: int = 0
    # peak_lr <= 0 means "derive from the pretraining checkpoint's recorded peak, divided
    # by INJECT_LR_RATIO". Pass a positive value to override explicitly.
    opt: OptConfig = field(default_factory=lambda: OptConfig(
        peak_lr=-1.0, schedule="constant"))
    # Deliberately-graded key (name-part) overlap between B and A, engineered rather than
    # left to chance (chance-level overlap already predicts damage robustly, but concentrates
    # ~75% of A at full 3-part overlap -- not a dose-response).
    # When True, build() overwrites only B's name assignment (assign_graded_b_names);
    # A/ballast/C and all value assignment are untouched, so the existing pretrained
    # checkpoint remains valid. Lives here, not on KIConfig, so it enters the injection
    # checkpoint's hash automatically via experiment_key(..., inject=inject_cfg, ...) --
    # the same pattern that closed the partition-content collision (see pretrain_key).
    graded_key_overlap: bool = False
    graded_key_seed: int = 11


@dataclass
class ProfileConfig:
    """Throughput probe. No evals, no checkpoints — steps and data generation only."""
    steps: int = 300
    batch_size: int = 256
    warmup_steps: int = 10  # excluded from timing so JIT compilation is not counted
    max_eval_people: int = 1  # the profile never evaluates; keep eval-set build trivial


@dataclass
class KIConfig:
    model: ModelConfig = field(default_factory=lambda: ModelConfig(dropout_rate=0.0))
    data: DataConfig = field(default_factory=lambda: DataConfig(
        task_type="biography", batch_size=256, eval_batch_size=256))
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    inject: InjectConfig = field(default_factory=InjectConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    # Pilot scale. Cut the population, not the steps: at 2000 people and 200 first names
    # there are still ~10 individuals per name part, which is ample sharing for the Gram
    # check, and compositional structure is forced by the architecture rather than crowding.
    num_a: int = 2000
    num_ballast: int = 2000
    num_b: int = 500
    # Held-out population C: a fourth disjoint index range, sampled by neither phase and
    # only ever evaluated. Exists for the hallucination corollary (the seen/unseen
    # confidence gap as keys crowd). 0 disables it.
    num_c: int = 500
    # Which half C's values come from. "X" (A's half) keeps C's facts in-distribution, so
    # a seen/unseen gap measures individual familiarity rather than value familiarity.
    # A science call, not a mechanical one — flagged rather than assumed settled.
    c_half: str = "X"
    partition_seed: int = 7
    partition_path: str = "data/biography/value_partition.npz"
    checkpoint_dir: str = "checkpoints"
    exposure_dir: str = "exposures"
    phase: str = "pretrain"  # pretrain | inject | sweep | profile | all
    seed: int = 42
    log_interval: int = 50
    wandb_project: str = "knowledge-injection"
    wandb_mode: str = "offline"


# --- optimizer ---

def make_optimizer(opt: OptConfig, total_steps: int) -> optax.GradientTransformation:
    """AdamW with the supplied betas and decay. No warmup leg at all."""
    if opt.schedule == "constant":
        schedule = optax.constant_schedule(opt.peak_lr)
    elif opt.schedule == "cosine":
        # alpha is a ratio, so the run ends at exactly final_lr rather than 0.
        schedule = optax.cosine_decay_schedule(
            init_value=opt.peak_lr, decay_steps=max(1, total_steps),
            alpha=opt.final_lr / opt.peak_lr)
    else:
        raise ValueError(f"Unknown schedule: {opt.schedule}")
    return optax.chain(
        optax.clip_by_global_norm(opt.grad_clip),
        optax.adamw(schedule, b1=opt.b1, b2=opt.b2, weight_decay=opt.weight_decay),
    )


def inherit_moments(fresh_opt_state, saved_opt_state):
    """Copy μ and ν from a saved optimizer state into a fresh one, leaving count at 0.

    The two states may differ in structure (cosine vs constant schedule); only the
    ScaleByAdamState moments are transferred.
    """
    def find(state):
        if isinstance(state, optax.ScaleByAdamState):
            return state
        if isinstance(state, tuple):
            for sub in state:
                found = find(sub)
                if found is not None:
                    return found
        return None

    saved = find(saved_opt_state)
    if saved is None:
        raise ValueError("No ScaleByAdamState in the saved optimizer state; cannot inherit moments.")

    def rebuild(state):
        if isinstance(state, optax.ScaleByAdamState):
            return state._replace(mu=saved.mu, nu=saved.nu)
        if isinstance(state, tuple):
            parts = tuple(rebuild(sub) for sub in state)
            return type(state)(*parts) if hasattr(state, "_fields") else parts
        return state

    return rebuild(fresh_opt_state)


# --- value partition ---

def build_partition(pop: BiographyPopulation, seed: int):
    """Seeded random split of each attribute's value indices into halves (X, Y)."""
    rng = np.random.default_rng(seed)
    halves = []
    for k in range(NUM_ATTRIBUTES):
        n = int(pop.num_values_per_attr[k])
        perm = rng.permutation(n)
        halves.append((np.sort(perm[: n // 2]), np.sort(perm[n // 2:])))
    return halves


def build_partition_fraction(pop: BiographyPopulation, seed: int, in_fraction: float):
    """Generalizes `build_partition` from a fixed 50/50 split to an arbitrary in/out
    fraction, for the exclusion-fraction sweep. X is the "in" region (size
    `int(in_fraction * n)`, what A and B share); Y is the "out" region (the rest, where the
    ballast-equivalent population lives) -- same X/Y roles `build_partition` already uses,
    so nothing downstream of the partition (assign_values, B_HALF, rank_ctx, ...) needs to
    change to support a non-50/50 split.

    `n_in = int(in_fraction * n)`, NOT `round(in_fraction * n)`: `round()` uses banker's
    rounding and disagrees with `n // 2` for some of this project's actual pool sizes (e.g.
    n=135: round(67.5) -> 68, but n // 2 -> 67). `int()` truncates toward zero, which matches
    `n // 2` exactly for positive n at in_fraction=0.5 -- the property `_self_test_fraction`
    below checks directly, not just argues.
    """
    rng = np.random.default_rng(seed)
    halves = []
    for k in range(NUM_ATTRIBUTES):
        n = int(pop.num_values_per_attr[k])
        perm = rng.permutation(n)
        n_in = int(in_fraction * n)
        halves.append((np.sort(perm[:n_in]), np.sort(perm[n_in:])))
    return halves


def build_partition_fraction_scaled(pop: BiographyPopulation, seed: int,
                                    in_fraction: float, pool_scale: float):
    """Like `build_partition_fraction`, but first restricts each attribute's pool to a
    random `pool_scale` fraction of its values before splitting -- the IPV (individuals-
    per-value) control for the exclusion-fraction sweep. Holds in_fraction at 0.5 (equal
    halves, no exclusion asymmetry) while shrinking both halves' pool size, so a change in
    trough depth here can only be about crowding, not about exclusion fraction.

    `rng.permutation(n)[:active_n]` is already both a random subset AND randomly ordered
    (permutation shuffles before truncating), so splitting it directly into `[:n_in]` /
    `[n_in:]` needs no second permutation call.

    Deliberately does NOT special-case pool_scale=1.0 to reproduce
    `build_partition_fraction`'s RNG consumption -- this function is never meant to
    reproduce that one's output (it has its own identity via the partition file's content
    hash in the checkpoint key, per `pretrain_key`), so there is nothing to preserve by
    matching call patterns. It happens to coincide anyway at pool_scale=1.0, since
    `rng.permutation(n)[:n] == rng.permutation(n)`; the self-test checks this as a bonus
    consistency property, not the guarantee actually needed.
    """
    rng = np.random.default_rng(seed)
    halves = []
    for k in range(NUM_ATTRIBUTES):
        n = int(pop.num_values_per_attr[k])
        active_n = int(pool_scale * n)
        active = rng.permutation(n)[:active_n]
        n_in = int(in_fraction * active_n)
        halves.append((np.sort(active[:n_in]), np.sort(active[n_in:])))
    return halves


def build_partition_fraction_windowed(pop: BiographyPopulation, seed: int,
                                      in_fraction: float, out_fraction: float):
    """The region-size-vs-exclusion-fraction control: A/B's region ("in") is fixed at
    `in_fraction` of the pool; ballast's region ("out") is a SEPARATE, independently-sized
    window taken immediately after it from the SAME seeded permutation, sized
    `out_fraction`. Anything beyond `in_fraction + out_fraction` is left unused (in neither
    population), matching the pattern `build_partition_fraction_scaled` already uses.

    Because "in" is always `perm[:in_n]` off the same seed, holding in_fraction fixed
    across several calls (e.g. 0.25 every time, varying only out_fraction) gives the SAME
    exact in-region values every time, not just the same size -- confirmed directly
    (see the self-test) against `build_partition_fraction_scaled(pop, seed, 0.5, 0.5)`,
    which is the in_fraction=out_fraction=0.25 case of this function by construction. And
    since out windows all start at the same `in_n`, a family of calls with fixed
    in_fraction and increasing out_fraction gives NESTED out-regions (each one a superset
    of the smaller one's), not just size-matched independent draws.
    """
    assert in_fraction + out_fraction <= 1.0 + 1e-9, (
        f"in_fraction ({in_fraction}) + out_fraction ({out_fraction}) exceeds 1 -- "
        f"the two regions would have to overlap.")
    rng = np.random.default_rng(seed)
    halves = []
    for k in range(NUM_ATTRIBUTES):
        n = int(pop.num_values_per_attr[k])
        perm = rng.permutation(n)
        n_in = int(in_fraction * n)
        n_out = int(out_fraction * n)
        halves.append((np.sort(perm[:n_in]), np.sort(perm[n_in:n_in + n_out])))
    return halves


def _self_test_partition_fraction(pop: BiographyPopulation, seed: int = 7):
    """Must run, and pass, before any exclusion-fraction sweep job is trusted."""
    old = build_partition(pop, seed)
    new_half = build_partition_fraction(pop, seed, 0.5)
    for k in range(NUM_ATTRIBUTES):
        assert np.array_equal(old[k][0], new_half[k][0]), (
            f"attr {k}: build_partition_fraction(..., 0.5) X differs from build_partition")
        assert np.array_equal(old[k][1], new_half[k][1]), (
            f"attr {k}: build_partition_fraction(..., 0.5) Y differs from build_partition")
    print("  self-test OK: build_partition_fraction(pop, seed, 0.5) == build_partition(pop, seed) exactly")

    new_f25 = build_partition_fraction(pop, seed, 0.25)
    scaled = build_partition_fraction_scaled(pop, seed, 0.5, 0.5)
    for k in range(NUM_ATTRIBUTES):
        want = int(0.25 * int(pop.num_values_per_attr[k]))
        got_f25 = len(new_f25[k][0])
        got_scaled = len(scaled[k][0])
        assert got_f25 == want, f"attr {k}: in_fraction=0.25 X size {got_f25} != {want}"
        assert got_scaled == want, (
            f"attr {k}: pool_scale=0.5,in_fraction=0.5 X size {got_scaled} != {want} "
            f"(the IPV control's derived match to the sweep's f=0.25 extreme)")
    print("  self-test OK: pool_scale=0.5 + in_fraction=0.5 gives the SAME half-pool-size as "
          "in_fraction=0.25 at full pool_scale, for every attribute -- the derived s=0.5 match")

    scaled_full = build_partition_fraction_scaled(pop, seed, 0.5, 1.0)
    for k in range(NUM_ATTRIBUTES):
        assert np.array_equal(old[k][0], scaled_full[k][0]) and np.array_equal(old[k][1], scaled_full[k][1]), (
            f"attr {k}: build_partition_fraction_scaled(..., 0.5, pool_scale=1.0) should "
            f"coincide with build_partition (bonus consistency check, not the primary guarantee)")
    print("  self-test OK (bonus): pool_scale=1.0 coincides with build_partition, as expected "
          "from rng.permutation(n)[:n] == rng.permutation(n)")

    # The region-size-vs-exclusion-fraction control (concentration follow-up). NOTE: an
    # earlier version of this self-test asserted build_partition_fraction_windowed(...,
    # 0.25, 0.25) reproduces the already-run IPV control exactly -- that is FALSE for
    # attributes with odd active_n under build_partition_fraction_scaled (major, company:
    # int(0.5 * int(0.5*n)) floors to one fewer than int(0.25*n) there), caught by this
    # very self-test before anything was built on the wrong assumption. Config 1 of the
    # concentration follow-up therefore reuses the IPV control's ALREADY-BUILT file
    # directly rather than regenerating it through this function. What this function
    # actually needs to guarantee -- and does -- is internal to itself: "in" stays
    # identical while out_fraction grows, and the out-regions nest.
    win_25 = build_partition_fraction_windowed(pop, seed, 0.25, 0.25)
    win_50 = build_partition_fraction_windowed(pop, seed, 0.25, 0.50)
    for k in range(NUM_ATTRIBUTES):
        assert np.array_equal(win_25[k][0], win_50[k][0]), (
            f"attr {k}: the 'in' region must stay identical as out_fraction grows")
        assert set(win_25[k][1].tolist()) <= set(win_50[k][1].tolist()), (
            f"attr {k}: the smaller 'out' region (out_fraction=0.25) must be a subset of "
            f"the larger one (out_fraction=0.5) -- both windows start at the same n_in")
    print("  self-test OK: build_partition_fraction_windowed's 'in' region is identical "
          "regardless of out_fraction, and 'out' regions nest as out_fraction grows")


def save_partition(path: str, halves, pop: BiographyPopulation, seed: int):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    arrays = {"seed": np.array(seed), "num_values_per_attr": np.asarray(pop.num_values_per_attr)}
    for k, (x_idx, y_idx) in enumerate(halves):
        arrays[f"attr_{k}_X"], arrays[f"attr_{k}_Y"] = x_idx, y_idx
    np.savez(path, **arrays)
    print(f"Wrote value partition to {path}")


def load_partition(path: str, pop: BiographyPopulation):
    data = np.load(path)
    stored = data["num_values_per_attr"]
    if not np.array_equal(stored, np.asarray(pop.num_values_per_attr)):
        raise ValueError(
            f"Partition at {path} was built for pools {stored.tolist()}, but the current "
            f"population has {np.asarray(pop.num_values_per_attr).tolist()}. The pools "
            f"changed; delete the partition and rebuild, and wipe checkpoints too."
        )
    return [(data[f"attr_{k}_X"], data[f"attr_{k}_Y"]) for k in range(NUM_ATTRIBUTES)]


def get_partition(cfg: KIConfig, pop: BiographyPopulation):
    if cfg.partition_path and os.path.exists(cfg.partition_path):
        halves = load_partition(cfg.partition_path, pop)
        print(f"Loaded value partition from {cfg.partition_path}")
        return halves
    halves = build_partition(pop, cfg.partition_seed)
    if cfg.partition_path:
        save_partition(cfg.partition_path, halves, pop, cfg.partition_seed)
    return halves


def assign_values(pop, person_ids, halves, which, rng, support_size):
    """Redraw `value_probs` for these people, restricted to one half of each pool.

    Redraw rather than mask-and-renormalise: with support_size=1 a person's single support
    value may sit in the excluded half, which would leave an all-zero probability row.
    `which="mixed"` picks a half per (person, attribute), giving ~50% expected overlap plus
    within-condition variation for the measured-overlap regression to exploit.
    """
    if which not in ("X", "Y", "mixed", "all"):
        raise ValueError(f"Unknown value half: {which}")
    person_ids = np.asarray(person_ids)
    for k in range(NUM_ATTRIBUTES):
        x_idx, y_idx = halves[k]
        rows = np.zeros((len(person_ids), pop.value_probs[k].shape[1]), dtype=np.float64)
        for r in range(len(person_ids)):
            if which == "mixed":
                allowed = x_idx if rng.random() < 0.5 else y_idx
            elif which == "all":
                allowed = np.concatenate([x_idx, y_idx])
            else:
                allowed = x_idx if which == "X" else y_idx
            s = min(support_size, len(allowed))
            rows[r, rng.choice(allowed, size=s, replace=False)] = 1.0 / s
        pop.value_probs[k][person_ids] = rows


NAME_PARTS = ("first", "middle", "last")


def assign_graded_b_names(pop: BiographyPopulation, ids_a: np.ndarray, ids_b: np.ndarray,
                           rng: np.random.Generator):
    """Overwrite B's (first, middle, last) name assignment so A's shared-part count with B
    is a real, engineered distribution across {0,1,2,3} rather than the chance
    distribution's ~75% concentration at 3.

    For each part independently, a "used-in-B" token subset is greedily built (random
    draw order, so no systematic bias toward low-index tokens) until it covers at least
    half of A's *actual* per-part usage -- not half the raw pool, which would only equal
    half of A's population if usage were perfectly uniform. B's 500 name triples are then
    drawn exclusively from these three subsets. The three subsets are chosen
    independently, so an A individual's shared-part count is ~Binomial(3, 0.5) --
    matching how names are drawn everywhere else in this project (three independent
    part draws). Forcing exactly-even quartiles would require constructing the three
    subsets non-independently, correlating which first name and which last name an
    individual has -- new structure this project doesn't otherwise assume.

    Only pop.person_names[ids_b] is touched; A's, ballast's, and C's rows are untouched,
    so a pretrained checkpoint built without this flag remains valid -- no re-pretraining
    needed.
    """
    names_a = pop.person_names[ids_a]
    used_in_b = []
    for j, part in enumerate(NAME_PARTS):
        vals, counts = np.unique(names_a[:, j], return_counts=True)
        order = rng.permutation(len(vals))
        vals, counts = vals[order], counts[order]
        cum = np.cumsum(counts)
        target = len(ids_a) / 2
        k = int(np.searchsorted(cum, target)) + 1
        used_in_b.append(vals[:k])
        print(f"  graded B names: {part} used-in-B size={len(used_in_b[-1])} "
              f"(of pool {len(pop.name_ids[part])}), covers {int(cum[k - 1])}/{len(ids_a)} of A")

    combos = set()
    triples = []
    while len(triples) < len(ids_b):
        triple = tuple(int(rng.choice(used_in_b[j])) for j in range(3))
        if triple not in combos:
            combos.add(triple)
            triples.append(triple)
    pop.person_names[ids_b] = np.array(triples, dtype=np.int32)

    shared = np.zeros((len(ids_a), 3), dtype=bool)
    for j in range(3):
        shared[:, j] = np.isin(names_a[:, j], used_in_b[j])
    shared_count = shared.sum(axis=1)
    counts_hist = {c: int((shared_count == c).sum()) for c in range(4)}
    print(f"  graded B names: shared_count distribution across A = {counts_hist}")


# --- logging helpers ---

def weight_norms(params) -> dict:
    """Per-leaf and global L2 norms. Logged from step 0: if weight decay 0.1 is wrong at
    this scale, it shows up here before it shows up in the loss."""
    out, total = {}, 0.0
    for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]:
        name = "/".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in path)
        norm = float(jnp.linalg.norm(leaf))
        out[f"wnorm/{name}"] = norm
        total += norm ** 2
    out["wnorm/global"] = float(np.sqrt(total))
    return out


def population_baseline_nats(pop: BiographyPopulation) -> float:
    """No-knowledge baseline: mean log of post-filter pool sizes (4.969 nats on the
    current pools). Computed from `pop`, not hardcoded, so it tracks the actual data."""
    return float(np.mean(np.log(pop.num_values_per_attr)))


def own_half_baseline_nats(halves, own_half: str) -> float:
    """No-knowledge baseline restricted to one value half — the reference point
    `retention_own` needs, analogous to `population_baseline_nats` but scoped."""
    sizes = [len(halves[k][0 if own_half == "X" else 1]) for k in range(NUM_ATTRIBUTES)]
    return float(np.mean(np.log(sizes)))


def token_half_groups(pop: BiographyPopulation, halves):
    """Vocab-wide token partition: unambiguous X, unambiguous Y, ambiguous (a token that is
    X for one attribute and Y for another -- birthplace/work_location share the CITIES pool
    but are partitioned independently, so ~half of city tokens carry no single well-defined
    half at the token level), and non-value (everything else: template/structural tokens).
    Shared by analyze_weight_localization.py and analyze_head_decomposition.py so the two
    diagnostics can't silently drift apart on what counts as which group.
    """
    x_union = set(int(t) for k in range(NUM_ATTRIBUTES) for t in pop.attr_first_token_ids[k][halves[k][0]])
    y_union = set(int(t) for k in range(NUM_ATTRIBUTES) for t in pop.attr_first_token_ids[k][halves[k][1]])
    ambiguous_ids = x_union & y_union
    x_ids = x_union - ambiguous_ids
    y_ids = y_union - ambiguous_ids
    non_value_ids = set(range(pop.vocab_size)) - x_ids - y_ids - ambiguous_ids
    return x_ids, y_ids, ambiguous_ids, non_value_ids


def first_token_positions(mask: np.ndarray, targets: np.ndarray, k: int):
    """Row-aligned (col, correct_token_id) for attribute k's first value-token position.
    Every biography has exactly one such position per attribute (all 6 appear once per
    biography, order permuted), so this must find exactly one match per row.
    """
    rows, cols = np.where(mask == (k + 1))
    order = np.argsort(rows, kind="stable")
    rows, cols = rows[order], cols[order]
    assert len(rows) == mask.shape[0] and np.array_equal(rows, np.arange(mask.shape[0])), (
        f"attribute {k}: expected exactly one first-token position per row; "
        f"got {len(rows)} for {mask.shape[0]} rows."
    )
    return cols, targets[rows, cols]


def rank_and_loss_within(row_logits: np.ndarray, correct_token: np.ndarray, candidate_ids: np.ndarray):
    """1-indexed rank (1=top) AND cross-entropy loss of the correct token, both computed
    with the softmax restricted to `candidate_ids` — not the full vocabulary. This is
    what separates suppression (argmax dies, rank/loss within the own half do not) from
    real corruption (both die together).
    """
    cand_logits = row_logits[:, candidate_ids]
    correct_logit = row_logits[np.arange(len(correct_token)), correct_token]
    rank = (cand_logits > correct_logit[:, None]).sum(axis=1) + 1
    m = cand_logits.max(axis=1)
    logsumexp = m + np.log(np.exp(cand_logits - m[:, None]).sum(axis=1))
    loss = logsumexp - correct_logit
    return rank, loss


def population_rank_metrics(state, ds, own_half: str, x_ids, y_ids, baseline_own: float,
                             l_pretrained_own: float = None, batch_size=256) -> dict:
    """Own-half-restricted rank/loss/retention for one dataset, aggregated over its full
    eval set. `own_half` must be "X" or "Y" — the half THIS population's people actually
    draw values from (never "mixed"; the `partial` condition needs a different, per-person
    treatment not implemented here).
    """
    n = ds.eval_inputs.shape[0]
    ranks, losses = [], []
    for start in range(0, n, batch_size):
        sl = slice(start, min(start + batch_size, n))
        logits = np.asarray(eval_forward(state.apply_fn, state.params, jnp.array(ds.eval_inputs[sl])))
        targets = np.asarray(ds.eval_targets[sl])
        mask = np.asarray(ds.eval_mask[sl])
        for k in range(NUM_ATTRIBUTES):
            cols, correct = first_token_positions(mask, targets, k)
            row_logits = logits[np.arange(sl.stop - sl.start), cols, :]
            own_ids = x_ids[k] if own_half == "X" else y_ids[k]
            rank, loss = rank_and_loss_within(row_logits, correct, own_ids)
            ranks.append(rank)
            losses.append(loss)
    rank_arr = np.concatenate(ranks)
    loss_own = float(np.concatenate(losses).mean())
    out = {
        f"{ds.name}/rank_own_mean": float(rank_arr.mean()),
        f"{ds.name}/rank_own_top1": float((rank_arr == 1).mean()),
        f"{ds.name}/attribute_loss_own": loss_own,
    }
    if l_pretrained_own is not None:
        denom = baseline_own - l_pretrained_own
        out[f"{ds.name}/retention_own"] = (
            (baseline_own - loss_own) / denom if abs(denom) > 1e-6 else float("nan"))
    return out


def evaluate_and_log(state, eval_sets, cfg, gstep, extra=None, baseline=None, l_pretrained=None,
                     rank_ctx=None, l_pretrained_own=None) -> dict:
    """Evaluate and log. When `baseline` is given, also logs `{name}/hallucination_gap` =
    attribute_loss - baseline for every dataset (Zucchet §4.1's signature: positive means
    confidently WRONG, i.e. worse than a uniform guess over the value pool — the natural
    read for dataC, which neither phase ever trains on). When `l_pretrained` is also given
    (a dict of each dataset's attribute_loss at this run's own starting point), additionally
    logs `{name}/retention` = (baseline - current) / (baseline - l_pretrained[name]).
    Retention is undefined (logged as NaN) wherever the denominator is near zero — chiefly
    B and C, which have nothing to "retain" going into injection.

    `rank_ctx` (a dict with keys "x_ids", "y_ids", "own_halves") additionally logs, for
    every dataset with an entry in "own_halves", the PRIMARY metric:
    `{name}/rank_own_mean`, `{name}/rank_own_top1`, `{name}/attribute_loss_own`, and — once
    `l_pretrained_own` fixes this run's own starting point — `{name}/retention_own`. This
    is `retention` restricted to a softmax over the population's OWN value half only, which
    is unaffected by cross-half output-marginal suppression; `{name}/suppression_gap` =
    `retention_own - retention` is logged alongside as the size of that artifact directly.
    Both `retention` (full-vocab) and `retention_own` are kept — never replace one metric
    with another; report both.
    """
    forward = lambda x: eval_forward(state.apply_fn, state.params, x)
    metrics = {}
    for ds in eval_sets:
        metrics.update(ds.evaluate(forward, cfg.data.eval_batch_size))
    if baseline is not None:
        for ds in eval_sets:
            attr_loss = metrics[f"{ds.name}/attribute_loss"]
            metrics[f"{ds.name}/hallucination_gap"] = attr_loss - baseline
            if l_pretrained is not None and ds.name in l_pretrained:
                denom = baseline - l_pretrained[ds.name]
                metrics[f"{ds.name}/retention"] = (
                    (baseline - attr_loss) / denom if abs(denom) > 1e-6 else float("nan"))
    if rank_ctx is not None:
        for ds in eval_sets:
            own_half = rank_ctx["own_halves"].get(ds.name)
            if own_half not in ("X", "Y"):
                continue  # "mixed" (partial condition) or unset: not implemented here
            l_pre_own = l_pretrained_own.get(ds.name) if l_pretrained_own else None
            metrics.update(population_rank_metrics(
                state, ds, own_half, rank_ctx["x_ids"], rank_ctx["y_ids"],
                rank_ctx["baseline_own"][own_half], l_pre_own))
            if f"{ds.name}/retention_own" in metrics and f"{ds.name}/retention" in metrics:
                metrics[f"{ds.name}/suppression_gap"] = (
                    metrics[f"{ds.name}/retention_own"] - metrics[f"{ds.name}/retention"])
    if extra:
        metrics.update(extra)
    wandb.log(metrics, step=gstep)
    def _fmt(ds):
        line = (f"{ds.name}: first_acc={metrics[f'{ds.name}/first_token_accuracy']:.3f} "
                f"attr_acc={metrics[f'{ds.name}/attribute_accuracy']:.3f}")
        if f"{ds.name}/retention_own" in metrics:
            line += f" ret_own={metrics[f'{ds.name}/retention_own']:.3f}"
        elif f"{ds.name}/retention" in metrics:
            line += f" retention={metrics[f'{ds.name}/retention']:.3f}"
        if f"{ds.name}/rank_own_top1" in metrics:
            line += f" rank_own_top1={metrics[f'{ds.name}/rank_own_top1']:.3f}"
        if f"{ds.name}/hallucination_gap" in metrics:
            line += f" halluc_gap={metrics[f'{ds.name}/hallucination_gap']:+.3f}"
        return line
    print(f"[step {gstep}] " + ", ".join(_fmt(ds) for ds in eval_sets))
    return metrics


def save_exposures(cfg: KIConfig, dataset, tag: str):
    if not cfg.exposure_dir:
        return
    os.makedirs(cfg.exposure_dir, exist_ok=True)
    path = os.path.join(cfg.exposure_dir, f"{tag}.npz")
    counts = dataset.exposure_counts
    served = counts[dataset.person_ids]
    np.savez(path, exposure_counts=counts, person_ids=dataset.person_ids)
    print(f"Exposures -> {path}  (per person: mean {served.mean():.1f}, "
          f"min {served.min()}, max {served.max()})")


def parse_steps(spec: str, total: int) -> set:
    steps = {int(s) for s in spec.split(",") if s.strip()}
    return {s for s in steps if 0 <= s <= total} | {total}


def reset_stream(dataset, seed: int):
    """Rewind a dataset's batch stream and exposure counters.

    Needed because the LR sweep reuses one dataset object across grid points: without this
    each successive LR would see a different batch sequence (making the comparison unfair)
    and inherit the previous point's exposure counts.

    This reaches into BiographyDataset's private attributes deliberately. If they are ever
    renamed upstream we want a loud failure here, not a silently unreset stream — that
    would surface as a spurious learning-rate effect in the sweep.
    """
    for attr in ("_np_rng", "_cache", "_cache_idx", "exposure_counts"):
        if not hasattr(dataset, attr):
            raise AttributeError(
                f"BiographyDataset has no '{attr}'; reset_stream can no longer rewind its "
                f"batch stream. Upstream renamed it — fix reset_stream before running a "
                f"sweep, or grid points will share a stream and exposure counts."
            )
    dataset._np_rng = np.random.default_rng(seed)
    dataset._cache = None
    dataset._cache_idx = 0
    dataset.exposure_counts[:] = 0


# --- checkpoint metadata ---

def meta_path(ckpt: str) -> str:
    return (ckpt[: -len(".msgpack")] if ckpt.endswith(".msgpack") else ckpt) + ".meta.json"


def save_meta(ckpt: str, **fields):
    with open(meta_path(ckpt), "w", encoding="utf-8") as f:
        json.dump(fields, f, indent=2, sort_keys=True)


def load_meta(ckpt: str) -> dict:
    path = meta_path(ckpt)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No metadata at {path}. The injection LR is derived from the pretraining peak "
            f"recorded there; rerun --phase pretrain, or set --inject.opt.peak_lr explicitly."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# --- training ---

def train_loop(state, dataset, steps, batch_size, eval_sets, cfg, *, step0=0,
               eval_interval=100, ckpt_steps=(), ckpt_fn=None, baseline=None, track_retention=False,
               rank_ctx=None):
    """NTP loop with dense checkpoint hooks and weight-norm logging.

    Returns (state, final_train_loss) where final_train_loss is the mean over the last 100
    steps — the LR-selection criterion. Training batches are freshly rendered every step
    (`_generate` redraws templates and values), so this is already a generalisation measure
    over renderings; only the facts are fixed.

    `baseline`: no-knowledge nats, enables `hallucination_gap` logging for every dataset.
    `track_retention`: additionally fixes `L_pretrained` from THIS call's own step-0 eval
    (not from random init) and logs `retention` from then on — only meaningful when this
    loop is continuing from an already-trained checkpoint, i.e. injection, not pretraining.
    `rank_ctx`: enables `retention_own` (the primary metric) the same way —
    fixed from this call's own step-0 own-half loss, not random init.
    """
    metrics = evaluate_and_log(state, eval_sets, cfg, step0, extra=weight_norms(state.params),
                               baseline=baseline, rank_ctx=rank_ctx)
    l_pretrained = None
    l_pretrained_own = None
    step0_extra = {}
    if track_retention and baseline is not None:
        l_pretrained = {ds.name: metrics[f"{ds.name}/attribute_loss"] for ds in eval_sets}
        step0_extra.update({f"{ds.name}/retention": 1.0 for ds in eval_sets})
    if rank_ctx is not None:
        l_pretrained_own = {ds.name: metrics[f"{ds.name}/attribute_loss_own"] for ds in eval_sets
                            if f"{ds.name}/attribute_loss_own" in metrics}
        # l_pretrained_own is {name: loss}; iterate its keys (already the names), not .name
        # on them again -- they're strings, not Dataset objects.
        step0_extra.update({f"{name}/retention_own": 1.0 for name in l_pretrained_own})
        step0_extra.update({f"{name}/suppression_gap": 0.0 for name in l_pretrained_own
                            if f"{name}/retention" in metrics or track_retention})
    if step0_extra:
        wandb.log(step0_extra, step=step0)
        metrics.update(step0_extra)
    if 0 in ckpt_steps and ckpt_fn is not None:
        ckpt_fn(0, state)

    recent, nan_check = [], max(1, steps // 100)
    for i in range(steps):
        rng, batch_rng = jax.random.split(state.rng)
        state = state.replace(rng=rng)
        state, step_metrics = train_step(
            state, dataset.get_batch(batch_rng, batch_size), BiographyDataset.loss_fn)
        step, gstep = i + 1, step0 + i + 1

        # Keep device scalars; only sync at log points and at the end.
        recent.append(step_metrics["loss"])
        if len(recent) > 100:
            recent.pop(0)
        if step % nan_check == 0 and not bool(jnp.isfinite(step_metrics["loss"])):
            print(f"Non-finite loss at step {gstep}; stopping.")
            break
        if gstep % cfg.log_interval == 0:
            wandb.log({"train_loss": float(step_metrics["loss"])}, step=gstep)
        if step in ckpt_steps and ckpt_fn is not None:
            ckpt_fn(step, state)
        if step % eval_interval == 0 or step == steps:
            metrics = evaluate_and_log(state, eval_sets, cfg, gstep,
                                       extra=weight_norms(state.params),
                                       baseline=baseline, l_pretrained=l_pretrained,
                                       rank_ctx=rank_ctx, l_pretrained_own=l_pretrained_own)

    final_loss = float(jnp.mean(jnp.stack(recent))) if recent else float("nan")
    return state, final_loss, metrics


# --- setup ---

def build(cfg: KIConfig, max_eval_people: int):
    """Population, value partition, datasets, model, and the post-replace data config.

    Value assignment happens before any BiographyDataset is constructed, because each
    dataset pre-generates its eval set from `value_probs` in __init__.

    `max_eval_people` is per-phase, not global: pretraining only ranks configurations, so
    a subsample is fine there, while injection reports retention as a result and must
    evaluate on all of A.
    """
    num_people = cfg.num_a + cfg.num_ballast + cfg.num_b + cfg.num_c
    pop = BiographyPopulation(
        cfg.data.biography_data_path, num_people, seed=cfg.seed,
        support_size=cfg.data.support_size, num_train_templates=cfg.data.num_train_templates)

    halves = get_partition(cfg, pop)
    ids_a = np.arange(cfg.num_a)
    ids_ballast = np.arange(cfg.num_a, cfg.num_a + cfg.num_ballast)
    ids_b = np.arange(cfg.num_a + cfg.num_ballast, cfg.num_a + cfg.num_ballast + cfg.num_b)
    ids_c = np.arange(cfg.num_a + cfg.num_ballast + cfg.num_b, num_people)

    if cfg.inject.condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {cfg.inject.condition}; expected one of {CONDITIONS}")
    b_half = B_HALF[cfg.inject.condition]

    if cfg.inject.graded_key_overlap:
        # Overwrites only pop.person_names[ids_b] -- must happen before any BiographyDataset
        # is constructed for B (dataset __init__ pre-generates eval sequences from names),
        # but is independent of value assignment below (person_names and value_probs are
        # separate arrays).
        names_rng = np.random.default_rng(cfg.inject.graded_key_seed)
        assign_graded_b_names(pop, ids_a, ids_b, names_rng)

    # Order matters: each group is drawn from one shared RNG in sequence, so appending C
    # last leaves A / ballast / B bit-identical to a run without C.
    rng = np.random.default_rng(cfg.seed + 900)
    assign_values(pop, ids_a, halves, "X", rng, cfg.data.support_size)
    assign_values(pop, ids_ballast, halves, "Y", rng, cfg.data.support_size)
    assign_values(pop, ids_b, halves, b_half, rng, cfg.data.support_size)
    if len(ids_c):
        assign_values(pop, ids_c, halves, cfg.c_half, rng, cfg.data.support_size)

    ds = lambda ids, name, seed, cap=None: BiographyDataset(
        pop, ids, name=name, seed=seed,
        max_eval_people=max_eval_people if cap is None else cap)

    # The union dataset is the training sampler only; its eval set is never used, so it is
    # capped to one person rather than regenerating A ∪ ballast. Note C is NOT in it.
    data_pre = ds(np.concatenate([ids_a, ids_ballast]), "pretrainset", cfg.seed + 1, cap=1)
    data_a = ds(ids_a, "dataA", cfg.seed + 2)
    data_ballast = ds(ids_ballast, "ballast", cfg.seed + 3)
    data_b = ds(ids_b, "dataB", cfg.seed + 4)
    data_c = ds(ids_c, "dataC", cfg.seed + 5) if len(ids_c) else None

    data_cfg = replace(cfg.data, sequence_length=pop.seq_len, vocab_size=pop.vocab_size)
    model = create_model(cfg.model, data_cfg)
    print(f"Vocab {pop.vocab_size}, seq_len {pop.seq_len}, prompt_len {pop.prompt_len}")
    print(f"People: A={len(ids_a)}, ballast={len(ids_ballast)}, B={len(ids_b)}, "
          f"C={len(ids_c)} (never trained); condition={cfg.inject.condition} "
          f"(B draws from {b_half}, C from {cfg.c_half})")
    return pop, model, data_cfg, data_pre, data_a, data_ballast, data_b, data_c


def init_state(model, cfg, data_cfg, tx, seed):
    full = Config(model=cfg.model, data=data_cfg, train=TrainConfig(), seed=seed)
    return create_train_state(model, full, jax.random.PRNGKey(seed), None, tx)


def pretrain_key(cfg, data_cfg):
    """Key for the pretrained params.

    `num_c` and `c_half` are in here even though C is never trained on: population size
    feeds `BiographyPopulation`, and changing it shifts RNG consumption in `_assign_values`,
    which moves every individual's train/eval template split — including A's. Verified in
    `tests/test_stream_determinism.py` [7]. `max_eval_people`, `eval_interval`, and
    `target_acc` are all stripped: none of them affect the trained parameters, only eval
    cadence and pass/fail reporting, so changing any of them must not force a retrain.
    Verified in `tests/test_stream_determinism.py` [8].

    `partition_hash` -- content hash of the file at `cfg.partition_path`, not just
    `partition_seed`. Two different partitions (e.g. two different exclusion fractions, or
    a pool-scaled partition for the IPV control) can share the same seed and population
    sizes while assigning completely different values to completely different people; pool
    sizes (hence every parameter shape) are identical either way, so the load-time shape
    assertion cannot catch this collision -- only the key can. Called after `build()`,
    which creates the partition file via `get_partition` if it does not already exist, so
    the file is guaranteed to be on disk by the time this hashes it.
    """
    pretrain_for_key = replace(cfg.pretrain, max_eval_people=0, eval_interval=0, target_acc=0.0)
    partition_hash = file_hash(cfg.partition_path) if cfg.partition_path and os.path.exists(cfg.partition_path) else None
    return experiment_key(
        cfg.model, data_cfg, cfg.data.biography_data_path, phase="pretrain",
        pretrain=pretrain_for_key, seed=cfg.seed,
        population=[cfg.num_a, cfg.num_ballast, cfg.num_b, cfg.num_c],
        c_half=cfg.c_half, partition_seed=cfg.partition_seed, partition_hash=partition_hash)


# --- phases ---

def phase_pretrain(cfg: KIConfig, built=None):
    pop, model, data_cfg, data_pre, data_a, data_ballast, _, data_c = built or build(
        cfg, cfg.pretrain.max_eval_people)
    reset_stream(data_pre, cfg.seed + 1)
    tx = make_optimizer(cfg.pretrain.opt, cfg.pretrain.total_steps)
    state = init_state(model, cfg, data_cfg, tx, cfg.seed + 10)
    print(f"Model parameters: {sum(x.size for x in jax.tree.leaves(state.params)):,}")

    key = pretrain_key(cfg, data_cfg)
    path = ckpt_path(cfg.checkpoint_dir, "pretrain", key)
    baseline = population_baseline_nats(pop)
    with wandb.init(project=cfg.wandb_project, mode=cfg.wandb_mode, name="pretrain",
                    config=vars_nested(cfg)):
        state, final_loss, metrics = train_loop(
            state, data_pre, cfg.pretrain.total_steps, cfg.pretrain.batch_size,
            [d for d in (data_a, data_ballast, data_c) if d], cfg,
            eval_interval=cfg.pretrain.eval_interval, baseline=baseline)
    acc = metrics["dataA/first_token_accuracy"]
    ballast_acc = metrics["ballast/first_token_accuracy"]
    if path:
        save_state(path, state)
        # ballast_first_token_accuracy is recorded so injection can gate on it too: the
        # A-vs-ballast contrast only isolates B's writes if BOTH start near ceiling.
        save_meta(path, peak_lr=cfg.pretrain.opt.peak_lr, schedule=cfg.pretrain.opt.schedule,
                  total_steps=cfg.pretrain.total_steps, batch_size=cfg.pretrain.batch_size,
                  final_train_loss=final_loss, dataA_first_token_accuracy=acc,
                  ballast_first_token_accuracy=ballast_acc, baseline_nats=baseline, key=key)
        print(f"Saved {path} (+ {os.path.basename(meta_path(path))})")
    save_exposures(cfg, data_pre, f"pretrain-{key}")

    gate_passed = acc >= cfg.pretrain.target_acc and ballast_acc >= cfg.pretrain.target_acc
    print(f"\n=== GATE === dataA/first_token_accuracy = {acc:.4f}, "
          f"ballast/first_token_accuracy = {ballast_acc:.4f} (target {cfg.pretrain.target_acc}), "
          f"final train loss = {final_loss:.4f}")
    if not gate_passed:
        failed = [n for n, a in [("dataA", acc), ("ballast", ballast_acc)] if a < cfg.pretrain.target_acc]
        print(f"BELOW TARGET ({', '.join(failed)}) — facts not learned, or the A-vs-ballast "
              f"contrast would be unfair. Injection results would be uninterpretable. Not proceeding.")
    return state, final_loss, acc, ballast_acc, key


def phase_inject(cfg: KIConfig, built=None):
    pop, model, data_cfg, _, data_a, data_ballast, data_b, data_c = built or build(
        cfg, cfg.inject.max_eval_people)
    pre_key = pretrain_key(cfg, data_cfg)
    pre_path = ckpt_path(cfg.checkpoint_dir, "pretrain", pre_key)
    if not pre_path or not os.path.exists(pre_path):
        raise FileNotFoundError(
            f"No pretraining checkpoint at {pre_path}. Run --phase pretrain first "
            f"(the key covers model, data content, population sizes, and partition seed).")

    # Read metadata whenever it exists, not only when deriving the LR: the ballast gate
    # below needs it regardless of how the LR was chosen.
    meta = load_meta(pre_path) if os.path.exists(meta_path(pre_path)) else None

    inject_opt = cfg.inject.opt
    if inject_opt.peak_lr <= 0:
        if meta is None:
            raise FileNotFoundError(
                f"No metadata at {meta_path(pre_path)}. The injection LR is derived from "
                f"the pretraining peak recorded there; rerun --phase pretrain, or set "
                f"--inject.opt.peak_lr explicitly.")
        derived = float(meta["peak_lr"]) / INJECT_LR_RATIO
        inject_opt = replace(inject_opt, peak_lr=derived)
        print(f"Injection LR derived: pretrain peak {float(meta['peak_lr']):g} / "
              f"{INJECT_LR_RATIO:.4g} = {derived:g} (constant)")
    else:
        print(f"Injection LR set explicitly: {inject_opt.peak_lr:g} ({inject_opt.schedule})")
    inject_cfg = replace(cfg.inject, opt=inject_opt)

    # Gate: the A-vs-ballast contrast in this run is only fair if BOTH started near
    # ceiling. Checked against each other directly, not just against target_acc in
    # isolation — "ballast had less to lose than A" is a claim about their relationship,
    # not about ballast alone, so it has to be verified against A's own recorded value.
    if meta is not None and "ballast_first_token_accuracy" in meta and "dataA_first_token_accuracy" in meta:
        a_acc = float(meta["dataA_first_token_accuracy"])
        ballast_acc = float(meta["ballast_first_token_accuracy"])
        print(f"Pretrain-time accuracy: dataA={a_acc:.4f}, ballast={ballast_acc:.4f} "
              f"(target {cfg.pretrain.target_acc})")
        a_ok, ballast_ok = a_acc >= cfg.pretrain.target_acc, ballast_acc >= cfg.pretrain.target_acc
        if a_ok and ballast_ok:
            print("Gate OK — both started near ceiling; the A-vs-ballast contrast is fair.")
        elif a_ok and not ballast_ok:
            print("WARNING: ballast started below target while A did not — ballast had "
                  "less knowledge to lose, so 'ballast stays flat' would not be strong "
                  "evidence of specificity.")
        elif ballast_ok and not a_ok:
            print("WARNING: A started below target while ballast did not — the measured "
                  "population itself is undertrained; the whole injection result would "
                  "be built on a checkpoint that failed its own gate.")
        else:
            print("WARNING: BOTH A and ballast started below target — this checkpoint "
                  "failed phase_pretrain's own gate. Injection results from it are not "
                  "just an unfair A-vs-ballast contrast, they are not interpretable at all.")
    elif meta is not None:
        print("NOTE: this checkpoint's metadata predates the ballast gate "
              "(no ballast_first_token_accuracy recorded) — rerun pretrain to get it checked.")

    tx = make_optimizer(inject_opt, inject_cfg.total_steps)
    state = init_state(model, cfg, data_cfg, tx, cfg.seed + 20)
    params, saved_opt_state, pre_step = load_state(pre_path, state)
    state = state.replace(params=params)
    if not inject_cfg.fresh_optimizer:
        state = state.replace(opt_state=inherit_moments(state.opt_state, saved_opt_state))
        print("Optimizer: inherited mu/nu from pretraining, step and schedule reset")
    else:
        print("Optimizer: fresh")

    inject_set = data_a if inject_cfg.condition == "identical" else data_b
    reset_stream(inject_set, cfg.seed + 5 + inject_cfg.seed)
    eval_sets = [d for d in (data_a, data_b, data_ballast, data_c) if d]
    ckpt_steps = parse_steps(inject_cfg.checkpoint_steps, inject_cfg.total_steps)
    key = experiment_key(
        cfg.model, data_cfg, cfg.data.biography_data_path, phase="inject",
        inject=replace(inject_cfg, max_eval_people=0),  # eval-only; see pretrain_key
        pretrain_key=pre_key, seed=cfg.seed)
    print(f"Injecting on {inject_set.name}, checkpoints at {sorted(ckpt_steps)}")

    def ckpt_fn(step, st):
        if not cfg.checkpoint_dir:
            return
        p = os.path.join(cfg.checkpoint_dir, f"inject-{key}-step{step:04d}.msgpack")
        if inject_cfg.save_opt_state:
            save_state(p, st)
        else:
            save_params(p, st.params)
        print(f"  checkpoint step {step} -> {p}")

    baseline = population_baseline_nats(pop)

    # rank_ctx: enables retention_own (the primary metric), logged for every population
    # whose own half is unambiguous. "partial" gives mixed X/Y per person
    # and is not handled by this population-level own_half — skipped there, not guessed at.
    halves = get_partition(cfg, pop)
    x_ids = [pop.attr_first_token_ids[k][halves[k][0]] for k in range(NUM_ATTRIBUTES)]
    y_ids = [pop.attr_first_token_ids[k][halves[k][1]] for k in range(NUM_ATTRIBUTES)]
    b_half = B_HALF[inject_cfg.condition]
    own_halves = {"dataA": "X", "ballast": "Y"}
    if b_half in ("X", "Y"):
        own_halves["dataB"] = b_half
    if cfg.c_half in ("X", "Y"):
        own_halves["dataC"] = cfg.c_half
    rank_ctx = {
        "x_ids": x_ids, "y_ids": y_ids, "own_halves": own_halves,
        "baseline_own": {"X": own_half_baseline_nats(halves, "X"),
                         "Y": own_half_baseline_nats(halves, "Y")},
    }

    with wandb.init(project=cfg.wandb_project, mode=cfg.wandb_mode, name="inject",
                    config=vars_nested(cfg)):
        state, final_loss, _ = train_loop(
            state, inject_set, inject_cfg.total_steps, inject_cfg.batch_size, eval_sets, cfg,
            step0=pre_step, eval_interval=inject_cfg.eval_interval,
            ckpt_steps=ckpt_steps, ckpt_fn=ckpt_fn, baseline=baseline, track_retention=True,
            rank_ctx=rank_ctx)
    save_exposures(cfg, inject_set, f"inject-{key}")
    print(f"Final inject train loss = {final_loss:.4f}")
    return state


def phase_sweep(cfg: KIConfig):
    """Pretraining peak-LR grid, selected on final training loss."""
    built = build(cfg, cfg.pretrain.max_eval_people)
    results = []
    for lr in PRETRAIN_LR_GRID:
        print(f"\n{'=' * 60}\nPretrain sweep: peak_lr = {lr:g}\n{'=' * 60}")
        sweep_cfg = replace(cfg, pretrain=replace(
            cfg.pretrain, opt=replace(cfg.pretrain.opt, peak_lr=lr)))
        _, final_loss, acc, ballast_acc, _ = phase_pretrain(sweep_cfg, built=built)
        results.append((lr, final_loss, acc, ballast_acc))

    print(f"\n{'=' * 60}\nSweep summary\n{'=' * 60}")
    print(f"{'peak_lr':>10}  {'final train loss':>17}  {'dataA first_acc':>16}  {'ballast first_acc':>17}")
    for lr, loss, acc, ballast_acc in results:
        print(f"{lr:>10.1e}  {loss:>17.4f}  {acc:>16.4f}  {ballast_acc:>17.4f}")
    best = min(results, key=lambda r: r[1])
    print(f"\nSelected peak_lr = {best[0]:g} (final train loss {best[1]:.4f})")
    if best[0] in (PRETRAIN_LR_GRID[0], PRETRAIN_LR_GRID[-1]):
        print("WARNING: the optimum is at a grid edge. Either extend the grid, or treat "
              "this as evidence that weight decay 0.1 is mistuned at this scale.")
    return best


def phase_profile(cfg: KIConfig):
    """Throughput profile, separating data generation from the training step.

    `get_batch` refills a cache of `batch_size * 64` biographies through a pure-Python
    loop in `_generate` — 16,384 rows at batch 256. On a fast accelerator that refill can
    become the real bottleneck, and it appears as a stall once every 64 steps rather than
    as uniformly slow steps, so averaging step time alone would hide it. Refills are
    predicted before each call and timed separately.
    """
    import time

    print(f"JAX {jax.__version__} | backend {jax.default_backend()} | devices {jax.devices()}")
    if jax.default_backend() == "cpu":
        print("WARNING: running on CPU. On a GPU node this means the CUDA jaxlib is "
              "missing (`uv sync` installs the CPU build) — the numbers below do not "
              "describe GPU throughput.")
    _, model, data_cfg, data_pre, *_ = build(cfg, cfg.profile.max_eval_people)
    reset_stream(data_pre, cfg.seed + 1)
    pcfg = cfg.profile
    tx = make_optimizer(cfg.pretrain.opt, max(1, pcfg.steps))
    state = init_state(model, cfg, data_cfg, tx, cfg.seed + 10)
    n_params = sum(x.size for x in jax.tree.leaves(state.params))
    print(f"Params {n_params:,} | batch {pcfg.batch_size} | seq {data_cfg.sequence_length - 1} "
          f"| cache {pcfg.batch_size * BiographyDataset.CACHE_MULTIPLIER:,} biographies/refill")

    refill_t, serve_t, step_t = [], [], []
    for i in range(pcfg.steps):
        cache = data_pre._cache
        will_refill = cache is None or data_pre._cache_idx + pcfg.batch_size > cache[0].shape[0]
        rng, batch_rng = jax.random.split(state.rng)
        state = state.replace(rng=rng)

        t0 = time.perf_counter()
        batch = data_pre.get_batch(batch_rng, pcfg.batch_size)
        t1 = time.perf_counter()
        state, _ = train_step(state, batch, BiographyDataset.loss_fn)
        jax.block_until_ready(state.params)  # JAX is async; without this we time dispatch
        t2 = time.perf_counter()

        if i >= pcfg.warmup_steps:
            (refill_t if will_refill else serve_t).append(t1 - t0)
            step_t.append(t2 - t1)
        if i == 0:
            print(f"  first step {t2 - t0:.2f}s (includes JIT compilation)")

    step = float(np.mean(step_t))
    serve = float(np.mean(serve_t)) if serve_t else 0.0
    per_refill = float(np.mean(refill_t)) if refill_t else float("nan")
    steps_served = BiographyDataset.CACHE_MULTIPLIER
    amortized = (per_refill / steps_served) if refill_t else 0.0
    wall = step + serve + amortized

    print(f"\n{'=' * 64}\nThroughput ({len(step_t)} timed steps, {pcfg.warmup_steps} warmup dropped)\n{'=' * 64}")
    print(f"  train step          {step * 1e3:8.2f} ms   ({1 / step:7.2f} steps/s compute-only)")
    print(f"  cached batch serve  {serve * 1e3:8.2f} ms")
    print(f"  cache refill        {per_refill:8.2f} s    x{len(refill_t)} observed, "
          f"serves {steps_served} steps -> {amortized * 1e3:.2f} ms/step amortized")
    print(f"  effective           {wall * 1e3:8.2f} ms   ({1 / wall:7.2f} steps/s end-to-end)")
    print(f"  generation share    {100 * (serve + amortized) / wall:8.1f} %  of wall time")

    if refill_t:
        budget = steps_served * step
        starving = per_refill > budget
        print(f"\n  refill {per_refill:.2f}s vs the {steps_served} steps it serves ({budget:.2f}s) -> "
              f"{'GPU IS STARVING — vectorizing _generate is the top optimization' if starving else 'generation keeps up'}")

    projected = 8000 * step + (8000 / steps_served) * per_refill if refill_t else 8000 * step
    print(f"\n  projected 8000-step pretrain: {projected / 60:.1f} min ({projected / 3600:.2f} h)")
    print(f"  projected 5-point LR sweep, parallel jobs: {projected / 60:.1f} min wall")

    try:
        stats = jax.local_devices()[0].memory_stats() or {}
        peak = stats.get("peak_bytes_in_use")
        if peak:
            print(f"\n  peak device memory  {peak / 2**30:.2f} GiB")
        else:
            print("\n  peak device memory  unavailable on this backend (CPU reports none)")
    except Exception as exc:  # noqa: BLE001 - diagnostics only, never fail the profile
        print(f"\n  peak device memory  unavailable ({type(exc).__name__})")


def main():
    cfg = parse_config(KIConfig, description="Knowledge injection and interference")
    print(cfg)
    if cfg.phase == "pretrain":
        phase_pretrain(cfg)
    elif cfg.phase == "inject":
        phase_inject(cfg)
    elif cfg.phase == "sweep":
        phase_sweep(cfg)
    elif cfg.phase == "profile":
        phase_profile(cfg)
    elif cfg.phase == "all":
        # 'all' reports retention, so it builds at the injection phase's eval size.
        built = build(cfg, cfg.inject.max_eval_people)
        _, _, acc, ballast_acc, _ = phase_pretrain(cfg, built=built)
        if acc < cfg.pretrain.target_acc or ballast_acc < cfg.pretrain.target_acc:
            return
        phase_inject(cfg, built=built)
    else:
        raise ValueError(f"Unknown phase: {cfg.phase}")


if __name__ == "__main__":
    main()
