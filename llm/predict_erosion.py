"""Can the base model, with no training, predict which old facts will be suppressed and which
will erode? (OLMo 2 1B; the toy's laws read from the pretrained representations.)

THE LAWS BEING TESTED (paper Section 3). With keys
k = sqrt(beta) mu + sqrt(1 - beta) g, one step on the new facts B moves old fact a by

    dh_a = -eta/|B| sum_b <k_a, k_b> r_b
         = common part (through the shared component, beta)  +  individual part (<g_a, g_b>)

so (i) the SUPPRESSION of fact a is the common write read against a's own margin over B's
answers, and (ii) the EROSION of fact a is proportional to its individual key overlap with
the new facts, <g_a, g_b>. Both are functions of the pretrained model and the two fact sets
alone. This script computes them from forward passes of the base model and tests them
against the per-fact trajectories already recorded by llm/inject_decomp.py.

PREDICTORS, per old fact a, at several taps (residual after layers 4/8/12, and the readout
state after the final norm), from the base model's states on A's prompts and B's prompts:

    beta_mean      mean_b cos(x_a, x_b)                     the shared component (beta-hat)
    cos_centered   the same after removing the global mean  the individual overlap <g_a, g_b>
    cos_max        max_b of the centered cosine             the nearest new fact
    proj_k         ||P_k x~_a|| / ||x~_a||, P_k = top-k principal directions of B's centered
                   states, k in PROJ_K                      how much of a's key B's keys span
    push           mean_b cos(x_a, x_b) (||u_{y_b}||^2 - <u_{y_a}, u_{y_b}>)   [readout tap]
                   the first-order rise of B's answers over a's own answer, from the linear
                   readout term of the gradient (paper eq. 1-2), in logit units
    push_ratio     push / margin0, margin0 = z_a[y_a] - max_{v in B's answer tokens} z_a[v]
                   the push against the margin it has to overcome

OUTCOMES, per old fact, from the saved readout states at every eval step of the decomposition
runs (three seeds), read through the pretrained unembedding U(0) as in decomp_offline.py:
correct (argmax over the vocabulary), rank within the fact's own pool, and margin_B(t) = the
logit margin of the fact's answer over the best of B's answer tokens. Suppression outcome:
correctness and margin drop at the trough. Erosion outcome: mean rank over steps >= EROSION_FROM
and rank at the last step.

STATISTICS. Spearman of every predictor with every outcome, on the non-copy stratum and on
all facts; the same within relation (rank-transformed inside each relation of >= MIN_GROUP
facts, then pooled) so relation membership cannot carry it; trivial controls alongside
(rank and margin at step 0, prompt length, pool size, ||x_a||); a permutation null (sd of
Spearman under shuffled predictor). Quartile trajectories of the predictors are left to the
plot script; the per-fact arrays are saved.

PRE-REGISTERED READING (written before the run). The laws are supported if, in every seed,
(a) push_ratio predicts suppression at the trough above the controls, and (b) cos_centered or
proj_k predicts the erosion outcome above the controls and does NOT predict the trough beyond
what the relation carries. If neither holds the result is recorded and the base-model
predictor claim is dropped; the laws stand as laws of the toy.

DATASET-LEVEL NUMBERS, also written: beta-hat (mean cosine over all A x B pairs, per tap),
n_B, the cosine between the unembedding centroids of B's and A's answers (value competition),
and the median push_ratio -- the quantities a "will suppression show up" prediction for a new
corpus would be made from. With one corpus they are reported, not calibrated.

Runs on CPU (the forward passes are the only cost: ~9k short prompts through the 1B model),
one job for all seeds:

  .venv-llm/bin/python -m llm.predict_erosion \
      --states llm/out/decomp_lr1e-05_seed0_states.npz,llm/out/decomp_lr1e-05_seed1_states.npz,llm/out/decomp_lr1e-05_seed2_states.npz
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.evalsets import build_a, build_b
from llm.inject import DATASET_ID, MODEL_ID

TAP_LAYERS = (4, 8, 12)          # residual stream after these blocks; plus "readout" (post final norm)
PROJ_K = (4, 16, 64)
MIN_GROUP = 15
EROSION_FROM = 400
N_PERM = 200


# ----------------------------------------------------------------------------- states ----
@torch.no_grad()
def tapped_states(model, tok, prompts, batch_size=32):
    """Last-position states at the taps: {tap: (n, d)} fp32. 'readout' is last_hidden_state
    (post final norm, what lm_head reads); 'L4' etc. are hidden_states[L] (residual after
    block L, before any norm)."""
    model.eval()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    out = {f"L{L}": [] for L in TAP_LAYERS}; out["readout"] = []
    t0 = time.time()
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False)
        o = model.model(**enc, output_hidden_states=True)
        for L in TAP_LAYERS:
            out[f"L{L}"].append(o.hidden_states[L][:, -1, :].float())
        out["readout"].append(o.last_hidden_state[:, -1, :].float())
        if (i // batch_size) % 20 == 0:
            print(f"    {i + len(enc['input_ids'])}/{len(prompts)} prompts  {time.time() - t0:.0f}s", flush=True)
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def render_train_prompt(template, name):
    """The training sentence up to the value slot: 'X worked at {v}.' -> 'X worked at'."""
    head = template.split("{v}")[0].replace("{name}", name).rstrip()
    return head


# -------------------------------------------------------------------------- predictors ----
def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def predictors_at_tap(xa, xb):
    """Per-A-fact overlap measures with B's states at one tap (torch for the big matmuls)."""
    A, B = torch.from_numpy(xa), torch.from_numpy(xb)
    Au, Bu = torch.from_numpy(unit(xa)), torch.from_numpy(unit(xb))
    beta_mean = (Au @ Bu.T).mean(1).numpy()
    mean_all = torch.cat([A, B]).mean(0, keepdim=True)
    Ac, Bc = A - mean_all, B - mean_all
    Acu, Bcu = torch.from_numpy(unit(Ac.numpy())), torch.from_numpy(unit(Bc.numpy()))
    C = Acu @ Bcu.T
    cos_centered = C.mean(1).numpy(); cos_max = C.max(1).values.numpy()
    # principal directions of B's centered states
    _, _, Vt = torch.linalg.svd(Bc - Bc.mean(0, keepdim=True), full_matrices=False)
    out = {"beta_mean": beta_mean, "cos_centered": cos_centered, "cos_max": cos_max}
    for k in PROJ_K:
        P = Vt[:k]                                         # (k, d)
        out[f"proj_{k}"] = ((Acu @ P.T).norm(dim=1)).numpy()
    return out


def push_predictors(xa, xb, U, ya, yb, b_pool_tokens):
    """The first-order rise of B's answers over a's own answer, from the linear readout term:
    push_a = mean_b cos(x_a, x_b) (||u_{y_b}||^2 - <u_{y_a}, u_{y_b}>); margin0_a = a's logit
    margin over the best B answer token; push_ratio = push / margin0."""
    Au, Bu = torch.from_numpy(unit(xa)), torch.from_numpy(unit(xb))
    cos = Au @ Bu.T                                          # (nA, nB)
    Ut = torch.from_numpy(U)
    u_yb = Ut[torch.from_numpy(yb)]                          # (nB, d)
    u_ya = Ut[torch.from_numpy(ya)]                          # (nA, d)
    g = (u_yb * u_yb).sum(1)[None, :] - (u_ya @ u_yb.T)      # (nA, nB): ||u_yb||^2 - <u_ya, u_yb>
    push = (cos * g).mean(1).numpy()
    z = torch.from_numpy(xa) @ Ut.T                          # (nA, V) base-model logits
    mine = z[torch.arange(len(ya)), torch.from_numpy(ya)]
    best_b = z[:, torch.from_numpy(np.array(sorted(b_pool_tokens)))].max(1).values
    margin0 = (mine - best_b).numpy()
    ratio = push / np.maximum(margin0, 0.05)
    return {"push": push, "margin0": margin0, "push_ratio": ratio}


# ---------------------------------------------------------------------------- outcomes ----
def read_all(U_T, x, items, pools, b_pool_tokens):
    """Per-fact correct / own-pool rank / margin over B's answer tokens for states x (n, d)."""
    logits = (torch.from_numpy(np.ascontiguousarray(x)) @ U_T).numpy()
    top = logits.argmax(1)
    n = len(items)
    correct = np.zeros(n); rank = np.full(n, np.nan); marginB = np.zeros(n)
    bp = np.array(sorted(b_pool_tokens))
    for j, it in enumerate(items):
        correct[j] = 1.0 if int(top[j]) == it["first_token"] else 0.0
        mine = logits[j, it["first_token"]]
        marginB[j] = mine - logits[j, bp].max()
        if it["rank_eligible"]:
            cand = logits[j, pools[it["relation_id"]]]
            rank[j] = 1.0 + int((cand > mine).sum()) + 0.5 * max(int((cand == mine).sum()) - 1, 0)
    return correct, rank, marginB


# -------------------------------------------------------------------------- statistics ----
def rankdata(v):
    v = np.asarray(v, float); n = len(v)
    order = np.argsort(v, kind="mergesort"); r = np.empty(n)
    r[order] = np.arange(n, dtype=float)
    # average ties
    sv = v[order]; i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return float("nan")
    rx, ry = rankdata(x[m]), rankdata(y[m])
    rx -= rx.mean(); ry -= ry.mean()
    d = math.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def spearman_within(x, y, groups):
    """Rank-transform x and y inside each group of >= MIN_GROUP facts, then one Spearman
    over the pooled within-group ranks (centred per group)."""
    xs, ys = [], []
    for g in set(groups):
        gi = np.where(groups == g)[0]
        m = gi[np.isfinite(x[gi]) & np.isfinite(y[gi])]
        if len(m) < MIN_GROUP:
            continue
        rx, ry = rankdata(x[m]), rankdata(y[m])
        xs.append((rx - rx.mean()) / max(len(m) - 1, 1)); ys.append((ry - ry.mean()) / max(len(m) - 1, 1))
    if not xs:
        return float("nan")
    return spearman(np.concatenate(xs), np.concatenate(ys))


def perm_sd(x, y, rng):
    m = np.isfinite(x) & np.isfinite(y)
    xs, ys = x[m], y[m]
    return float(np.std([spearman(rng.permutation(xs), ys) for _ in range(N_PERM)]))


# --------------------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="", help="comma-separated decomp_*_states.npz, one per seed")
    ap.add_argument("--gate", default="llm/out/gate.json")
    ap.add_argument("--facts", default="llm/out/b_facts.json")
    ap.add_argument("--out", default="llm/out/predict_erosion_lr1e-05.json")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_outcomes", action="store_true",
                    help="predictors and dataset-level numbers only (no state dumps needed): "
                         "the prediction for a corpus BEFORE its run")
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    rng = np.random.default_rng(a.seed); random.seed(a.seed)
    t_start = time.time()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    model.eval()
    U0 = model.lm_head.weight.detach().numpy().astype(np.float32)
    U_T = torch.from_numpy(np.ascontiguousarray(U0.T))
    ds = load_dataset(DATASET_ID, split="train")
    a_items, a_pools, a_info = build_a(tok, a.gate, ds)
    b_items, b_pools, facts = build_b(tok, a.facts)
    b_pool_tokens = set(t for ids in b_pools.values() for t in ids)
    print(f"A: {len(a_items)} facts; B: {len(b_items)} eval items, {len(b_pool_tokens)} answer tokens; "
          f"model+data loaded in {time.time() - t_start:.0f}s", flush=True)

    # ---- base-model states: A prompts, B eval prompts, B training renderings (one per fact)
    print("A prompts", flush=True)
    XA = tapped_states(model, tok, [it["prompt"] for it in a_items], a.batch)
    print("B eval prompts", flush=True)
    XB = tapped_states(model, tok, [it["prompt"] for it in b_items], a.batch)
    tt = facts["train_templates"]
    # names: the eval table carries them; the 4,000 eval prompts are unique, so index by prompt
    name_of = {e["prompt"]: e["name"] for e in facts["eval"]}
    b_train_prompts = [render_train_prompt(random.choice(tt[it["attr"]]), name_of[it["prompt"]]) for it in b_items]
    print("B training renderings, e.g. " + repr(b_train_prompts[0]), flush=True)
    XBt = tapped_states(model, tok, b_train_prompts, a.batch)
    del model
    ya = np.array([it["first_token"] for it in a_items]); yb = np.array([it["first_token"] for it in b_items])
    taps = list(XA.keys())
    np.savez_compressed(a.out.replace(".json", "_states.npz"),
                        **{f"A/{t}": XA[t].astype(np.float16) for t in taps},
                        **{f"B/{t}": XB[t].astype(np.float16) for t in taps},
                        **{f"Btrain/{t}": XBt[t].astype(np.float16) for t in taps})

    # ---- predictors
    pred = {}
    for t in taps:
        for src, XBx in (("eval", XB), ("train", XBt)):
            for k, v in predictors_at_tap(XA[t], XBx[t]).items():
                pred[f"{t}/{src}/{k}"] = v
    for src, XBx in (("eval", XB), ("train", XBt)):
        for k, v in push_predictors(XA["readout"], XBx["readout"], U0, ya, yb, b_pool_tokens).items():
            pred[f"readout/{src}/{k}"] = v
    # controls
    pred["ctrl/prompt_len"] = np.array([len(tok(it["prompt"], add_special_tokens=False)["input_ids"]) for it in a_items], float)
    pred["ctrl/pool_size"] = np.array([it["pool_size"] for it in a_items], float)
    pred["ctrl/x_norm"] = np.linalg.norm(XA["readout"], axis=1)
    print(f"predictors computed at {time.time() - t_start:.0f}s", flush=True)

    # dataset-level numbers
    level = {"n_B": len(facts["eval"]), "n_B_people": len(facts["people"])}
    for t in taps:
        level[f"beta_hat/{t}"] = float(pred[f"{t}/eval/beta_mean"].mean())
        level[f"cos_centered_mean/{t}"] = float(pred[f"{t}/eval/cos_centered"].mean())
    ua = unit(U0[ya]).mean(0); ub = unit(U0[yb]).mean(0)
    level["value_competition_cos"] = float(ua @ ub / (np.linalg.norm(ua) * np.linalg.norm(ub)))
    level["push_ratio_median"] = float(np.median(pred["readout/eval/push_ratio"]))
    level["margin0_median"] = float(np.median(pred["readout/eval/margin0"]))

    copy = np.array([it["copy"] for it in a_items]); rel = np.array([it["relation_id"] for it in a_items])
    noncopy = copy == "noncopy"
    # the prediction, per relation: facts whose answer is inside B's answer set have margin <= 0
    # (the write lifts their own answer); the rest face B's region across their margin
    m0 = pred["readout/eval/margin0"]; inside = np.array([it["first_token"] in b_pool_tokens for it in a_items])
    level["frac_A_inside_B_region"] = float(inside.mean())
    level["frac_A_noncopy_inside_B_region"] = float(inside[noncopy].mean())
    by_rel = {}
    for r_ in sorted(set(rel), key=lambda r_: -int((rel == r_).sum())):
        mi = rel == r_
        by_rel[r_] = {"n": int(mi.sum()), "n_noncopy": int((mi & noncopy).sum()),
                      "inside_B_region": float(inside[mi].mean()),
                      "margin_median": float(np.median(m0[mi])),
                      "push_ratio_median": float(np.median(pred["readout/eval/push_ratio"][mi]))}
    level["by_relation"] = by_rel
    print("prediction by relation (n, frac inside B's region, median margin over B's answers):")
    for r_, v in by_rel.items():
        if v["n"] >= 10:
            print(f"  {r_:>6} n={v['n']:>4} inside={v['inside_B_region']:.2f} margin={v['margin_median']:+.2f} push/margin={v['push_ratio_median']:+.2f}")
    print(f"A inside B's region: {level['frac_A_inside_B_region']:.2f} (non-copy {level['frac_A_noncopy_inside_B_region']:.2f})")
    if a.no_outcomes:
        json.dump({"facts": a.facts, "taps": taps, "level": level,
                   "predictors": {k: v.tolist() for k, v in pred.items()},
                   "copy": copy.tolist(), "relation": rel.tolist(),
                   "inside_B_region": inside.astype(int).tolist()},
                  open(a.out, "w"), indent=1)
        print(f"saved (predictors only) -> {a.out}  ({time.time() - t_start:.0f}s)")
        return

    # ---- outcomes from the decomposition runs' saved states
    per_seed = []
    steps_ref = None
    for path in a.states.split(","):
        z = np.load(path, allow_pickle=True)
        steps = [int(s) for s in z["steps"]]; X = z["x"].astype(np.float32); x0 = z["x0"].astype(np.float32)
        assert X.shape[1] == len(a_items), (X.shape, len(a_items))
        if steps_ref is None:
            steps_ref = steps
            # tap check: the saved readout state against the fresh one
            c = (unit(x0) * unit(XA["readout"])).sum(1)
            print(f"tap check: cos(saved x0, fresh readout) mean {c.mean():.4f} min {c.min():.4f}", flush=True)
        assert steps == steps_ref
        C = np.zeros((len(steps), len(a_items))); R = np.zeros_like(C); M = np.zeros_like(C)
        for ti in range(len(steps)):
            C[ti], R[ti], M[ti] = read_all(U_T, X[ti], a_items, a_pools, b_pool_tokens)
        per_seed.append((C, R, M))
        acc = C[:, noncopy].mean(1)
        print(f"{os.path.basename(path)}: noncopy through U(0): " + " ".join(f"{s}:{v:.2f}" for s, v in zip(steps, acc)), flush=True)
    Cm = np.mean([p[0] for p in per_seed], 0); Rm = np.mean([p[1] for p in per_seed], 0); Mm = np.mean([p[2] for p in per_seed], 0)
    t_tr = int(np.argmin(Cm[:, noncopy].mean(1)))
    late = [i for i, s in enumerate(steps_ref) if s >= EROSION_FROM]
    outcomes = {
        "trough/correct": Cm[t_tr], "trough/rank": Rm[t_tr], "trough/margin_drop": Mm[0] - Mm[t_tr],
        "erosion/rank_late_mean": np.nanmean(Rm[late], 0), "erosion/rank_last": Rm[-1],
        "erosion/correct_late_mean": Cm[late].mean(0), "erosion/margin_drop_last": Mm[0] - Mm[-1],
    }
    pred["ctrl/rank0"] = Rm[0]; pred["ctrl/margin0_run"] = Mm[0]
    print(f"trough at step {steps_ref[t_tr]}; erosion window steps {[steps_ref[i] for i in late]}", flush=True)

    # ---- statistics
    stats = {}
    for pname, pv in pred.items():
        for oname, ov in outcomes.items():
            row = {}
            for sname, mask in (("noncopy", noncopy), ("all", np.ones(len(a_items), bool))):
                x, y = pv[mask], ov[mask]
                row[f"{sname}/rho"] = spearman(x, y)
                row[f"{sname}/rho_within_relation"] = spearman_within(x, y, rel[mask])
                if sname == "noncopy":
                    row[f"{sname}/perm_sd"] = perm_sd(x, y, rng)
            # per seed, non-copy, for the pre-registered "in every seed" criterion
            for si, (C, R, M) in enumerate(per_seed):
                ov_s = {"trough/correct": C[t_tr], "trough/rank": R[t_tr], "trough/margin_drop": M[0] - M[t_tr],
                        "erosion/rank_late_mean": np.nanmean(R[late], 0), "erosion/rank_last": R[-1],
                        "erosion/correct_late_mean": C[late].mean(0), "erosion/margin_drop_last": M[0] - M[-1]}[oname]
                row[f"seed{si}/rho"] = spearman(pv[noncopy], ov_s[noncopy])
            stats[f"{pname} -> {oname}"] = row

    # ---- report the headline table
    print("\nSpearman, non-copy (within-relation in brackets; perm sd ~ {:.3f})".format(
        np.median([v["noncopy/perm_sd"] for v in stats.values()])))
    heads = ["trough/correct", "trough/margin_drop", "erosion/rank_late_mean", "erosion/margin_drop_last"]
    print(f"{'predictor':40s}" + "".join(f"{h:>28s}" for h in heads))
    for pname in pred:
        cells = []
        for h in heads:
            r = stats[f"{pname} -> {h}"]
            cells.append(f"{r['noncopy/rho']:+.3f} [{r['noncopy/rho_within_relation']:+.3f}]")
        print(f"{pname:40s}" + "".join(f"{c:>28s}" for c in cells))
    print("\ndataset level: " + json.dumps(level))

    json.dump({"states": a.states.split(","), "steps": steps_ref, "trough_step": steps_ref[t_tr],
               "erosion_from": EROSION_FROM, "taps": taps, "level": level, "stats": stats,
               "predictors": {k: v.tolist() for k, v in pred.items()},
               "outcomes": {k: np.nan_to_num(v, nan=-1).tolist() for k, v in outcomes.items()},
               "traj_correct": Cm.tolist(), "traj_rank": np.nan_to_num(Rm, nan=-1).tolist(),
               "copy": copy.tolist(), "relation": rel.tolist()},
              open(a.out, "w"), indent=1)
    print(f"saved -> {a.out}  ({time.time() - t_start:.0f}s total)")


if __name__ == "__main__":
    main()
