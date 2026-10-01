"""Every knob of the real-entity injection run, in one place.

Separate from llm/ on purpose: llm/ is the original OLMo replication and stays frozen. The
only things shared are data, never code -- llm/out/gate.json (the A set) is read here, and
the scoring definitions are re-implemented, not imported, so a change on either side cannot
silently move the other.
"""
import argparse
from dataclasses import asdict, dataclass, field, fields


@dataclass
class RunConfig:
    # --- model ------------------------------------------------------------------------------
    model_id: str = "allenai/OLMo-2-0425-1B"
    # --- data -------------------------------------------------------------------------------
    b_path: str = "llm_real/out/b_real_gated.json"      # from `run.py gate`
    generic_path: str = "llm_real/out/generic_tokens.npy"  # from `run.py fetch-generic`
    a_gate: str = "llm/out/gate.json"                     # the A set, shared with llm/
    counterfact_id: str = "NeelNanda/counterfact-tracing"
    b_ratio: float = 1.0          # fraction of training tokens drawn from B documents (1.0 = llm/ regime)
    augment: bool = True          # article prose + templated fact sentences; False = article prose only
    corpus_b: bool = False        # b_path is a document corpus ({docs, held}), not a fact table:
                                  # the real-text arm; B is read as token accuracy / loss on its
                                  # own documents instead of fact accuracy
    audit_format: bool = True     # refuse corpora with QA / list / key-value register;
                                  # off for domain corpora whose native register trips it (PubMed's
                                  # "BACKGROUND:" headers, code's "else:" lines) -- the domain shift
                                  # IS the experiment there, so the register rule does not apply
    # --- optimisation -----------------------------------------------------------------------
    lr: float = 1e-5
    steps: int = 6000
    warmup: int = 100
    tokens_per_step: int = 32768
    micro_batch: int = 16         # rows per forward/backward; memory only
    seq_len: int = 512
    grad_checkpoint: bool = True
    clip: float = 1.0
    betas: tuple = (0.9, 0.95)
    seed: int = 0
    # --- evaluation and checkpoints ---------------------------------------------------------
    eval_batch: int = 256
    eval_dense_until: int = 300
    eval_geometric: float = 1.35
    ckpt_every_eval: bool = True  # save model+optimizer at every geometric eval point (resume)
    keep_last_ckpt_only: bool = True
    out_dir: str = "llm_real/out"
    tag: str = ""                 # free-form suffix for the run name

    @property
    def name(self):
        aug = "corpus" if self.corpus_b else ("aug" if self.augment else "strict")
        t = f"_{self.tag}" if self.tag else ""
        return f"real_{aug}_r{self.b_ratio:g}_lr{self.lr:g}_seed{self.seed}{t}"

    def to_dict(self):
        d = asdict(self)
        d["betas"] = list(self.betas)
        return d


def add_args(ap: argparse.ArgumentParser, cfg: RunConfig = RunConfig()):
    """One flag per field, typed from the default. Booleans take --flag / --no-flag."""
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        if isinstance(v, bool):
            ap.add_argument(f"--{f.name}", dest=f.name, action="store_true", default=v)
            ap.add_argument(f"--no-{f.name}", dest=f.name, action="store_false")
        elif isinstance(v, tuple):
            ap.add_argument(f"--{f.name}", type=float, nargs=len(v), default=v)
        else:
            ap.add_argument(f"--{f.name}", type=type(v), default=v)
    return ap


def from_args(ns) -> RunConfig:
    kw = {f.name: getattr(ns, f.name) for f in fields(RunConfig) if hasattr(ns, f.name)}
    if "betas" in kw:
        kw["betas"] = tuple(kw["betas"])
    return RunConfig(**kw)
