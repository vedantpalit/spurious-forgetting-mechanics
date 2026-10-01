"""Inject B into OLMo 2 1B by continued pretraining, and watch A.

THE QUESTION. Does A's accuracy crash and then spontaneously recover? Nothing else.

REGISTER. Training is continued pretraining on prose documents rendered from B's fact table --
the same operation and the same kind of string OLMo saw throughout pretraining. A is read out
with CounterFact's cloze probe, which is NEVER trained on: training on the probe would practise
the measuring instrument and contaminate the crash depth. The residual gap (prose in, probe out)
is a stated property of the design, and the generic-text control is what tests it.

DOCUMENTS ARE RENDERED FRESH EVERY STEP. A fixed corpus is memorisable -- the first B wrote one
document per person with one phrasing per fact, hit training loss 0.006 by rote, and scored at
chance on a held-out phrasing. Here the fact repeats and the string does not: sentence order is
permuted and each sentence draws a random training phrasing at render time. Training loss should
therefore plateau well above zero; if it collapses toward zero, memorisation is still winning and
that number is where it shows.

WHAT IS MEASURED, every eval:
  * A accuracy, NONCOPY stratum -- THE HEADLINE. 62% of A is answerable by copying the answer out
                       of the prompt, which the gate cannot filter. The copy stratum is reported
                       beside it because copying is an in-context operation rather than
                       parametric recall, and whether it crashes differently is informative.
  * A mean rank     -- of the correct answer within its own-region candidate pool. Accuracy
                       crashing while rank holds is the suppression signature, and it is what
                       turns "the curve reproduces" into "the mechanism reproduces".
  * A per stratum   -- copy/noncopy, P176/P178/other, and b_typed/untouched (relations of a
                       semantic type B injects into, versus the rest, which is a built-in
                       unrelated contrast inside the same run).
  * B accuracy      -- on HELD-OUT phrasings, which is what caught the memorisation failure.
                       Recovery tracks B's acquisition rather than step count, so a run
                       where B never learns is not scoreable either way.

EVAL SCHEDULE. Heavily dense early: the whole event lives before step ~300.

Run:
  .venv-llm/bin/python -m llm.inject --lr 3e-5 --steps 300
"""
import argparse
import json
import math
import os
import random

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.evalsets import build_a, build_b

MODEL_ID = "allenai/OLMo-2-0425-1B"
DATASET_ID = "NeelNanda/counterfact-tracing"

# Dense before 300, where the small model's whole event lives, then geometric so a long run is
# readable at all. Extended past 1000 because B's acquisition rate now sets the run length and it
# is not yet known -- guessing the length was never the plan, measuring it is.
_DENSE = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100, 125, 150, 175, 200,
          250, 300, 400, 500, 700, 1000]


def eval_schedule(total):
    pts, s = [p for p in _DENSE if p <= total], 1000
    while s < total:
        s = int(s * 1.4)
        pts.append(min(s, total))
    return sorted(set(pts) | {0, total})

# The small experiments injected at peak_lr / (40/3): 4e-4 pretrain, 3e-5 inject. Pass
# --pretrain_lr to have the matched injection LR printed for whatever OLMo 2 1B documents.
INJECT_LR_RATIO = 40 / 3


class DocSampler:
    """Renders B's documents fresh, so no string is ever trained on twice by construction."""

    def __init__(self, facts, tok, seq_len, seed):
        self.people = facts["people"]
        self.attrs = facts["attrs"]
        self.templates = facts["train_templates"]
        self.tok, self.seq_len = tok, seq_len
        self.rng = random.Random(seed)

    def document(self):
        p = self.rng.choice(self.people)
        sents = [self.rng.choice(self.templates[at]).format(name=p["name"], v=p[at])
                 for at in self.attrs if at in p]          # real people carry only the facts they have
        self.rng.shuffle(sents)
        return " ".join(sents)

    def batch(self, batch_size):
        rows = []
        for _ in range(batch_size):
            ids = []
            while len(ids) < self.seq_len:
                ids.extend(self.tok(self.document() + "\n\n",
                                    add_special_tokens=False).input_ids)
            rows.append(ids[:self.seq_len])
        return torch.tensor(rows)


@torch.no_grad()
def score(model, tok, items, pools, device, batch_size=128):
    """First-token accuracy over the full vocabulary, and mean within-pool rank."""
    model.eval()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = [it["prompt"] for it in items]
    acc, rank = [], []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            out = model(**enc)
        logits = out.logits[:, -1, :].float()
        top = logits.argmax(-1).tolist()
        for j, it in enumerate(items[i:i + batch_size]):
            acc.append(1.0 if top[j] == it["first_token"] else 0.0)
            if not it["rank_eligible"]:
                rank.append(float("nan"))
                continue
            ids = pools[it.get("relation_id") or it["attr"]]
            mine = logits[j, it["first_token"]]
            cand = logits[j, ids]
            greater = int((cand > mine).sum().item())
            tied = int((cand == mine).sum().item()) - 1
            rank.append(1.0 + greater + 0.5 * max(tied, 0))
    return acc, rank


def summarise(acc, rank, items, label, key):
    """Accuracy with SE and mean rank, overall and per level of `key`."""
    out = {}
    for s in ["ALL"] + sorted({it[key] for it in items}):
        idx = [k for k, it in enumerate(items) if s == "ALL" or it[key] == s]
        if not idx:
            continue
        p = sum(acc[k] for k in idx) / len(idx)
        out[f"{label}/{s}/acc"] = p
        out[f"{label}/{s}/acc_se"] = math.sqrt(max(p * (1 - p), 0) / len(idx))
        out[f"{label}/{s}/n"] = len(idx)
        rs = [rank[k] for k in idx if not math.isnan(rank[k])]
        if rs:
            out[f"{label}/{s}/rank"] = sum(rs) / len(rs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=4, help="micro-batch; memory only")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--tokens_per_step", type=int, default=32768,
                    help="optimizer step size in tokens, reached by accumulation")
    ap.add_argument("--warmup", type=int, default=100, help="linear LR warmup steps")
    ap.add_argument("--eval_batch", type=int, default=128)
    ap.add_argument("--pretrain_lr", type=float, default=None,
                    help="OLMo 2 1B's documented peak LR; prints the matched injection LR")
    ap.add_argument("--gate", default="llm/out/gate.json")
    ap.add_argument("--facts", default="llm/out/b_facts.json")
    ap.add_argument("--model_id", default=MODEL_ID,
                    help="HF id; the gate (--gate) must have been built on the same model")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out_path = a.out or f"llm/out/inject_lr{a.lr:g}_seed{a.seed}.json"

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  lr={a.lr:g}  steps={a.steps}  batch={a.batch_size}x{a.seq_len}")
    if a.pretrain_lr:
        print(f"matched ratio {INJECT_LR_RATIO:.4g}: pretrain {a.pretrain_lr:g} -> "
              f"inject {a.pretrain_lr / INJECT_LR_RATIO:g}  (running at {a.lr:g})")

    tok = AutoTokenizer.from_pretrained(a.model_id)
    # fp32 MASTER WEIGHTS, bf16 only for the matmuls. Loading the model in bf16 and running
    # AdamW on those parameters directly is what broke the first sweep: measured 2026-09-04,
    # ten steps at lr=1e-6 moved 0.82% of parameters against fp32's 86%, and at 3e-5 moved 13%
    # against 86% -- and the survivors were the small-magnitude weights whose ULP the update
    # could clear, so it was a biased sparse update rather than a small one. embed_tokens was
    # frozen outright (0.00% vs fp32's 0.52%).
    model = AutoModelForCausalLM.from_pretrained(a.model_id, dtype=torch.float32).to(device)

    ds = load_dataset(DATASET_ID, split="train")
    a_items, a_pools, a_info = build_a(tok, a.gate, ds)
    b_items, b_pools, facts = build_b(tok, a.facts)
    # A facts whose answer TYPE matches an attribute B actually carries: the facts B competes
    # with (companies: P176/P178/P108 <- employer or maker; cities: P159/P36 <- birth_city;
    # countries: P27 <- citizenship; languages: P103/P364 <- language)
    TYPE_OF = {"employer": ("P176", "P178", "P108"), "maker": ("P176", "P178", "P108"),
               "birth_city": ("P159", "P36"), "citizenship": ("P27",), "language": ("P103", "P364"),
               "country": ("P27",)}       # EntityQuestions P17: places -> countries
    comp = {rid for at in facts["attrs"] for rid in TYPE_OF.get(at, ())}
    for it in a_items:
        it["compete"] = "same_type" if it["relation_id"] in comp else "other_type"
    print(f"A facts of B's answer types (compete=same_type): {sum(it['compete'] == 'same_type' for it in a_items)} "
          f"of {len(a_items)}  (relations {sorted(comp)})")
    print(f"A: {a_info['n_A']} facts, rank-eligible {a_info['rank_eligible']} "
          f"({a_info['rank_coverage']:.1%})")
    print(f"   copy {a_info['copy']}  |  strata {a_info['strata']}  |  {a_info['touched']}")
    print(f"B: {len(b_items)} eval items on held-out phrasings over {len(b_pools)} attributes")

    sampler = DocSampler(facts, tok, a.seq_len, a.seed)
    print(f"documents rendered fresh each step from {len(facts['people'])} people x "
          f"{len(facts['attrs'])} attributes; {a.batch_size * a.seq_len:,} tokens/step")

    micro_tokens = a.batch_size * a.seq_len
    accum = max(a.tokens_per_step // micro_tokens, 1)
    print(f"  {micro_tokens:,} tokens/micro-batch x {accum} accumulated = "
          f"{micro_tokens * accum:,} tokens/step; {a.warmup}-step linear warmup")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    schedule = eval_schedule(a.steps)
    curve = []

    def evaluate(step):
        aa, ar = score(model, tok, a_items, a_pools, device, a.eval_batch)
        ba, br = score(model, tok, b_items, b_pools, device, a.eval_batch)
        rec = {"step": step}
        rec.update(summarise(aa, ar, a_items, "A", "copy"))
        rec.update(summarise(aa, ar, a_items, "Arel", "stratum"))
        rec.update(summarise(aa, ar, a_items, "Atouch", "touched"))
        rec.update(summarise(aa, ar, a_items, "Acomp", "compete"))
        rec.update(summarise(ba, br, b_items, "B", "stratum"))
        # per-fact A outcomes, so any stratum can be read offline (1106 ints + floats per eval)
        rec["A/correct"] = [int(v) for v in aa]
        rec["A/rank_each"] = [None if math.isnan(v) else round(float(v), 2) for v in ar]
        curve.append(rec)
        print(f"  [step {step:>4}] A noncopy={rec['A/noncopy/acc']:.4f}"
              f"+-{rec['A/noncopy/acc_se']:.4f} rank={rec.get('A/noncopy/rank', float('nan')):.3f}"
              f" | copy={rec['A/copy/acc']:.4f} all={rec['A/ALL/acc']:.4f}"
              f" | b_typed={rec['Atouch/b_typed/acc']:.4f} "
              f"untouched={rec['Atouch/untouched/acc']:.4f}"
              f" | B={rec['B/ALL/acc']:.4f}")
        model.train()

    evaluate(0)
    for step in range(1, a.steps + 1):
        # Warmup because Adam's bias correction makes step 1 a full-size lr*sign(grad) update on
        # every parameter from a cold optimizer state -- a shock to a pretrained model, and a
        # plausible contributor to A collapsing inside the first ten steps.
        for gp in opt.param_groups:
            gp["lr"] = a.lr * min(step / max(a.warmup, 1), 1.0)
        total = 0.0
        for _ in range(accum):
            batch = sampler.batch(a.batch_size).to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(input_ids=batch, labels=batch).loss / accum
            loss.backward()
            total += float(loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step in schedule:
            print(f"    train loss {total:.4f}  (lr {opt.param_groups[0]['lr']:.3g})")
            evaluate(step)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"model": a.model_id, "facts": a.facts, "lr": a.lr, "steps": a.steps, "seed": a.seed, "batch_size": a.batch_size,
                   "a_relation": [it["relation_id"] for it in a_items],
                   "a_copy": [it["copy"] for it in a_items],
                   "a_compete": [it["compete"] for it in a_items],
                   "seq_len": a.seq_len, "tokens_per_step": micro_tokens * accum,
                   "warmup": a.warmup, "a_info": a_info, "curve": curve}, fh, indent=2)
    print(f"\nwrote {out_path}")

    # --- the reading, against the pre-registered options, on the NONCOPY headline -------------
    accs = [c["A/noncopy/acc"] for c in curve]
    steps = [c["step"] for c in curve]
    i = int(min(range(len(accs)), key=lambda k: accs[k]))
    peak = max(accs[i:])
    frac = (peak - accs[i]) / max(accs[0] - accs[i], 1e-12)
    print(f"A noncopy: baseline {accs[0]:.4f} -> trough {accs[i]:.4f}@{steps[i]} -> "
          f"peak-after {peak:.4f}  (recovery fraction {frac:.3f})")
    print(f"A copy:    baseline {curve[0]['A/copy/acc']:.4f} -> end "
          f"{curve[-1]['A/copy/acc']:.4f}")
    print(f"B: end {curve[-1]['B/ALL/acc']:.4f}")
    if curve[-1]["B/ALL/acc"] < 0.5:
        print("  READING 4: B never learned. UNINTERPRETABLE -- not scoreable either way.")
    elif accs[0] - accs[i] < 0.05:
        print("  READING 3 (at this LR): no crash while B learns.")
    elif frac > 0.1:
        print("  READING 1: crash AND spontaneous recovery in non-copy A.")
    else:
        print("  READING 2: crash, no recovery. Do NOT reach for a gentler LR; "
              "and the generic-text control is now REQUIRED before this negative "
              "means anything.")


if __name__ == "__main__":
    main()
