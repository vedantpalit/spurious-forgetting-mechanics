"""GATE: can OLMo 2 1B answer enough CounterFact facts to form population A?

Forward passes only. No training, no gradients. This runs before anything else, and if it fails
the answer is a different fact source, not a training run.

WHAT IT DECIDES
  * how many relations have >= 3 distinct templates (paraphrases are obtained by POOLING the
    distinct (relation_prefix, relation_suffix) pairs that appear for a relation_id -- the
    dataset has no paraphrase_prompts field, and that pooling is an ASSUMPTION this gate
    checks rather than relies on);
  * how many facts the base model answers correctly under BOTH >= 3 templates AND >= 2/3
    of its relation's templates -- a relative criterion, because template counts run 3
    to 16 and an absolute '>= 3 correct' would let 3-of-16 (19%) pass the same bar as
    3-of-3 (100%), which does not filter luck at all;
  * RELATION CONCENTRATION -- if 80% of the surviving set is one relation, the later curve
    measures that relation rather than factual recall generally;
  * the answer-pool size per relation, which sets the rank cap used later.

VERDICT
  |A| >= 300      proceed. 300 gives a binomial SE of ~2.6% at 50% accuracy, and the small
                  model's crash (1.00 -> 0.12) clears that by an order of magnitude.
  |A| <  300      STOP. Fallback ladder: the original ROME CounterFact JSON (which has real
                  paraphrase_prompts), then a different source, then a larger model.

FORMAT DISCIPLINE. Prompts are fed VERBATIM as cloze continuations -- no system prompt, no
instruction wrapper, no chat template, no added BOS beyond the tokenizer's own. This is the same
operation the model performed throughout pretraining on the same kind of string, and it is what
keeps a later crash from being attributable to a format shift.

FIRST-TOKEN SCORING. Correct means the argmax next token equals the first token of
the target rendered WITH ITS LEADING SPACE -- the prompts end mid-sentence ("... is"), so the
answer's first token is " Paris", not "Paris". Getting this wrong silently scores everything as
incorrect, so it is asserted at startup rather than assumed.

Run (on the cluster, after prefetch):
  HF_HOME=<cache-with-working-locks> HF_HUB_OFFLINE=1 \\
  .venv-llm/bin/python -m llm.gate_a_set --out llm/out/gate.json
"""
import argparse
import json
import os
from collections import Counter, defaultdict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "allenai/OLMo-2-0425-1B"          # base; NOT -SFT / -DPO / -Instruct
DATASET_ID = "NeelNanda/counterfact-tracing"
MIN_TEMPLATES = 3
MIN_FRACTION = 2 / 3      # see the CRITERION note in main()
MIN_A = 300


def first_token_id(tok, target: str) -> int:
    """First token of the target as it would CONTINUE the prompt, i.e. with a leading space.

    CounterFact prompts end mid-sentence ("The mother tongue of X is"), so the continuation the
    model must produce starts with a space. Tokenizing the bare string instead yields a
    different id for most BPE vocabularies and scores every fact as wrong -- silently.
    """
    ids = tok(" " + target.strip(), add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(f"target {target!r} tokenized to nothing")
    return ids[0]


def self_check(tok):
    """Fail loudly at startup if the leading-space convention is not what this code assumes."""
    bare = tok("Paris", add_special_tokens=False).input_ids
    lead = tok(" Paris", add_special_tokens=False).input_ids
    print(f"  tokenizer self-check: 'Paris' -> {bare}, ' Paris' -> {lead}")
    if bare == lead:
        print("  NOTE: this tokenizer does not distinguish a leading space; first_token_id is "
              "still correct but the distinction is moot.")
    ids = tok("The mother tongue of Danielle Darrieux is", add_special_tokens=False).input_ids
    assert len(ids) > 3, "prompt tokenization looks wrong"
    print(f"  sample prompt -> {len(ids)} tokens, no special tokens added")


@torch.no_grad()
def argmax_next(model, tok, prompts, device, batch_size=32):
    """Greedy next-token id for each prompt. Left-padded so the last position is the real one."""
    out = []
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        logits = model(**enc).logits[:, -1, :]
        out.extend(logits.argmax(-1).tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="llm/out/gate.json")
    ap.add_argument("--model_id", default=MODEL_ID, help="HF id; default the base model")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap the number of rows")
    ap.add_argument("--verify_batching", type=int, default=64,
                    help="rescore this many prompts at batch_size=1 and require "
                         "agreement; 0 disables")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  model={a.model_id}")
    tok = AutoTokenizer.from_pretrained(a.model_id)
    self_check(tok)
    model = AutoModelForCausalLM.from_pretrained(
        a.model_id, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device)
    model.eval()

    ds = load_dataset(DATASET_ID, split="train")
    ds_full_rows = len(ds)
    if a.limit:
        ds = ds.select(range(a.limit))
    print(f"dataset rows: {len(ds)}")

    # --- templates per relation, by pooling distinct (prefix, suffix) pairs -----------------
    templates = defaultdict(set)
    for r in ds:
        templates[r["relation_id"]].add((r["relation_prefix"], r["relation_suffix"]))
    tmpl_counts = {k: len(v) for k, v in templates.items()}
    usable = {k for k, n in tmpl_counts.items() if n >= MIN_TEMPLATES}
    print(f"\nrelations: {len(templates)}   with >= {MIN_TEMPLATES} templates: {len(usable)}")
    print(f"  template-count distribution: {dict(sorted(Counter(tmpl_counts.values()).items()))}")
    if not usable:
        print("\nGATE FAILED: pooling produced no relation with enough templates. The paraphrase "
              "assumption does not hold; fall back to the ROME CounterFact JSON.")
        return

    # --- score every fact under every template of its relation -----------------------------
    rows = [r for r in ds if r["relation_id"] in usable]
    print(f"facts in usable relations: {len(rows)}")
    flat_prompts, index = [], []
    for i, r in enumerate(rows):
        for (pre, suf) in sorted(templates[r["relation_id"]]):
            flat_prompts.append(f"{pre}{r['subject']}{suf}")
            index.append(i)
    print(f"forward passes: {len(flat_prompts)}")
    preds = argmax_next(model, tok, flat_prompts, device, a.batch_size)

    # BATCHING SELF-CHECK. Prompts are left-padded, and transformers assigns position ids
    # 0..L-1 regardless of padding, so real tokens sit at shifted absolute positions. For a
    # RoPE model that SHOULD cancel in the relative QK product -- but that is an argument,
    # not evidence, and it would corrupt every score silently rather than raise. So a slice
    # is rescored at batch_size=1 (no padding at all) and compared.
    n_chk = min(a.verify_batching, len(flat_prompts))
    if n_chk:
        solo = argmax_next(model, tok, flat_prompts[:n_chk], device, 1)
        mism = [j for j in range(n_chk) if solo[j] != preds[j]]
        print(f"  batching self-check: {n_chk - len(mism)}/{n_chk} agree with batch_size=1")
        if mism:
            raise SystemExit(
                f"BATCHING IS NOT SAFE: {len(mism)}/{n_chk} predictions differ between "
                f"batched (left-padded) and unbatched scoring. Every number this gate "
                f"produces would be padding-contaminated. Fix by right-padding and gathering "
                f"the logit at each sequence's true last index, or by passing explicit "
                f"position_ids derived from the attention mask.")

    want = [first_token_id(tok, r["target_true"]) for r in rows]
    n_correct = Counter()
    n_seen = Counter()
    for p, i in zip(preds, index):
        n_seen[i] += 1
        if p == want[i]:
            n_correct[i] += 1

    # CRITERION. Not an ABSOLUTE ">= 3 correct". Template counts
    # run 3 to 16, so 3-of-3 (100%) and 3-of-16 (19%) would both qualify, and 3-of-16 does
    # not filter luck -- which is the entire point of requiring paraphrases. The criterion is
    # therefore RELATIVE: correct on at least MIN_TEMPLATES templates AND on at least
    # MIN_FRACTION of that relation's templates, so every fact faces a comparable test. All
    # four thresholds are reported, since the choice changes |A| and should be visible.
    def keep_under(min_n, frac):
        return [i for i in range(len(rows))
                if n_correct[i] >= min_n and n_correct[i] >= frac * n_seen[i]]

    variants = [("all templates", MIN_TEMPLATES, 1.0),
                (f">= {MIN_FRACTION:.0%} of templates", MIN_TEMPLATES, MIN_FRACTION),
                (">= 50% of templates", MIN_TEMPLATES, 0.5),
                (f">= {MIN_TEMPLATES} absolute (NOT used)", MIN_TEMPLATES, 0.0)]
    print()
    print(f"|A| under each criterion (all require >= {MIN_TEMPLATES} correct):")
    for _name, _mn, _fr in variants:
        print(f"  {_name:>34}: {len(keep_under(_mn, _fr)):>6}")

    keep = keep_under(MIN_TEMPLATES, MIN_FRACTION)
    A = [rows[i] for i in keep]
    print()
    print(f"|A| (CRITERION: >= {MIN_TEMPLATES} correct AND >= {MIN_FRACTION:.0%} of "
          f"that relation's templates): {len(A)}")
    if len(rows):
        se = (0.25 / max(len(A), 1)) ** 0.5
        print(f"  binomial SE at 50% accuracy: {se:.4f}  ({se*100:.2f} percentage points)")
    # --- relation concentration ------------------------------------------------------------
    share = Counter(r["relation_id"] for r in A)
    tot = max(sum(share.values()), 1)
    hhi = sum((v / tot) ** 2 for v in share.values())
    print(f"\nrelation concentration: {len(share)} relations, Herfindahl index {hhi:.4f}")
    print(f"  {'relation_id':>14} {'n':>6} {'share':>8}")
    for rid, n in share.most_common(5):
        print(f"  {rid:>14} {n:>6} {n/tot:>7.1%}")
    top = share.most_common(1)[0][1] / tot if share else 0.0
    if top > 0.30:
        print(f"  WARNING: the largest relation is {top:.1%} of A (> 30%). The later curve would "
              f"partly measure that relation rather than factual recall generally. Propose "
              f"capping per-relation contributions before proceeding.")

    # --- answer pools, which set the rank cap -----------------------------------------------
    pools = defaultdict(set)
    for r in A:
        pools[r["relation_id"]].add(r["target_true"].strip())
    sizes = sorted(len(v) for v in pools.values())
    print(f"\nanswer-pool sizes per relation: min {sizes[0] if sizes else 0}, "
          f"median {sizes[len(sizes)//2] if sizes else 0}, max {sizes[-1] if sizes else 0}")
    print("  (the rank cap is set from these, not chosen in advance)")

    if a.limit:
        scale = ds_full_rows / max(len(ds), 1)
        print()
        print(f"SUBSAMPLE RUN (--limit {a.limit}). The {MIN_A}-fact floor is defined against "
              f"the FULL set, so NO VERDICT is issued here.")
        print(f"  hit rate {len(A)}/{len(rows)} = {len(A)/max(len(rows),1):.2%}  ->  "
              f"extrapolated |A| over all {ds_full_rows} rows ~ {int(len(A) * scale)}")
        verdict = "SUBSAMPLE"
    else:
        verdict = "PROCEED" if len(A) >= MIN_A else "STOP"
        print()
        print(f"GATE VERDICT: {verdict}  (|A| = {len(A)}, floor = {MIN_A})")
    if verdict == "STOP":
        print("  Fallback ladder: ROME CounterFact JSON (real paraphrase_prompts), then a "
              "different fact source, then a larger model. Do NOT proceed with an undersized A.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({
            "model": a.model_id, "dataset": DATASET_ID,
            "n_relations": len(templates), "n_relations_usable": len(usable),
            "template_counts": tmpl_counts,
            "n_rows_scored": len(rows), "n_A": len(A), "verdict": verdict,
            "relation_share": dict(share), "herfindahl": hhi,
            "answer_pool_sizes": {k: len(v) for k, v in pools.items()},
            "A": [{"row": i, "subject": r["subject"],
                   "relation_id": r["relation_id"], "target_true": r["target_true"],
                   "n_correct": n_correct[i], "n_templates": n_seen[i]}
                  for i, r in zip(keep, A)],
        }, fh, indent=2)
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
