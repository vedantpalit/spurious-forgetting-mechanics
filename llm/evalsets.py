"""Evaluation sets for A and B, and the rank pools.

A comes from the gate and is read out with CounterFact's own cloze probe, fed verbatim -- no
system prompt, no instruction wrapper, no chat template. B comes from build_b.py and is read out
with phrasings held out of its training documents. Both are scored the same way: first-token
argmax over the full vocabulary, plus a mean rank restricted to a fixed candidate pool.

RANK POOLS ARE BUILT FROM THE FULL DATASET, NOT FROM A. In the biography setup the value pool is
every value the attribute could take, not the subset A happened to use, which is what makes "rank
within the region" a statement about the model still preferring the right answer among plausible
alternatives. Building it from A would also make the metric's resolution depend on how many facts
survived the gate. Pools are capped at POOL_CAP so a rank of 5 means the same thing everywhere,
and selection above the cap is deterministic.

WHY RANK AT ALL. Accuracy crashing while rank holds is the suppression signature -- mass leaving
the answer region with relative order inside it preserved. It is the difference between "the
curve reproduces" and "the mechanism reproduces", so it is logged at every eval, not derived
afterwards from top-1.

THREE STRATIFICATIONS OF A, each with its own SEs:

  copy / noncopy -- THE HEADLINE IS NONCOPY. 62% of A is answerable by copying the answer out of
    the prompt ("Ferrari F40, developed by" -> "Ferrari"), which the gate cannot filter because
    copying works across every paraphrase. Those facts are not junk: copying is an in-context
    operation rather than parametric recall, and whether it crashes differently is informative on
    its own. Defined operationally as COPY_LOOSE below -- never a judgement call per fact.

  P176 / P178 / other -- the gate found A heavily skewed toward "who makes X" (both relations
    mean that), and "other" is ~350 facts and visibly noisier.

  b_attr / untouched -- whether the fact's relation is of a semantic type B injects into. B is
    measured against all of A regardless, so the relations B does not touch are a built-in
    unrelated contrast inside the same run.
"""
import json
import random
from collections import Counter, defaultdict

POOL_CAP = 20        # distinct answers per relation
MIN_POOL = 4         # below this a rank carries essentially no information
DUP_RELATIONS = ("P176", "P178")     # both mean "who makes X"

# Relations whose answers are of the same semantic type as one of B's attributes. Companies
# (P176/P178 makers, P108 employer), cities (P159), countries (P27), languages (P103/P364).
B_TYPED_RELATIONS = ("P176", "P178", "P108", "P159", "P27", "P103", "P364")


def first_token_id(tok, target: str, lead: bool = True):
    """First token of the target, by default as a CONTINUATION (with its leading space)."""
    ids = tok((" " if lead else "") + target.strip(), add_special_tokens=False).input_ids
    return ids[0] if ids else None


def is_copy(tok, prompt: str, target: str) -> bool:
    """COPY_LOOSE: either form of the answer's first token already appears in the prompt.

    Space-insensitive on purpose. BPE gives ' Ferrari' and 'Ferrari' different ids, and in the
    copy-heavy templates the subject sits at position 0 with no leading space -- the strict
    space-sensitive test caught 34 facts (3.1%) where this catches 684 (61.8%), against 686 for a
    substring test. Mechanical, and it reproduces the substring intuition without depending on it.
    """
    pt = set(tok(prompt, add_special_tokens=False).input_ids)
    return (first_token_id(tok, target, True) in pt
            or first_token_id(tok, target, False) in pt)


def build_a(tok, gate_path, dataset, seed=0):
    """A's eval items, prompts and rank pools. `dataset` is the whole split, not A."""
    A = json.load(open(gate_path, encoding="utf-8"))["A"]

    full_pool, prompt_by = defaultdict(set), {}
    for r in dataset:
        full_pool[r["relation_id"]].add(r["target_true"].strip())
        prompt_by.setdefault((r["subject"], r["relation_id"]), r["prompt"])

    rng = random.Random(seed)
    pool_tokens = {}
    for rid, vals in full_pool.items():
        v = sorted(vals)                       # deterministic before any sampling
        if len(v) > POOL_CAP:
            v = sorted(rng.sample(v, POOL_CAP))
        ids, seen = [], set()
        for s in v:                            # distinct first tokens, else rank is ill-posed
            t = first_token_id(tok, s)
            if t is not None and t not in seen:
                seen.add(t)
                ids.append(t)
        pool_tokens[rid] = ids

    items = []
    for r in A:
        rid = r["relation_id"]
        prompt = prompt_by.get((r["subject"], rid))
        if prompt is None:
            raise KeyError(f"no prompt for {r['subject']!r} / {rid}")
        n_pool = len(pool_tokens.get(rid, []))
        items.append({
            "subject": r["subject"], "relation_id": rid, "target": r["target_true"],
            "prompt": prompt,
            "first_token": first_token_id(tok, r["target_true"]),
            "stratum": rid if rid in DUP_RELATIONS else "other",
            "copy": "copy" if is_copy(tok, prompt, r["target_true"]) else "noncopy",
            "touched": "b_typed" if rid in B_TYPED_RELATIONS else "untouched",
            "rank_eligible": n_pool >= MIN_POOL, "pool_size": n_pool,
        })

    n_elig = sum(i["rank_eligible"] for i in items)
    info = {
        "n_A": len(items),
        "rank_eligible": n_elig, "rank_coverage": n_elig / max(len(items), 1),
        "pool_cap": POOL_CAP, "min_pool": MIN_POOL,
        "strata": dict(Counter(i["stratum"] for i in items)),
        "copy": dict(Counter(i["copy"] for i in items)),
        "touched": dict(Counter(i["touched"] for i in items)),
    }
    return items, pool_tokens, info


def build_b(tok, b_facts_path):
    """B's eval items and pools. The pool is B's own value set for that attribute -- closed and
    known, unlike A's, because we chose it."""
    d = json.load(open(b_facts_path, encoding="utf-8"))
    pool_tokens = {}
    for at, vals in d["pools"].items():
        ids, seen = [], set()
        for s in vals:
            t = first_token_id(tok, s)
            if t is not None and t not in seen:
                seen.add(t)
                ids.append(t)
        pool_tokens[at] = ids
    items = [{"prompt": e["prompt"], "attr": e["attr"], "target": e["value"],
              "first_token": e["first_token"], "stratum": e["attr"],
              "copy": "noncopy", "touched": "b_typed",
              "rank_eligible": len(pool_tokens[e["attr"]]) >= MIN_POOL,
              "pool_size": len(pool_tokens[e["attr"]])}
             for e in d["eval"]]
    return items, pool_tokens, d
