"""Cloze scoring for A and B: first-token accuracy over the full vocabulary, and mean rank of
the correct answer within a fixed candidate pool. Same definitions as llm/inject.py (so the
curves are comparable), re-implemented here rather than imported so the two pipelines cannot
drift into each other, and vectorized: one gather per batch for the ranks instead of a Python
loop per item.

A's items and pools come from llm/out/gate.json through `build_a`, which mirrors
llm/evalsets.build_a's stratification (copy / noncopy is the headline split). B's items come
from the gated B json.
"""
import json
import math
import random
from collections import defaultdict

import numpy as np
import torch

POOL_CAP = 20
MIN_POOL = 4


def first_token_id(tok, s):
    ids = tok(" " + s.strip(), add_special_tokens=False).input_ids
    return ids[0] if ids else None


def _copy_loose(tok, prompt, answer):
    """Either surface form of the answer's first token appears among the prompt's tokens."""
    pt = set(tok(prompt, add_special_tokens=False).input_ids)
    a = answer.strip()
    for form in (" " + a, a):
        ids = tok(form, add_special_tokens=False).input_ids
        if ids and ids[0] in pt:
            return True
    return False


DUP_RELATIONS = ("P176", "P178")                                 # both mean "who makes X"
# Relations whose answers are of the same semantic type as one of B's attributes
# (llm/evalsets.py's B_TYPED_RELATIONS): makers/employers, cities, countries, languages.
B_TYPED_RELATIONS = ("P176", "P178", "P108", "P159", "P27", "P103", "P364")


def build_a(tok, gate_path, dataset, seed=0):
    """A's eval items and rank pools from llm/out/gate.json -- the same construction as
    llm/evalsets.build_a, step for step (same pool cap and sampling seed, same copy test, same
    strata), so A is literally the same 1,106 facts scored the same way as in llm/inject.py."""
    A = json.load(open(gate_path, encoding="utf-8"))["A"]
    full_pool, prompt_by = defaultdict(set), {}
    for r in dataset:
        full_pool[r["relation_id"]].add(r["target_true"].strip())
        prompt_by.setdefault((r["subject"], r["relation_id"]), r["prompt"])
    rng = random.Random(seed)
    pools = {}
    for rid, vals in full_pool.items():
        v = sorted(vals)
        if len(v) > POOL_CAP:
            v = sorted(rng.sample(v, POOL_CAP))
        ids, seen = [], set()
        for s in v:
            t = first_token_id(tok, s)
            if t is not None and t not in seen:
                seen.add(t); ids.append(t)
        pools[rid] = ids
    items = []
    for r in A:
        rid = r["relation_id"]
        prompt = prompt_by[(r["subject"], rid)]
        items.append({"subject": r["subject"], "relation_id": rid, "target": r["target_true"],
                      "prompt": prompt, "first_token": first_token_id(tok, r["target_true"]),
                      "stratum": rid if rid in DUP_RELATIONS else "other",
                      "copy": "copy" if _copy_loose(tok, prompt, r["target_true"]) else "noncopy",
                      "touched": "b_typed" if rid in B_TYPED_RELATIONS else "untouched",
                      "rank_eligible": len(pools.get(rid, [])) >= MIN_POOL})
    info = {"n_A": len(items),
            "copy": sum(i["copy"] == "copy" for i in items),
            "noncopy": sum(i["copy"] == "noncopy" for i in items),
            "rank_eligible": sum(i["rank_eligible"] for i in items)}
    return items, pools, info


def build_b_items(tok, b):
    """B's eval items (held-out phrasings) and per-attribute pools (B's own value set)."""
    pools = {}
    for attr, vals in b["pools"].items():
        ids, seen = [], set()
        for s in vals:
            t = first_token_id(tok, s)
            if t is not None and t not in seen:
                seen.add(t); ids.append(t)
        pools[attr] = ids
    items = [{"prompt": e["prompt"], "first_token": e["first_token"], "target": e["value"],
              "attr": e["attr"], "stratum": e["attr"], "copy": "noncopy", "touched": "b_typed",
              "rank_eligible": len(pools[e["attr"]]) >= MIN_POOL}
             for e in b["eval"]]
    return items, pools


@torch.no_grad()
def score(model, tok, items, pools, device, batch_size=256, top_out=None):
    """Returns (acc[n], rank[n]) as float arrays; rank is nan where not eligible. If `top_out` is a
    list, the argmax token id of every item is appended to it (what displaced the answer).

    Ranks are vectorized: every item's pool is padded to a fixed width once, and a batch's
    ranks are one gather and two masked comparisons -- no per-item Python."""
    model.eval()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    n = len(items)
    width = max((len(v) for v in pools.values()), default=1)
    pool_idx = torch.full((n, width), 0, dtype=torch.long)
    pool_mask = torch.zeros((n, width), dtype=torch.bool)
    elig = torch.zeros(n, dtype=torch.bool)
    for i, it in enumerate(items):
        if it["rank_eligible"]:
            ids = pools[it.get("relation_id") or it["attr"]]
            pool_idx[i, :len(ids)] = torch.tensor(ids); pool_mask[i, :len(ids)] = True
            elig[i] = True
    pool_idx, pool_mask, elig = pool_idx.to(device), pool_mask.to(device), elig.to(device)
    tgt = torch.tensor([it["first_token"] for it in items], device=device)
    acc = np.zeros(n); rank = np.full(n, np.nan)
    for i in range(0, n, batch_size):
        chunk = items[i:i + batch_size]; m = len(chunk)
        enc = tok([it["prompt"] for it in chunk], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = model(**enc).logits[:, -1, :].float()
        t = tgt[i:i + m]
        top = logits.argmax(-1)
        acc[i:i + m] = (top == t).cpu().numpy()
        if top_out is not None:
            top_out.extend(top.tolist())
        mine = logits.gather(1, t[:, None])                                   # (m, 1)
        cand = logits.gather(1, pool_idx[i:i + m])                           # (m, width)
        msk = pool_mask[i:i + m]
        greater = ((cand > mine) & msk).sum(1)
        tied = ((cand == mine) & msk).sum(1) - 1
        r = 1.0 + greater.float() + 0.5 * tied.clamp(min=0).float()
        r = torch.where(elig[i:i + m], r, torch.full_like(r, float("nan")))
        rank[i:i + m] = r.cpu().numpy()
    return acc, rank


def summarise(acc, rank, items, label, key):
    out = {}
    levels = ["ALL"] + sorted({it[key] for it in items})
    for s in levels:
        idx = [k for k, it in enumerate(items) if s == "ALL" or it[key] == s]
        if not idx:
            continue
        p = float(acc[idx].mean())
        out[f"{label}/{s}/acc"] = p
        out[f"{label}/{s}/acc_se"] = math.sqrt(max(p * (1 - p), 0) / len(idx))
        out[f"{label}/{s}/n"] = len(idx)
        rs = rank[idx]; rs = rs[~np.isnan(rs)]
        if len(rs):
            out[f"{label}/{s}/rank"] = float(rs.mean())
    return out
