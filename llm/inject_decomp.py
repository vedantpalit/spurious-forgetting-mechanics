"""The two-process decomposition, measured on OLMo 2 1B itself.

THE QUESTION. Is the LLM's crash-and-recovery produced by a shift that every old fact receives
in common, and its erosion by what each fact receives on its own? In the small transformer and
the minimal model the answer is yes (delta / epsilon). In OLMo the curve was shown to reproduce
and the mechanism was never tested; the LLM's crash also moves the within-pool rank, which a pure
common shift would not. This run tests it directly.

WHAT IS MEASURED, at every eval step, for every old fact a (the A set of llm/inject.py):
    x_a(t)   the state the unembedding reads: the base model's last_hidden_state at the
             answer position (post final norm in HF's OLMo-2). Checked at step 0 against the
             model's own logits; the run stops if they disagree.
    dx_a     = x_a(t) - x_a(0)
    delta    = mean_a dx_a                over the stratum being scored (copy / non-copy)
    eps_a    = dx_a - delta
and four sets of logits through the CURRENT unembedding U(t):
    actual        x_a(t) U(t)                         the model
    common        (x_a(0) + delta) U(t)               the common shift alone
    individual    (x_a(0) + eps_a) U(t)               the individual displacement alone
    readout       x_a(0) U(t)                         the unembedding's own change alone
each read out as first-token accuracy over the full vocabulary and mean within-pool rank,
exactly as llm/inject.py does. Also logged: ||delta||, rms ||eps_a||, mean ||x_a||, and
||U(t) - U(0)|| / ||U(0)||. B's accuracy on held-out phrasings gives the clock.

SECOND GROUPING (added after seed 0's first run, 2026-09-16). A spans a dozen CounterFact
relations with different prompt frames; the toy's population has one. A shift that is common
within a relation but points differently across relations is 'individual' under the
all-facts mean. So delta is also taken within each relation group of >= MIN_GROUP facts
("common_rel" / "individual_rel"), and the variance share of the change's top principal
directions is logged. The readout states themselves are saved at every eval
(decomp_*_states.npz, fp16, ~110 MB) so any other grouping can be tested offline.

The training loop, data, optimiser and eval schedule are llm/inject.py's, imported; llm/ stays
the replication and this file adds the decomposition beside it. Same command line:

  .venv-llm/bin/python -m llm.inject_decomp --lr 1e-5 --steps 1000 --seed 0
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
from llm.inject import DATASET_ID, MODEL_ID, DocSampler, eval_schedule, score, summarise


@torch.no_grad()
def states(model, tok, items, device, batch_size=128):
    """x_a for every item: the base model's last_hidden_state at the answer position, which
    in HF's OLMo-2 is the output of the final norm, i.e. exactly what lm_head reads. (n, d)
    fp32. The tap check in main() verifies this against the model's own logits."""
    model.eval()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    xs = []
    for i in range(0, len(items), batch_size):
        enc = tok([it["prompt"] for it in items[i:i + batch_size]], return_tensors="pt",
                  padding=True, add_special_tokens=False).to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            out = model.model(**enc)
        xs.append(out.last_hidden_state[:, -1, :].detach().float())
    return torch.cat(xs)


@torch.no_grad()
def read(model, x, items, pools, batch_size=256):
    """Accuracy and within-pool rank of logits = lm_head(x), the same rules as inject.score."""
    acc, rank = [], []
    for i in range(0, len(items), batch_size):
        logits = model.lm_head(x[i:i + batch_size].to(model.lm_head.weight.dtype)).float()
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


MIN_GROUP = 15     # a within-relation mean over fewer facts is mostly the facts themselves


def group_key(it):
    return it.get("relation_id") or it["attr"]


def decompose(model, x0, xt, items, pools, U0):
    """The readings per stratum. delta is taken (a) over the whole stratum and (b) within
    each relation group of at least MIN_GROUP facts ("rel"): a shift that is common within a
    relation but points differently across relations is 'individual' under (a) and 'common'
    under (b). Also the share of the change's variance in its top principal directions."""
    rec = {}
    U = model.lm_head.weight.detach().float()
    rec["dU_rel"] = float((U - U0).norm() / U0.norm())
    for stratum in ("noncopy", "copy"):
        idx = [k for k, it in enumerate(items) if it["copy"] == stratum]
        if not idx:
            continue
        sub = [items[k] for k in idx]
        a0, at = x0[idx], xt[idx]
        dx = at - a0
        delta = dx.mean(0, keepdim=True)
        eps = dx - delta
        rec[f"{stratum}/delta_norm"] = float(delta.norm())
        rec[f"{stratum}/eps_rms"] = float(eps.norm(dim=1).pow(2).mean().sqrt())
        rec[f"{stratum}/x_norm"] = float(at.norm(dim=1).mean())
        rec[f"{stratum}/dx_cos_delta"] = float(
            torch.nn.functional.cosine_similarity(dx, delta.expand_as(dx), dim=1).mean())
        # how low-dimensional is the change: variance share of the top 1 / 5 directions of dx
        if dx.shape[0] > 5 and float(dx.norm()) > 0:
            s = torch.linalg.svdvals(dx - dx.mean(0, keepdim=True))
            tot = float((s ** 2).sum())
            rec[f"{stratum}/pc1_share"] = float(s[0] ** 2 / tot)
            rec[f"{stratum}/pc5_share"] = float((s[:5] ** 2).sum() / tot)
        # (b) within-relation means
        delta_rel = torch.zeros_like(dx); covered = torch.zeros(len(sub), dtype=torch.bool)
        for g in {group_key(it) for it in sub}:
            gi = [k for k, it in enumerate(sub) if group_key(it) == g]
            if len(gi) >= MIN_GROUP:
                delta_rel[gi] = dx[gi].mean(0, keepdim=True); covered[gi] = True
        eps_rel = dx - delta_rel
        rec[f"{stratum}/rel_covered"] = int(covered.sum())
        rec[f"{stratum}/rel_delta_rms"] = float(delta_rel[covered].norm(dim=1).pow(2).mean().sqrt()) if covered.any() else float("nan")
        rec[f"{stratum}/rel_eps_rms"] = float(eps_rel[covered].norm(dim=1).pow(2).mean().sqrt()) if covered.any() else float("nan")
        for name, x in (("actual", at), ("common", a0 + delta), ("individual", a0 + eps),
                        ("readout", a0), ("common_rel", a0 + delta_rel),
                        ("individual_rel", a0 + eps_rel)):
            acc, rank = read(model, x, sub, pools)
            if name.endswith("_rel"):
                acc = [v for v, c in zip(acc, covered.tolist()) if c]
                rank = [v for v, c in zip(rank, covered.tolist()) if c]
            rec[f"{stratum}/{name}/acc"] = sum(acc) / len(acc) if acc else float("nan")
            rs = [r for r in rank if not math.isnan(r)]
            rec[f"{stratum}/{name}/rank"] = sum(rs) / len(rs) if rs else float("nan")
        rec[f"{stratum}/n"] = len(idx)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--tokens_per_step", type=int, default=32768)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval_batch", type=int, default=128)
    ap.add_argument("--gate", default="llm/out/gate.json")
    ap.add_argument("--facts", default="llm/out/b_facts.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_states", action="store_true",
                    help="do not save the readout states (fp16, ~110 MB per run)")
    a = ap.parse_args()
    out_path = a.out or f"llm/out/decomp_lr{a.lr:g}_seed{a.seed}.json"
    states_path = out_path.replace(".json", "_states.npz")

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  lr={a.lr:g}  steps={a.steps}  batch={a.batch_size}x{a.seq_len}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device)

    ds = load_dataset(DATASET_ID, split="train")
    a_items, a_pools, a_info = build_a(tok, a.gate, ds)
    b_items, b_pools, facts = build_b(tok, a.facts)
    print(f"A: {a_info['n_A']} facts ({a_info['copy']}); B: {len(b_items)} eval items")

    sampler = DocSampler(facts, tok, a.seq_len, a.seed)
    micro_tokens = a.batch_size * a.seq_len
    accum = max(a.tokens_per_step // micro_tokens, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    schedule = eval_schedule(a.steps)
    curve = []

    # pretrained state and unembedding, the reference for every decomposition
    x0 = states(model, tok, a_items, device, a.eval_batch)
    U0 = model.lm_head.weight.detach().float().clone()
    # the tap must reproduce the model's own logits: check once, loudly
    acc_tap, _ = read(model, x0, a_items, a_pools)
    acc_ref, _ = score(model, tok, a_items, a_pools, device, a.eval_batch)
    agree = sum(1 for u, v in zip(acc_tap, acc_ref) if u == v) / len(acc_ref)
    print(f"tap check: per-item accuracy agreement with inject.score {agree:.4f} "
          f"(bf16 autocast vs fp32 head; expect > 0.99)")
    if agree < 0.95:
        raise SystemExit("last_hidden_state does not reproduce the model's logits; "
                         "the state must be taken after the final norm -- stop.")

    saved_steps, saved_x = [], []
    meta = dict(copy=[it["copy"] for it in a_items], group=[group_key(it) for it in a_items],
                first_token=[it["first_token"] for it in a_items],
                rank_eligible=[bool(it["rank_eligible"]) for it in a_items])

    def evaluate(step):
        xt = states(model, tok, a_items, device, a.eval_batch)
        rec = {"step": step}
        rec.update(decompose(model, x0, xt, a_items, a_pools, U0))
        ba, br = score(model, tok, b_items, b_pools, device, a.eval_batch)
        rec.update(summarise(ba, br, b_items, "B", "stratum"))
        curve.append(rec)
        s = "noncopy"
        print(f"  [step {step:>4}] noncopy actual={rec[f'{s}/actual/acc']:.3f} "
              f"common={rec[f'{s}/common/acc']:.3f} individual={rec[f'{s}/individual/acc']:.3f} "
              f"readout={rec[f'{s}/readout/acc']:.3f} | within-relation common={rec[f'{s}/common_rel/acc']:.3f} "
              f"individual={rec[f'{s}/individual_rel/acc']:.3f} (n={rec[f'{s}/rel_covered']}) "
              f"| rank actual={rec[f'{s}/actual/rank']:.2f} common={rec[f'{s}/common/rank']:.2f} "
              f"| ||delta||={rec[f'{s}/delta_norm']:.2f} rms||eps||={rec[f'{s}/eps_rms']:.2f} "
              f"rel: {rec[f'{s}/rel_delta_rms']:.2f}/{rec[f'{s}/rel_eps_rms']:.2f} "
              f"||x||={rec[f'{s}/x_norm']:.1f} cos={rec[f'{s}/dx_cos_delta']:.2f} "
              f"pc1={rec.get(f'{s}/pc1_share', float('nan')):.2f} pc5={rec.get(f'{s}/pc5_share', float('nan')):.2f} "
              f"dU/U={rec['dU_rel']:.4f} | B={rec['B/ALL/acc']:.3f}")
        if not a.no_states:
            import numpy as np
            saved_steps.append(step); saved_x.append(xt.cpu().to(torch.float16).numpy())
            np.savez_compressed(states_path, steps=np.array(saved_steps), x=np.stack(saved_x),
                                x0=x0.cpu().to(torch.float16).numpy(),
                                U=model.lm_head.weight.detach().cpu().to(torch.float16).numpy(),
                                **{k: np.array(v) for k, v in meta.items()})
        model.train()

    evaluate(0)
    for step in range(1, a.steps + 1):
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
            print(f"    train loss {total:.4f}")
            evaluate(step)
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump({"lr": a.lr, "steps": a.steps, "seed": a.seed,
                           "tokens_per_step": micro_tokens * accum, "warmup": a.warmup,
                           "a_info": a_info, "curve": curve}, fh, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
