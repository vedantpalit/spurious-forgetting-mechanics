"""The training loop: continued pretraining on the mixed stream, A and B scored on a schedule,
resumable from a checkpoint.

Efficiency against llm/inject.py, each measured on the first job and written to the run log:
  * pre-tokenized pools (tokens.py): no tokenizer calls inside the step;
  * micro-batch 16 x 512 with gradient checkpointing: 4 accumulation passes per 32k-token
    step instead of 16;
  * fused AdamW, TF32 matmuls, bf16 autocast, fp32 master weights (the precision lesson of
    llm/ is kept: never run the optimizer on bf16 parameters);
  * vectorized rank scoring, eval batch 256;
  * checkpoints (model + optimizer + RNG + curve) at the geometric eval points, so a
    preempted job resumes at the last one instead of restarting. `--resume` finds it.

Same optimisation as llm/: 100-step linear warmup then constant LR, betas (0.9, 0.95), no
weight decay, grad clip 1.0, loss on every token.
"""
import json
import math
import os
import random
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_real.config import RunConfig
from llm_real.scoring import build_a, build_b_items, score, summarise
from llm_real.tokens import BRenderer, Mixer, SEP, TokenPool, audit_format


def eval_schedule(cfg: RunConfig):
    dense = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100, 125, 150, 175, 200, 250, 300]
    pts = [p for p in dense if p <= min(cfg.eval_dense_until, cfg.steps)]
    s = pts[-1]
    while s < cfg.steps:
        s = int(math.ceil(s * cfg.eval_geometric / 5.0) * 5)
        pts.append(min(s, cfg.steps))
    return sorted(set(pts) | {0, cfg.steps})


def _ckpt_path(cfg, step):
    return os.path.join(cfg.out_dir, "ckpt", f"{cfg.name}_step{step:05d}.pt")


def _latest_ckpt(cfg):
    d = os.path.join(cfg.out_dir, "ckpt")
    if not os.path.isdir(d):
        return None
    cands = sorted(f for f in os.listdir(d) if f.startswith(cfg.name + "_step") and f.endswith(".pt"))
    return os.path.join(d, cands[-1]) if cands else None


def run(cfg: RunConfig, resume: bool = False):
    os.makedirs(cfg.out_dir, exist_ok=True)
    out_path = os.path.join(cfg.out_dir, f"{cfg.name}.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(cfg.seed); random.seed(cfg.seed); np.random.seed(cfg.seed)
    print(f"run {cfg.name}: device={device} lr={cfg.lr:g} steps={cfg.steps} "
          f"b_ratio={cfg.b_ratio} augment={cfg.augment}")
    print(json.dumps(cfg.to_dict()))

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=torch.float32).to(device)
    if cfg.grad_checkpoint:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # --- data --------------------------------------------------------------------------------
    ds = load_dataset(cfg.counterfact_id, split="train")
    a_items, a_pools, a_info = build_a(tok, cfg.a_gate, ds)
    b = json.load(open(cfg.b_path, encoding="utf-8"))
    print(f"A: {a_info}")
    sep_ids = tok(SEP, add_special_tokens=False).input_ids
    generic = None
    if cfg.b_ratio < 1.0 or (cfg.corpus_b and os.path.exists(cfg.generic_path)):
        g = np.load(cfg.generic_path, allow_pickle=True)
        generic = TokenPool(list(g), cfg.seed + 3)
        print(f"generic pool: {len(generic.docs)} documents, {generic.n_tokens:,} tokens")
    if cfg.corpus_b:
        # the real-text arm: B is a corpus of documents, trained as packed rows; read as
        # token accuracy and loss on a fixed sample of its training documents (the clock)
        # and on held-out documents never trained on; generic prose loss alongside.
        b_items, b_pools = [], {}
        train_ids = [tok(d["text"], add_special_tokens=False).input_ids for d in b["docs"]]
        held_ids = [tok(d["text"], add_special_tokens=False).input_ids for d in b["held"]]
        renderer = TokenPool(train_ids, cfg.seed)
        print(f"B corpus: {len(train_ids)} training documents ({renderer.n_tokens:,} tokens, "
              f"{renderer.n_tokens / cfg.tokens_per_step:.1f} steps per epoch), {len(held_ids)} held out")

        def pack(ids_list, n_rows):
            pool = TokenPool(ids_list, cfg.seed + 23)
            return np.asarray([pool.row(cfg.seq_len, sep_ids) for _ in range(n_rows)], dtype=np.int64)
        eval_rows = {"train": pack(train_ids, 64), "held": pack(held_ids, 64)}
        if generic is not None:
            # evaluation only (never trained when b_ratio = 1): keep the leads that pass the
            # register rule, so the generic loss is read on prose of the same register as B
            from llm_real.tokens import violation
            ok = [d.tolist() for d in generic.docs[:4000] if not violation(tok.decode(d))]
            print(f"generic eval sample: {len(ok)} of 4000 leads pass the register rule")
            eval_rows["generic"] = pack(ok[:2000], 64)
        if cfg.audit_format:
            audit_format([d["text"] for d in b["docs"][:200]], "B documents (as trained)")
        else:
            print("format audit SKIPPED (--no-audit_format): domain corpus in its native register")
    else:
        b_items, b_pools = build_b_items(tok, b)
        print(f"B: {len(b['people'])} people, {sum(len(p['facts']) for p in b['people'])} facts, "
              f"{len(b_items)} eval items over {len(b_pools)} attributes")
        renderer = BRenderer(b, tok, cfg.augment, cfg.seed)
        audit_format([renderer.document_string(b, i) for i in range(min(200, renderer.n_people))],
                     "B documents (as trained)")
        eval_rows = {}
    mixer = Mixer(renderer, generic, cfg.b_ratio, cfg.seq_len, sep_ids, cfg.seed)
    if generic is not None and cfg.b_ratio < 1.0:        # audited only when it is trained on
        audit_format([tok.decode(d[:400]) for d in generic.docs[:200]], "generic documents")

    micro_tokens = cfg.micro_batch * cfg.seq_len
    accum = max(cfg.tokens_per_step // micro_tokens, 1)
    print(f"{micro_tokens:,} tokens/micro-batch x {accum} = {micro_tokens * accum:,} tokens/step")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=0.0,
                            fused=(device == "cuda"))
    schedule = eval_schedule(cfg)
    curve, start = [], 0

    # --- resume ------------------------------------------------------------------------------
    if resume:
        p = _latest_ckpt(cfg)
        if p:
            ck = torch.load(p, map_location=device, weights_only=False)
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            curve, start = ck["curve"], ck["step"]
            random.setstate(ck["py_rng"]); torch.set_rng_state(ck["torch_rng"])
            mixer.rng.setstate(ck["mixer_rng"]); renderer.rng.setstate(ck["renderer_rng"])
            if generic is not None:
                generic.rng.setstate(ck["generic_rng"])
            print(f"resumed from {p} at step {start}")
        else:
            print("no checkpoint to resume from; starting fresh")

    @torch.no_grad()
    def token_eval(rows):
        """Mean next-token loss and teacher-forced token accuracy on fixed packed rows."""
        model.eval()
        losses, correct, count = [], 0, 0
        for i in range(0, len(rows), 16):
            batch = torch.from_numpy(rows[i:i + 16]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(input_ids=batch).logits.float()
            tgt = batch[:, 1:]; pred = logits[:, :-1]
            losses.append(torch.nn.functional.cross_entropy(pred.reshape(-1, pred.shape[-1]), tgt.reshape(-1), reduction="sum").item())
            correct += int((pred.argmax(-1) == tgt).sum()); count += tgt.numel()
        return sum(losses) / count, correct / count

    def evaluate(step):
        t0 = time.time()
        a_top = []
        aa, ar = score(model, tok, a_items, a_pools, device, cfg.eval_batch, top_out=a_top)
        rec = {"step": step}
        # per-fact outcomes and the displacing token, so the region that takes over can be read offline
        rec["A/correct"] = [int(v) for v in aa]
        rec["A/rank_each"] = [None if math.isnan(v) else round(float(v), 2) for v in ar]
        rec["A/top_each"] = a_top
        rec.update(summarise(aa, ar, a_items, "A", "copy"))
        rec.update(summarise(aa, ar, a_items, "Arel", "stratum"))
        rec.update(summarise(aa, ar, a_items, "Atouch", "touched"))
        if cfg.corpus_b:
            for k, rows in eval_rows.items():
                rec[f"B/{k}/loss"], rec[f"B/{k}/tok_acc"] = token_eval(rows)
            rec["B/ALL/acc"] = rec["B/train/tok_acc"]          # the clock, in the same slot
            b_str = (f"B train loss={rec['B/train/loss']:.3f} tok_acc={rec['B/train/tok_acc']:.3f} "
                     f"held loss={rec['B/held/loss']:.3f}"
                     + (f" generic loss={rec['B/generic/loss']:.3f}" if "B/generic/loss" in rec else ""))
        else:
            ba, br = score(model, tok, b_items, b_pools, device, cfg.eval_batch)
            rec.update(summarise(ba, br, b_items, "B", "stratum"))
            b_str = f"B={rec['B/ALL/acc']:.4f} rank={rec.get('B/ALL/rank', float('nan')):.3f}"
        rec["eval_seconds"] = time.time() - t0
        curve.append(rec)
        print(f"  [step {step:>5}] A noncopy={rec['A/noncopy/acc']:.4f}"
              f"+-{rec['A/noncopy/acc_se']:.4f} rank={rec.get('A/noncopy/rank', float('nan')):.3f}"
              f" | copy={rec['A/copy/acc']:.4f} | {b_str}  ({rec['eval_seconds']:.0f}s)",
              flush=True)
        model.train()

    def checkpoint(step):
        if not cfg.ckpt_every_eval:
            return
        os.makedirs(os.path.join(cfg.out_dir, "ckpt"), exist_ok=True)
        p = _ckpt_path(cfg, step)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "curve": curve,
                    "step": step, "py_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                    "mixer_rng": mixer.rng.getstate(), "renderer_rng": renderer.rng.getstate(),
                    "generic_rng": generic.rng.getstate() if generic is not None else None,
                    "cfg": cfg.to_dict()}, p)
        if cfg.keep_last_ckpt_only:
            for f in os.listdir(os.path.dirname(p)):
                q = os.path.join(os.path.dirname(p), f)
                if f.startswith(cfg.name + "_step") and q != p:
                    os.remove(q)

    def save_curve():
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"cfg": cfg.to_dict(), "a_info": a_info,
                       "n_b_people": len(b["docs"]) if cfg.corpus_b else len(b["people"]),
                       "tokens_per_step": micro_tokens * accum, "curve": curve}, fh, indent=1)

    if start == 0:
        evaluate(0)
    model.train()
    t_step = time.time()
    for step in range(start + 1, cfg.steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = cfg.lr * min(step / max(cfg.warmup, 1), 1.0)
        total = 0.0
        for _ in range(accum):
            batch = torch.from_numpy(mixer.batch(cfg.micro_batch)).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = model(input_ids=batch, labels=batch).loss / accum
            loss.backward()
            total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step in schedule:
            dt = (time.time() - t_step) / max(step - start, 1)
            print(f"    step {step}: train loss {total:.4f}  lr {opt.param_groups[0]['lr']:.3g}  "
                  f"({dt:.2f} s/step)", flush=True)
            evaluate(step)
            save_curve()
            if step > cfg.eval_dense_until and cfg.ckpt_every_eval:
                checkpoint(step)
    save_curve()
    print(f"wrote {out_path}")
    reading(curve)


def reading(curve):
    """The pre-registered read, on the non-copy headline."""
    accs = [c["A/noncopy/acc"] for c in curve]; steps = [c["step"] for c in curve]
    i = int(np.argmin(accs)); peak = max(accs[i:])
    frac = (peak - accs[i]) / max(accs[0] - accs[i], 1e-12)
    bend = curve[-1]["B/ALL/acc"]
    print(f"A noncopy: baseline {accs[0]:.4f} -> trough {accs[i]:.4f}@{steps[i]} -> peak-after "
          f"{peak:.4f} (recovery fraction {frac:.3f}); B end {bend:.4f}"
          + (" (B = token accuracy on its training documents)" if "B/train/loss" in curve[-1] else ""))
    if bend < 0.5:
        print("  READING 4: B never learned -- not scoreable either way.")
    elif accs[0] - accs[i] < 0.05:
        print("  READING 3: no crash while B learns.")
    elif frac > 0.1:
        print("  READING 1: crash and spontaneous recovery in non-copy A.")
    else:
        print("  READING 2: crash, no recovery.")
