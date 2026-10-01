"""Does a bf16 optimizer step actually write anything?

THE SUSPICION. `inject.py` loaded OLMo in bfloat16 and ran AdamW directly on those bf16
parameters -- no fp32 master weights. bf16 carries 8 mantissa bits, so its relative resolution is
about 0.39%. Adam normalises its update to roughly `lr` in magnitude, and a 1B transformer's
weights are around 0.02, so the relative change per step is about lr/0.02: 0.005% at lr=1e-6 and
0.5% at lr=1e-4. Everything below roughly 3e-5 should round away entirely, and everything above
should land *distorted* -- surviving on small-magnitude weights and vanishing on large ones.

WHY IT MATTERS. That would reproduce the LR sweep exactly: the 1e-6 and 3e-6 arms did nothing,
B only began learning at 1e-4, and the same update demolished A. If it holds, every negative in
that sweep is an artifact of a regime where the only writable updates were destructive ones --
the rank movement, the absent relatedness effect, and the 10-50x gap between A dying and B
learning all measured rounding rather than learning.

WHAT THIS MEASURES. The same ten optimizer steps on the same documents, in bf16 and in fp32, at
two learning rates. fp32 is the ground truth for what should have been written. Reported per
configuration: what fraction of parameters moved at all, and the realised ||dW||/||W||. If bf16
at 1e-6 moves almost nothing while fp32 does, the diagnosis is confirmed.

Run:
  .venv-llm/bin/python -m llm.check_precision
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm.inject import DocSampler

MODEL_ID = "allenai/OLMo-2-0425-1B"
WATCH = ["model.embed_tokens.weight",
         "model.layers.0.mlp.down_proj.weight",
         "model.layers.8.mlp.down_proj.weight",
         "model.layers.8.self_attn.o_proj.weight"]


def run(dtype, lr, facts, tok, steps, batch_size, seq_len, seed, device):
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device)
    model.train()
    # Snapshot on the CPU in fp32. On the GPU it cost 4GB next to fp32 weights, grads and Adam
    # state, which is what OOMed the fp32 arm on a 40GB card.
    before = {n: p.detach().to("cpu", torch.float32).clone() for n, p in model.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    sampler = DocSampler(facts, tok, seq_len, seed)      # same seed => identical documents
    for _ in range(steps):
        batch = sampler.batch(batch_size).to(device)
        model(input_ids=batch, labels=batch).loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

    n_tot = n_moved = 0
    dsq = wsq = 0.0
    per_tensor = {}
    for n, p in model.named_parameters():
        d = p.detach().to("cpu", torch.float32) - before[n]
        moved = int((d != 0).sum().item())
        n_tot += d.numel()
        n_moved += moved
        dsq += float((d * d).sum().item())
        wsq += float((before[n].float() ** 2).sum().item())
        if n in WATCH:
            w = before[n]
            per_tensor[n] = (moved / d.numel(),
                             float(d.norm().item()) / max(float(w.norm().item()), 1e-12))
    del model, before, opt
    torch.cuda.empty_cache()
    return n_moved / n_tot, (dsq ** 0.5) / max(wsq ** 0.5, 1e-12), per_tensor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", default="llm/out/b_facts.json")
    ap.add_argument("--steps", type=int, default=10)
    # Small on purpose: Adam normalises m/sqrt(v) to ~1, so the UPDATE MAGNITUDE this measures
    # is nearly independent of batch size, while fp32 logits cost batch x seq x 100k x 4 bytes.
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("needs a GPU; on CPU the fp32 arm is unusably slow and bf16 is emulated")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    facts = json.load(open(a.facts, encoding="utf-8"))
    print(f"{a.steps} AdamW steps, batch {a.batch_size}x{a.seq_len}, identical documents "
          f"across every configuration\n")

    print(f"{'dtype':>10} {'lr':>8} {'params moved':>13} {'||dW||/||W||':>13}")
    results = {}
    for dtype, label in [(torch.bfloat16, "bfloat16"), (torch.float32, "float32")]:
        for lr in (1e-6, 3e-5):
            frac, rel, per = run(dtype, lr, facts, tok, a.steps, a.batch_size, a.seq_len,
                                 a.seed, device)
            results[(label, lr)] = (frac, rel, per)
            print(f"{label:>10} {lr:>8.0e} {frac:>12.2%} {rel:>13.3e}")

    print("\nper-tensor (fraction of elements moved):")
    print(f"  {'tensor':<44} {'bf16 1e-6':>10} {'fp32 1e-6':>10} {'bf16 3e-5':>10} "
          f"{'fp32 3e-5':>10}")
    for n in WATCH:
        row = [results[(d, lr)][2].get(n, (float('nan'),))[0]
               for d in ("bfloat16", "float32") for lr in (1e-6, 3e-5)]
        row = [row[0], row[2], row[1], row[3]]      # bf16/fp32 at 1e-6, then at 3e-5
        print(f"  {n:<44} " + " ".join(f"{x:>9.2%}" for x in row))

    b16, f32 = results[("bfloat16", 1e-6)][1], results[("float32", 1e-6)][1]
    print(f"\nAt lr=1e-6, bf16 wrote {b16 / max(f32, 1e-30):.1%} of what fp32 wrote.")
    if results[("bfloat16", 1e-6)][0] < 0.05:
        print("CONFIRMED: bf16 rounds the update away. The LR sweep measured rounding, not "
              "learning, and its negatives are artifacts. Fix is fp32 master weights.")
    else:
        print("NOT CONFIRMED: bf16 is writing at 1e-6. The sweep's negatives need another "
              "explanation -- do not attribute them to precision.")


if __name__ == "__main__":
    main()
