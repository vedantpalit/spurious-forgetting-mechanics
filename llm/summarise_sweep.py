"""One table for the whole LR sweep.

Reads every llm/out/inject_lr*.json and reports each arm against the pre-registered readings.
The headline is A's NONCOPY stratum; `copy` is shown beside it because in-context copying may
crash differently from parametric recall, and B's curve is shown because recovery tracks B's
acquisition rather than step count, so an arm where B never learns is not scoreable either way.

Run:
  .venv-llm/bin/python -m llm.summarise_sweep
"""
import argparse
import glob
import json
import math


def at(curve, key):
    return [c.get(key, float("nan")) for c in curve]


def crossing(steps, vals, level):
    """First step where `vals` reaches `level`, or None."""
    for s, v in zip(steps, vals):
        if not math.isnan(v) and v >= level:
            return s
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="llm/out/inject_lr*.json")
    a = ap.parse_args()
    runs, skipped = [], []
    for path in sorted(glob.glob(a.glob)):
        d = json.load(open(path, encoding="utf-8"))
        # Runs predating fp32 master weights measured rounding, not learning. They
        # have no tokens_per_step field, and mixing them into this table would be silent.
        if not d.get("tokens_per_step"):
            skipped.append(path)
            continue
        runs.append((d["lr"], path, d))
    for path in skipped:
        print(f"SKIPPED (pre-fp32, voided): {path}")
    if not runs:
        raise SystemExit(f"no runs matched {a.glob}")
    runs.sort(key=lambda r: r[0])

    print(f"{'lr':>8} {'base':>6} {'trough':>7} {'@':>5} {'peak':>6} {'@':>5} {'rec':>6} "
          f"{'end':>6} | {'rankT':>6} {'rankP':>6} | {'B@tr':>6} {'B@pk':>6} {'B_end':>6} "
          f"{'B>.5':>5} | reading")
    for lr, path, d in runs:
        c = d["curve"]
        steps = [x["step"] for x in c]
        nc = at(c, "A/noncopy/acc")
        # THE TROUGH IS THE CRASHED POINT WITH THE LARGEST SUBSEQUENT REBOUND.
        # Not the global minimum: these curves crash, recover partially, then decline for
        # thousands of steps, so the global minimum is the final step and a naive reading
        # reports zero recovery everywhere. And not simply the largest rebound either -- every
        # arm RISES ~0.05 before it crashes (format priming), and at 3e-6 that rise
        # outsizes the real recovery, which made the rule pick step 0. A trough must therefore
        # first be a crash: at least CRASH below the running maximum up to that point.
        CRASH = 0.05
        cand = [k for k in range(len(nc)) if nc[k] <= max(nc[:k + 1]) - CRASH]
        i = max(cand, key=lambda k: max(nc[k:]) - nc[k]) if cand else 0
        peak = max(nc[i:])
        pk = i + nc[i:].index(peak)
        drop = nc[0] - nc[i]
        rec = (peak - nc[i]) / drop if drop >= 0.05 else float("nan")
        cp = at(c, "A/copy/acc")
        rk = at(c, "A/noncopy/rank")
        b = at(c, "B/ALL/acc")
        b50, b90 = crossing(steps, b, 0.5), crossing(steps, b, 0.9)

        if math.isnan(rec):
            rec = float("nan")
        if b[-1] < 0.5:
            read = "4: B never learned -- UNSCOREABLE"
        elif nc[0] - nc[i] < 0.05:
            read = "3: no crash while B learns"
        elif rec == rec and rec > 0.1:
            read = "1: CRASH AND RECOVERY"
        else:
            read = "2: crash, no recovery"
        print(f"{lr:>8.0e} {nc[0]:>6.3f} {nc[i]:>7.3f} {steps[i]:>5} {peak:>6.3f} "
              f"{steps[pk]:>5} {rec:>6.3f} {nc[-1]:>6.3f} | {rk[i]:>6.2f} {rk[pk]:>6.2f} | "
              f"{b[i]:>6.3f} {b[pk]:>6.3f} {b[-1]:>6.3f} {str(b50):>5} | {read}")

    print("\nper-arm A curves (noncopy accuracy, and B beside it):")
    for lr, path, d in runs:
        c = d["curve"]
        print(f"\n  lr={lr:g}")
        for x in c:
            print(f"    step {x['step']:>4}  noncopy={x['A/noncopy/acc']:.4f}"
                  f"+-{x['A/noncopy/acc_se']:.4f}  copy={x['A/copy/acc']:.4f}"
                  f"  rank={x.get('A/noncopy/rank', float('nan')):.3f}"
                  f"  b_typed={x['Atouch/b_typed/acc']:.4f}"
                  f"  untouched={x['Atouch/untouched/acc']:.4f}"
                  f"  B={x['B/ALL/acc']:.4f}")


if __name__ == "__main__":
    main()
