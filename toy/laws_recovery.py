"""The recovery-level law, refitted on the minimal model.

Claim: the shift's between-half content <s, w> is withdrawn to a level K that does not
depend on beta, n_B or the rate (the balance), while its peak carries all the variation; so
the reversal fraction is  f = 1 - K / x_peak  with K fitted once.

    python laws_recovery.py     # beta, n_B, dose sweeps + the ten-seed base; matched cells
                                # with a collapse (peak <s,w> >= 0.3)
"""
import json
import numpy as np

rows = []
for f, tag in (("sw_beta.json", "beta"), ("sw_nb.json", "nb"), ("sw_dose.json", "dose"), ("paper_base.json", "base")):
    for r in json.load(open(f)).values():
        if r["norm"] != 1 or (not r.get("matched", True) and tag != "dose"):
            continue
        sw = np.array([x["s_on_w"] for x in r["rows"]]); pk = sw.argmax()
        if sw[pk] < 0.3:
            continue
        rows.append(dict(tag=tag, beta=r["beta"], nb=r.get("nb", 128), peak=sw[pk], end=sw[-1], f=1 - sw[-1] / sw[pk]))
R = {k: np.array([x[k] for x in rows]) for k in ("beta", "nb", "peak", "end", "f")}
K = R["end"].mean(); fp = 1 - K / R["peak"]; fm = R["f"]
r2 = 1 - ((fm - fp) ** 2).sum() / ((fm - fm.mean()) ** 2).sum()
print(f"{len(rows)} runs | end level K = {K:.3f} +/- {R['end'].std():.3f} (cv {R['end'].std() / K:.2f}); "
      f"corr(end, beta) {np.corrcoef(R['end'], R['beta'])[0, 1]:+.2f}, corr(end, log n_B) {np.corrcoef(R['end'], np.log(R['nb']))[0, 1]:+.2f}")
print(f"peak: cv {R['peak'].std() / R['peak'].mean():.2f}; corr(peak, beta) {np.corrcoef(R['peak'], R['beta'])[0, 1]:+.2f}, "
      f"corr(peak, log n_B) {np.corrcoef(R['peak'], np.log(R['nb']))[0, 1]:+.2f}")
print(f"f = 1 - K/x_peak: R^2 {r2:.3f}, corr {np.corrcoef(fm, fp)[0, 1]:.3f}, bias {np.mean(fp - fm):+.3f}")
