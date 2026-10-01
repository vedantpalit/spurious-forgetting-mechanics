"""Static PNG/PDF versions of the three before-scaling figures, all built around population A
so the three read as one connected story rather than three separate metrics on three separate
populations:

  Plot 1 -- A in `disjoint` (A plays the opposite-half role in this condition -- the same
            structural role ballast played in high_overlap, confirmed symmetric). Whole-population
            decomposition: raw first_acc vs. rank_own_top1.
  Plot 2 -- A in `high_overlap` (A's own-half role, the mild-suppression contrast case),
            split by shared_count, in the SAME top1_rate units as Plot 1's rank_own_top1 line.
  Plot 3 -- A in `disjoint` (same condition as Plot 1), split by shared_count, in the same
            units -- this one is a literal decomposition of Plot 1's aggregate curve, and
            overlays that curve as a dashed reference so the connection is visible, not implied.

Re-parsed directly from results/*.out -- same source files as before. No new runs.
"""
import os
import re

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.35
plt.rcParams["grid.linewidth"] = 0.6

OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

C_RAW = "#b8502c"
C_RANKOWN = "#2c5f72"
C_B = "#8b93a0"
GROUP_COLORS = ["#3a6e8c", "#6e8a6b", "#b08a44", "#b0402e"]


def parse_crossover(path, pop_key):
    text = open(path, encoding="utf-8").read()
    out = {}
    for line in text.splitlines():
        mm = re.search(r"\[step (\d+)\]", line)
        if not mm:
            continue
        step = int(mm.group(1))
        pm = re.search(rf"{pop_key}: first_acc=([\d.]+)[^,]*rank_own_top1=([\d.]+)", line)
        dm = re.search(r"dataB: first_acc=([\d.]+)", line)
        if pm and dm:
            out[step] = {
                "pop_first_acc": float(pm.group(1)),
                "pop_rank_own_top1": float(pm.group(2)),
                "dataB_first_acc": float(dm.group(1)),
            }
    return out


def parse_graded(path):
    text = open(path, encoding="utf-8").read()
    blocks = re.split(r"=== step (\d+)", text)[1:]
    out = {}
    for i in range(0, len(blocks), 2):
        step = int(blocks[i])
        body = blocks[i + 1]
        primary = body.split("-- continuous regression")[0]
        groups = {}
        for m in re.finditer(
            r"shared_count=(\d)\s+n=\s*(\d+)\s+mean_rank_own=\s*([\-\d.]+)\s+top1_rate=([\-\d.]+)",
            primary,
        ):
            c, n, mr, t1 = m.groups()
            groups[int(c)] = {"n": int(n), "mean_rank_own": float(mr), "top1_rate": float(t1)}
        out[step] = groups
    return out


# ---- Plot 1 source: A in disjoint (opposite-half role), original crossover, 3 seeds ----
CROSS_FILES = {
    "seed0": "results/inject.disjoint.seed0.17464657.out",
    "seed1": "results/inject.disjoint.seed1.17464658.out",
    "seed2": "results/inject.disjoint.seed2.17464659.out",
}
cross = {k: parse_crossover(v, "dataA") for k, v in CROSS_FILES.items()}
abs_steps = sorted(cross["seed0"].keys())
pooled_cross = {}
for astep in abs_steps:
    step = astep - 8000
    pooled_cross[step] = {
        key: sum(cross[s][astep][key] for s in cross) / 3
        for key in ("pop_first_acc", "pop_rank_own_top1", "dataB_first_acc")
    }

# ---- Plot 2 / 3 source: graded A, split by shared_count, high_overlap and disjoint ----
GRADED_FILES = {
    "ho_seed0": "results/analyze_graded_key_overlap.ho_seed0.17467511.out",
    "ho_seed1": "results/analyze_graded_key_overlap.ho_seed1.17467512.out",
    "ho_seed2": "results/analyze_graded_key_overlap.ho_seed2.17467513.out",
    "dj_seed0": "results/analyze_graded_key_overlap.dj_seed0.17467514.out",
    "dj_seed1": "results/analyze_graded_key_overlap.dj_seed1.17467515.out",
    "dj_seed2": "results/analyze_graded_key_overlap.dj_seed2.17467516.out",
}
graded = {k: parse_graded(v) for k, v in GRADED_FILES.items()}
graded_steps = sorted(graded["ho_seed0"].keys())


def pool_condition(prefix, field):
    out = {}
    for step in graded_steps:
        out[step] = {
            c: sum(graded[f"{prefix}_{s}"][step][c][field] for s in ("seed0", "seed1", "seed2")) / 3
            for c in range(4)
        }
    return out


pooled_ho_rank = pool_condition("ho", "mean_rank_own")
pooled_dj_rank = pool_condition("dj", "mean_rank_own")
group_n = {c: graded["ho_seed0"][graded_steps[0]][c]["n"] for c in range(4)}


def weighted_mean(pooled, step):
    total_n = sum(group_n.values())
    return sum(group_n[c] * pooled[step][c] for c in range(4)) / total_n


# ================= Plot 1: A in disjoint, whole-population decomposition =================
fig, ax = plt.subplots(figsize=(7.5, 5), dpi=200)
steps0 = sorted(pooled_cross.keys())
xs = [s + 1 for s in steps0]
ax.plot(xs, [pooled_cross[s]["dataB_first_acc"] for s in steps0],
        color=C_B, linestyle="--", linewidth=1.6, label="B's own first_acc (context)")
ax.plot(xs, [pooled_cross[s]["pop_first_acc"] for s in steps0],
        color=C_RAW, linewidth=2.2, label="A raw first_acc (argmax, full vocab)")
ax.plot(xs, [pooled_cross[s]["pop_rank_own_top1"] for s in steps0],
        color=C_RANKOWN, linewidth=2.2, label="A rank_own_top1 (own-half-restricted)")
ax.set_xscale("log")
ax.set_xticks([1, 11, 101, 1001])
ax.set_xticklabels(["0", "10", "100", "1000"])
ax.set_xlim(1, 1300)
ax.set_ylim(0, 1.03)
ax.set_xlabel("injection step (log scale)")
ax.set_ylabel("accuracy / rank_own_top1")
ax.set_title("Plot 1 — Suppression vs. erosion: population A, disjoint\n"
              "(A is opposite-half here; crossover runs, mean of 3 seeds)")
ax.legend(loc="center right", frameon=True, fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "plot1_decomposition.png"), dpi=200)
fig.savefig(os.path.join(OUT_DIR, "plot1_decomposition.pdf"))
plt.close(fig)

# ================= Plot 2: A in high_overlap by shared_count (contrast case) =================
fig, ax = plt.subplots(figsize=(7.5, 5), dpi=200)
ref = [weighted_mean(pooled_ho_rank, s) for s in graded_steps]
ax.plot(graded_steps, ref, color="0.4", linestyle="--", linewidth=1.6,
        label="population mean (all shared_count pooled)")
for c in range(4):
    ax.plot(graded_steps, [pooled_ho_rank[s][c] for s in graded_steps],
             color=GROUP_COLORS[c], linewidth=2.2, marker="o", markersize=3.5,
             label=f"shared_count = {c}  (n={group_n[c]})")
ax.axhline(1, color="0.75", linewidth=0.8)
ax.set_xscale("log")
ax.set_xticks([10, 100, 1000])
ax.set_xticklabels(["10", "100", "1000"])
ax.set_xlim(9, 1300)
ax.set_ylim(1, 4.3)
ax.set_xlabel("injection step (log scale)")
ax.set_ylabel("mean rank_own (1 = best, higher = worse)")
ax.set_title("Plot 2 — A in high_overlap, by shared_count (contrast: A's own-half role)\n"
              "same population as Plot 1; graded population, mean of 3 seeds")
ax.legend(loc="upper left", frameon=True, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "plot2_key_overlap_high_overlap.png"), dpi=200)
fig.savefig(os.path.join(OUT_DIR, "plot2_key_overlap_high_overlap.pdf"))
plt.close(fig)

# ================= Plot 3: A in disjoint by shared_count (decomposes Plot 1) =================
fig, ax = plt.subplots(figsize=(7.5, 5), dpi=200)
# reference line: the n-weighted mean across all four shared_count groups, in the SAME
# mean_rank_own units as the group lines (Plot 1 uses a different metric -- first_acc /
# rank_own_top1 -- so its curve can't be overlaid directly once this reverts to mean_rank_own).
ref = [weighted_mean(pooled_dj_rank, s) for s in graded_steps]
ax.plot(graded_steps, ref, color="0.4", linestyle="--", linewidth=1.6,
        label="population mean (all shared_count pooled) — same shape as Plot 1")
for c in range(4):
    ax.plot(graded_steps, [pooled_dj_rank[s][c] for s in graded_steps],
             color=GROUP_COLORS[c], linewidth=2.2, marker="o", markersize=3.5,
             label=f"shared_count = {c}  (n={group_n[c]})")
ax.axhline(1, color="0.75", linewidth=0.8)
ax.set_xscale("log")
ax.set_xticks([10, 100, 1000])
ax.set_xticklabels(["10", "100", "1000"])
ax.set_xlim(9, 1300)
ax.set_ylim(1, 14.5)
ax.set_xlabel("injection step (log scale)")
ax.set_ylabel("mean rank_own (1 = best, higher = worse)")
ax.set_title("Plot 3 — A in disjoint, by shared_count (decomposes Plot 1's curve)\n"
              "same population and condition as Plot 1; graded population, mean of 3 seeds")
leg3 = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, fontsize=9)
fig.savefig(os.path.join(OUT_DIR, "plot3_key_overlap_disjoint.png"), dpi=200,
            bbox_extra_artists=(leg3,), bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "plot3_key_overlap_disjoint.pdf"),
            bbox_extra_artists=(leg3,), bbox_inches="tight")
plt.close(fig)

print("Wrote:", sorted(os.listdir(OUT_DIR)))
