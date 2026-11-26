"""Figures of the controlled rotation study, drawn from results/analysis_* only. Writes figures/*.png."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
FIG.mkdir(exist_ok=True)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.axisbelow": True})
dt = pd.read_csv(RES / "analysis_detection_table.csv")
est = pd.read_csv(RES / "analysis_estimands.csv")
prim = json.loads((RES / "analysis_primary.json").read_text())

# 1. Det per arm, ROT vs BLK (H146, regime 1, iso-FPR 3%)
arms = ["B", "C", "C+", "A-cv", "D"]
d = dt[(dt.regime == 1) & (dt.cohort == "H146") & (dt.L == "pooled") & (dt.fpr == 0.03)].set_index(["arm", "variant"])
fig, ax = plt.subplots(figsize=(7, 4))
for off, v, c in ((-0.09, "ROT", S1), (0.09, "BLK", S2)):
    x = [i + off for i in range(len(arms))]
    y = [d.loc[(a, v), "det"] for a in arms]
    lo = [d.loc[(a, v), "det"] - d.loc[(a, v), "ci_low"] for a in arms]
    hi = [d.loc[(a, v), "ci_high"] - d.loc[(a, v), "det"] for a in arms]
    ax.errorbar(x, y, yerr=[lo, hi], fmt="o", ms=7, color=c, ecolor=c, elinewidth=2, capsize=0,
                markeredgecolor=SURF, markeredgewidth=2, label={"ROT": "Rotating (ROT)", "BLK": "Block (BLK)"}[v])
ax.set_xticks(range(len(arms)), ["B\nper-channel", "C\n+ aggregated", "C+\n+ distributional", "A-cv\n+ transition (MUBSIR)",
                                 "D\n+ new switching"])
ax.set_ylabel("Cases detected (iso-FPR 3%)")
ax.set_ylim(0.4, 0.95)
ax.legend(frameon=False, loc="lower right")
ax.set_title("Same days, same per-channel deviation, different order: ROT and BLK detected alike", loc="left", fontsize=10.5)
fig.tight_layout()
fig.savefig(FIG / "fig1_detection_by_arm_rot_vs_blk.png", dpi=160)
plt.close(fig)

# 2. Forest plot of the registered estimands
ids = ["PRIMARY", "S1", "S2", "S3", "S4", "S5", "S6_L6", "S6_L12", "S7", "S8"]
lab = {"PRIMARY": "Primary  Δ = [D−C+](ROT) − [D−C+](BLK), 3%", "S1": "S1  Δ at iso-FPR 1%",
       "S2": "S2  D − C+ on ROT", "S3": "S3  Δ with A-cv instead of D", "S4": "S4  Δ, regime 2 (trained on cases)",
       "S5": "S5  [C+−C](BLK) − [C+−C](FIX)", "S6_L6": "S6  Δ, L = 6", "S6_L12": "S6  Δ, L = 12",
       "S7": "S7  Δ, relaxed cohort (249 hosts)", "S8": "S8  Δ, strict natural-activity rule (36 hosts)"}
e = est.set_index("id").loc[ids]
fig, ax = plt.subplots(figsize=(10, 5.4))
ax.axvspan(-0.05, 0.05, color="#f0efec", zorder=0, label="±5 points (Outcome 3 band)")
ax.axvline(0, color=INK2, lw=1)
ax.axvline(0.02, color=INK2, lw=1, ls=(0, (3, 3)), label="+2 points (Outcome 1 minimum)")
for i, k in enumerate(ids):
    r = e.loc[k]
    c = S2 if k == "S5" else S1
    ax.plot([r.ci_low, r.ci_high], [i, i], color=c, lw=2, solid_capstyle="round")
    ax.plot(r.estimate, i, "o", ms=7, color=c, markeredgecolor=SURF, markeredgewidth=2)
    note = f"{r.estimate * 100:+.1f} pts" + ("" if k == "PRIMARY" else f"   Holm p = {r.p_holm:.3f}")
    ax.text(0.085, i, note, va="center", fontsize=8.5, color=INK2)
ax.set_yticks(range(len(ids)), [lab[k] for k in ids])
ax.invert_yaxis()
ax.set_xlim(-0.12, 0.2)
ax.set_xticks([-0.1, -0.05, 0, 0.02, 0.05, 0.1])
ax.set_xlabel("Difference in share of cases detected (95% host-bootstrap CI)")
fig.legend(frameon=False, loc="lower center", ncol=2, fontsize=8.5)
ax.set_title(f"Registered estimands (frozen rules give Outcome {prim['outcome']})", loc="left", fontsize=10.5)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(FIG / "fig2_registered_estimands.png", dpi=160)
plt.close(fig)

# 3. Det by variant and length for arms C, C+, D
vs = ["ROT", "BLK", "FIX-EMAIL", "FIX-USB", "FIX-HTTP", "SIM"]
d = dt[(dt.regime == 1) & (dt.cohort == "H146") & (dt.fpr == 0.03) & (dt.L != "pooled")].copy()
d["L"] = d["L"].astype(int)
d = d.set_index(["arm", "variant", "L"])
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, L in zip(axes, (6, 12)):
    for off, a, c in ((-0.18, "C", S1), (0, "C+", S2), (0.18, "D", S3)):
        x = [i + off for i in range(len(vs))]
        ax.bar(x, [d.loc[(a, v, L), "det"] for v in vs], width=0.16, color=c, label=a, edgecolor=SURF, linewidth=1)
    ax.set_xticks(range(len(vs)), ["ROT", "BLK", "FIX\nEmail", "FIX\nUSB", "FIX\nHTTP", "SIM*"])
    ax.set_title(f"Attack length L = {L} workdays", loc="left", fontsize=10)
    ax.set_ylim(0, 1.02)
axes[0].set_ylabel("Cases detected (iso-FPR 3%)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, ["C (aggregated)", "C+ (+ distributional)", "D (+ new switching)"], frameon=False, loc="upper center", ncol=3, fontsize=9)
fig.text(0.01, 0.01, "*SIM carries more deviation than the matched variants (descriptive only).", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.92))
fig.savefig(FIG / "fig3_detection_by_variant_and_length.png", dpi=160)
plt.close(fig)
print("figures written:", sorted(p.name for p in FIG.glob("*.png")))
