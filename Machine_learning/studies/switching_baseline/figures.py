"""Figures of the switching-baseline study, drawn from results/analysis_* only. Writes figures/*.png."""

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
S1, S2 = "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.axisbelow": True})
est = pd.read_csv(RES / "analysis_estimands.csv").set_index("id")
prim = json.loads((RES / "analysis_primary.json").read_text())

# 1. forest plot of the registered estimands
ids = ["PRIMARY", "S2", "S1", "S3", "S4", "S5", "S6", "S7"]
lab = {"PRIMARY": "Primary  Δ₁ = H − A (personal habit vs 100 features), 3%",
       "S2": "Key  Δ₂ = H − R (personal vs population reference), 3%", "S1": "S1  Δ₁ at iso-FPR 1%",
       "S3": "S3  Δ₂ low-habit minus high-habit attackers", "S4": "S4  Δ₁ High- minus Low-switching attackers",
       "S5": "S5  existing cases: [H−A](ROT) − [H−A](BLK)", "S6": "S6  existing cases: [H−R](ROT) − [H−R](BLK)",
       "S7": "S7  row-level PR-AUC, H − A"}
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.axvspan(-0.03, 0.03, color="#f0efec", zorder=0, label="±3 points (Outcome 3 band)")
ax.axvline(0, color=INK2, lw=1)
ax.axvline(0.02, color=INK2, lw=1, ls=(0, (3, 3)), label="+2 points (Outcome 1 minimum)")
for i, k in enumerate(ids):
    r = est.loc[k]
    c = S2 if k in ("PRIMARY", "S2") else S1
    ax.plot([r.ci_low, r.ci_high], [i, i], color=c, lw=2, solid_capstyle="round")
    ax.plot(r.estimate, i, "o", ms=7, color=c, markeredgecolor=SURF, markeredgewidth=2)
    note = f"{r.estimate * 100:+.2f} pts" if k != "S7" else f"{r.estimate:+.4f} AP"
    if k not in ("PRIMARY",) and pd.notna(r.get("p_holm")):
        note += f"   Holm p = {r.p_holm:.3f}"
    ax.text(0.055, i, note, va="center", fontsize=8.5, color=INK2)
ax.set_yticks(range(len(ids)), [lab[k] for k in ids])
ax.invert_yaxis()
ax.set_xlim(-0.06, 0.12)
ax.set_xlabel("Difference (share detected, or average precision for S7), 95% bootstrap CI")
fig.legend(frameon=False, loc="lower center", ncol=2, fontsize=8.5)
ax.set_title(f"Registered estimands (frozen rules give Outcome {prim['outcome']})", loc="left", fontsize=10.5)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(FIG / "fig1_registered_estimands.png", dpi=160)
plt.close(fig)

# 2. detection by arm within habit tertiles and attack switching levels (P1, 3%)
d = pd.read_csv(RES / "analysis_p1_detection_table.csv")
d = d[d.fpr == 0.03].set_index("group")
groups = ["all", "habit_low", "habit_mid", "habit_high", "switching_Low", "switching_Medium", "switching_High"]
names = ["All\nattackers", "Habit\nlow", "Habit\nmid", "Habit\nhigh", "Switching\nLow", "Switching\nMedium", "Switching\nHigh"]
fig, ax = plt.subplots(figsize=(9.5, 4))
for off, a, c, nm in ((-0.2, "A", "#8a8984", "A: 100 features"), (0, "R", S1, "R: + population reference"),
                      (0.2, "H", S2, "H: + personal 2010 habit")):
    ax.bar([i + off for i in range(len(groups))], [d.loc[g, f"det_{a}"] for g in groups], width=0.18, color=c,
           edgecolor=SURF, linewidth=1, label=nm)
ax.set_xticks(range(len(groups)), [f"{n}\n(n={int(d.loc[g, 'attackers'])})" for n, g in zip(names, groups)], fontsize=8.5)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Attackers detected (iso-FPR 3%)")
ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, 1.13))
fig.tight_layout()
fig.savefig(FIG / "fig2_detection_by_habit_and_switching.png", dpi=160)
plt.close(fig)
print("figures written:", sorted(p.name for p in FIG.glob("*.png")))
