"""Figures of the switching-weight study, drawn from results/ only. Writes figures/*.png."""

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
prim = json.loads((RES / "analysis_primary.json").read_text())
shares = pd.read_csv(RES / "shap_shares.csv").set_index("arm")
ARMS = ["W0", "A0", "W2", "W5", "W10"]
lab = [f"{a}\n{shares.loc[a, 'shap_share_switching'] * 100:.0f}% share" for a in ARMS]

# 1. dose-response: dev attackers (left) and existing matched cases (right), iso-FPR 3%
d = pd.read_csv(RES / "analysis_p1_detection_table.csv")
d = d[(d.fpr == 0.03) & (d.group == "all")].set_index("arm")
p2 = pd.read_csv(RES / "analysis_p2_detection_table.csv")
p2 = p2[p2.fpr == 0.03].pivot_table(index="arm", columns="variant", values="det")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
ax = axes[0]
x = range(len(ARMS))
ax.errorbar(x, d.loc[ARMS, "det"], yerr=[d.loc[ARMS, "det"] - d.loc[ARMS, "ci_low"], d.loc[ARMS, "ci_high"] - d.loc[ARMS, "det"]],
            fmt="o-", color=S1, ms=7, lw=2, elinewidth=2, capsize=0, markeredgecolor=SURF, markeredgewidth=2)
ax.set_title("Existing dev attacks (348 attackers)", loc="left", fontsize=10)
ax.set_xticks(list(x), lab, fontsize=8.5)
ax.set_ylabel("Detected (iso-FPR 3%)")
ax.set_ylim(0.5, 1.0)
ax = axes[1]
for v, c, nm in (("BLK", S2, "Block (BLK)"), ("ROT", S1, "Rotating (ROT)"), ("SIM", S3, "Simultaneous (SIM)")):
    ax.plot(list(x), p2.loc[ARMS, v], "o-", color=c, ms=7, lw=2, markeredgecolor=SURF, markeredgewidth=2, label=nm)
ax.set_title("Existing matched cases (145 hosts)", loc="left", fontsize=10)
ax.set_xticks(list(x), lab, fontsize=8.5)
ax.legend(frameon=False, loc="lower left", fontsize=8.5)
fig.suptitle("Forcing reliance on switching features lowers detection — most for rotating attacks", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig(FIG / "fig1_dose_response.png", dpi=160)
plt.close(fig)

# 2. forest plot of the registered estimands
est = pd.read_csv(RES / "analysis_estimands.csv").set_index("id")
ids = ["PRIMARY_W2", "PRIMARY_W5", "PRIMARY_W10", "S1", "S2", "S3"]
names = {"PRIMARY_W2": "Primary  W2 − A0 (3%)", "PRIMARY_W5": "Primary  W5 − A0 (3%)", "PRIMARY_W10": "Primary  W10 − A0 (3%)",
         "S1": "S1  W5 − A0 at 1%", "S2": "S2  W5 − A0: High- minus Low-switching", "S3": "S3  cases: [W5 − A0](ROT) − [W5 − A0](BLK)"}
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.axvspan(-0.02, 0.02, color="#f0efec", zorder=0, label="±2 points (equivalence band)")
ax.axvline(0, color=INK2, lw=1)
for i, k in enumerate(ids):
    r = est.loc[k]
    c = S2 if k.startswith("PRIMARY") else S1
    ax.plot([r.ci_low, r.ci_high], [i, i], color=c, lw=2, solid_capstyle="round")
    ax.plot([r.ci90_low, r.ci90_high], [i, i], color=c, lw=5, alpha=0.45, solid_capstyle="round")
    ax.plot(r.estimate, i, "o", ms=7, color=c, markeredgecolor=SURF, markeredgewidth=2)
    hp = r.p_holm if pd.notna(r.get("p_holm")) else r.p_holm_over_w
    ax.text(0.03, i, f"{r.estimate * 100:+.2f} pts   Holm p = {hp:.3f}", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(range(len(ids)), [names[k] for k in ids])
ax.invert_yaxis()
ax.set_xlim(-0.13, 0.09)
ax.set_xlabel("Difference in share detected (thin: 95% CI, thick: 90% CI)")
fig.legend(frameon=False, loc="lower center", fontsize=8.5)
ax.set_title(f"Registered estimands (frozen rules give Outcome {prim['outcome']})", loc="left", fontsize=10.5)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(FIG / "fig2_registered_estimands.png", dpi=160)
plt.close(fig)
print("figures written:", sorted(p.name for p in FIG.glob("*.png")))
