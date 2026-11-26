"""Learning-curve figure, drawn from results/ only. Writes figures/fig1_learning_curve.png."""

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
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.axisbelow": True})
cv = pd.read_csv(RES / "analysis_curve.csv")
sub = pd.read_csv(RES / "analysis_subgroups.csv")
prim = json.loads((RES / "analysis_primary.json").read_text())
p = prim["primary"]
x = cv.fit_users.to_numpy()

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
ax = axes[0]
for col, c, nm in (("det_3pct", S1, "iso-FPR 3%"), ("det_1pct", S2, "iso-FPR 1%")):
    ax.plot(x, cv[col], "o-", color=c, lw=2, ms=7, markeredgecolor=SURF, markeredgewidth=2)
    ax.annotate(nm, (x[-1], cv[col].iloc[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK2)
ax.axvspan(cv.fit_users[cv.arm == "F86"].iloc[0], x[-1], color="#f0efec", zorder=0)
ax.text((cv.fit_users[cv.arm == "F86"].iloc[0] + x[-1]) / 2, 0.705,
        f"cost of reserving\nthe test (3%):\n{p['estimate'] * 100:+.2f} pts\n90% CI {p['ci90_low'] * 100:+.2f} to {p['ci90_high'] * 100:+.2f}",
        ha="center", va="bottom", fontsize=8.5, color=INK2)
ax.set_xticks(x, [f"{a}\n{n:.0f}" for a, n in zip(cv.arm, x)], fontsize=8.5)
ax.set_xlabel("arm and mean fit users per fold")
ax.set_ylabel("Attackers detected (dev, 348)")
ax.set_title("All dev attackers", loc="left", fontsize=10)
ax.set_xlim(x[0] - 25, x[-1] + 95)
ax.set_ylim(0.4, 1.0)

ax = axes[1]
order = [("all attackers", S1, 0), ("High stealth", S2, 0), ("U+H (no email)", S3, 7), ("5-day attacks", S4, -7)]
for g, c, dy in order:
    d = sub[sub.group == g].set_index("arm").loc[cv.arm]
    ax.plot(x, d.det_3pct, "o-", color=c, lw=2, ms=6, markeredgecolor=SURF, markeredgewidth=2)
    ax.annotate(f"{g} (n={int(d.n.iloc[0])})", (x[-1], d.det_3pct.iloc[-1]), xytext=(8, dy), textcoords="offset points",
                va="center", fontsize=8.5, color=INK2)
ax.set_xticks(x, [f"{a}\n{n:.0f}" for a, n in zip(cv.arm, x)], fontsize=8.5)
ax.set_xlabel("arm and mean fit users per fold")
ax.set_title("Weakest groups (iso-FPR 3%, descriptive)", loc="left", fontsize=10)
ax.set_xlim(x[0] - 25, x[-1] + 230)
fig.suptitle(f"Learning curve: detection still rises slowly with more data; reserving 13.7% for test costs little "
             f"(Outcome {prim['outcome']})", x=0.01, ha="left", fontsize=10.5)
fig.tight_layout()
fig.savefig(FIG / "fig1_learning_curve.png", dpi=160)
plt.close(fig)
print("figures written:", sorted(q.name for q in FIG.glob("*.png")))
