"""Figures for the user-risk aggregation study (reads results/*.csv only; dev OOF)."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})
BLUE, GREY, LIGHT, RED = "#2a6fdb", "#7a7a7a", "#bdbdbd", "#c0392b"

cmp = pd.read_csv(RES / "policy_comparison.csv")
dur = pd.read_csv(RES / "duration_bias_analysis.csv")
ecdf = pd.read_csv(RES / "benign_max_by_horizon_long.csv")
grp = pd.read_csv(RES / "policy_by_duration_stealth.csv")

# 1 ---------------------------------------------------------------- risk_peak vs observation duration
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
cs = dur[dur.section == "cross_sectional_by_duration_group"]
order = ["(0, 30]", "(30, 60]", "(60, 90]", "(90, 104]", "(104, 200]"]
counts = {}
for who, c, off in (("benign", GREY, -0.12), ("attackers", BLUE, 0.12)):
    g = cs[cs.subset == who].set_index("group").reindex(order)
    counts[who] = g.n.fillna(0).astype(int).tolist()
    x = np.arange(len(order)) + off
    ax[0].errorbar(x, g.share_risk_peak_gt_85, yerr=[g.share_risk_peak_gt_85 - g.ci_low, g.ci_high - g.share_risk_peak_gt_85],
                   fmt="o", color=c, capsize=3, label=f"{who} (95% CI)")
ax[0].set_xticks(range(len(order)))
ax[0].set_xticklabels([f"{o}\nben {b} / att {a}" for o, b, a in zip(order, counts["benign"], counts["attackers"])], fontsize=8)
ax[0].set_xlabel("observed workdays (dev users, natural variation) and users per group")
ax[0].set_ylabel("share with risk_peak > 85")
ax[0].set_ylim(-0.05, 1.05)
ax[0].set_title("Cross-section: 838 of 859 users have 105+ days; too few short users to test", fontsize=10)
ax[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.9))
hc = dur[(dur.section == "benign_horizon_curve")]
for T, c in ((80, LIGHT), (85, BLUE), (90, GREY)):
    g = hc[(hc.threshold == T) & (hc.horizon != "full")].copy()
    g["H"] = g.horizon.astype(int)
    ax[1].plot(g.H, g.share_first_H_max_above, "o-", color=c, label=f"benign, max over first H days > {T}")
    if T == 85:
        ax[1].fill_between(g.H, g.ci_low, g.ci_high, color=c, alpha=0.15)
        ax[1].plot(g.H, g.independence_prediction, ":", color=RED, label="if days were independent (85)")
ax[1].set_xlabel("observation horizon H (workdays, controlled)")
ax[1].set_ylabel("benign users alerted (share)")
ax[1].set_title("Controlled exposure: benign alert rate grows with H", fontsize=10)
ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig1_risk_peak_vs_duration.png")
plt.close(fig)

# 2 ---------------------------------------------------------------- benign max-risk ECDF by horizon
fig, ax = plt.subplots(figsize=(7, 4.2))
order = ["10", "20", "30", "60", "90", "full"]
shades = np.linspace(0.25, 1.0, len(order))
for H, s in zip(order, shades):
    v = np.sort(ecdf.loc[ecdf.horizon == H, "max_risk"].to_numpy())
    ax.plot(v, 1 - np.arange(1, len(v) + 1) / len(v), color=BLUE, alpha=s, label=f"H = {H}")
ax.axvline(85, color=RED, lw=1, ls="--")
ax.text(85.5, 0.5, "85", color=RED)
ax.set_yscale("log")
ax.set_xlabel("benign user's maximum daily risk over the horizon")
ax.set_ylabel("share of benign users above x (log)")
ax.set_title("Benign maximum risk shifts right as the horizon grows", fontsize=10)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig2_benign_max_by_horizon.png")
plt.close(fig)

# 3 ---------------------------------------------------------------- recall vs FPR
fig, ax = plt.subplots(figsize=(8, 5.5))
fam_colors = {"A current max": BLUE, "G persistence": "#16a085", "G persistence + escalation": "#27ae60",
              "E EWMA": "#8e44ad", "C top-k": "#d35400", "D percentile": "#b7950b", "B fixed window": LIGHT,
              "F exposure-adjusted": RED}
for fam, c in fam_colors.items():
    g = cmp[cmp.family == fam]
    ax.scatter(g.FPR * 100, g.recall * 100, color=c, s=22, label=fam, alpha=0.85)
iso = cmp[cmp.family.str.contains("iso-FPR")]
ax.scatter(iso.FPR * 100, iso.recall * 100, marker="x", color="black", s=25, label="any method at cross-fitted iso-FPR")
b = cmp[cmp.policy_id == "max_full>85.00"].iloc[0]
ax.errorbar(b.FPR * 100, b.recall * 100, xerr=[[100 * (b.FPR - b.FPR_ci_low)], [100 * (b.FPR_ci_high - b.FPR)]],
            yerr=[[100 * (b.recall - b.recall_ci_low)], [100 * (b.recall_ci_high - b.recall)]], fmt="*", ms=14,
            color=BLUE, mec="black", label="current: risk_peak > 85")
ax.set_xlim(-0.3, 9)
ax.set_ylim(40, 101)
ax.set_xlabel("benign employees alerted (FPR, %)")
ax.set_ylabel("attackers alerted (user-level recall, %)")
ax.set_title("Recall vs FPR of candidate policies (dev OOF, mean of 3 repeats)", fontsize=10)
ax.legend(frameon=False, fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "fig3_recall_vs_fpr.png")
plt.close(fig)

# 4 ---------------------------------------------------------------- SOC volume vs threshold
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
lines = [("A current max", "max_full>", "single day > T", BLUE, "-"),
         ("G persistence", "persist_2of3>", ">= 2 of 3 days > T", "#16a085", "-"),
         ("G persistence", "persist_3of5>", ">= 3 of 5 days > T", "#16a085", "--"),
         ("E EWMA", "ewma_h5>", "EWMA(h=5) > T", "#8e44ad", "-")]
for fam, pre, lab, c, ls in lines:
    g = cmp[cmp.policy_id.str.startswith(pre) & ~cmp.policy_id.str.contains("_or_")].sort_values("threshold")
    ax[0].plot(g.threshold, g.benign_employees_alerted_per_100_benign, ls, marker="o", color=c, label=lab)
    ax[1].plot(g.threshold, g.benign_incidents_per_100_benign_employee_months, ls, marker="o", color=c, label=lab)
ax[0].set_ylabel("benign employees alerted per 100 benign")
ax[1].set_ylabel("benign incidents per 100 benign employee-months")
for a in ax:
    a.set_xlabel("threshold T on daily risk")
    a.axvline(85, color=RED, lw=0.8, ls=":")
ax[0].legend(frameon=False, fontsize=8)
fig.suptitle("SOC workload vs threshold (dev OOF; ~5 months of monitoring per employee)", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "fig4_alert_volume_vs_threshold.png")
plt.close(fig)

# 5 ---------------------------------------------------------------- latency by policy
sel = cmp[cmp.latency_median.notna() & cmp.threshold.isin([80, 85, 90])].copy()
sel = sel.sort_values("latency_median")
fig, ax = plt.subplots(figsize=(8, 0.28 * len(sel) + 1.2))
yy = np.arange(len(sel))
ax.hlines(yy, sel.latency_median, sel.latency_p90, color=LIGHT, lw=2, label="median to P90")
ax.plot(sel.latency_p75, yy, "|", color=GREY, ms=10, label="P75")
ax.plot(sel.latency_median, yy, "o", color=BLUE, label="median")
ax.set_yticks(yy)
ax.set_yticklabels([f"{p}  (window det. {d:.0%})" for p, d in zip(sel.policy_id, sel.attack_window_detection)], fontsize=7)
ax.set_xlabel("workdays from the first injected day to the first in-window alert")
ax.set_title("Detection latency by alert policy (detected attackers only)", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "fig5_latency_by_policy.png")
plt.close(fig)

# 6 ---------------------------------------------------------------- detection by duration x stealth
pols = ["max_full>85.00", "max_full>90.00", "persist_2of3>85", "persist_3of5>85", "persist_2of3>85_or_day>98", "ewma_h5>80"]
g = grp[grp.policy_id.isin(pols) & (grp.group_type == "attack_duration")]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
cols = [BLUE, GREY, "#16a085", "#1abc9c", "#27ae60", "#8e44ad"]
for p, c in zip(pols, cols):
    h = g[g.policy_id == p].copy()
    h["d"] = h.group.astype(float)
    h = h.sort_values("d")
    ax[0].plot(h.d, h.attack_window_detection, "o-", color=c, label=p)
s = grp[grp.policy_id.isin(pols) & (grp.group_type == "stealth")]
w = 0.13
for i, (p, c) in enumerate(zip(pols, cols)):
    h = s[s.policy_id == p].set_index("group").reindex(["Low", "Medium", "High"])
    ax[1].bar(np.arange(3) + (i - 2.5) * w, h.attack_window_detection, width=w, color=c, label=p)
ax[1].set_xticks(range(3))
ax[1].set_xticklabels(["Low (0.75)", "Medium (0.50)", "High (0.25)"])
ax[0].set_xlabel("attack duration (workdays)")
ax[1].set_xlabel("stealth level (injection multiplier)")
for a in ax:
    a.set_ylabel("attack-window detection")
    a.set_ylim(0, 1.05)
ax[0].legend(frameon=False, fontsize=7)
fig.suptitle("Short and high-stealth attacks under each alert policy (dev OOF)", fontsize=10)
fig.tight_layout()
fig.savefig(FIG / "fig6_detection_by_duration_stealth.png")
plt.close(fig)
print("figures written:", sorted(p.name for p in FIG.glob("*.png")))
