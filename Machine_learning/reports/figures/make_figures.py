"""
Figures for Final_Detailed_Report.md, drawn only from existing result files (and, for the threshold trade-off,
from the frozen development OOF scores). Nothing is re-trained and no decision is taken. Writes reports/figures/*.png.
Run from Model/:  .venv/Scripts/python reports/figures/make_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FIG = Path(__file__).resolve().parent
M = FIG.parent.parent
RES = M / "results"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.titlelocation": "left", "axes.titlesize": 11})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=160)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- 1. confusion matrix (test, score > 85)
u = pd.read_csv(RES / "13_user_classification_metrics.csv").iloc[0]
cells = [[("Caught", int(u.TP), True), ("Missed", int(u.FN), False)],
         [("False alarm", int(u.FP), False), ("Correctly left alone", int(u.TN), True)]]
fig, ax = plt.subplots(figsize=(6.2, 4.4))
ax.grid(False)
for i, row in enumerate(cells):
    for j, (lab, n, ok) in enumerate(row):
        ax.add_patch(plt.Rectangle((j + 0.02, 1 - i + 0.02), 0.96, 0.96, color=BLUE if ok else ORANGE, alpha=0.9 if ok else 0.85))
        ax.text(j + 0.5, 1 - i + 0.58, str(n), ha="center", va="center", fontsize=26, color="white", weight="bold")
        ax.text(j + 0.5, 1 - i + 0.30, lab, ha="center", va="center", fontsize=10, color="white")
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_xticks([0.5, 1.5], ["Predicted attacker", "Predicted benign"])
ax.set_yticks([1.5, 0.5], ["Actual attacker\n(52)", "Actual benign\n(84)"])
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title(f"Test set, alert threshold 85: {int(u.TP + u.TN)} of {int(u.n_users)} correct, "
             f"{int(u.FN + u.FP)} errors")
save(fig, "fig01_confusion_matrix.png")

# ---------------------------------------------------------------- 2. development vs test
cv = pd.read_csv(RES / "04_cv_summary.csv")
cv = cv[(cv.algo == "lightgbm") & (cv.h == 10)].iloc[0]
tf = json.loads((RES / "08_test_threshold_free.json").read_text())
op = pd.read_csv(RES / "07_dev_operating_points.csv")
op85 = op[op.threshold == "band_high_85"]
tt = pd.read_csv(RES / "08_test_fixed_thresholds.csv").iloc[0]
perf = [("Daily PR-AUC", cv.pr_auc, tf["row PR-AUC (y_10)"]),
        ("User ROC-AUC", cv.user_auc, tf["user AUC (peak risk)"]),
        ("Attackers detected at 85\n(inside attack window)", op85.user_recall.mean(), tt.recall)]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [2.2, 1]})
ax = axes[0]
y = np.arange(len(perf))[::-1]
for k, (lab, d, t) in enumerate(perf):
    ax.plot([d * 100, t * 100], [y[k]] * 2, color=GRID, lw=3, zorder=1)
    ax.plot(d * 100, y[k], "o", color=BLUE, ms=10, markeredgecolor=SURF, markeredgewidth=2, zorder=2)
    ax.plot(t * 100, y[k], "o", color=ORANGE, ms=10, markeredgecolor=SURF, markeredgewidth=2, zorder=2)
    ax.text(max(d, t) * 100 + 0.4, y[k], f"dev {d * 100:.1f} · test {t * 100:.1f}", va="center", fontsize=9, color=INK2)
ax.set_yticks(y, [p[0] for p in perf])
ax.set_xlim(88, 102.5)
ax.set_xlabel("%")
ax.set_title("Higher is better")
ax.plot([], [], "o", color=BLUE, label="Development (cross-validated)")
ax.plot([], [], "o", color=ORANGE, label="Test (seen once)")
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax = axes[1]
vals = [op85.fp_benign_users.mean() / 511 * 100, tt.FPR * 100]
bars = ax.bar(["Development", "Test"], vals, color=[BLUE, ORANGE], width=0.55)
for b, v, n in zip(bars, vals, [f"{op85.fp_benign_users.mean():.1f} of 511", f"{int(tt.FP)} of 84"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v:.1f}%\n({n})", ha="center", va="bottom", fontsize=9, color=INK2)
ax.set_ylim(0, 4.6)
ax.set_ylabel("Benign employees alarmed (%)")
ax.set_title("Lower is better")
fig.suptitle("Test results match the development estimates (no sign of overfitting)", x=0.01, ha="left", fontsize=11.5)
save(fig, "fig02_dev_vs_test.png")

# ---------------------------------------------------------------- 3. algorithm comparison (Stage A)
cvall = pd.read_csv(RES / "04_cv_summary.csv")
alg = cvall.groupby("algo")[["recall@1%fpr", "pr_auc"]].mean().sort_values("recall@1%fpr")
names = {"catboost": "CatBoost", "lightgbm": "LightGBM (chosen)", "xgboost": "XGBoost", "random_forest": "Random forest",
         "decision_tree": "Decision tree"}
fig, ax = plt.subplots(figsize=(8.2, 3.8))
cols = [ORANGE if a == "lightgbm" else BLUE for a in alg.index]
bars = ax.barh([names[a] for a in alg.index], alg["recall@1%fpr"] * 100, color=cols, height=0.6)
for b, (a, r) in zip(bars, alg.iterrows()):
    ax.text(b.get_width() + 0.4, b.get_y() + b.get_height() / 2, f"{r['recall@1%fpr'] * 100:.1f}%   (PR-AUC {r.pr_auc:.3f})",
            va="center", fontsize=9, color=INK2)
ax.set_xlim(0, 105)
ax.set_xlabel("Daily detection at 1% false alarms (development, mean of the 5, 10 and 20-day windows)")
ax.set_title("Algorithm comparison (CatBoost +0.4 points: below the +0.5 margin)")
save(fig, "fig03_algorithms.png")

# ---------------------------------------------------------------- 4. detection by duration x stealth (dev)
ds = pd.read_csv(RES / "09_dev_duration_x_stealth.csv").set_index("duration")
ds = ds[["High (0.25)", "Medium (0.50)", "Low (0.75)"]]
fig, ax = plt.subplots(figsize=(6.6, 4.6))
ax.grid(False)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", SEQ)
ax.imshow(ds.to_numpy() * 100, cmap=cmap, vmin=0, vmax=100, aspect="auto")
for i in range(ds.shape[0]):
    for j in range(ds.shape[1]):
        v = ds.iloc[i, j] * 100
        ax.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=10.5, color="white" if v > 55 else INK, weight="bold")
ax.set_xticks(range(3), ["High stealth", "Medium stealth", "Low stealth"])
ax.set_yticks(range(len(ds)), [f"{int(d)} days" for d in ds.index])
ax.set_ylabel("Attack duration")
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Attackers detected at 85 (development, 348 attackers)")
save(fig, "fig04_duration_x_stealth.png")

# ---------------------------------------------------------------- 5. time to detection by stealth (dev)
lt = pd.read_csv(RES / "11_latency_dev_stealth_85.csv").set_index("stealth")
order = [("Low (0.75)", "Low stealth"), ("Medium (0.50)", "Medium stealth"), ("High (0.25)", "High stealth")]
fig, ax = plt.subplots(figsize=(8, 3.4))
for k, (key, lab) in enumerate(order):
    r = lt.loc[key]
    yk = len(order) - 1 - k
    ax.plot([r.p25, r.p90], [yk] * 2, color=GRID, lw=2)
    ax.plot([r.p25, r.p75], [yk] * 2, color=BLUE, lw=8, solid_capstyle="round")
    ax.plot(r["median"], yk, "o", color=ORANGE, ms=10, markeredgecolor=SURF, markeredgewidth=2)
    ax.text(r.p90 + 0.3, yk, f"median {r['median']:.0f} · 90th pct {r.p90:.0f} days · detected {r.detected * 100:.0f}%",
            va="center", fontsize=9, color=INK2)
ax.set_yticks(range(3), [o[1] for o in order][::-1])
ax.set_xlim(-0.5, 30)
ax.set_xlabel("Workdays from the first exfiltration day to the first alert (development, threshold 85)")
ax.plot([], [], color=BLUE, lw=8, label="middle half (25th–75th percentile)")
ax.plot([], [], "o", color=ORANGE, label="median")
ax.plot([], [], color=GRID, lw=2, label="up to the 90th percentile")
ax.legend(frameon=False, loc="upper right", fontsize=8.5)
ax.set_title("Time to detection: about 3 workdays overall, longer for high stealth")
save(fig, "fig05_time_to_detection.png")

# ---------------------------------------------------------------- 6. SHAP share by feature family (test)
sh = pd.read_csv(RES / "10_test_share_by_family.csv", index_col=0)
rot = [i for i in sh.index if i.startswith("Rotation")]
fam = sh.drop(index=rot)
fam.loc["All rotation and multi-channel features"] = sh.loc[rot].sum()
fam = fam.rename(index={"Accumulation & shift (all channels)": "Accumulated deviation", "HTTP / web": "Web upload", "Email": "E-mail",
                        "USB / file": "USB / files"}).sort_values("all rows")
fig, ax = plt.subplots(figsize=(8.6, 4.2))
yy = np.arange(len(fam))
ax.barh(yy + 0.2, fam["all rows"] * 100, height=0.38, color=BLUE, label="All rows")
ax.barh(yy - 0.2, fam["y_10 = 1 rows"] * 100, height=0.38, color=ORANGE, label="Attack-window rows")
for k, (_, r) in enumerate(fam.iterrows()):
    ax.text(r["all rows"] * 100 + 0.6, k + 0.2, f"{r['all rows'] * 100:.1f}%", va="center", fontsize=8.5, color=INK2)
    ax.text(r["y_10 = 1 rows"] * 100 + 0.6, k - 0.2, f"{r['y_10 = 1 rows'] * 100:.1f}%", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(yy, fam.index)
ax.set_xlim(0, 64)
ax.set_xlabel("Share of the model's explanation (mean |SHAP|, test set)")
ax.legend(frameon=False, loc="lower right", fontsize=9)
ax.set_title("The model relies mostly on accumulated deviation, not on rotation")
save(fig, "fig06_shap_families.png")

# ---------------------------------------------------------------- 7. calibration (development, leave-fold-out)
rel = pd.read_csv(RES / "07_reliability.csv")
fig, ax = plt.subplots(figsize=(5.4, 5))
ax.plot([0, 100], [0, 100], ls="--", color=INK2, lw=1, label="perfect calibration")
ax.plot(rel.mean_pred * 100, rel.observed * 100, "o-", color=BLUE, lw=2, ms=7, markeredgecolor=SURF, markeredgewidth=2,
        label="MUBSIR score")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xlabel("Average score in the bin")
ax.set_ylabel("Share of rows that were actually positive (%)")
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.set_title("Calibration: a score of 85 means about an 85% chance")
save(fig, "fig07_calibration.png")

# ---------------------------------------------------------------- 8. threshold trade-off (dev OOF, descriptive)
man = json.loads((M / "model" / "manifest.json").read_text())
A, B = man["calibration"]["a"], man["calibration"]["b"]
split = pd.read_csv(M / "config" / "final_split.csv")
dev = set(split.loc[split.part == "dev", "user"].astype(str))
meta = pd.read_parquet(M / "work" / "eval_meta.parquet")
meta["user"] = meta.user.astype(str)
Md = meta[meta.user.isin(dev)].reset_index(drop=True)
oof = pd.read_parquet(M / "work" / "oof" / "lightgbm_h10.parquet")
um = pd.read_csv(M / "config" / "user_meta.csv")
um["user"] = um.user.astype(str)
att = Md.user.map(um.set_index("user").is_attacker).to_numpy() == 1
ts = [40, 50, 55, 60, 65, 70, 75, 80, 85, 90]
rec, ben = np.zeros(len(ts)), np.zeros(len(ts))
for c in oof.columns:
    p = np.clip(oof[c].to_numpy(), 1e-7, 1 - 1e-7)
    r = 100 / (1 + np.exp(-(A * np.log(p / (1 - p)) + B)))
    win = pd.Series(np.where(att & Md.in_window.to_numpy(), r, -1)).groupby(Md.user).max()
    pk = pd.Series(np.where(~att, r, -1)).groupby(Md.user).max()
    a_users = win[win.index.isin(set(Md.user[att]))]
    b_users = pk[pk.index.isin(set(Md.user[~att]))]
    for k, t in enumerate(ts):
        rec[k] += (a_users > t).mean() / len(oof.columns)
        ben[k] += (b_users > t).mean() / len(oof.columns)
fig, ax = plt.subplots(figsize=(8.4, 4.2))
ax.plot(ts, rec * 100, "o-", color=BLUE, lw=2, ms=6, markeredgecolor=SURF, markeredgewidth=1.5)
ax.plot(ts, ben * 100, "o-", color=ORANGE, lw=2, ms=6, markeredgecolor=SURF, markeredgewidth=1.5)
ax.annotate("attackers reaching the threshold", (ts[-1], rec[-1] * 100), xytext=(6, 0), textcoords="offset points",
            va="center", fontsize=9, color=INK2)
ax.annotate("benign employees reaching it", (ts[-1], ben[-1] * 100), xytext=(6, 0), textcoords="offset points",
            va="center", fontsize=9, color=INK2)
for t, lab in ((60, "medium starts (60)"), (85, "alert (85)")):
    k = ts.index(t)
    ax.axvline(t, color=INK2, lw=1, ls=":")
    ax.text(t + 0.6, 52, f"{lab}\n{rec[k] * 100:.1f}% attackers\n{ben[k] * 100:.1f}% benign", fontsize=8.5, color=INK2)
ax.set_xlim(38, 107)
ax.set_ylim(0, 102)
ax.set_xlabel("Threshold (score)")
ax.set_ylabel("% of employees (development)")
ax.set_title("Threshold trade-off: a smooth curve, no sharp break at 60")
save(fig, "fig08_threshold_tradeoff.png")
