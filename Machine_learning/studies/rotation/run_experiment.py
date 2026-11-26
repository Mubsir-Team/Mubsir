"""
Rotation-evasion experiment (see PREREGISTRATION.md). Dev users only; frozen artefacts are read, never written.

Arms: M = Stage A MUBSIR OOF (100 features, not retrained); S = three per-channel LightGBM monitors (max of the three);
P = one LightGBM on per-channel features without the engineered cross-channel features.
Outputs: results/*.csv, results/run_log.json
"""

import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "pipeline"))
import models  # noqa: E402
from common import SEED, WORK, NAS  # noqa: E402

T = importlib.import_module("05_tune")      # data loading (dev only) + frozen rules-hash check
RES = HERE / "results"
RES.mkdir(exist_ok=True)
rng = np.random.default_rng(42)
B = 2000
PREREG_SHA = hashlib.sha256((HERE / "PREREGISTRATION.md").read_bytes()).hexdigest()
assert PREREG_SHA == "b7ca05e51c65c1483b017d6546077f76d6aeee76b850522efcc192e9a28bd8b2", "pre-registration changed"

D = T.load()
X, y, users_rows = D["X"], D["y"], D["users"]
feats = list(X.columns)
um, folds = D["umeta"], D["folds"]

# ------------------------------------------------------------------ feature groups (partition of the 100)
EMAIL = [c for c in feats if c.startswith(("email_", "external_email_count", "internal_email_count", "E_email"))]
USB = [c for c in feats if c.startswith(("usb_", "file_", "E_usb"))]
HTTP = [c for c in feats if c.startswith(("http_", "www_", "unique_domains", "E_http"))]
CONTEXT = ["logon_count", "logoff_count"]
CROSS_PREFIX = ("cumulative_deviation", "combined_channel_intensity", "channel_distribution_shift",
                "consecutive_elevated_days", "cross_channel_deviation_days", "dominant_channel_switches",
                "multi_channel_days", "active_channel_count", "nr_")
CROSS = [c for c in feats if c.startswith(CROSS_PREFIX)]
groups = EMAIL + USB + HTTP + CONTEXT + CROSS
assert sorted(groups) == sorted(feats) and len(groups) == len(set(groups)) == 100, set(feats) ^ set(groups)
SILOS = {"email": EMAIL + CONTEXT, "usb_file": USB + CONTEXT, "http": HTTP + CONTEXT}
PERCH = EMAIL + USB + HTTP + CONTEXT
fg = pd.DataFrame([(c, g) for g, cols in (("email", EMAIL), ("usb_file", USB), ("http", HTTP), ("context", CONTEXT),
                                          ("cross_channel", CROSS)) for c in cols], columns=["feature", "group"])
fg.to_csv(RES / "feature_groups.csv", index=False)
print({g: len(c) for g, c in (("email", EMAIL), ("usb", USB), ("http", HTTP), ("context", CONTEXT), ("cross", CROSS))})

# ------------------------------------------------------------------ train arms S and P on the 15 outer folds
R = 3
oofS = {k: np.full((R, len(y)), np.nan) for k in SILOS}
oofP = np.full((R, len(y)), np.nan)
t0 = time.time()
for (r, k), f in folds.groupby(["repeat", "fold"]):
    fit_u, es_u, val_u = (set(f.loc[f.role == ro, "user"]) for ro in ("fit", "es", "valid"))
    fr, er, vr = T.rows(D, fit_u), T.rows(D, es_u), T.rows(D, val_u)
    for name, cols in SILOS.items():
        s, _ = models.fit_predict("lightgbm", X.iloc[fr][cols], y[fr], X.iloc[er][cols], y[er], X.iloc[vr][cols], seed=SEED + r)
        oofS[name][r, vr] = s
    s, _ = models.fit_predict("lightgbm", X.iloc[fr][PERCH], y[fr], X.iloc[er][PERCH], y[er], X.iloc[vr][PERCH], seed=SEED + r)
    oofP[r, vr] = s
    print(f"r{r} k{k} done {time.time() - t0:.0f}s", flush=True)
assert not np.isnan(oofP).any() and all(not np.isnan(v).any() for v in oofS.values())
SCORES = {"M": pd.read_parquet(WORK / "oof" / "lightgbm_h10.parquet").to_numpy().T,     # (R, rows)
          "S": np.max(np.stack(list(oofS.values())), axis=0),
          "P": oofP}
for name, s in oofS.items():
    SCORES[f"S_{name}_only"] = s

# ------------------------------------------------------------------ per-user tables
users = sorted(set(users_rows))
uidx = pd.Series(np.arange(len(users)), index=users)
row_u = uidx.loc[users_rows].to_numpy()
inw = D["in_window"]
U = um.loc[users, ["is_attacker", "switching_level", "stealth_level", "window_workdays", "channel_count",
                   "channel_configuration"]].copy()
gt = pd.read_parquet(NAS / "11_Ground_Truth_and_Quality_Validation" / "ground_truth_user_days_r62.parquet",
                     columns=["user", "is_switch", "executed_channel_count"])
gt["user"] = gt["user"].astype(str)
U["executed_switches"] = gt.groupby("user")["is_switch"].sum().reindex(users).fillna(0).astype(int)
U["multi_channel_days"] = gt.assign(m=gt.executed_channel_count > 1).groupby("user")["m"].sum().reindex(users).fillna(0)
att = np.where(U.is_attacker.to_numpy() == 1)[0]
ben = np.where(U.is_attacker.to_numpy() == 0)[0]
fold_of = np.zeros((R, len(users)), int)
for r in range(R):
    fold_of[r] = folds[(folds.repeat == r) & (folds.role == "valid")].set_index("user")["fold"].loc[users].to_numpy()


def threshold_at_fpr(neg, fpr):
    neg = np.sort(np.asarray(neg, dtype=float))[::-1]
    return neg[min(int(np.floor(fpr * len(neg))), len(neg) - 1)]


def detection(score, fpr):
    """Per repeat: user-level detected (peak > cross-fitted thr) and window detected; plus achieved benign FPR."""
    det_u = np.zeros((R, len(users)), bool)
    det_w = np.zeros((R, len(users)), bool)
    for r in range(R):
        peak = pd.Series(score[r]).groupby(row_u).max().to_numpy()
        winmax = pd.Series(np.where(inw, score[r], -np.inf)).groupby(row_u).max().to_numpy()
        for k in range(5):
            inn = fold_of[r] == k
            t = threshold_at_fpr(peak[(~inn) & (U.is_attacker.to_numpy() == 0)], fpr)
            det_u[r, inn] = peak[inn] > t
            det_w[r, inn] = winmax[inn] > t
    return det_u, det_w


DET = {(a, fpr): detection(s, fpr) for a, s in SCORES.items() for fpr in (0.03, 0.01)}

# overall table
rows = []
for (a, fpr), (du, dw) in DET.items():
    peaks = [pd.Series(SCORES[a][r]).groupby(row_u).max().to_numpy() for r in range(R)]
    rows.append(dict(arm=a, iso_fpr=fpr, user_auc=np.mean([roc_auc_score(U.is_attacker, p) for p in peaks]),
                     recall_user=du[:, att].mean(), recall_window=dw[:, att].mean(), achieved_benign_fpr=du[:, ben].mean()))
overall = pd.DataFrame(rows)
overall.to_csv(RES / "overall_by_arm.csv", index=False)

# detection by group for every arm
grp_rows = []
U["switch_bin"] = pd.cut(U["executed_switches"], [-1, 1, 3, 6, 100], labels=["0-1", "2-3", "4-6", "7+"]).astype(str)
for (a, fpr), (du, dw) in DET.items():
    for col in ("switching_level", "channel_count", "switch_bin", "stealth_level", "window_workdays", "channel_configuration"):
        for g in sorted(U.loc[U.index[att], col].dropna().unique(), key=str):
            m = (U[col].to_numpy() == g) & (U.is_attacker.to_numpy() == 1)
            grp_rows.append(dict(arm=a, iso_fpr=fpr, dimension=col, group=g, n_attackers=int(m.sum()),
                                 recall_user=du[:, m].mean(), recall_window=dw[:, m].mean()))
grp = pd.DataFrame(grp_rows)
grp.to_csv(RES / "detection_by_group.csv", index=False)

# ------------------------------------------------------------------ pre-registered decision (paired bootstrap)
sw = U["switching_level"].to_numpy()


def D_(du, mask):
    return du[:, mask].mean()


def gap_stats(armA, armB, fpr, gcol="switching_level", hi="High", lo="Low"):
    da, db = DET[(armA, fpr)][0], DET[(armB, fpr)][0]
    g = U[gcol].to_numpy()
    H = np.where((g == hi) & (U.is_attacker.to_numpy() == 1))[0]
    L = np.where((g == lo) & (U.is_attacker.to_numpy() == 1))[0]
    point = dict(DA_high=da[:, H].mean(), DA_low=da[:, L].mean(), DB_high=db[:, H].mean(), DB_low=db[:, L].mean())
    point["gap"] = (point["DA_high"] - point["DB_high"]) - (point["DA_low"] - point["DB_low"])
    point["B_high_minus_low"] = point["DB_high"] - point["DB_low"]
    point["A_high_minus_low"] = point["DA_high"] - point["DA_low"]
    bs = []
    for _ in range(B):
        h = rng.choice(H, len(H)); l = rng.choice(L, len(L))
        ah, al, bh, bl = da[:, h].mean(), da[:, l].mean(), db[:, h].mean(), db[:, l].mean()
        bs.append(((ah - bh) - (al - bl), bh - bl, ah - al))
    bs = np.array(bs)
    for j, name in enumerate(("gap", "B_high_minus_low", "A_high_minus_low")):
        point[f"{name}_ci_low"], point[f"{name}_ci_high"] = np.quantile(bs[:, j], [0.025, 0.975])
    return point


primary = gap_stats("M", "S", 0.03)
c1 = primary["gap"] > 0 and primary["gap_ci_low"] > 0
c2 = primary["B_high_minus_low"] < 0 and primary["B_high_minus_low_ci_high"] < 0
c3 = primary["A_high_minus_low"] >= -0.03
verdict = ("SUPPORTED" if (c1 and c2 and c3) else "PARTIAL" if c1 else "NOT SUPPORTED")
robust = []
for label, args in (("iso-FPR 1%, M vs S, switching High vs Low", ("M", "S", 0.01)),
                    ("iso-FPR 3%, M vs P, switching High vs Low", ("M", "P", 0.03)),
                    ("iso-FPR 3%, M vs S, configured channels 3 vs 2", ("M", "S", 0.03, "channel_count", 3, 2)),
                    ("iso-FPR 3%, M vs S, executed switches 7+ vs 0-1", ("M", "S", 0.03, "switch_bin", "7+", "0-1"))):
    robust.append(dict(check=label, **gap_stats(*args)))
robust = pd.DataFrame([dict(check="PRIMARY: iso-FPR 3%, M vs S, switching High vs Low", **primary)] + robust)
robust.to_csv(RES / "decision_and_robustness.csv", index=False)

# logistic regression: detected ~ arm(S) + switches + arm:switches + stealth + duration (attacker x arm rows)
du_M, du_S = DET[("M", 0.03)][0], DET[("S", 0.03)][0]
A = U.iloc[att]
sw_z = (A.executed_switches - A.executed_switches.mean()) / A.executed_switches.std()
base_cols = pd.get_dummies(A[["stealth_level"]].astype(str), drop_first=True).astype(float)
base_cols["log_duration"] = np.log(A["window_workdays"].astype(float))


def fit_lr(idx):
    Xs, ys = [], []
    for arm_flag, du in ((0, du_M), (1, du_S)):
        det = du[:, att].mean(0)[idx]                   # share of repeats detected (0..1)
        Xi = np.c_[np.full(len(idx), arm_flag), sw_z.to_numpy()[idx], arm_flag * sw_z.to_numpy()[idx], base_cols.to_numpy()[idx]]
        for rep_w in (det, 1 - det):                    # weighted binary rows
            pass
        Xs.append(np.r_[Xi, Xi]); ys.append(np.r_[np.ones(len(idx)), np.zeros(len(idx))])
        if arm_flag == 0:
            w0 = np.r_[det, 1 - det]
        else:
            w1 = np.r_[det, 1 - det]
    lr = LogisticRegression(C=1e4, max_iter=5000).fit(np.vstack(Xs), np.concatenate(ys), sample_weight=np.r_[w0, w1] + 1e-9)
    return lr.coef_[0][:3]


coef = fit_lr(np.arange(len(att)))
bs = np.array([fit_lr(rng.integers(0, len(att), len(att))) for _ in range(300)])
lr_tab = pd.DataFrame({"term": ["arm = siloed (S)", "executed switches (z)", "siloed x switches (interaction)"],
                       "coef": coef, "ci_low": np.quantile(bs, .025, axis=0), "ci_high": np.quantile(bs, .975, axis=0)})
lr_tab.to_csv(RES / "logistic_interaction.csv", index=False)

log = dict(preregistration_sha256=PREREG_SHA, dev_users=len(users), attackers=int(len(att)), benign=int(len(ben)),
           feature_counts=dict(email=len(EMAIL), usb_file=len(USB), http=len(HTTP), context=len(CONTEXT), cross=len(CROSS)),
           conditions=dict(primary_gap_ci_excludes_0=bool(c1), siloed_hurt_by_rotation=bool(c2),
                           mubsir_not_hurt=bool(c3)), verdict=verdict, primary={k: float(v) for k, v in primary.items()},
           seconds=round(time.time() - t0))
(RES / "run_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
print(json.dumps(log, indent=1))
print(overall.round(4).to_string(index=False))
print(robust.round(4).to_string(index=False))
print(lr_tab.round(4).to_string(index=False))
