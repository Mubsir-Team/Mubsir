"""
Registered analysis of the learning-curve study (PREREGISTRATION.md §5-7). Dev OOF scores only; the test set is never read.
Writes results/analysis_*.{json,csv}. Descriptive extrapolation is labelled as such.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent.parent
sys.path.insert(0, str(MODEL_DIR / "pipeline"))
from metrics import threshold_at_fpr  # noqa: E402

PRE = (HERE / "PREREGISTRATION.sha256").read_text().strip()
assert hashlib.sha256((HERE / "PREREGISTRATION.md").read_bytes()).hexdigest() == PRE, "pre-registration changed"
gates = json.loads((HERE / "results" / "gates.json").read_text())
assert gates["consequence"] == "proceed", "gates did not pass"
RES, SC = HERE / "results", HERE / "scores"
ARMS = ["F30", "F45", "F60", "F75", "F86", "F100"]
FRAC = {"F30": 0.30, "F45": 0.45, "F60": 0.60, "F75": 0.75, "F86": 859 / 995, "F100": 1.0}
SEEDS = {a: ((0,) if a == "F100" else (0, 1)) for a in ARMS}
R, B, FPRS = 3, 2000, (0.03, 0.01)

um = pd.read_csv(MODEL_DIR / "config" / "user_meta.csv").set_index("user")
folds = pd.read_csv(MODEL_DIR / "config" / "cv_folds.csv")
val = folds[folds.role == "valid"]
base = pd.read_parquet(SC / "F100_s0.parquet")
users = base["user"].to_numpy()
y10 = base["y_10"].to_numpy().astype(int)
U = np.array(sorted(set(users)))
uidx = pd.Series(np.arange(len(U)), index=U)
row_u = uidx[users].to_numpy()
is_att = um.loc[U, "is_attacker"].to_numpy() == 1
assert len(U) == 859 and is_att.sum() == 348
fold_of = {r: val[val.repeat == r].set_index("user").loc[U, "fold"].to_numpy() for r in range(R)}
log = pd.read_csv(RES / "training_log.csv")
n_fit = log.groupby("arm").fit_users.mean()


def peaks(score, mask=None):
    s = pd.Series(score if mask is None else np.where(mask, score, -np.inf))
    return s.groupby(row_u).max().reindex(range(len(U))).to_numpy()


# ------------------------------------------------------------------ per-user detection flags (thresholds fixed once)
det = {}      # (arm, fpr) -> array [runs, users] of 0/1 for attackers (NaN for benign)
bfp = {}      # (arm, fpr) -> realised benign FPR (held-out benign users above their fold threshold), mean over runs
auc_user, prauc = {}, {}
for a in ARMS:
    for s in SEEDS[a]:
        df = pd.read_parquet(SC / f"{a}_s{s}.parquet")
        assert (df["user"].to_numpy() == users).all()
        for r in range(R):
            sc = df[f"rep{r}"].to_numpy()
            pk = peaks(sc)                               # whole-period peak per user
            pk_att = peaks(sc, y10 == 1)                 # peak over y_10 = 1 rows
            prauc.setdefault(a, []).append(average_precision_score(y10, sc))
            auc_user.setdefault(a, []).append(roc_auc_score(is_att, pk))
            for f in FPRS:
                flags = np.full(len(U), np.nan)
                fp = []
                for k in range(5):
                    in_k = fold_of[r] == k
                    t = threshold_at_fpr(pk[(~is_att) & (~in_k)], f)
                    flags[in_k & is_att] = (pk_att[in_k & is_att] > t).astype(float)
                    fp.append(pk[in_k & ~is_att] > t)
                det.setdefault((a, f), []).append(flags)
                bfp.setdefault((a, f), []).append(np.concatenate(fp).mean())
D = {k: np.nanmean(np.vstack(v), axis=0) for k, v in det.items()}   # per-user detection share over runs (attackers)
Det = {k: float(np.nanmean(v)) for k, v in D.items()}


# ------------------------------------------------------------------ weighted AP (identical to sklearn for unit weights)
def ap_prep(score):
    o = np.argsort(-score, kind="mergesort")
    s_sorted = score[o]
    ends = np.r_[np.flatnonzero(np.diff(s_sorted) != 0), len(o) - 1]
    return o, ends


def ap_w(yv, w, prep):
    o, ends = prep
    tp = np.cumsum((w * yv)[o])[ends]
    fp = np.cumsum((w * (1 - yv))[o])[ends]
    prec = tp / np.maximum(tp + fp, 1e-12)
    rec = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


preps = {}
for a in ("F86", "F100"):
    for s in SEEDS[a]:
        df = pd.read_parquet(SC / f"{a}_s{s}.parquet")
        for r in range(R):
            sc = df[f"rep{r}"].to_numpy()
            preps.setdefault(a, []).append(ap_prep(sc))
            if r == 0 and s == 0:
                assert abs(ap_w(y10, np.ones(len(y10)), preps[a][-1]) - average_precision_score(y10, sc)) < 1e-12

# ------------------------------------------------------------------ bootstrap (users, stratified attacker/benign)
rng = np.random.default_rng(2026)
att_i, ben_i = np.flatnonzero(is_att), np.flatnonzero(~is_att)
est_fun = {
    "PRIMARY": lambda c: np.average(D[("F100", .03)][att_i], weights=c[att_i]) - np.average(D[("F86", .03)][att_i], weights=c[att_i]),
    "S1": lambda c: np.average(D[("F100", .01)][att_i], weights=c[att_i]) - np.average(D[("F86", .01)][att_i], weights=c[att_i]),
    "S3": lambda c: np.average(D[("F100", .03)][att_i], weights=c[att_i]) - np.average(D[("F60", .03)][att_i], weights=c[att_i]),
}


def s2(c):
    w = c[row_u]
    return np.mean([ap_w(y10, w, p) for p in preps["F100"]]) - np.mean([ap_w(y10, w, p) for p in preps["F86"]])


est_fun["S2"] = s2
ones = np.ones(len(U))
point = {k: float(f(ones)) for k, f in est_fun.items()}
boot = {k: np.empty(B) for k in est_fun}
for b in range(B):
    c = np.bincount(rng.choice(att_i, len(att_i)), minlength=len(U)) + np.bincount(rng.choice(ben_i, len(ben_i)), minlength=len(U))
    for k, f in est_fun.items():
        boot[k][b] = f(c.astype(float))
rows = []
for k in ("PRIMARY", "S1", "S2", "S3"):
    v = boot[k]
    p = min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))
    rows.append(dict(id=k, estimate=point[k], ci_low=np.percentile(v, 2.5), ci_high=np.percentile(v, 97.5),
                     ci90_low=np.percentile(v, 5), ci90_high=np.percentile(v, 95), p=p))
est = pd.DataFrame(rows).set_index("id")
sec = est.loc[["S1", "S2", "S3"], "p"].sort_values()
holm, run = {}, 0.0
for i, (k, p) in enumerate(sec.items()):
    run = max(run, min(1.0, p * (len(sec) - i)))
    holm[k] = run
est["p_holm"] = pd.Series(holm)
est.to_csv(RES / "analysis_estimands.csv")

# ------------------------------------------------------------------ outcome (PREREGISTRATION §7)
pr = est.loc["PRIMARY"]
if pr.ci_low > 0 and pr.estimate >= 0.02:
    outcome, text = 2, "reserving the test has a meaningful cost"
elif pr.ci90_low > -0.02 and pr.ci90_high < 0.02:
    outcome, text = 1, "reserving the test costs little"
else:
    outcome, text = 3, "inconclusive"

# ------------------------------------------------------------------ descriptive: curve, subgroups, benign FPR
curve = pd.DataFrame([dict(arm=a, fraction=FRAC[a], fit_users=n_fit[a],
                           det_3pct=Det[(a, .03)], det_1pct=Det[(a, .01)],
                           row_pr_auc=np.mean(prauc[a]), user_auc=np.mean(auc_user[a]),
                           benign_fpr_3pct=np.mean(bfp[(a, .03)]), benign_fpr_1pct=np.mean(bfp[(a, .01)])) for a in ARMS])
curve.to_csv(RES / "analysis_curve.csv", index=False)

meta = um.loc[U]
groups = {"all attackers": is_att,
          "5-day attacks": is_att & (meta.window_workdays.to_numpy() == 5),
          "High stealth": is_att & (meta.stealth_level.to_numpy() == "High"),
          "U+H (no email)": is_att & (meta.channel_configuration.to_numpy() == "U+H"),
          "3 channels (E+U+H)": is_att & (meta.channel_configuration.to_numpy() == "E+U+H")}
sub = pd.DataFrame([dict(group=g, n=int(m.sum()), arm=a, det_3pct=float(np.mean(D[(a, .03)][m])))
                    for g, m in groups.items() for a in ARMS])
sub.to_csv(RES / "analysis_subgroups.csv", index=False)


# ------------------------------------------------------------------ descriptive extrapolation (labelled)
def powerlaw(n, a, b, c):
    return a - b * np.power(n, -c)


try:
    xs, ys = curve.fit_users.to_numpy(), curve.det_3pct.to_numpy()
    par, _ = curve_fit(powerlaw, xs, ys, p0=(0.95, 1.0, 0.5), maxfev=20000)
    extrap = dict(params=dict(a=par[0], b=par[1], c=par[2]),
                  predicted_det_3pct_at_fit_sizes={"cv_scale_588": powerlaw(588, *par),
                                                   "final_scale_859_users": powerlaw(859, *par),
                                                   "final_scale_995_users": powerlaw(995, *par)},
                  predicted_gain_adding_test_users_at_final_scale=powerlaw(995, *par) - powerlaw(859, *par),
                  note="MODEL-BASED EXTRAPOLATION, not a measurement. Det(n) = a - b*n^-c fitted to the six arm means "
                       "(n = mean fit users per fold).")
except Exception as e:  # noqa: BLE001
    extrap = dict(error=str(e))
(RES / "analysis_extrapolation.json").write_text(json.dumps(extrap, indent=1, default=float), encoding="utf-8")

out = dict(preregistration_sha256=PRE, outcome=outcome, outcome_text=text,
           det_3pct={a: Det[(a, .03)] for a in ARMS}, det_1pct={a: Det[(a, .01)] for a in ARMS},
           primary=est.loc["PRIMARY"].to_dict(), secondary=est.loc[["S1", "S2", "S3"]].to_dict("index"),
           bootstrap=dict(resamples=B, rng=2026, stratified="attacker vs benign users"))
(RES / "analysis_primary.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")

pd.set_option("display.width", 200)
print(curve.round(4).to_string(index=False))
print()
print(est.round(4).to_string())
print(f"\nOUTCOME {outcome}: {text}")
print("\nSubgroups (Det @3%):")
print(sub.pivot(index="group", columns="arm", values="det_3pct")[ARMS].round(3).to_string())
print("\nExtrapolation:", json.dumps(extrap, indent=1, default=float))
