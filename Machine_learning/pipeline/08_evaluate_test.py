"""
08 — The single evaluation of the frozen pipeline on the locked test users (136: 52 attackers, 84 benign).

Guards: runs only if results/pretest/freeze.json exists and every frozen hash still matches; refuses to run
if the test has already been scored (work/test/test_scores.parquet exists). Uses only frozen artefacts:
model/lgbm_y10.txt, the manifest calibration and thresholds. Nothing here feeds back into any decision.

Outputs: work/test/test_scores.parquet (the only test scoring; later analyses read it),
         results/08_test_*.csv|json, reports/08_test_evaluation.md
"""

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

from common import (WORK, RESULTS, REPORTS, MODEL_DIR, DATASET, SELECTION_RULES, FINAL_SPLIT, LABELS, EVAL_META,
                    USER_META, sha256, load_dataset, write_json, md_table)
from metrics import duration_macro_row_pr_auc, threshold_at_fpr, DURATIONS

FREEZE = RESULTS / "pretest" / "freeze.json"
OUT = WORK / "test"
SCORES = OUT / "test_scores.parquet"
EPS = 1e-7
rng = np.random.default_rng(42)

# ------------------------------------------------------------------ guards
fz = json.loads(FREEZE.read_text())
for name, path in (("model_file", MODEL_DIR / "model" / "lgbm_y10.txt"), ("manifest", MODEL_DIR / "model" / "manifest.json"),
                   ("final_split", FINAL_SPLIT), ("selection_rules", SELECTION_RULES), ("dataset", DATASET)):
    assert sha256(path) == fz[name], f"frozen artefact changed: {name}"
OUT.mkdir(parents=True, exist_ok=True)
RERUN = SCORES.exists()   # a re-run may only recompute the same report from the same frozen scores
man = json.loads((MODEL_DIR / "model" / "manifest.json").read_text(encoding="utf-8"))
A, B = man["calibration"]["a"], man["calibration"]["b"]
assert (A, B) == (fz["calibration"]["a"], fz["calibration"]["b"])
T1 = man["thresholds"]["dev_1pct_user_fpr_risk"]
assert T1 == fz["thresholds"]["dev_1pct_user_fpr"]
THRESHOLDS = {"High band (risk > 85)": 85.0, "Medium+ (risk > 60)": 60.0, f"Dev 1% point (risk > {T1:.2f})": T1}

# ------------------------------------------------------------------ the one scoring of the test users
split = pd.read_csv(FINAL_SPLIT)
test_users = set(split.loc[split.part == "test", "user"])
df = load_dataset()
lab = pd.read_parquet(LABELS)
meta = pd.read_parquet(EVAL_META)
um = pd.read_csv(USER_META).set_index("user")
te = df["user"].astype(str).isin(test_users).to_numpy()
X = df.loc[te, man["features"]].astype("float32")
bst = lgb.Booster(model_file=str(MODEL_DIR / "model" / "lgbm_y10.txt"))
p = bst.predict(X.to_numpy())
pc = np.clip(p, EPS, 1 - EPS)
risk = 100.0 / (1.0 + np.exp(-(A * np.log(pc / (1 - pc)) + B)))
S = pd.DataFrame({"user": df.loc[te, "user"].astype(str).to_numpy(), "day": df.loc[te, "day"].to_numpy(),
                  "p_raw": p, "risk": risk, "y": lab.loc[te, "y"].to_numpy(), "y_10": lab.loc[te, "y_10"].to_numpy(),
                  "in_window": meta.loc[te, "in_window"].to_numpy(), "day_type": meta.loc[te, "day_type"].to_numpy(),
                  "wd_from_first_attack": meta.loc[te, "wd_from_first_attack"].to_numpy()})
S["is_attacker"] = S["user"].map(um["is_attacker"]).to_numpy()
S["duration"] = S["user"].map(um["window_workdays"]).to_numpy()
if RERUN:
    prev = pd.read_parquet(SCORES)
    assert np.array_equal(prev["risk"].to_numpy(), S["risk"].to_numpy()), "frozen test scores differ"
    print("report re-run on the identical frozen test scores (no new decision)")
else:
    S.to_parquet(SCORES, index=False)
    print(f"test scored once: {S.user.nunique()} users, {len(S):,} rows -> {SCORES}")

att = sorted(S.loc[S.is_attacker == 1, "user"].unique())
ben = sorted(S.loc[S.is_attacker == 0, "user"].unique())
assert len(att) == 52 and len(ben) == 84


def cp(k, n):
    lo = beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo, hi


def user_view(s):
    peak = s.groupby("user")["risk"].max()
    inwin = s[s.in_window].groupby("user")["risk"].max()
    return peak, inwin


def metrics_all(s, users_att, users_ben):
    y10, r = s["y_10"].to_numpy(), s["risk"].to_numpy()
    benign_row = (s["is_attacker"] == 0).to_numpy()
    peak, inwin = user_view(s)
    ua = peak.loc[users_att].to_numpy()
    ub = peak.loc[users_ben].to_numpy()
    out = {"row_pr_auc": average_precision_score(y10, r), "row_roc_auc": roc_auc_score(y10, r),
           "duration_macro_row_pr_auc": duration_macro_row_pr_auc(y10, r, s["duration"].to_numpy(), benign_row)[0],
           "user_auc": roc_auc_score(np.r_[np.ones(len(ua)), np.zeros(len(ub))], np.r_[ua, ub])}
    for t in (85.0, 60.0, T1):
        out[f"recall_{t:.2f}"] = float((inwin.reindex(users_att).fillna(-1) > t).mean())
        out[f"fp_{t:.2f}"] = int((ub > t).sum())
    return out


# ------------------------------------------------------------------ threshold-free (row and user level)
y10, yd, r = S["y_10"].to_numpy(), S["y"].to_numpy(), S["risk"].to_numpy()
benign_row = (S.is_attacker == 0).to_numpy()
neg = r[benign_row]
tf = {
    "row PR-AUC (y_10)": average_precision_score(y10, r),
    "row ROC-AUC (y_10)": roc_auc_score(y10, r),
    "duration-macro row PR-AUC (y_10)": duration_macro_row_pr_auc(y10, r, S["duration"].to_numpy(), benign_row)[0],
    "row PR-AUC on attack-day label y": average_precision_score(yd, r),
    "row recall@1% FPR (benign-user rows of test)": float(np.mean(r[y10 == 1] > threshold_at_fpr(neg, 0.01))),
    "row recall@5% FPR (benign-user rows of test)": float(np.mean(r[y10 == 1] > threshold_at_fpr(neg, 0.05))),
}
peak, inwin = user_view(S)
lab_u = np.r_[np.ones(len(att)), np.zeros(len(ben))]
sc_u = np.r_[peak.loc[att].to_numpy(), peak.loc[ben].to_numpy()]
tf["user AUC (peak risk)"] = roc_auc_score(lab_u, sc_u)
tf["user PR-AUC (peak risk)"] = average_precision_score(lab_u, sc_u)
q = r / 100
tf["Brier (risk/100 vs y_10)"] = brier_score_loss(y10, q)
tf["mean calibrated prediction"] = float(q.mean())
tf["observed y_10 rate"] = float(y10.mean())
per_dur = dict(zip(DURATIONS, duration_macro_row_pr_auc(y10, r, S["duration"].to_numpy(), benign_row)[1]))

# ------------------------------------------------------------------ frozen thresholds (user level)
fixed = []
for name, t in THRESHOLDS.items():
    det = inwin.reindex(att).fillna(-1) > t
    anytime = peak.loc[att] > t
    fp = int((peak.loc[ben] > t).sum())
    tp = int(det.sum())
    first = S[S.in_window & (S.risk > t)].groupby("user")["wd_from_first_attack"].min().astype(float)
    pre_alarm = S[(S.day_type == "pre_window") & (S.risk > t)]["user"].nunique()
    lo_r, hi_r = cp(tp, len(att))
    lo_f, hi_f = cp(fp, len(ben))
    fixed.append({"threshold": name, "TP": tp, "FN": len(att) - tp, "FP": fp, "TN": len(ben) - fp,
                  "recall": tp / len(att), "recall_95CI": f"{lo_r:.3f}-{hi_r:.3f}",
                  "FPR": fp / len(ben), "FPR_95CI": f"{lo_f:.3f}-{hi_f:.3f}",
                  "precision": tp / (tp + fp) if tp + fp else np.nan,
                  "attackers_alarmed_any_time": int(anytime.sum()),
                  "attackers_alarmed_before_window": int(pre_alarm),
                  "latency_p25": first.quantile(.25), "latency_median": first.median(),
                  "latency_p75": first.quantile(.75), "latency_p90": first.quantile(.9),
                  "row_alarm_rate_benign": float((neg > t).mean())})
fixed = pd.DataFrame(fixed)

# descriptive only: recall at k benign false positives (threshold read from test benign peaks)
bpk = np.sort(peak.loc[ben].to_numpy())[::-1]
at_k = [{"benign_FP_allowed": k, "threshold_risk": float(bpk[k]),
         "recall": float((inwin.reindex(att).fillna(-1) > bpk[k]).mean())} for k in (0, 1, 2, 4, 8)]

# ------------------------------------------------------------------ bootstrap (users, stratified)
boot = []
groups = {u: g for u, g in S.groupby("user")}
for _ in range(2000):
    a_s = rng.choice(att, len(att), replace=True)
    b_s = rng.choice(ben, len(ben), replace=True)
    parts = [groups[u].assign(user=f"{u}#{i}") for i, u in enumerate(np.r_[a_s, b_s])]
    s = pd.concat(parts, ignore_index=True)
    ua = [f"{u}#{i}" for i, u in enumerate(a_s)]
    ub = [f"{u}#{i + len(a_s)}" for i, u in enumerate(b_s)]
    boot.append(metrics_all(s, ua, ub))
boot = pd.DataFrame(boot)
ci = boot.quantile([0.025, 0.975]).T.rename(columns={0.025: "ci_low", 0.975: "ci_high"})
ci["point"] = pd.Series(metrics_all(S, att, ben))

# ------------------------------------------------------------------ dev reference (from dev-only results)
dev = {"row PR-AUC (y_10)": "0.9439 (15-fold mean, Stage A)", "duration-macro row PR-AUC (y_10)": "0.8209 (nested, 05)",
       "user AUC (peak risk)": "0.9873 (Stage A OOF)", "High band recall / FP users": "0.906 / 15.7 of 511 (3.1%)",
       "Medium+ recall / FP users": "0.954 / 54.7 of 511 (10.7%)",
       "Dev 1% point recall / FP users": "0.871 / 5.33 of 511 at the frozen threshold (1.04%)"}

# ------------------------------------------------------------------ write
S_bands = pd.cut(S["risk"], [-1, 60, 85, 101], labels=["Low (<=60)", "Medium (60-85]", "High (>85)"])
bands = S.assign(band=S_bands).groupby("band", observed=True).agg(rows=("y", "size"), share_attack_days=("y", "mean"),
                                                                 share_y10=("y_10", "mean"))
bands["share_of_rows"] = bands["rows"] / len(S)
write_json(RESULTS / "08_test_threshold_free.json", tf)
fixed.to_csv(RESULTS / "08_test_fixed_thresholds.csv", index=False)
ci.to_csv(RESULTS / "08_test_bootstrap_ci.csv")
pd.DataFrame(at_k).to_csv(RESULTS / "08_test_recall_at_k_fp_descriptive.csv", index=False)
bands.to_csv(RESULTS / "08_test_bands.csv")

text = f"""# 08 — Test evaluation (single run of the frozen pipeline)

Frozen artefacts verified against `results/pretest/freeze.json` before scoring (model, manifest, split, rules, dataset).
Test: 136 users (52 attackers, 84 benign), {len(S):,} rows. Scored once; scores in `work/test/test_scores.parquet`.
No model, feature, calibration or threshold decision uses these results.

## Threshold-free
{md_table(pd.Series(tf, name='test').to_frame())}

Row PR-AUC per attack duration (test): {', '.join(f'{d} d: {v:.3f}' for d, v in per_dur.items())}

## Frozen thresholds (user level; observation period = each user's 2011 workdays, median 105)
{md_table(fixed, index=False)}

- TP: attackers with an in-window day above the threshold. FP: benign users whose peak daily risk exceeds it.
- Precision is attacker-level on this test mix (38% attackers); it does not transfer to production.
- Latency is in workdays from the first injected day.

## Bootstrap 95% CIs (2,000 user resamples, stratified)
{md_table(ci[['point', 'ci_low', 'ci_high']])}

## Descriptive only: recall at k benign false positives (threshold read from the test benign peaks)
{md_table(pd.DataFrame(at_k), index=False)}

## Rows per band (test)
{md_table(bands)}

## Dev reference (dev-only estimates, for comparison)
{md_table(pd.Series(dev, name='dev').to_frame())}
"""
(REPORTS / "08_test_evaluation.md").write_text(text, encoding="utf-8")
print(text)
