"""
13 — User-level binary classification evaluation of the frozen model on the frozen test predictions.

Read-only: uses work/test/test_scores.parquet (the single test scoring from 08) and model/manifest.json.
Nothing is re-scored, re-calibrated or re-thresholded; no threshold is chosen on test.

Unit of evaluation: ONE USER. Each test user (136) gets ONE score, risk_peak = the maximum daily calibrated risk
over all of that user's rows in the observation period (2011 workdays; the same rows 08 scored).
Ground truth: is_attacker (from the approved attacker scenarios), cross-checked against the row-level
attack-day label y (a user is an attacker iff at least one of its rows has y = 1).
Prediction rule: positive iff risk_peak > t, for each frozen operating point:
  t = 85     registered user-level alert rule (manifest thresholds.alert; mubsir_infer.score_users)
  t = 60     frozen Medium band lower edge (manifest bands)
  t = 92.63  frozen dev 1% user-FPR point (manifest; reporting only, not an alert rule)
Each threshold gets its own confusion matrix. Threshold-free metrics use risk_peak directly.

Outputs: results/13_user_classification_metrics.csv, results/13_user_threshold_free.csv,
         results/13_user_predictions.csv, reports/13_user_classification.md
"""

import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import roc_auc_score, average_precision_score

from common import WORK, RESULTS, REPORTS, MODEL_DIR, md_table

SCORES = WORK / "test" / "test_scores.parquet"
EPS = 1e-7
rng = np.random.default_rng(42)

# ------------------------------------------------------------------ frozen inputs (read only)
sha_before = hashlib.sha256(SCORES.read_bytes()).hexdigest()
man = json.loads((MODEL_DIR / "model" / "manifest.json").read_text(encoding="utf-8"))
A, B = man["calibration"]["a"], man["calibration"]["b"]
S = pd.read_parquet(SCORES)
pc = np.clip(S["p_raw"].to_numpy(), EPS, 1 - EPS)
assert np.allclose(100 / (1 + np.exp(-(A * np.log(pc / (1 - pc)) + B))), S["risk"].to_numpy(), atol=1e-9), \
    "stored risk does not follow the manifest calibration"
assert man["thresholds"]["alert"] == "risk > 85"
POINTS = [("registered alert rule: risk_peak > 85", 85.0),
          ("frozen Medium band edge: risk_peak > 60", 60.0),
          (f"frozen dev 1% user-FPR point: risk_peak > {man['thresholds']['dev_1pct_user_fpr_risk']:.2f}",
           man["thresholds"]["dev_1pct_user_fpr_risk"])]

# ------------------------------------------------------------------ one row per user
U = S.groupby("user").agg(risk_peak=("risk", "max"), n_days=("risk", "size"), is_attacker=("is_attacker", "first"),
                          any_attack_day=("y", "max"), duration=("duration", "first"))
U["peak_day"] = S.loc[S.groupby("user")["risk"].idxmax(), ["user", "day"]].set_index("user")["day"]
U["peak_in_attack_window"] = S.loc[S.groupby("user")["risk"].idxmax()].set_index("user")["in_window"]
assert S.groupby("user")["is_attacker"].nunique().max() == 1
assert (U["is_attacker"] == U["any_attack_day"]).all(), "attacker label disagrees with the attack-day ground truth"
y_true = U["is_attacker"].to_numpy().astype(int)
score = U["risk_peak"].to_numpy()
P, N = int(y_true.sum()), int((1 - y_true).sum())
assert (P, N) == (52, 84)


def cp(k, n):
    if n == 0:
        return (np.nan, np.nan)
    return (beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0, beta.ppf(0.975, k + 1, n - k) if k < n else 1.0)


def metrics(y, s, t):
    pred = s > t
    tp = int(((y == 1) & pred).sum()); fn = int(((y == 1) & ~pred).sum())
    fp = int(((y == 0) & pred).sum()); tn = int(((y == 0) & ~pred).sum())
    rec = tp / (tp + fn); spec = tn / (tn + fp)
    prec = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) else 0.0
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    return dict(TP=tp, FN=fn, FP=fp, TN=tn, accuracy=(tp + tn) / len(y), precision=prec, recall=rec,
                specificity=spec, F1=f1, F2=f2, balanced_accuracy=(rec + spec) / 2, MCC=mcc, NPV=npv,
                FPR=fp / (fp + tn), FNR=fn / (fn + tp))


# stratified user bootstrap for the composite metrics
att_idx, ben_idx = np.where(y_true == 1)[0], np.where(y_true == 0)[0]
boots = [np.r_[rng.choice(att_idx, P), rng.choice(ben_idx, N)] for _ in range(2000)]

rows = []
for name, t in POINTS:
    m = metrics(y_true, score, t)
    ci = {"accuracy": cp(m["TP"] + m["TN"], P + N), "precision": cp(m["TP"], m["TP"] + m["FP"]),
          "recall": cp(m["TP"], P), "specificity": cp(m["TN"], N), "NPV": cp(m["TN"], m["TN"] + m["FN"]),
          "FPR": cp(m["FP"], N), "FNR": cp(m["FN"], P)}
    bm = pd.DataFrame([metrics(y_true[i], score[i], t) for i in boots])
    for k in ("F1", "F2", "balanced_accuracy", "MCC"):
        ci[k] = tuple(bm[k].quantile([0.025, 0.975]))
    row = {"operating_point": name, "threshold_risk": t, "rule": f"positive iff risk_peak > {t:.6g}",
           "n_users": P + N, "n_attackers": P, "n_benign": N}
    for k, v in m.items():
        row[k] = v
        if k in ci:
            row[f"{k}_ci95_low"], row[f"{k}_ci95_high"] = ci[k]
    # contrast with the 08 in-window detection count (attacker counted only if an in-window day exceeds t)
    inwin = S[S.in_window].groupby("user")["risk"].max()
    row["attackers_with_in_window_alarm"] = int((inwin.reindex(U.index[y_true == 1]).fillna(-1) > t).sum())
    rows.append(row)
cls = pd.DataFrame(rows)

auc = roc_auc_score(y_true, score)
ap = average_precision_score(y_true, score)
b_auc = [roc_auc_score(y_true[i], score[i]) for i in boots]
b_ap = [average_precision_score(y_true[i], score[i]) for i in boots]
tf = pd.DataFrame([
    {"metric": "user-level ROC-AUC (risk_peak)", "value": auc, "ci95_low": np.quantile(b_auc, .025), "ci95_high": np.quantile(b_auc, .975),
     "chance_level": 0.5},
    {"metric": "user-level Average Precision / PR-AUC (risk_peak)", "value": ap, "ci95_low": np.quantile(b_ap, .025),
     "ci95_high": np.quantile(b_ap, .975), "chance_level": P / (P + N)}])

# ------------------------------------------------------------------ write
cls.to_csv(RESULTS / "13_user_classification_metrics.csv", index=False)
tf.to_csv(RESULTS / "13_user_threshold_free.csv", index=False)
out_u = U.reset_index()[["user", "is_attacker", "risk_peak", "n_days", "peak_day", "peak_in_attack_window", "duration"]]
for name, t in POINTS:
    out_u[f"pred_risk_peak_gt_{t:.2f}"] = (out_u["risk_peak"] > t).astype(int)
out_u.to_csv(RESULTS / "13_user_predictions.csv", index=False)
assert hashlib.sha256(SCORES.read_bytes()).hexdigest() == sha_before   # frozen scores untouched


def fmt(v, ci_lo=None, ci_hi=None, pct=True):
    if pct:
        s = f"{v:.3f}"
    else:
        s = f"{v:.3f}"
    return s if ci_lo is None or np.isnan(ci_lo) else f"{s} ({ci_lo:.3f}–{ci_hi:.3f})"


cm_blocks = []
for _, r in cls.iterrows():
    cm_blocks.append(f"""### {r.operating_point}

| | Predicted attacker | Predicted benign | Total |
|---|---:|---:|---:|
| **Actual attacker** | TP = {r.TP} | FN = {r.FN} | {r.n_attackers} |
| **Actual benign** | FP = {r.FP} | TN = {r.TN} | {r.n_benign} |
| **Total** | {r.TP + r.FP} | {r.FN + r.TN} | {r.n_users} |
""")
mtab = pd.DataFrame({
    r.operating_point.split(":")[0]: {
        "Threshold (risk_peak >)": f"{r.threshold_risk:.2f}",
        "Accuracy": fmt(r.accuracy, r.accuracy_ci95_low, r.accuracy_ci95_high),
        "Precision (PPV)": fmt(r.precision, r.precision_ci95_low, r.precision_ci95_high),
        "Recall / Sensitivity (TPR)": fmt(r.recall, r.recall_ci95_low, r.recall_ci95_high),
        "Specificity (TNR)": fmt(r.specificity, r.specificity_ci95_low, r.specificity_ci95_high),
        "F1": fmt(r.F1, r.F1_ci95_low, r.F1_ci95_high),
        "F2": fmt(r.F2, r.F2_ci95_low, r.F2_ci95_high),
        "Balanced accuracy": fmt(r.balanced_accuracy, r.balanced_accuracy_ci95_low, r.balanced_accuracy_ci95_high),
        "MCC": fmt(r.MCC, r.MCC_ci95_low, r.MCC_ci95_high),
        "NPV": fmt(r.NPV, r.NPV_ci95_low, r.NPV_ci95_high),
        "FPR": fmt(r.FPR, r.FPR_ci95_low, r.FPR_ci95_high),
        "FNR": fmt(r.FNR, r.FNR_ci95_low, r.FNR_ci95_high),
    } for _, r in cls.iterrows()})
nd = U["n_days"]
text = f"""# 13 — User-level classification evaluation (frozen model, frozen test predictions)

Read-only evaluation of `work/test/test_scores.parquet` (SHA-256 `{sha_before[:16]}…`, unchanged) with the frozen
calibration and thresholds of `model/manifest.json`. No threshold was chosen or tuned on test.

## Unit of evaluation
- **One row = one user.** 136 test users: **52 attackers (positive), 84 benign (negative)**; prevalence 38.2%.
- **Score:** `risk_peak` = the maximum daily calibrated risk over the user's whole observation period
  (2011 workdays; {int(nd.sum()):,} daily rows; per user median {int(nd.median())}, min {int(nd.min())}, max {int(nd.max())} workdays).
- **Ground truth:** `is_attacker` from the approved attacker scenarios. Cross-checked: it equals "at least one
  injected attack day (y = 1)" for all 136 users.
- **Prediction rule:** positive iff `risk_peak > t`. The registered user-level rule is **t = 85**
  (manifest `alert: risk > 85`; `mubsir_infer.score_users` sets `alert = risk_peak > 85`). The other two frozen operating
  points (60, 92.63) are reported separately. No other binary threshold was registered.
- No user-level number below is computed from daily rows: each user contributes exactly one score and one label.

## Threshold-independent (user level, risk_peak)
{md_table(tf.set_index('metric'))}

Chance level: ROC-AUC 0.5; AP = prevalence = 0.382.

## Confusion matrices (one per frozen operating point; not combined)
{''.join(cm_blocks)}
## Metrics at each operating point (95% CI: Clopper–Pearson for proportions; 2,000 stratified user bootstrap for F1, F2, balanced accuracy, MCC)
{md_table(mtab)}

## Reading notes
- These matrices classify on the **peak over the whole period**, so an attacker counts as a true positive even
  if its peak falls outside its attack window. The detection counts in `08_test_evaluation.md` count an attacker only if a day
  **inside** its window exceeds the threshold: {', '.join(f"{r.attackers_with_in_window_alarm} vs TP {r.TP} at {r.threshold_risk:.2f}" for _, r in cls.iterrows())}.
  Both definitions use the same users and labels; they answer different questions (who is flagged vs whether the attack itself was caught).
- Accuracy, precision, NPV, F1, F2 and MCC depend on the test prevalence (38.2% attackers). Production prevalence is far
  lower, so these values do not transfer; recall, specificity, FPR, FNR and balanced accuracy do not depend on prevalence.
- 84 benign users: one false positive moves FPR by 1.19 points; the CIs reflect this.
- Row-level metrics (daily PR-AUC etc.) are in `08_test_evaluation.md` and are not comparable with these user-level values.
"""
(REPORTS / "13_user_classification.md").write_text(text, encoding="utf-8")
print(text)
