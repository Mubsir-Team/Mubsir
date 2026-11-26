# 13 — User-level classification evaluation (frozen model, frozen test predictions)

Read-only evaluation of `work/test/test_scores.parquet` (SHA-256 `bed37ad0e93e9f6b…`, unchanged) with the frozen
calibration and thresholds of `model/manifest.json`. No threshold was chosen or tuned on test.

## Unit of evaluation
- **One row = one user.** 136 test users: **52 attackers (positive), 84 benign (negative)**; prevalence 38.2%.
- **Score:** `risk_peak` = the maximum daily calibrated risk over the user's whole observation period
  (2011 workdays; 13,924 daily rows; per user median 105, min 7, max 107 workdays).
- **Ground truth:** `is_attacker` from the approved attacker scenarios. Cross-checked: it equals "at least one
  injected attack day (y = 1)" for all 136 users.
- **Prediction rule:** positive iff `risk_peak > t`. The registered user-level rule is **t = 85**
  (manifest `alert: risk > 85`; `mubsir_infer.score_users` sets `alert = risk_peak > 85`). The other two frozen operating
  points (60, 92.63) are reported separately. No other binary threshold was registered.
- No user-level number below is computed from daily rows: each user contributes exactly one score and one label.

## Threshold-independent (user level, risk_peak)
| metric | value | ci95_low | ci95_high | chance_level |
|---|---|---|---|---|
| user-level ROC-AUC (risk_peak) | 0.9886 | 0.9684 | 0.9993 | 0.5000 |
| user-level Average Precision / PR-AUC (risk_peak) | 0.9867 | 0.9665 | 0.9989 | 0.3824 |

Chance level: ROC-AUC 0.5; AP = prevalence = 0.382.

## Confusion matrices (one per frozen operating point; not combined)
### registered alert rule: risk_peak > 85

| | Predicted attacker | Predicted benign | Total |
|---|---:|---:|---:|
| **Actual attacker** | TP = 48 | FN = 4 | 52 |
| **Actual benign** | FP = 1 | TN = 83 | 84 |
| **Total** | 49 | 87 | 136 |
### frozen Medium band edge: risk_peak > 60

| | Predicted attacker | Predicted benign | Total |
|---|---:|---:|---:|
| **Actual attacker** | TP = 51 | FN = 1 | 52 |
| **Actual benign** | FP = 6 | TN = 78 | 84 |
| **Total** | 57 | 79 | 136 |
### frozen dev 1% user-FPR point: risk_peak > 92.63

| | Predicted attacker | Predicted benign | Total |
|---|---:|---:|---:|
| **Actual attacker** | TP = 45 | FN = 7 | 52 |
| **Actual benign** | FP = 1 | TN = 83 | 84 |
| **Total** | 46 | 90 | 136 |

## Metrics at each operating point (95% CI: Clopper–Pearson for proportions; 2,000 stratified user bootstrap for F1, F2, balanced accuracy, MCC)
| index | registered alert rule | frozen Medium band edge | frozen dev 1% user-FPR point |
|---|---|---|---|
| Threshold (risk_peak >) | 85.00 | 60.00 | 92.63 |
| Accuracy | 0.963 (0.916–0.988) | 0.949 (0.897–0.979) | 0.941 (0.887–0.974) |
| Precision (PPV) | 0.980 (0.891–0.999) | 0.895 (0.785–0.960) | 0.978 (0.885–0.999) |
| Recall / Sensitivity (TPR) | 0.923 (0.815–0.979) | 0.981 (0.897–1.000) | 0.865 (0.742–0.944) |
| Specificity (TNR) | 0.988 (0.935–1.000) | 0.929 (0.851–0.973) | 0.988 (0.935–1.000) |
| F1 | 0.950 (0.905–0.990) | 0.936 (0.889–0.972) | 0.918 (0.860–0.970) |
| F2 | 0.934 (0.873–0.985) | 0.962 (0.923–0.989) | 0.886 (0.806–0.953) |
| Balanced accuracy | 0.956 (0.917–0.990) | 0.955 (0.919–0.982) | 0.927 (0.879–0.971) |
| MCC | 0.922 (0.859–0.984) | 0.896 (0.821–0.955) | 0.877 (0.797–0.954) |
| NPV | 0.954 (0.886–0.987) | 0.987 (0.931–1.000) | 0.922 (0.846–0.968) |
| FPR | 0.012 (0.000–0.065) | 0.071 (0.027–0.149) | 0.012 (0.000–0.065) |
| FNR | 0.077 (0.021–0.185) | 0.019 (0.000–0.103) | 0.135 (0.056–0.258) |

## Reading notes
- These matrices classify on the **peak over the whole period**, so an attacker counts as a true positive even
  if its peak falls outside its attack window. The detection counts in `08_test_evaluation.md` count an attacker only if a day
  **inside** its window exceeds the threshold: 47 vs TP 48 at 85.00, 49 vs TP 51 at 60.00, 45 vs TP 45 at 92.63.
  Both definitions use the same users and labels; they answer different questions (who is flagged vs whether the attack itself was caught).
- Accuracy, precision, NPV, F1, F2 and MCC depend on the test prevalence (38.2% attackers). Production prevalence is far
  lower, so these values do not transfer; recall, specificity, FPR, FNR and balanced accuracy do not depend on prevalence.
- 84 benign users: one false positive moves FPR by 1.19 points; the CIs reflect this.
- Row-level metrics (daily PR-AUC etc.) are in `08_test_evaluation.md` and are not comparable with these user-level values.
