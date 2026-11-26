# 07 — Calibration, final fit, dev-frozen thresholds

Adopted configuration: LightGBM defaults, y_10, 100 predictors (05: tuning not adopted; 06: no ablation adopted).
Dev users only (859); test users never loaded.

## Platt calibration
Stacked fit (3 repeats, 268,413 rows): **a = 0.9549, b = 0.0760**.

| repeat | a | b |
|---|---|---|
| 0 | 0.9649 | 0.0824 |
| 1 | 0.9369 | 0.0615 |
| 2 | 0.9629 | 0.0843 |

Quality, with each outer fold calibrated by a fit on the other 4 folds of its repeat:

| repeat | brier_raw | brier_calibrated_lfo | ece_raw | ece_calibrated_lfo | mean_pred_raw | mean_pred_cal | observed_rate | pr_auc_raw | pr_auc_cal |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.0257 | 0.0258 | 0.0042 | 0.0027 | 0.1322 | 0.1355 | 0.1354 | 0.9415 | 0.9413 |
| 1 | 0.0252 | 0.0253 | 0.0052 | 0.0027 | 0.1320 | 0.1357 | 0.1354 | 0.9442 | 0.9434 |
| 2 | 0.0249 | 0.0249 | 0.0044 | 0.0019 | 0.1322 | 0.1354 | 0.1354 | 0.9447 | 0.9445 |

Reliability (calibrated, leave-fold-out, mean of 3 repeats):

| bin | rows | mean_pred | observed |
|---|---|---|---|
| 0.0-0.1 | 73249.6667 | 0.0105 | 0.0111 |
| 0.1-0.2 | 2625.6667 | 0.1418 | 0.1509 |
| 0.2-0.3 | 1194.3333 | 0.2459 | 0.2447 |
| 0.3-0.4 | 784.6667 | 0.3478 | 0.3391 |
| 0.4-0.5 | 636.3333 | 0.4500 | 0.4295 |
| 0.5-0.6 | 534.6667 | 0.5501 | 0.4921 |
| 0.6-0.7 | 572.6667 | 0.6507 | 0.6052 |
| 0.7-0.8 | 616.6667 | 0.7516 | 0.7108 |
| 0.8-0.9 | 924.6667 | 0.8545 | 0.8511 |
| 0.9-1.0 | 8331.6667 | 0.9851 | 0.9887 |

## Final model
- LightGBM defaults, **237 trees** (mean best iteration of the 15 Stage A folds), seed 42, all 859 dev users, 89,471 rows.
- Saved as `model/lgbm_y10.txt` (native text). Reloaded predictions match in-memory predictions (max diff 0.0e+00).
- `model/manifest.json` holds the feature order, calibration, thresholds, and provenance hashes.
- The OOF-fitted calibration does not see the final model's slightly larger training set (documented limitation).

## Dev operating points on the risk scale (OOF, calibrated, mean of 3 repeats)
| threshold | value | user_recall | fp_benign_users | row_alarm_rate_benign |
|---|---|---|---|---|
| band_high_85 | 85.0000 | 0.9061 | 15.6667 | 0.0009 |
| band_medium_60 | 60.0000 | 0.9540 | 54.6667 | 0.0041 |
| dev_1pct_user_fpr | 92.6320 | 0.8707 | 5.0000 | 0.0002 |

- `user_recall`: attackers with an in-window day above the threshold.
- `fp_benign_users`: of 511 benign dev users, those whose peak risk exceeds it.
- Frozen for test: bands 60 / 85, and dev 1% user-FPR point = **risk 92.63**.

Share of dev rows per band (OOF, mean of 3 repeats):

| band | share_of_rows |
|---|---|
| Low (<=60) | 0.8833 |
| Medium (60-85] | 0.0178 |
| High (>85) | 0.0989 |
