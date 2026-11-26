# 08 — Test evaluation (single run of the frozen pipeline)

Frozen artefacts verified against `results/pretest/freeze.json` before scoring (model, manifest, split, rules, dataset).
Test: 136 users (52 attackers, 84 benign), 13,924 rows. Scored once; scores in `work/test/test_scores.parquet`.
No model, feature, calibration or threshold decision uses these results.

## Threshold-free
| index | test |
|---|---|
| row PR-AUC (y_10) | 0.9551 |
| row ROC-AUC (y_10) | 0.9860 |
| duration-macro row PR-AUC (y_10) | 0.7853 |
| row PR-AUC on attack-day label y | 0.7774 |
| row recall@1% FPR (benign-user rows of test) | 0.8978 |
| row recall@5% FPR (benign-user rows of test) | 0.9535 |
| user AUC (peak risk) | 0.9886 |
| user PR-AUC (peak risk) | 0.9867 |
| Brier (risk/100 vs y_10) | 0.0207 |
| mean calibrated prediction | 0.1326 |
| observed y_10 rate | 0.1328 |

Row PR-AUC per attack duration (test): 5 d: 0.142, 10 d: 0.794, 20 d: 0.906, 30 d: 0.964, 40 d: 0.937, 50 d: 0.970

## Frozen thresholds (user level; observation period = each user's 2011 workdays, median 105)
| threshold | TP | FN | FP | TN | recall | recall_95CI | FPR | FPR_95CI | precision | attackers_alarmed_any_time | attackers_alarmed_before_window | latency_p25 | latency_median | latency_p75 | latency_p90 | row_alarm_rate_benign |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| High band (risk > 85) | 47 | 5 | 1 | 83 | 0.9038 | 0.790-0.968 | 0.0119 | 0.000-0.065 | 0.9792 | 48 | 1 | 1.0000 | 3.0000 | 5.0000 | 9.0000 | 0.0001 |
| Medium+ (risk > 60) | 49 | 3 | 6 | 78 | 0.9423 | 0.841-0.988 | 0.0714 | 0.027-0.149 | 0.8909 | 51 | 5 | 0.0000 | 1.0000 | 3.0000 | 5.0000 | 0.0018 |
| Dev 1% point (risk > 92.63) | 45 | 7 | 1 | 83 | 0.8654 | 0.742-0.944 | 0.0119 | 0.000-0.065 | 0.9783 | 45 | 0 | 2.0000 | 4.0000 | 6.0000 | 12.0000 | 0.0001 |

- TP: attackers with an in-window day above the threshold. FP: benign users whose peak daily risk exceeds it.
- Precision is attacker-level on this test mix (38% attackers); it does not transfer to production.
- Latency is in workdays from the first injected day.

## Bootstrap 95% CIs (2,000 user resamples, stratified)
| index | point | ci_low | ci_high |
|---|---|---|---|
| row_pr_auc | 0.9551 | 0.9312 | 0.9734 |
| row_roc_auc | 0.9860 | 0.9773 | 0.9924 |
| duration_macro_row_pr_auc | 0.7853 | 0.7387 | 0.8334 |
| user_auc | 0.9886 | 0.9684 | 0.9993 |
| recall_85.00 | 0.9038 | 0.8269 | 0.9808 |
| fp_85.00 | 1.0000 | 0.0000 | 3.0000 |
| recall_60.00 | 0.9423 | 0.8654 | 1.0000 |
| fp_60.00 | 6.0000 | 2.0000 | 11.0000 |
| recall_92.63 | 0.8654 | 0.7692 | 0.9423 |
| fp_92.63 | 1.0000 | 0.0000 | 3.0000 |

## Descriptive only: recall at k benign false positives (threshold read from the test benign peaks)
| benign_FP_allowed | threshold_risk | recall |
|---|---|---|
| 0 | 95.1732 | 0.8462 |
| 1 | 84.8231 | 0.9038 |
| 2 | 76.8965 | 0.9038 |
| 4 | 65.0881 | 0.9423 |
| 8 | 47.5844 | 0.9808 |

## Rows per band (test)
| band | rows | share_attack_days | share_y10 | share_of_rows |
|---|---|---|---|---|
| Low (<=60) | 12333 | 0.0126 | 0.0255 | 0.8857 |
| Medium (60-85] | 215 | 0.5070 | 0.8140 | 0.0154 |
| High (>85) | 1376 | 0.6817 | 0.9884 | 0.0988 |

## Dev reference (dev-only estimates, for comparison)
| index | dev |
|---|---|
| row PR-AUC (y_10) | 0.9439 (15-fold mean, Stage A) |
| duration-macro row PR-AUC (y_10) | 0.8209 (nested, 05) |
| user AUC (peak risk) | 0.9873 (Stage A OOF) |
| High band recall / FP users | 0.906 / 15.7 of 511 (3.1%) |
| Medium+ recall / FP users | 0.954 / 54.7 of 511 (10.7%) |
| Dev 1% point recall / FP users | 0.871 / 5.33 of 511 at the frozen threshold (1.04%) |
