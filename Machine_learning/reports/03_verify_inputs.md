# 03 — Input, label and split verification

**25/25 checks PASS.** Model dataset SHA-256 `1b26162ba13fd53f049e47608ea86efeb70d6ec42ba5839c9d7dda548337ddbc`.

## Strongest single predictors for y_10 (dev rows only, AUC folded to >= 0.5)
| feature | auc_y10_dev |
|---|---|
| cumulative_deviation_10wd | 0.9436 |
| cumulative_deviation_20wd | 0.9328 |
| consecutive_elevated_days_20wd | 0.9048 |
| cumulative_deviation_30wd | 0.9012 |
| cumulative_deviation_5wd | 0.8937 |
| consecutive_elevated_days_10wd | 0.8918 |
| consecutive_elevated_days_30wd | 0.8880 |
| nr_excl_days_20wd | 0.8774 |
| nr_excl_days_10wd | 0.8746 |
| external_email_count_10wd | 0.8632 |

Full list: `results/03_single_feature_auc_dev.csv`.

## Checks
- [PASS] Model dataset SHA-256 matches prepare_dataset.py report 1b26162ba13fd53f
- [PASS] 100 predictors 100
- [PASS] Dataset columns = keys + y + predictors set()
- [PASS] No forbidden column among predictors 
- [PASS] No excluded allowlist column (C1/C2/ids) among predictors set()
- [PASS] No raw baseline column (p50/p90/p95/mad/active_rate) among predictors 
- [PASS] All predictors numeric 
- [PASS] No inf in predictors 
- [PASS] NaN only in channel_distribution_shift_* ['channel_distribution_shift_5wd', 'channel_distribution_shift_10wd', 'channel_distribution_shift_20wd', 'channel_distribution_shift_30wd']
- [PASS] No constant predictor []
- [PASS] No predictor constant within every user (identity column) []
- [PASS] Labels aligned with dataset keys 
- [PASS] y_h equals an independent trailing-window loop 0
- [PASS] No predictor equals y_5 []
- [PASS] No predictor equals y_10 []
- [PASS] No predictor equals y_20 []
- [PASS] Evaluation metadata aligned with dataset keys 
- [PASS] Evaluation metadata holds no predictor column 
- [PASS] Split covers the 995 dataset users 
- [PASS] Dev and test users disjoint 
- [PASS] Dev and test schedule groups disjoint 
- [PASS] No (user, day) shared by dev and test 
- [PASS] CV folds use dev users only 
- [PASS] Legacy-val users all in dev 
- [PASS] No single predictor separates y_10 almost perfectly (AUC < 0.99, dev only) {'feature': 'cumulative_deviation_10wd', 'auc_y10_dev': 0.9435642062046374}
