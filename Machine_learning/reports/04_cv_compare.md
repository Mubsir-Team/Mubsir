# 04 — Stage A: algorithms x label horizons (dev CV, 15 folds)

Rules: `config/selection_rules.json` (written before training). Test users not used.

**Selected algorithm: `lightgbm` · selected label horizon: y_10**

## Summary (fold means; user metrics pooled per repeat, mean of 3)
| algo | h | pr_auc | pr_auc_sd | recall@1%fpr | recall@1%fpr_sd | recall@5%fpr | roc_auc | day_pr_auc | user_auc | user_recall@1%ufpr | latency_median@1%ufpr | user_recall@5%ufpr | latency_median@5%ufpr | mean_trees | seconds_per_fold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| catboost | 5 | 0.9479 | 0.0036 | 0.8868 | 0.0124 | 0.9526 | 0.9865 | 0.8525 | 0.9915 | 0.8860 | 3.0000 | 0.9406 | 1.0000 | 288.3333 | 5.8267 |
| lightgbm | 5 | 0.9477 | 0.0043 | 0.8857 | 0.0121 | 0.9538 | 0.9870 | 0.8369 | 0.9884 | 0.8640 | 3.0000 | 0.9243 | 1.3333 | 275.6667 | 2.3533 |
| xgboost | 5 | 0.9464 | 0.0045 | 0.8829 | 0.0133 | 0.9523 | 0.9868 | 0.8321 | 0.9879 | 0.8726 | 3.0000 | 0.9205 | 2.0000 | 325.8000 | 5.6533 |
| random_forest | 5 | 0.9061 | 0.0089 | 0.8165 | 0.0182 | 0.9190 | 0.9756 | 0.7496 | 0.9740 | 0.7701 | 6.0000 | 0.8659 | 4.0000 | nan | 3.8400 |
| decision_tree | 5 | 0.8786 | 0.0095 | 0.7715 | 0.0184 | 0.8827 | 0.9630 | 0.6989 | 0.9691 | 0.7414 | 5.3333 | 0.8554 | 4.0000 | nan | 1.9533 |
| catboost | 10 | 0.9443 | 0.0047 | 0.8708 | 0.0124 | 0.9427 | 0.9836 | 0.8034 | 0.9914 | 0.8841 | 3.0000 | 0.9406 | 1.1667 | 313.8667 | 6.1867 |
| lightgbm | 10 | 0.9439 | 0.0046 | 0.8630 | 0.0143 | 0.9432 | 0.9840 | 0.7729 | 0.9873 | 0.8707 | 3.0000 | 0.9195 | 2.0000 | 237.3333 | 1.7667 |
| xgboost | 10 | 0.9428 | 0.0041 | 0.8621 | 0.0153 | 0.9420 | 0.9837 | 0.7757 | 0.9874 | 0.8621 | 3.6667 | 0.9224 | 2.0000 | 272.0667 | 5.3533 |
| random_forest | 10 | 0.9100 | 0.0104 | 0.8071 | 0.0229 | 0.9089 | 0.9733 | 0.6884 | 0.9744 | 0.7807 | 6.0000 | 0.8678 | 4.0000 | nan | 3.8533 |
| decision_tree | 10 | 0.8771 | 0.0084 | 0.7501 | 0.0233 | 0.8746 | 0.9561 | 0.6204 | 0.9617 | 0.0000 | nan | 0.8305 | 4.6667 | nan | 1.9133 |
| catboost | 20 | 0.9377 | 0.0050 | 0.8322 | 0.0146 | 0.9197 | 0.9774 | 0.7078 | 0.9890 | 0.8716 | 3.0000 | 0.9272 | 2.0000 | 231.9333 | 4.9800 |
| lightgbm | 20 | 0.9375 | 0.0085 | 0.8303 | 0.0216 | 0.9191 | 0.9778 | 0.6610 | 0.9849 | 0.8506 | 4.0000 | 0.9061 | 2.0000 | 203.0667 | 1.6000 |
| xgboost | 20 | 0.9374 | 0.0054 | 0.8272 | 0.0175 | 0.9176 | 0.9779 | 0.6748 | 0.9838 | 0.8295 | 4.3333 | 0.9042 | 2.6667 | 226.0667 | 5.0467 |
| random_forest | 20 | 0.9144 | 0.0111 | 0.7851 | 0.0332 | 0.8902 | 0.9695 | 0.6086 | 0.9743 | 0.7864 | 6.0000 | 0.8745 | 4.0000 | nan | 3.2933 |
| decision_tree | 20 | 0.8758 | 0.0140 | 0.7117 | 0.0404 | 0.8529 | 0.9507 | 0.5190 | 0.9585 | 0.0000 | nan | 0.8170 | 6.0000 | nan | 1.8267 |

- `pr_auc`, `recall@*%fpr`, `roc_auc`: row level on the model's own label y_h (not comparable across h).
- `day_pr_auc`: row level on the attack-day label y (same target for every h).
- `user_*`, `latency_*`: comparable across h. Threshold = share of benign dev users that ever alarm (1% / 5%).
  Latency in workdays from the first injected day to the first in-window alarm.

## Rule A — algorithm (mean over h)
| algo | recall@1%fpr | pr_auc |
|---|---|---|
| catboost | 0.8633 | 0.9433 |
| decision_tree | 0.7444 | 0.8772 |
| lightgbm | 0.8597 | 0.9431 |
| random_forest | 0.8029 | 0.9101 |
| xgboost | 0.8574 | 0.9422 |

Qualifying challengers: none -> **lightgbm**.

## Rule B — horizon (lightgbm)
| h | user_auc | user_recall@1%ufpr | latency_median@1%ufpr | user_recall@5%ufpr | day_pr_auc |
|---|---|---|---|---|---|
| 5 | 0.9884 | 0.8640 | 3.0000 | 0.9243 | 0.8369 |
| 10 | 0.9873 | 0.8707 | 3.0000 | 0.9195 | 0.7729 |
| 20 | 0.9849 | 0.8506 | 4.0000 | 0.9061 | 0.6610 |

Qualifying candidates: none -> **y_10**.
