# 09 — Breakdown by simulation dimension

Model: frozen LightGBM y_10 + Platt. Threshold for detection and latency: alert band risk > 85 (frozen).
**dev** = Stage A out-of-fold scores (348 attackers / 511 benign; mean of 3 repeats).
**test** = the single frozen test scoring (52 / 84). Test cells with fewer than 10 attackers are flagged; read them from dev.
`row_pr_auc` = y_10 average precision over the group's attackers' rows plus all benign rows.
Latency = workdays from the first injected day to the first in-window alarm.

### By stealth level (injection multiplier)
| stealth_level | dev_attackers | dev_detected@85 | dev_range_3_repeats | dev_latency_median | dev_latency_p90 | dev_row_pr_auc | test_attackers | test_detected@85 | test_rate_95CI | test_latency_median | test_row_pr_auc | small_cell |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| High (0.25) | 117 | 0.7493 | 0.73-0.77 | 6.0000 | 14.8000 | 0.7982 | 17 | 14/17 | 0.57-0.96 | 6.0000 | 0.8359 |  |
| Low (0.75) | 115 | 1.0000 | 1.00-1.00 | 1.0000 | 4.0000 | 0.9819 | 18 | 17/18 | 0.73-1.00 | 2.0000 | 0.9823 |  |
| Medium (0.50) | 116 | 0.9713 | 0.97-0.97 | 3.0000 | 6.0000 | 0.9514 | 17 | 16/17 | 0.71-1.00 | 1.5000 | 0.9722 |  |

### By attack duration (workdays)
| window_workdays | dev_attackers | dev_detected@85 | dev_range_3_repeats | dev_latency_median | dev_latency_p90 | dev_row_pr_auc | test_attackers | test_detected@85 | test_rate_95CI | test_latency_median | test_row_pr_auc | small_cell |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5.0000 | 46 | 0.6159 | 0.59-0.65 | 1.0000 | 4.0000 | 0.4991 | 4 | 1/4 | 0.01-0.81 | 4.0000 | 0.1423 | yes |
| 10.0000 | 60 | 0.8722 | 0.87-0.88 | 3.0000 | 7.0000 | 0.7535 | 10 | 8/10 | 0.44-0.97 | 2.5000 | 0.7935 |  |
| 20.0000 | 60 | 0.9444 | 0.93-0.95 | 3.0000 | 12.0000 | 0.8592 | 10 | 10/10 | 0.69-1.00 | 4.0000 | 0.9058 |  |
| 30.0000 | 60 | 0.9556 | 0.95-0.97 | 3.0000 | 7.0000 | 0.9116 | 10 | 10/10 | 0.69-1.00 | 2.0000 | 0.9636 |  |
| 40.0000 | 61 | 1.0000 | 1.00-1.00 | 3.0000 | 8.8000 | 0.9521 | 9 | 9/9 | 0.66-1.00 | 2.0000 | 0.9372 | yes |
| 50.0000 | 61 | 0.9781 | 0.97-0.98 | 2.0000 | 11.0000 | 0.9497 | 9 | 9/9 | 0.66-1.00 | 3.0000 | 0.9695 | yes |

### By switching level
| switching_level | dev_attackers | dev_detected@85 | dev_range_3_repeats | dev_latency_median | dev_latency_p90 | dev_row_pr_auc | test_attackers | test_detected@85 | test_rate_95CI | test_latency_median | test_row_pr_auc | small_cell |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| High | 115 | 0.9014 | 0.89-0.91 | 2.0000 | 8.0000 | 0.9119 | 18 | 14/18 | 0.52-0.94 | 3.5000 | 0.8910 |  |
| Low | 117 | 0.8689 | 0.85-0.89 | 2.0000 | 7.0000 | 0.9184 | 16 | 15/16 | 0.70-1.00 | 2.0000 | 0.9364 |  |
| Medium | 116 | 0.9483 | 0.95-0.95 | 3.0000 | 8.0000 | 0.9302 | 18 | 18/18 | 0.81-1.00 | 3.0000 | 0.9689 |  |

### By channel configuration
| channel_configuration | dev_attackers | dev_detected@85 | dev_range_3_repeats | dev_latency_median | dev_latency_p90 | dev_row_pr_auc | test_attackers | test_detected@85 | test_rate_95CI | test_latency_median | test_row_pr_auc | small_cell |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E+H | 104 | 0.9679 | 0.96-0.97 | 2.0000 | 6.0000 | 0.9489 | 13 | 12/13 | 0.64-1.00 | 2.0000 | 0.9726 |  |
| E+U | 101 | 0.8515 | 0.83-0.87 | 3.0000 | 8.0000 | 0.9009 | 16 | 14/16 | 0.62-0.98 | 3.5000 | 0.9040 |  |
| E+U+H | 119 | 0.9524 | 0.94-0.96 | 3.0000 | 9.0000 | 0.9265 | 20 | 20/20 | 0.83-1.00 | 3.5000 | 0.9617 |  |
| U+H | 24 | 0.6389 | 0.58-0.67 | 3.0000 | 5.5000 | 0.5463 | 3 | 1/3 | 0.01-0.91 | 12.0000 | 0.5225 | yes |

### Detection by duration x stealth (dev, share detected at 85, mean of 3 repeats)
| duration | High (0.25) | Low (0.75) | Medium (0.50) |
|---|---|---|---|
| 5.0000 | 0.2833 | 1.0000 | 0.7917 |
| 10.0000 | 0.6349 | 1.0000 | 1.0000 |
| 20.0000 | 0.8667 | 1.0000 | 1.0000 |
| 30.0000 | 0.8596 | 1.0000 | 1.0000 |
| 40.0000 | 1.0000 | 1.0000 | 1.0000 |
| 50.0000 | 0.9111 | 1.0000 | 1.0000 |

Attackers per cell (dev): {'High (0.25)': {5.0: 20, 10.0: 21, 20.0: 25, 30.0: 19, 40.0: 17, 50.0: 15}, 'Low (0.75)': {5.0: 10, 10.0: 15, 20.0: 18, 30.0: 21, 40.0: 27, 50.0: 24}, 'Medium (0.50)': {5.0: 16, 10.0: 24, 20.0: 17, 30.0: 20, 40.0: 17, 50.0: 22}}

### Attack days by injected channel set (share of attack days above the threshold)
| group | dev_attack_days | test_attack_days | dev_share_above_85 | test_share_above_85 | dev_share_above_60 | test_share_above_60 | dev_median_risk | test_median_risk |
|---|---|---|---|---|---|---|---|---|
| EMAIL | 5045.0000 | 685.0000 | 0.8131 | 0.8307 | 0.8858 | 0.9124 | 99.4757 | 99.5461 |
| USB | 1144.0000 | 270.0000 | 0.5862 | 0.6370 | 0.6824 | 0.7556 | 93.9821 | 97.0786 |
| HTTP | 816.0000 | 113.0000 | 0.7222 | 0.6903 | 0.8150 | 0.7699 | 98.8998 | 99.4673 |
| EMAIL+HTTP | 426.0000 | 60.0000 | 0.9382 | 0.9500 | 0.9781 | 1.0000 | 99.7718 | 99.8560 |
| EMAIL+USB | 384.0000 | 66.0000 | 0.8342 | 0.8333 | 0.8958 | 0.9545 | 99.6040 | 99.4516 |
| EMAIL+USB+HTTP | 34.0000 | 3.0000 | 0.8725 | 1.0000 | 0.9412 | 1.0000 | 99.8297 | 99.8547 |
| USB+HTTP | 29.0000 | 5.0000 | 0.8161 | 0.8000 | 0.8966 | 1.0000 | 97.2400 | 98.0888 |

### Attack days by same-day synchronisation
| group | dev_attack_days | test_attack_days | dev_share_above_85 | test_share_above_85 | dev_share_above_60 | test_share_above_60 | dev_median_risk | test_median_risk |
|---|---|---|---|---|---|---|---|---|
| SINGLE | 7005.0000 | 1068.0000 | 0.7655 | 0.7669 | 0.8443 | 0.8577 | 99.2207 | 99.2659 |
| TWO_CHANNEL | 839.0000 | 131.0000 | 0.8864 | 0.8855 | 0.9376 | 0.9771 | 99.7072 | 99.7896 |
| THREE_CHANNEL | 34.0000 | 3.0000 | 0.8725 | 1.0000 | 0.9412 | 1.0000 | 99.8297 | 99.8547 |

### Non-attack days: alarm rate by day type
| day_type | dev_rows | test_rows | dev_alarm_rate_85 | test_alarm_rate_85 | dev_alarm_rate_60 | test_alarm_rate_60 | dev_y10_positive_share | test_y10_positive_share |
|---|---|---|---|---|---|---|---|---|
| benign | 53243.0000 | 8510.0000 | 0.0009 | 0.0001 | 0.0041 | 0.0018 | 0.0000 | 0.0000 |
| offday | 1442.0000 | 228.0000 | 0.6810 | 0.7061 | 0.7550 | 0.7456 | 0.9619 | 0.9561 |
| post_window | 15371.0000 | 2266.0000 | 0.1083 | 0.1209 | 0.1503 | 0.1531 | 0.1852 | 0.1893 |
| pre_window | 11537.0000 | 1718.0000 | 0.0015 | 0.0012 | 0.0077 | 0.0070 | 0.0000 | 0.0000 |

`y10_positive_share`: forced off-days and early post-window days are y_10-positive by definition (an attack day in the
last 10 workdays); pre-window and benign rows never are.

### Active days only (audit M2: rows without any channel activity are easy negatives; about 26-28% of rows in the 995-user scope)
| source | row_pr_auc_all_rows | row_pr_auc_active_rows | share_active_rows |
|---|---|---|---|
| dev | 0.9434 | 0.9472 | 0.7436 |
| test | 0.9551 | 0.9598 | 0.7225 |
