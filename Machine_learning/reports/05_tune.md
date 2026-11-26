# 05 — Stage B nested tuning (LightGBM, y_10)

Rules: `config/selection_rules.json` stage_B (SHA-256 `41de4354e861fb08…`, verified at start). Dev users only; test untouched.

## Decision: **KEEP DEFAULT configuration**
| Condition | Holds |
|---|---|
| tuned primary >= default primary + 0.005 | no |
| tuned row_pr_auc (15-fold mean) >= default row_pr_auc - 0.002 | yes |
| tuned frozen_inner_user_recall_1pct >= default frozen_inner_user_recall_1pct | yes |

## Nested estimates
| index | primary | row_pr_auc | user_recall | fp_users_per_repeat | primary_delta_vs_default |
|---|---|---|---|---|---|
| default | 0.8209 | 0.9439 | 0.8554 | 4.3333 | 0.0000 |
| tuned | 0.8185 | 0.9437 | 0.8640 | 5.3333 | -0.0024 |

- `primary`: duration-macro row PR-AUC on pooled outer OOF per repeat, mean of 3.
- `row_pr_auc`: 15-fold mean.
- `user_recall`: frozen inner-OOF threshold (1% of benign outer-train users), pooled per repeat, mean of 3.
- `fp_users_per_repeat`: realised false-positive benign users out of 511 per repeat.

## Per repeat
| repeat | default_duration_macro | default_user_recall | default_fp_users | default_pooled_row_pr_auc | default_recall@1%fpr_row | tuned_duration_macro | tuned_user_recall | tuned_fp_users | tuned_pooled_row_pr_auc | tuned_recall@1%fpr_row |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.8189 | 0.8592 | 5 | 0.9415 | 0.8617 | 0.8193 | 0.8649 | 5 | 0.9408 | 0.8652 |
| 1 | 0.8224 | 0.8563 | 4 | 0.9442 | 0.8657 | 0.8174 | 0.8764 | 6 | 0.9441 | 0.8641 |
| 2 | 0.8212 | 0.8506 | 4 | 0.9447 | 0.8629 | 0.8186 | 0.8506 | 5 | 0.9439 | 0.8627 |

## Per duration (row PR-AUC, mean of 3 repeats)
| duration | default | tuned |
|---|---|---|
| 5 | 0.4991 | 0.4898 |
| 10 | 0.7535 | 0.7510 |
| 20 | 0.8592 | 0.8589 |
| 30 | 0.9116 | 0.9109 |
| 40 | 0.9521 | 0.9519 |
| 50 | 0.9497 | 0.9483 |

## Per outer fold
| repeat | fold | best_trial | inner_trial0 | inner_default_exact | inner_best | trees_default | trees_tuned | default_fp_benign_valid | tuned_fp_benign_valid | n_benign_valid | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 59 | 0.8025 | 0.8046 | 0.8113 | 245 | 159 | 2 | 2 | 101 | 1111.0000 |
| 0 | 1 | 56 | 0.8198 | 0.8208 | 0.8238 | 166 | 292 | 0 | 0 | 103 | 1404.9000 |
| 0 | 2 | 33 | 0.8079 | 0.8087 | 0.8138 | 198 | 689 | 0 | 0 | 103 | 1381.0000 |
| 0 | 3 | 53 | 0.8160 | 0.8165 | 0.8210 | 240 | 140 | 2 | 2 | 102 | 754.6000 |
| 0 | 4 | 44 | 0.8104 | 0.8103 | 0.8115 | 210 | 204 | 1 | 1 | 102 | 1024.4000 |
| 1 | 0 | 25 | 0.8045 | 0.8044 | 0.8107 | 257 | 196 | 1 | 1 | 101 | 991.3000 |
| 1 | 1 | 27 | 0.8239 | 0.8259 | 0.8284 | 232 | 292 | 2 | 2 | 103 | 1034.9000 |
| 1 | 2 | 46 | 0.8107 | 0.8112 | 0.8146 | 436 | 798 | 0 | 0 | 102 | 1600.3000 |
| 1 | 3 | 44 | 0.8135 | 0.8132 | 0.8168 | 238 | 465 | 0 | 1 | 102 | 1269.9000 |
| 1 | 4 | 25 | 0.8079 | 0.8039 | 0.8106 | 173 | 615 | 1 | 2 | 103 | 1275.6000 |
| 2 | 0 | 14 | 0.8097 | 0.8092 | 0.8110 | 196 | 416 | 1 | 0 | 102 | 1357.9000 |
| 2 | 1 | 22 | 0.8186 | 0.8190 | 0.8233 | 243 | 216 | 0 | 0 | 102 | 809.8000 |
| 2 | 2 | 55 | 0.8149 | 0.8128 | 0.8203 | 331 | 395 | 1 | 1 | 103 | 1405.6000 |
| 2 | 3 | 18 | 0.8200 | 0.8202 | 0.8240 | 174 | 332 | 1 | 2 | 102 | 1171.8000 |
| 2 | 4 | 49 | 0.8134 | 0.8133 | 0.8181 | 221 | 259 | 1 | 2 | 102 | 1151.8000 |

`inner_trial0` = trial 0 (defaults with reg_alpha = 1e-3); `inner_default_exact` = exact defaults on the same inner folds.

## Chosen parameters per outer fold (stability)
| index | mean | std | min | max |
|---|---|---|---|---|
| repeat | 1.0000 | 0.8452 | 0.0000 | 2.0000 |
| fold | 2.0000 | 1.4639 | 0.0000 | 4.0000 |
| learning_rate | 0.0421 | 0.0221 | 0.0136 | 0.0832 |
| num_leaves | 42.8000 | 38.0098 | 17.0000 | 174.0000 |
| max_depth | -0.4667 | 2.0656 | -1.0000 | 7.0000 |
| min_child_samples | 43.3333 | 26.1498 | 10.0000 | 99.0000 |
| subsample | 0.7719 | 0.1375 | 0.5340 | 0.9783 |
| subsample_freq | 1.0000 | 0.0000 | 1.0000 | 1.0000 |
| colsample_bytree | 0.7228 | 0.0956 | 0.5835 | 0.9354 |
| reg_lambda | 0.7102 | 1.3115 | 0.0011 | 4.7334 |
| reg_alpha | 0.1286 | 0.2491 | 0.0013 | 0.8322 |
| min_split_gain | 0.2162 | 0.1153 | 0.0137 | 0.4167 |
