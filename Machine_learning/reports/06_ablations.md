# 06 — Stage B ablations (LightGBM defaults, y_10, nested 15 outer folds)

Rule: `stage_B.ablation_adoption` — each variant vs the adopted configuration (LightGBM defaults, nested results from 05):
primary (duration-macro row PR-AUC) >= +0.005, row PR-AUC >= −0.002, frozen inner-OOF user recall >= default.
Dev users only; test untouched.

| variant | primary | d_primary | row_pr_auc | d_row_pr_auc | user_recall | d_user_recall | fp_users_per_repeat | row_recall_1pct | conditions_met | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| default | 0.8209 | 0.0000 | 0.9439 | 0.0000 | 0.8554 | 0.0000 | 4.3333 | 0.8634 |  |  |
| no_nr | 0.8214 | 0.0005 | 0.9440 | 0.0001 | 0.8649 | 0.0096 | 5.3333 | 0.8641 | 2/3 (n/y/y) | not adopted |
| spw | 0.8115 | -0.0093 | 0.9418 | -0.0021 | 0.8467 | -0.0086 | 4.3333 | 0.8611 | 0/3 (n/n/n) | not adopted |
| c1_doc | 0.8377 | 0.0169 | 0.9520 | 0.0081 | 0.8898 | 0.0345 | 5.0000 | 0.8818 | 3/3 (y/y/y) | documentation only |

Conditions order: primary / row PR-AUC / user recall.

## Row PR-AUC per attack duration (mean of 3 repeats)
| variant | ap_d5 | ap_d10 | ap_d20 | ap_d30 | ap_d40 | ap_d50 |
|---|---|---|---|---|---|---|
| default | 0.4991 | 0.7535 | 0.8592 | 0.9116 | 0.9521 | 0.9497 |
| no_nr | 0.5038 | 0.7498 | 0.8596 | 0.9113 | 0.9527 | 0.9511 |
| spw | 0.4711 | 0.7378 | 0.8525 | 0.9085 | 0.9511 | 0.9480 |
| c1_doc | 0.5211 | 0.7843 | 0.8782 | 0.9240 | 0.9606 | 0.9583 |

- `no_nr`: drops the 24 rotation features (`nr_*`).
- `spw`: scale_pos_weight = negatives/positives of each training set (no other change).
- `c1_doc`: adds 11 non-static C1 companion columns that the injection did not update; documents the size of the
  simulation artefact, never adopted.
