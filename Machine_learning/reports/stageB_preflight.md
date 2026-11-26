# Stage B — pre-flight verification (before any Optuna trial)

Script: `audit/stageB_preflight.py`. Outputs: `results/preflight_stageB/`.
No Optuna, calibration or test scoring was run. The only computation was in-memory refits of the Stage A default folds; nothing in `work/`, `config/` or the pipeline was written.

**Result: 18 PASS, 1 MISMATCH (V3).**

## V1 — Default baseline never trained on its outer-validation users
| Check | Result |
|---|---|
| fit / es / valid users disjoint, all 15 outer folds | PASS (0 overlaps) |
| fit / es / valid schedule groups disjoint | PASS (0) |
| fit + es + valid = all 859 dev users per outer fold | PASS |
| In-memory refit of all 15 folds reproduces the cached default OOF (`work/oof/lightgbm_h10.parquet`) | PASS, max abs diff **0.0** |
| Refit tree counts = Stage A best iterations | PASS |
| Label-poison test: validation labels randomised before refit | PASS, predictions change by **0.0** |
| Every dev row scored exactly once per repeat, by its own outer fold | PASS (268,413 = 3 × 89,471) |

## V2 — Early stopping uses training-side es users only
| Check | Result |
|---|---|
| `models.fit_predict` passes only `eval_set=[(X_es, y_es)]` | PASS |
| Booster `evals_result_` has a single set (`valid_0` = es); best iteration reproduced in all 15 folds | PASS |
| es users disjoint from outer-validation users, also by schedule group | PASS |

**For the tuned arm (not yet implemented):**
- The registered design uses the outer es users for the outer fit, and inner es users (fold 0 of SGKF(7) inside each inner-train set) for Optuna's inner fits.
- Planned inner splits were generated (splitting only, no training): no group overlap between inner fit / es / valid; es share 14.2–14.9%.
- The implementation will assert both conditions before every fit.

## V3 — User-recall threshold: **MISMATCH**
- The registered `frozen_lfo_user_recall_1pct` (adoption condition 3) takes the threshold for outer fold k from the OOF peak scores of benign users in the other four folds (408–410 users). The **users** are training-side: PASS.
- But each of those scores was produced by a sibling model j ≠ k, and **every one of those sibling models was trained on fold-k users** (`V3_lfo_threshold_dependencies.csv`: 4 of 4 in all 15 folds).
- So the threshold depends indirectly on the outer-validation users through the sibling models' parameters. It is also on another model's score scale.
- It is therefore **not** derived exclusively from training-side information.
- This affects both arms identically and was also used in the "frozen leave-fold-out" latency of audit part 2, where it gave the same numbers as the in-sample threshold.

**Proposed correction (amendment 2, needs approval):** for outer fold k, take the threshold from **inner out-of-fold scores of the outer-train users only**:
- The 5 inner folds are the same planned inner splits.
- The models are trained only on outer-train users, with the arm's own configuration:
  - Tuned arm: the inner OOF of the best Optuna trial, which Optuna produces anyway.
  - Default arm: 5 inner default fits per outer fold, 75 LightGBM fits in total.
- Threshold = 1% benign-user FPR on those inner peaks (408–410 benign users, k = 4).
- The threshold is frozen, then applied to the outer model's scores on fold k.

Remaining caveat (documented, not leakage): inner models are trained on about 80% of the outer-train users, so their score scale can differ slightly from the outer model's. The realised FP count on each outer fold will be reported next to the recall.

## V4 — Same objective calculation for default and tuned
| Check | Result |
|---|---|
| Two independent duration-macro implementations (sklearn AP vs explicit step sum) agree on the default OOF | PASS (1.1e-16) |
| Equal to the audit-2 values on which the objective decision was based | PASS (0.0) |
| Duration groups taken from `user_meta.window_workdays`; not a predictor; values {5,10,20,30,40,50} | PASS |
| All 6 durations present in every outer-validation fold and every planned inner-validation fold | PASS |

**Procedure parity, to be enforced in code:**
- One function computes the duration-macro row PR-AUC for (a) Optuna's inner objective, (b) the tuned arm and (c) the default arm.
- Both arms use the same outer folds, seeds (42 + repeat), fit/es users, pooled-per-repeat aggregation and threshold procedure.

Minor: LightGBM 4.7 emits a deprecation warning for the `eval_set` argument. It has no effect on results (the refit reproduced the scores exactly).
