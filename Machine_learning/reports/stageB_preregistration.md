# Stage B — pre-registration record

Date: 2026-11-24. Approved by the project owner in chat before any Stage B computation.

`config/selection_rules.json` SHA-256 at registration:
`977e93a43527bc0660ba238d19a005520f1e30cfc4ade30c5f0b568edb4f3fb3`

State at registration: Stage A finished (LightGBM, y_10). No Optuna trial, calibration, ablation, final fit or test score existed (`work/` and `results/` held only Stage A outputs and the two audits).

## What was registered (`stage_B` section)
- **Optuna objective:** duration-macro row PR-AUC on y_10, i.e. the mean of the average precision for each of the 6 attack-duration groups (each group = its attackers + all benign users). It is computed once on pooled inner OOF over 5 schedule-grouped inner folds of the dev users. TPE seed 42, 60 trials, default configuration as trial 0.
- **Adoption (15 outer folds):** all three conditions must hold:
  - duration-macro row PR-AUC ≥ default + 0.005;
  - row PR-AUC ≥ default − 0.002;
  - frozen leave-fold-out user recall at 1% benign-user FPR ≥ default.
- **Calibration:** Platt scaling on the stacked OOF scores of the 3 repeats, row level, target y_10, with a and b also reported per repeat.
- **Test protocol:** test opened once after the final fit; all thresholds frozen from dev; FP counts out of 84 benign test users with exact CIs.

The original `later_stages` text is kept under `later_stages_superseded`. Rationale: `reports/audit_stageA.md`, `reports/audit_stageA2.md`.

The SHA-256 above is checked again before Optuna starts; any change to the rules file after this point must be recorded here with a new hash.

---

## Amendment 1 — nested CV (2026-11-24, approved by the project owner before any Optuna trial)

New `config/selection_rules.json` SHA-256 (this is the one checked before Optuna starts):
`b93ae393d388a6672ef779a7c0307a863e8f83d94cb179b8e0184c3dffb0c803`

State at amendment: unchanged. No Optuna trial, calibration, ablation, final fit or test score existed (`work/` held only Stage A OOF, labels and evaluation metadata).

**Problem fixed.** In the first registration, Optuna's inner CV and the 15-fold adoption comparison used the same 859 dev users. Every outer-validation user had also been an Optuna validation user, so the comparison would have favoured the tuned configuration through selection on the same data.

**New design (`stage_B.nested_cv`):**
- For each of the 15 outer folds, Optuna runs on the outer-train users only. Inner CV is 5 schedule-grouped folds, with fold-specific seeds; the objective, search space, 60 trials and trial 0 (defaults) are unchanged.
- The winning configuration is trained on the outer fit users (early stopping on the outer es users) and scores the outer-validation users, whom Optuna never saw.
- The default arm is the Stage A LightGBM y_10 OOF on the same folds. It is recomputed and must match.
- The adoption rule (three conditions) is unchanged. It is now applied to the nested estimate of the tuning procedure.
- If tuning is adopted, Optuna runs once on all dev users to produce the final parameters. Their reported expected performance is the nested estimate, never their own tuning score.
- Calibration data for the tuned case: the final parameters re-run on the 15 outer folds, with the mild optimism documented.

---

## Amendment 2 — threshold for user recall (2026-11-24, approved by the project owner before any Optuna trial)

New `config/selection_rules.json` SHA-256 (supersedes the amendment-1 hash; this is the one checked before Optuna starts):
`41de4354e861fb08f2018d76ee1d35fe45cbd3b3b4af3a37a96fba70fdf5deb0`

State at amendment: unchanged. No Optuna trial, calibration, ablation, final fit or test score existed.

**Problem fixed** (pre-flight check V3, `reports/stageB_preflight.md`): the leave-fold-out threshold for outer fold k used OOF peaks of benign users in the other folds. Those scores came from sibling models that had all been trained on fold-k users, so the threshold depended indirectly on the outer-validation users.

**New definition (`definitions.frozen_inner_user_recall_1pct`):**
- For each outer fold and each arm, the arm's configuration is trained on the 5 inner folds of the outer-train users only.
- The threshold is the 1% benign-user FPR point of those inner out-of-fold peaks (408–410 benign users, k = 4). It is frozen, then applied to the same arm's outer model on the outer-validation users.
- The default arm needs 75 extra inner fits; the tuned arm reuses the inner OOF of its best trial.
- The realised FP count per outer fold is reported next to the recall.
- Adoption condition 3 now uses this definition. The old definition is kept as `frozen_lfo_user_recall_1pct_superseded`.
