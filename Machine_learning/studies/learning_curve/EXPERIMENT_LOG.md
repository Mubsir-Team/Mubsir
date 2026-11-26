# Experiment log: learning curve study

## 2026-11-26: Request and draft

**Owner request.** After the split-justification analysis, the owner asked for a pre-registered learning curve
("Yes, preregister it"). The question: what detection is lost by reserving 13.7% of the users as a locked test set?

**Draft.** `PREREGISTRATION.md` was written as a DRAFT. No model has been trained.

**Integrity baseline.** `hash_artifacts.py before` hashed 561 files. Excluded from the hash: this folder, `.venv`,
`__pycache__`, `mubsir_system/` and `deliverables/`. The last two are being built by another session and are not
frozen artefacts.

**Feasibility check, before the freeze.** This check counted users only; no training and no scores were used.
- The first draft rule took groups up to `ceil(f × n_stratum)` users per stratum. With 21 strata per fit set (sizes 4 to
  184), it overshot the target by 6.4–11.3% at f = 0.30. That would fail the planned G2 tolerance of ±3%.
- With `max(1, round(f × n_stratum))`, the realised size versus the target (min, max over 15 folds × 2 seeds) is:

  | f | min | max |
  |---|---:|---:|
  | 0.30 | −1.4% | +5.6% |
  | 0.45 | −0.2% | +4.9% |
  | 0.60 | −0.2% | +3.0% |
  | 0.75 | −0.4% | +1.4% |
  | 0.863 | −0.5% | +0.7% |

- The draft was changed **before the freeze**:
  - the rule is now `round`;
  - G2 tolerance is ±6% for all arms and ±2% for F86, the primary arm.

**Next step.** Owner review. On approval: freeze (`PREREGISTRATION.sha256`), then gates and training.

## 2026-11-26: Freeze
- Owner: "Freeze as written".
- The status line was set to FROZEN, then the SHA-256 was written to `PREREGISTRATION.sha256`: `5be9e0289975bcc33025dd7abc430b153ac7506fb2a382e4b97eec8c98a96d45`.
- No model had been trained before the freeze.

## 2026-11-26: Training script and dry run
- `train.py` was written. It uses the Stage A fit replica, and only the fold's `fit` users are subsampled.
- `train.py --dry-run` checks G3 and G2 without fitting any model:
  - **G3 PASS:** all pipeline files, `cv_folds.csv`, `user_meta.csv`, `selection_rules.json`, the dataset and the
    labels are identical to `hashes_before.json`.
  - **G2 PASS:**
    - realised size versus target is −1.4% to +5.6% for all arms (tolerance ±6%);
    - F86 is −0.5% to +0.7% (tolerance ±2%);
    - the subsamples are nested;
    - every stratum is kept;
    - no user comes from outside the fit set.
- Output: `results/subsample_sizes.csv`. No model was fitted, and no `__pycache__` was created.
- **Next step:** owner go-ahead to train (165 fits), then G1.

## 2026-11-26: Gates and training
- Owner: "Run gates and train".
- Command: `train.py` (stdout in `results/train_stdout.log`). 165 fits in 170 s, exit 0.
- **G3 PASS** (integrity) and **G2 PASS** (subsampling), with the same values as the dry run.
- **G1 PASS:** F100 OOF equals Stage A OOF exactly in all 3 repeats (max |diff| 0.00e+00).
- Mean fit users and trees per arm (training log only, not an estimand):

  | Arm | F30 | F45 | F60 | F75 | F86 | F100 |
  |---|---:|---:|---:|---:|---:|---:|
  | Fit users | 178.9 | 267.5 | 356.2 | 442.6 | 507.8 | 587.5 |
  | Trees | 154.4 | 182.4 | 187.6 | 200.2 | 215.2 | 237.3 |

- Outputs: 11 score files in `scores/`, plus `results/gates.json` and `results/training_log.csv`. No `__pycache__` was
  created.
- No estimand has been computed. **Next step:** owner go-ahead for the registered analysis.

## 2026-11-26: Registered analysis
- Owner: "Run the analysis".
- `analyze.py` (stdout in `results/analyze_stdout.log`):
  - **Primary:** Det(F100) − Det(F86) @3% = +0.0057; 95% CI [−0.0005, +0.0125]; 90% CI [+0.0005, +0.0115]; p = 0.093.
  - **S1:** @1% = +0.0120 [+0.0053, +0.0196], Holm p < 0.001.
  - **S2:** row PR-AUC = +0.0031 [+0.0023, +0.0039], Holm p < 0.001.
  - **S3:** F100 − F60 @3% = +0.0172 [+0.0077, +0.0278], Holm p < 0.001.
  - **OUTCOME 1** (reserving the test costs little): the 90% CI of the primary estimand lies within ±0.02. S1 is
    significant but below 2 points, so it is reported and does not change the outcome.
  - Benign attackers have no detection flag, so `np.nanmean` warns once about benign columns. This is harmless and
    expected.
- `figures.py` drew `fig1_learning_curve.png`. The first draft had overlapping labels; they were offset and the
  figure redrawn. `inputs_manifest.py` lists 14 inputs.
- `REPORT.md`, `README.md` and `run_all.ps1` were written.
- `hash_artifacts.py after`: 561 files compared; 0 changed, 0 removed, 0 added outside the study; all key artefacts
  unchanged.
