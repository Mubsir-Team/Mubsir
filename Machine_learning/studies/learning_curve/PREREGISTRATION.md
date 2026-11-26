# Pre-registration: Learning curve and the cost of reserving the test set

**Status:** FROZEN on 2026-11-26 by the owner ("Freeze as written"). The SHA-256 of this file is in
`PREREGISTRATION.sha256`, and no rule below may change. Every script checks the hash before it runs.

**Date:** 2026-11-26. **Folder:** `Model/studies/learning_curve/` (self-contained, English only).

## 1. Question

MUBSIR v2 reserves 136 of 995 users (13.7%) as a locked test set, so the final model was trained on 859 users.

Does training on 13.7% fewer users measurably lower detection? More generally, how does detection change with the
amount of training data?

If the curve is flat at this size, reserving the test set cost little, and the split is justified by evidence rather
than by convention alone.

## 2. Constraints (owner rules, unchanged)

- **Data:** dev users only (859). The test set is not read: no test feature, label or score.
- **Frozen files:** no frozen file is modified (model, pipeline, config, data, results, other studies).
- **Attack data:** no attack data is generated, simulated, injected or modified. Users are only subsampled.
- **Integrity:** every project file is hashed before and after the study (`hash_artifacts.py`). Excluded from the
  hash, besides this folder, `.venv` and `__pycache__`:
  - `mubsir_system/` and `deliverables/`, which are being built by another session and are not frozen artefacts.
- **Rules:** frozen before training and never changed after results are seen.

## 3. Design

**Folds.** The frozen Stage A folds in `config/cv_folds.csv` are reused: 3 repeats × 5 folds. Each fold has about:
- 588 `fit` users;
- 98 `es` users (early stopping);
- 173 `valid` users.

**What varies.** Only the `fit` users are subsampled. The `es` and `valid` users of every fold are unchanged, so every
arm is evaluated on exactly the same users.

**Fractions of the fit users (arms):**

| Arm | Fraction f | About how many fit users |
|---|---:|---:|
| F30 | 0.30 | 176 |
| F45 | 0.45 | 265 |
| F60 | 0.60 | 353 |
| F75 | 0.75 | 441 |
| **F86** | **0.863** (= 859 / 995) | 507 |
| **F100** | 1.00 | 588 |

F86 / F100 is the same ratio as "dev only" / "dev + test" (859 / 995). It represents the data lost by reserving the test.

**Subsampling (nested and grouped):**
- Fit users are sampled by **schedule group** (`config/user_meta.csv` column `group`; attackers sharing a schedule stay
  together, and each benign user is its own group), **within each stratum** (`stratum`).
- For each (repeat r, fold k, subsample seed s), the groups of each stratum are put in one random order with
  `numpy.random.default_rng(2026 + 1000·s + 10·r + k)`.
- An arm takes whole groups in that order until it has at least `max(1, round(f × n_stratum))` users of the stratum.
  Smaller arms are therefore subsets of larger ones (nested).
- Two subsample seeds (s = 0, 1) are used for every f < 1. F100 has no subsampling.

**Model.** The model is exactly the Stage A configuration: `models._lgbm(SEED + r, models.MAX_TREES, None)` with early
stopping on the fold's `es` users (`models.ES_ROUNDS`). This is the same fit replica used in
`studies/switching_weight/train.py`. Features (100), label (`y_10`) and hyperparameters are unchanged.

**Cost.** 15 folds × (5 fractions × 2 seeds + 1) = 165 LightGBM fits, a few minutes.

## 4. Gates (checked before any estimand is computed)

- **G1, reproduction:** F100 OOF scores equal Stage A OOF (`work/oof/lightgbm_h10.parquet`) exactly, in all 3 repeats.
  If not, the study stops (outcome 4).
- **G2, subsampling:**
  - the realised fit size of every arm is within ±6% of `f × n_fit` per fold, and within ±2% for F86, the primary arm;
  - smaller arms are subsets of larger ones for the same (r, k, s);
  - every stratum keeps at least one user;
  - no user outside the fold's fit users is ever used.

  If not, the study stops.
- **G3, integrity:** `pipeline/*.py`, `config/cv_folds.csv`, `config/user_meta.csv`, `config/selection_rules.json`,
  `data/model_dataset_995.parquet` and `work/labels.parquet` are identical to `hashes_before.json`. If not, the study
  stops.

## 5. Evaluation (same definitions as the closed studies)

**Thresholds.**
- For arm a, repeat r, seed s and fold k: `metrics.threshold_at_fpr(peaks, f)`.
- The peaks are the whole-period peak OOF scores of the benign dev users **not** in fold k, from the same arm,
  repeat and seed.
- f = 0.03 is primary; f = 0.01 is secondary.

**Detection.**
- An attacker is detected if at least one of its rows with `y_10 = 1` scores above the threshold of its validation fold.
- Det = share of the 348 dev attackers detected. It is averaged over the 3 repeats, and over the 2 subsample seeds
  for f < 1.

**Bootstrap.**
- Thresholds and per-user detection flags are computed once.
- The bootstrap then resamples users: 2,000 resamples with `numpy.random.default_rng(2026)`, stratified by attacker
  vs benign.
- Estimands are recomputed in each resample. Arms are paired, because they share the same users.
- CIs: the 95% CI is the 2.5–97.5 percentile range, and the 90% CI is the 5–95 percentile range.
- p = min(1, 2 · min(share ≤ 0, share ≥ 0)).

## 6. Estimands

**Primary (iso-FPR 3%):** **Δ_cost = Det(F100) − Det(F86)**. This is the detection lost by training on 13.7% fewer
users.

**Secondary (Holm–Bonferroni across S1–S3, α = 0.05):**

| ID | Estimand |
|---|---|
| S1 | Δ_cost at iso-FPR 1% |
| S2 | row-level PR-AUC(F100) − PR-AUC(F86) on the OOF rows of dev users. PR-AUC is computed per repeat (and seed) and averaged; the bootstrap weights rows by the resampled user counts |
| S3 | Det(F100) − Det(F60) at iso-FPR 3%. This checks whether the amount of data matters at all in this range |

**Descriptive (no tests):**
- The full curve, F30 → F100: Det (3% and 1%), row PR-AUC and user AUC against the number of fit users.
- The curve for the weakest groups: 5-day attacks, High stealth, and USB-only or HTTP-only channel use where available.
- The realised benign FPR per arm.
- **Extrapolation (labelled as such):**
  - fit Det(n) = a − b·n^(−c) to the six arm means;
  - report the predicted difference between the final model's training size and that size plus the test users.
  - This is a model-based extrapolation, not a measurement.

## 7. Decision rules (applied in this order)

| Outcome | Condition |
|---|---|
| **4: inconclusive (gate)** | G1, G2 or G3 fails |
| **2: reserving the test has a meaningful cost** | the 95% CI of Δ_cost lies entirely above 0 **and** Δ_cost ≥ +0.02 |
| **1: reserving the test costs little** | the 90% CI of Δ_cost lies within (−0.02, +0.02): TOST equivalence, α = 0.05 per side |
| **3: inconclusive** | none of the above |

- A 2-point change in detection is the smallest one considered practically meaningful, as in the earlier studies.
- Significant changes smaller than 2 points are reported, but they do not change the outcome.

## 8. Known limitations (stated before results)

- **Scale.** The curve is measured at cross-validation scale: up to 588 fit users. The final model was trained on all
  859 dev users. Learning curves are usually concave, so the cost at the final scale is expected to be no larger than
  Δ_cost. This is an assumption, and it is not tested.
- **Fixed early-stopping set.** The `es` set is not subsampled. Every arm therefore keeps the same early-stopping
  signal; only the data the trees learn from shrinks.
- **Simulated attacks.** All attacks are simulated. The curve describes this dataset, not real-world data needs.
- **No change to the model.** Whatever the outcome, the frozen model and split are not changed. The result only
  documents how the split affects detection.

## 9. Outputs (all inside this folder)

| Path | Content |
|---|---|
| `train.py` | gates, the 6 arms, OOF scores |
| `analyze.py` | estimands, outcome, curve |
| `figures.py` | learning-curve figure |
| `scores/`, `results/`, `figures/` | outputs |
| `REPORT.md`, `README.md`, `EXPERIMENT_LOG.md`, `INPUTS.md` | documentation |
| `hashes_before.json`, `hashes_after.json`, `hashes_comparison.json` | integrity |
