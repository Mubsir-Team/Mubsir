# Research Study: Controlled Channel Rotation and Independent Switching Signal in MUBSIR

Self-contained study folder. Everything this study produced — protocol, code, generated data, model scores, results,
figures, logs and integrity hashes — is in this folder. Nothing in any other MUBSIR folder was created or modified
(verified by `hashes_comparison.json`). Files outside this folder are only **read**; they are listed with their SHA-256 in
`INPUTS.md`. The test set and test scores were never read.

**Result:** see `REPORT.md`. Short version: explicit switching features add no measurable value (primary Δ = −0.8 points,
95% CI −2.8 to +1.2); by the frozen decision rules the study ends in **Outcome 2**, with the caveat explained in the report.

## Read in this order
| File | What it is |
|---|---|
| `REPORT.md` | Final report: question, design, results, outcome, recommendation, limitations |
| `PREREGISTRATION.md` (+ `.sha256`) | Frozen protocol, written and hashed before any data generation or training |
| `AUDIT.md` | Phase 0 audit: exact feature definitions, why existing data cannot isolate switching |
| `EXPERIMENT_LOG.md` | Every step in order: command, seeds, gate results, implementation notes |
| `INPUTS.md` | External read-only inputs with SHA-256 |

## Code (run order; `run_all.ps1` runs all of it)
| Script | Step | Main outputs |
|---|---|---|
| `hash_artifacts.py before/after` | integrity baseline and check of all pre-existing `Model/` files | `hashes_before.json`, `hashes_after.json`, `hashes_comparison.json` |
| `features.py` | re-implementation of the 100 MUBSIR features from daily counts + baselines (library) | — |
| `gate0.py` | Gate 0: exact feature reproduction (stop rule) | `results/gate0*.{json,csv}` |
| `gen_data.py` | matched counterfactual cases ROT / BLK / FIX / SIM | `data/counterfactual_cases.parquet`, `results/data_*.{json,csv}` |
| `rot_features.py` | new switching features (§5.1) and personal-norm features (§5.2) (library) | — |
| `train_arms.py` | shortcut gate; arms B, C, C+, A-cv, D (regime 1) and C+, D (regime 2) | `scores/*.parquet`, `results/shortcut_gate.json`, `results/training_*` |
| `analyze.py` | registered analysis: primary, S1–S8 + Holm, outcome rules, descriptives, A-frozen | `results/analysis_*` |
| `posthoc_descriptive.py` | post-hoc descriptive decomposition (no tests) | `results/posthoc_arm_differences.csv` |
| `figures.py` | figures | `figures/*.png` |
| `inputs_manifest.py` | hashes of external inputs | `INPUTS.md`, `results/inputs_manifest.json` |

## Folders
- `data/` — generated counterfactual cases (2,252 cases, 236,796 rows; features, labels, raw counts).
- `scores/` — out-of-fold scores of every arm on the original dev rows and on the cases.
- `results/` — every table and JSON the report cites (`analysis_primary.json` holds the outcome).
- `figures/` — `fig1` detection by arm (ROT vs BLK), `fig2` registered estimands, `fig3` detection by variant and length.

## Main result files
| File | Content |
|---|---|
| `results/analysis_primary.json` | primary estimand, S1–S8 with Holm p, outcome conditions, outcome |
| `results/analysis_estimands.csv` | the same estimands as a table |
| `results/analysis_detection_table.csv` | detection per arm × variant × length × FPR × cohort, with CIs and by host type |
| `results/analysis_arm_ladder.csv` | gains of each feature family on ROT vs BLK (descriptive) |
| `results/analysis_latency.csv` | detection latency (workdays) |
| `results/analysis_unique_detections.csv` | cases detected by D but not C+ (and the reverse) |
| `results/analysis_a_frozen.csv` | frozen MUBSIR on the cases at its fixed operating points (descriptive) |
| `results/analysis_threshold_check.csv` | realised held-out benign FPR of every cross-fitted threshold |
| `results/analysis_case_detection.csv` | case-level detection and latency for every arm, repeat and FPR |
| `results/posthoc_arm_differences.csv` | post-hoc per-variant arm differences with CIs |
