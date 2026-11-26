# Research Study: Habit-Relative Channel Switching (2010 Switching Baseline) in MUBSIR

Self-contained study folder. All of this study's protocol, code, derived features, scores, results, figures, logs and
integrity hashes are here. No file in any other MUBSIR folder was created or modified (`hashes_comparison.json`).
External files are only **read**; they are listed with SHA-256 in `INPUTS.md`.

No attack data was generated, simulated, injected or modified. The test set was not used.

**Result:** Outcome 3, no meaningful added value. Habit-relative switching changes detection by −0.5 points
(95% CI −1.3 to +0.4). See `REPORT.md`.

## Read in this order
| File | What it is |
|---|---|
| `REPORT.md` | Final report |
| `PREREGISTRATION.md` (+ `.sha256`) | Frozen protocol (written after the audit and the owner's approval of `PLAN.md`) |
| `PLAN.md` | Revised analysis plan reviewed by the owner (no new attack data) |
| `AUDIT.md` | Phase 0 audit of the 2010 switching habit (§6–7 superseded by PLAN.md) |
| `EXPERIMENT_LOG.md` | Every step, command, seed, gate result and owner decision, in order |
| `INPUTS.md` | External read-only inputs with SHA-256 |

## Code (run order; `run_all.ps1` runs all of it)
| Script | Step | Outputs |
|---|---|---|
| `hash_artifacts.py before/after` | integrity baseline and check (closed studies included) | `hashes_*.json` |
| `audit_2010.py` | Phase 0 audit (read-only) | `results/audit_*` |
| `population_check.py` | 2010 habit per dev user, eligibility, shortcut screen | `results/habit_2010_dev_users.csv` (frozen), `results/population_check.json` |
| `rot_features.py` | linked-pair and change definitions (byte-identical copy from the closed study) | — |
| `hab_features.py` | the 8 R and H features (library) | — |
| `gates.py` | gates G1–G4, features for P1 and P2 | `results/gates.json`, `results/gate3_folds.csv`, `data/features_p*.parquet` |
| `train.py` | arms A, R, H, OOF scores | `scores/*.parquet`, `results/training_*` |
| `analyze.py` | registered analysis and outcome | `results/analysis_*` |
| `figures.py` | figures | `figures/*.png` |
| `inputs_manifest.py` | external input hashes | `INPUTS.md`, `results/inputs_manifest.json` |

## Main result files
| File | Content |
|---|---|
| `results/analysis_primary.json` | detection per arm, primary, key secondary, S1–S7 with Holm, outcome |
| `results/analysis_estimands.csv` | the estimands as a table |
| `results/analysis_p1_detection_table.csv` | detection per arm by habit tertile, switching level, channel count and stealth, with CIs |
| `results/analysis_p2_detection_table.csv` | existing matched cases per variant and habit tertile |
| `results/analysis_p1_attacker_detection.csv` | per-attacker detection (mean over repeats) |
| `results/analysis_latency.csv`, `analysis_unique_detections.csv`, `analysis_threshold_check.csv` | descriptives and threshold calibration |
