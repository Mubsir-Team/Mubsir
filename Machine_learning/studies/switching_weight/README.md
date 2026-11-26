# Research Study: Up-weighting Switching Features in MUBSIR Training

This folder is self-contained. It holds everything the study produced: the protocol, code, scores, results, figures,
logs and integrity hashes.
- No file in any other MUBSIR folder was created or modified (`hashes_comparison.json`).
- External files are only **read**. They are listed with SHA-256 in `INPUTS.md`.
- No attack data was generated, simulated, injected or modified.
- The test set was not used.

**Result:** Outcome 2, up-weighting harms detection.
- Weights 2 and 5 stay within ±2 points of the original.
- Weight 10 lowers detection by 2.8 points (95% CI −4.3 to −1.4).
- The loss is largest for rotating attacks.

See `REPORT.md`.

## Read in this order
| File | What it is |
|---|---|
| `REPORT.md` | Final report |
| `PREREGISTRATION.md` (+ `.sha256`) | Frozen protocol |
| `PLAN.md` | Analysis plan approved by the owner |
| `EXPERIMENT_LOG.md` | Every step, command, seed, gate result and owner decision |
| `INPUTS.md` | External read-only inputs with SHA-256 |

## Code (run order; `run_all.ps1` runs all of it)
| Script | Step | Outputs |
|---|---|---|
| `hash_artifacts.py before/after` | integrity baseline and check | `hashes_*.json` |
| `train.py` | gates G1–G3; arms A0, W2, W5, W10, W0, SW-only; OOF scores; SHAP shares | `scores/*.parquet`, `results/gates.json`, `results/shap_shares.csv`, `results/training_log.csv` |
| `analyze.py` | registered analysis and outcome | `results/analysis_*` |
| `figures.py` | figures | `figures/fig1_dose_response.png`, `figures/fig2_registered_estimands.png` |
| `inputs_manifest.py` | external input hashes | `INPUTS.md`, `results/inputs_manifest.json` |

## Main result files
| File | Content |
|---|---|
| `results/analysis_primary.json` | detection and PR-AUC per arm, primary estimands (3 weights), S1–S4 with Holm, outcome |
| `results/analysis_estimands.csv` | estimands with 95% and 90% CIs |
| `results/shap_shares.csv` | manipulation check: switching share of SHAP and of tree splits per arm |
| `results/analysis_p1_detection_table.csv` | detection per arm by switching level, stealth level and channel count, with CIs |
| `results/analysis_p2_detection_table.csv` | existing matched cases per variant and arm |
| `results/analysis_benign_alarms_vs_habit.csv` | which benign users are alarmed, vs 2010 switching habit |
| `results/analysis_unique_detections.csv`, `analysis_threshold_check.csv`, `analysis_p1_attacker_detection.csv` | descriptives and threshold calibration |
