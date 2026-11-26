# Research Study: Learning Curve and the Cost of Reserving the Test Set

This folder is self-contained. It holds the protocol, code, scores, results, figure, logs and integrity hashes.
- Dev users only. The test set was not used.
- No attack data was generated, simulated, injected or modified.
- No frozen file was changed (`hashes_comparison.json`).
- External files are only read; they are listed with SHA-256 in `INPUTS.md`.

**Result: Outcome 1, reserving the test costs little.**
- Training on 86% instead of 100% of the users (the same ratio as 859 / 995) lowers detection by **+0.57 points** at
  iso-FPR 3% (90% CI +0.05 to +1.15).
- That is inside the registered ±2-point equivalence band.
- More data still helps slowly: +1.7 points from 60% to 100%.

See `REPORT.md`.

## Read in this order
| File | What it is |
|---|---|
| `REPORT.md` | Final report |
| `PREREGISTRATION.md` (+ `.sha256`) | Frozen protocol |
| `EXPERIMENT_LOG.md` | Every step, gate result and owner decision |
| `INPUTS.md` | External read-only inputs with SHA-256 |

## Code (run order; `run_all.ps1` runs all of it)
| Script | Step | Outputs |
|---|---|---|
| `hash_artifacts.py before/after` | integrity baseline and check | `hashes_*.json` |
| `train.py` (`--dry-run` = gates G3 + G2 only) | gates; 6 arms × 3 repeats × 5 folds (× 2 subsample seeds) | `scores/*.parquet`, `results/gates.json`, `results/subsample_sizes.csv`, `results/training_log.csv` |
| `analyze.py` | registered estimands and outcome, curve, subgroups, extrapolation | `results/analysis_*` |
| `figures.py` | figure | `figures/fig1_learning_curve.png` |
| `inputs_manifest.py` | external input hashes | `INPUTS.md`, `results/inputs_manifest.json` |
