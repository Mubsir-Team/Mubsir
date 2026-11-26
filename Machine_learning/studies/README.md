# MUBSIR research studies

Five studies were run after the model was frozen. Each folder is self-contained, holding:
- the protocol (`PREREGISTRATION.md`, where the study was pre-registered);
- the code, scores, results and figures;
- a final `REPORT.md`;
- integrity hashes of every frozen file, taken before and after the study (`hashes_comparison.json`: 0 changed in all five).

None of the studies used the test set. None modified the frozen model, the pipeline or the data. The switching studies did not generate, inject or modify any attack data.

| Folder | Date | Question | Answer |
|---|---|---|---|
| `user_risk/` | 2026-11-25 | How should daily risk be aggregated into one alert per employee? | Keep `risk_peak > 85`. No aggregation beats max at iso-FPR, and benign alarms grow with observation length. |
| `rotation/` | 2026-11-26 | Does MUBSIR beat siloed per-channel monitors when attackers rotate channels? | Partial: +2.9 recall points at 3% FPR, and +8.1 points for high-switching attackers. The gain comes from modelling all channels together. |
| `controlled_rotation/` | 2026-11-26 | Do new explicit switching features add an independent signal? | No: Δ −0.8 points [−2.8, +1.2]. MUBSIR detects rotation through aggregate deviation. |
| `switching_baseline/` | 2026-11-26 | Does switching measured against the employee's own 2010 habit help? | No meaningful added value (Outcome 3): Δ −0.5 points [−1.3, +0.4]. |
| `switching_weight/` | 2026-11-26 | Does up-weighting the switching features in training help? | No (Outcome 2). At weight 10, detection falls by 2.8 points [−4.3, −1.4], and the loss is largest for rotating attacks. |

Read-only links between studies (each listed with SHA-256 in that study's `INPUTS.md`):
- `controlled_rotation` reads `rotation/results/feature_groups.csv`.
- `switching_baseline` and `switching_weight` read `controlled_rotation/data/counterfactual_cases.parquet`.
- `switching_weight` reads `switching_baseline/results/habit_2010_dev_users.csv`.

**Folder names:** until 2026-11-26 these folders were `Model/study_<name>/`. Text inside the reports, logs, `INPUTS.md` and the hash files keeps the old names, because they are records of the runs. See `../RESTRUCTURE.md`.

**Running a study:** each folder with a `run_all.ps1` reproduces its study. Run it from inside that folder. It uses `..\..\.venv`.
