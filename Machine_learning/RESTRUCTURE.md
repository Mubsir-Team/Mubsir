# v2 folder restructure (2026-11-26)

The `Model/` folder was reorganised into core files, studies and a local archive, to prepare it as the
**MUBSIR Model v2** repository. **No file was deleted, and no result was re-computed.**

## What moved

| Before | After |
|---|---|
| `study_user_risk/` | `studies/user_risk/` |
| `study_rotation/` | `studies/rotation/` |
| `study_controlled_rotation/` | `studies/controlled_rotation/` |
| `study_switching_baseline/` | `studies/switching_baseline/` |
| `study_switching_weight/` | `studies/switching_weight/` |
| `PROJECT_HANDOVER.md`, `audit_summary.md`, `images/`, `exports/` | `archive/` (local only, in `.gitignore`) |

The following did not move: `pipeline/`, `model/`, `config/`, `data/`, `results/`, `reports/`, `work/`, `audit/`,
`prepare_dataset.py`, `requirements.txt`, `mubsir_system/` and `deliverables/`.
The last two are excluded from the repository until the production-system step is finished.

## What changed inside files

Only path lines in 25 study scripts changed, so that they still find `Model/` from one level deeper:

- `MODEL_DIR = HERE.parent` became `MODEL_DIR = HERE.parent.parent` (19 scripts).
- `HERE.parent / "pipeline"` became `HERE.parent.parent / "pipeline"` (3 scripts).
- Paths to sibling studies use the new names: `studies/controlled_rotation`, `studies/switching_baseline` and `rotation`.
- The `hash_artifacts.py` skip lists use the new folder names.
- In `inputs_manifest.py`, the input paths use the new folder names.
- `run_all.ps1` (3 studies) now uses `..\..\.venv\Scripts\python.exe`.
- `switching_weight/train.py` gate G3 checks the new path. It falls back to the pre-move key in its `hashes_before.json`.

Changed files:
- `studies/controlled_rotation/`: `analyze.py`, `gate0.py`, `gen_data.py`, `hash_artifacts.py`, `inputs_manifest.py`,
  `run_all.ps1`, `train_arms.py`
- `studies/rotation/`: `hash_artifacts.py`, `run_experiment.py`
- `studies/switching_baseline/`: `analyze.py`, `audit_2010.py`, `gates.py`, `hash_artifacts.py`, `inputs_manifest.py`,
  `population_check.py`, `run_all.ps1`, `train.py`
- `studies/switching_weight/`: `analyze.py`, `hash_artifacts.py`, `inputs_manifest.py`, `posthoc_switching_only.py`,
  `run_all.ps1`, `train.py`
- `studies/user_risk/`: `hash_artifacts.py`, `run_study.py`

After the check below, a dated "Folder move" note was appended to the `EXPERIMENT_LOG.md` of `controlled_rotation`,
`switching_baseline` and `switching_weight`.

New files: `README.md`, `.gitignore`, `RESTRUCTURE.md`, `studies/README.md`, `archive/README.md`.

Some records still use the old folder names, because they describe the runs as they happened: reports, logs,
`INPUTS.md`, `hashes_*.json` and docstrings.

## Verification

- Every file was SHA-256 hashed before the move: 595 files, excluding `.venv` and `__pycache__`.
- After the move, 570 files were byte-identical at their new paths. The 25 that changed are exactly the files listed above.
- No file is missing. The only new files are the 4 documents listed above; this record was written after the check.
- Frozen model: `model/verify_bundle.py` gives 8/8 OK. The 5 hashes in `results/pretest/freeze.json` all match.
- Every path that a study script reads outside its own folder resolves to an existing file: 147 paths, 0 missing.
- The frozen pipeline imports correctly from inside a study folder.
- No `__pycache__` was created.
