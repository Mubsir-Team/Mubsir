# Reproduces the switching-baseline study in the registered order. Run from this folder:
#   powershell -ExecutionPolicy Bypass -File run_all.ps1
# Every script after the freeze checks the PREREGISTRATION.md hash. Nothing outside study_switching_baseline/ is written.
# No attack data is generated, injected or modified. Byte-code writing is disabled so no __pycache__ appears in frozen folders.
# Note: re-running 'hash_artifacts.py before' overwrites the integrity baseline with the then-current state.
$ErrorActionPreference = "Stop"
$env:PYTHONDONTWRITEBYTECODE = "1"
$py = "..\..\.venv\Scripts\python.exe"
Set-Location $PSScriptRoot

& $py hash_artifacts.py before      # integrity baseline (all pre-existing Model/ files, closed studies included)
& $py audit_2010.py                 # Phase 0 audit (read-only)
& $py population_check.py           # habit per dev user, eligibility (frozen file results/habit_2010_dev_users.csv)
# --- PREREGISTRATION.md frozen here (sha256 in PREREGISTRATION.sha256) ---
& $py gates.py                      # gates G1-G4 + features for P1 (dev rows) and P2 (existing cases)
& $py train.py                      # arms A, R, H on the 714 habit-eligible users, existing folds
& $py analyze.py                    # registered analysis: primary, key secondary, S1-S7 with Holm, outcome
& $py figures.py                    # figures/*.png
& $py inputs_manifest.py            # INPUTS.md
& $py hash_artifacts.py after       # integrity check
