# Reproduces the switching-weight study in the registered order. Run from this folder:
#   powershell -ExecutionPolicy Bypass -File run_all.ps1
# Every script checks the PREREGISTRATION.md hash. Nothing outside study_switching_weight/ is written.
# No attack data is generated, injected or modified. Byte-code writing is disabled so no __pycache__ appears in frozen folders.
# Note: re-running 'hash_artifacts.py before' overwrites the integrity baseline with the then-current state.
$ErrorActionPreference = "Stop"
$env:PYTHONDONTWRITEBYTECODE = "1"
$py = "..\..\.venv\Scripts\python.exe"
Set-Location $PSScriptRoot

& $py hash_artifacts.py before      # integrity baseline (all pre-existing Model/ files, closed studies included)
# --- PREREGISTRATION.md frozen here (sha256 in PREREGISTRATION.sha256) ---
& $py train.py                      # gates G1-G3 + six arms (A0, W2, W5, W10, W0, SW-only), OOF scores, SHAP shares
& $py analyze.py                    # registered analysis: primary (3 weights), S1-S4 with Holm, outcome, descriptives
& $py figures.py                    # figures/*.png
& $py inputs_manifest.py            # INPUTS.md
& $py hash_artifacts.py after       # integrity check
