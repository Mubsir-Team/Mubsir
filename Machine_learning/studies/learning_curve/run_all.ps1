# Reproduces the learning-curve study in the registered order. Run from this folder:
#   powershell -ExecutionPolicy Bypass -File run_all.ps1
# Every script checks the PREREGISTRATION.md hash. Nothing outside studies/learning_curve/ is written.
# Dev users only; no attack data is generated, injected or modified. Byte-code writing is disabled.
# Note: re-running 'hash_artifacts.py before' overwrites the integrity baseline with the then-current state.
$ErrorActionPreference = "Stop"
$env:PYTHONDONTWRITEBYTECODE = "1"
$py = "..\..\.venv\Scripts\python.exe"
Set-Location $PSScriptRoot

& $py hash_artifacts.py before      # integrity baseline
# --- PREREGISTRATION.md frozen here (sha256 in PREREGISTRATION.sha256) ---
& $py train.py                      # gates G3, G2, then 165 fits, then G1 (F100 == Stage A OOF)
& $py analyze.py                    # registered analysis: primary, S1-S3 with Holm, outcome, descriptives
& $py figures.py                    # figures/fig1_learning_curve.png
& $py inputs_manifest.py            # INPUTS.md
& $py hash_artifacts.py after       # integrity check
