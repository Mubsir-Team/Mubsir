# Reproduces the controlled rotation study end to end, in the registered order. Run from this folder:
#   powershell -ExecutionPolicy Bypass -File run_all.ps1
# Every script checks the PREREGISTRATION.md hash first. Nothing outside study_controlled_rotation/ is written.
# Byte-code writing is disabled so importing the frozen pipeline/model modules creates no __pycache__ there.
$ErrorActionPreference = "Stop"
$env:PYTHONDONTWRITEBYTECODE = "1"
$py = "..\..\.venv\Scripts\python.exe"
Set-Location $PSScriptRoot

& $py hash_artifacts.py before      # integrity baseline of all pre-existing Model/ files
& $py gate0.py                      # Gate 0: exact re-implementation of the 100 features (stop rule)
& $py gen_data.py                   # counterfactual cases (H146 rng 42, R253 rng 43, S8 rng 44)
& $py train_arms.py                 # shortcut gate + regime 1 (B, C, C+, A-cv, D) + regime 2 (C+, D)
& $py analyze.py                    # registered analysis: primary, S1-S8 with Holm, outcome, descriptives
& $py posthoc_descriptive.py        # POST-HOC descriptive decomposition (no tests)
& $py figures.py                    # figures/*.png
& $py inputs_manifest.py            # INPUTS.md: every external read-only input with SHA-256
& $py hash_artifacts.py after       # integrity check: hashes_comparison.json
