# MUBSIR Model v2

MUBSIR is a UEBA model for low-and-slow, multi-channel data exfiltration (email, HTTP upload, USB).
It is one LightGBM model that gives each eligible employee one calibrated daily risk score from 0 to 100.
The score is the probability that the employee had at least one attack day in the last 10 workdays (label `y_10`).
The model uses 100 behavioural features, computed against each employee's own 2010 baseline.

v2 is trained on the New_Attack_Simulation dataset: 995 eligible employees, of whom 400 are simulated attackers.
The model is frozen in `model/`.

## Result on the locked test set (opened once, 2026-11-24)

The test set has 136 employees: 52 attackers and 84 benign.

| Metric | Value |
|---|---|
| Row PR-AUC (`y_10`) | 0.955 |
| User ROC-AUC (peak risk) | 0.989 |
| Alert rule `risk_peak > 85`: attackers detected | 48 of 52 (recall 92.3%) |
| Alert rule `risk_peak > 85`: benign employees alarmed | 1 of 84 (specificity 98.8%) |
| Precision at 85 | 98.0% |

Sources: `reports/08_test_evaluation.md`, `reports/13_user_classification.md`.
Test prevalence (38%) is far higher than in production, so precision will be lower there.

## Repository layout

| Path | What it is |
|---|---|
| `model/` | **Frozen model bundle**: model file, manifest, `mubsir_infer.py`, `verify_bundle.py`, SHA-256 sums |
| `pipeline/` | The pipeline, scripts `01`–`14`, run in order, plus `common.py`, `models.py` and `metrics.py` |
| `prepare_dataset.py` | Builds `data/model_dataset_995.parquet` from the simulation output |
| `config/` | Feature allowlist, eligible users, the grouped train/test split, CV folds, pre-registered selection rules |
| `data/` | `model_dataset_995.parquet`: 100 features and labels for every employee-workday |
| `results/` | Machine-readable results of every pipeline step, including the pre-test freeze (`results/pretest/freeze.json`) |
| `reports/` | One Markdown report per pipeline step, plus the stage audits and pre-registrations. `reports/dataset_audit.md` is the pre-training dataset audit, with the feature decisions and the reason 147 users stay out of the test set |
| `work/` | Intermediate outputs (labels, out-of-fold scores, tuning runs, frozen test scores) that the studies read |
| `audit/` | Independent audit scripts run before Stage A, Stage B and the test |
| `studies/` | Five pre-registered research studies run after the model was frozen (see `studies/README.md`) |
| `RESTRUCTURE.md` | Record of the v2 folder restructure (what moved where, and the hash check) |

The following stay **local and are not part of this repository** (see `.gitignore`):
- `archive/`: superseded handover documents, images and a large CSV export; nothing reads them.
- `mubsir_system/` and `deliverables/`: the production system and the database package, which are still being built.
- `.venv/`.

## Using the model

```python
import pandas as pd
from mubsir_infer import MubsirScorer          # run from model/
s = MubsirScorer(".")                          # refuses to load if any file hash differs
rows = pd.read_parquet("example_input.parquet")
s.score(rows)         # daily risk per employee-workday
s.score_users(rows)   # peak risk per employee, alert = risk_peak > 85
s.explain(rows, k=5)  # top SHAP contributions
```

Integrity check: `python model/verify_bundle.py` prints `8/8 files OK`.
The input contract and the meaning of the outputs are in `model/README.md`.

## Reproducing the pipeline

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python pipeline\01_build_labels.py    # then 02 ... 14 in order
```

Some steps read the simulation output from the sibling folder `../New_Attack_Simulation/`, which is not in this
repository: `prepare_dataset.py`, `pipeline/common.py` and some audits and studies.
Everything after `data/model_dataset_995.parquet` uses files included here.

Methodology rules followed throughout:
- grouped split by attack-schedule signature;
- 5×3 grouped CV on dev users only;
- every decision rule pre-registered before results were seen;
- the test set opened once, after a 24/24 pre-test audit;
- every frozen file hash-checked before and after each study.

## Research studies (after the freeze)

Each study is self-contained in `studies/<name>/`, with its pre-registration, code, scores, results, report and
integrity hashes. Studies use development data only and never modify a frozen file.

| Study | Question | Answer |
|---|---|---|
| `user_risk` | How should daily risk be aggregated per employee? | Keep `max > 85`; no aggregation beats it at equal FPR |
| `rotation` | Does MUBSIR beat per-channel monitors against channel rotation? | Yes, partially: +2.9 recall points at 3% FPR, more for high switching |
| `controlled_rotation` | Do explicit switching features add a signal? | No; rotation is detected through aggregate deviation |
| `switching_baseline` | Does switching relative to the employee's 2010 habit help? | No meaningful added value |
| `switching_weight` | Does up-weighting switching features in training help? | No; at weight 10 it lowers detection by 2.8 points |
