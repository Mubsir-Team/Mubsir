# MUBSIR model bundle (single model, y_10)

| File | Purpose |
|---|---|
| `lgbm_y10.txt` | LightGBM model, native text format (237 trees, defaults) |
| `manifest.json` | Feature order (100), Platt calibration, thresholds, provenance hashes |
| `mubsir_infer.py` | `MubsirScorer`: `score`, `score_users`, `explain`, `thresholds` |
| `verify_bundle.py` | Standard-library SHA-256 check against `SHA256SUMS.txt` |
| `example_input.parquet` | 60 dev feature rows (2 users) to try the scorer |
| `requirements.txt` | Versions the bundle was built and verified with |
| `SHA256SUMS.txt` | Hashes of every file above |

## Usage
```python
import pandas as pd
from mubsir_infer import MubsirScorer
s = MubsirScorer(".")                         # refuses to load if any hash differs
rows = pd.read_parquet("example_input.parquet")
s.score(rows)          # user, day, risk, level, alert, model_version
s.score_users(rows)    # risk_peak, n_days, peak_day, first_alert_day, level, alert
s.explain(rows, k=5)   # dicts for model_prediction_outputs.shap_explanation_jsonb
```

## Input contract
- One row per (eligible employee, workday) with the 100 columns named in `manifest.json`, computed exactly as in training (team feature definitions + `E_*_today`).
- NaN is allowed only in `channel_distribution_shift_{5,10,20,30}wd` (a window without any activity). It must never be replaced by 0. Any other NaN, a missing column or an infinite value raises `ValueError`.
- Scope: the 995 eligible multi-channel employees (USB or HTTP baseline in 2010). Other employees are out of scope.

## Output meaning
- `risk` (0–100): calibrated daily probability ×100 that the employee had at least one attack day in the last 10 workdays, under the development population mix. It is **not** the probability that the employee is malicious.
- `level`: Low `risk <= 60`, Medium `60 < risk <= 85`, High `risk > 85`. `alert` = `risk > 85`.
- `risk_peak`: maximum daily risk over the rows given. It is a ranking statistic and grows with the number of days observed.

## Interface change from the previous bundle
One model replaces the three horizon models. There are no `risk_5`, `risk_10`, `risk_20` or `alert_windows` fields any more; `risk` is the single score.
