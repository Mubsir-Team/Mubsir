"""
14 — Row-level (daily) binary metrics of the frozen model at the frozen thresholds, on the frozen test predictions.

Read-only: uses work/test/test_scores.parquet (the single test scoring from 08) and model/manifest.json.
Nothing is re-scored, re-calibrated or re-thresholded; no threshold is chosen on test. Descriptive only.

Unit: ONE ROW = one user-day (13,924 test rows). Prediction: risk > t, t in {85 (alert), 60 (Medium edge),
92.63 (frozen dev 1% user-FPR point)}. Two labels are reported separately:
  y_10  the training label (attack day within the last 10 workdays)
  y     the injected attack day itself
Output: results/14_row_threshold_metrics.csv
"""

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from common import WORK, RESULTS, MODEL_DIR

SCORES = WORK / "test" / "test_scores.parquet"
sha_before = hashlib.sha256(SCORES.read_bytes()).hexdigest()
man = json.loads((MODEL_DIR / "model" / "manifest.json").read_text(encoding="utf-8"))
assert man["thresholds"]["alert"] == "risk > 85"
THRESHOLDS = [("alert rule (risk > 85)", 85.0), ("Medium band edge (risk > 60)", 60.0),
              ("frozen dev 1% user-FPR point (risk > 92.63)", man["thresholds"]["dev_1pct_user_fpr_risk"])]

d = pd.read_parquet(SCORES)
benign_rows = (d["is_attacker"] == 0).to_numpy()
out = []
for label in ["y_10", "y"]:
    y = d[label].to_numpy().astype(int)
    for name, t in THRESHOLDS:
        p = (d["risk"] > t).to_numpy()
        tp, fp = int((p & (y == 1)).sum()), int((p & (y == 0)).sum())
        fn, tn = int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum())
        prec, rec = tp / (tp + fp), tp / (tp + fn)
        out.append(dict(label=label, threshold=name, t=t, rows=len(d), positives=int(y.sum()),
                        TP=tp, FN=fn, FP=fp, TN=tn, precision=prec, recall=rec,
                        f1=2 * prec * rec / (prec + rec), fpr=fp / (fp + tn), specificity=tn / (fp + tn),
                        accuracy=(tp + tn) / len(y), mcc=matthews_corrcoef(y, p),
                        alarm_rate_benign_user_rows=float(p[benign_rows].mean())))
res = pd.DataFrame(out)
res.to_csv(RESULTS / "14_row_threshold_metrics.csv", index=False)
assert hashlib.sha256(SCORES.read_bytes()).hexdigest() == sha_before   # frozen scores untouched
print(res.round(4).to_string(index=False))
