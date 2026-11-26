"""
MUBSIR inference (single model, label horizon 10 workdays).

Scores READY-MADE feature rows (the 100 features listed in manifest.json, computed exactly as in training);
it does not compute features from raw logs.

    from mubsir_infer import MubsirScorer
    s = MubsirScorer("path/to/model")            # verifies SHA256SUMS.txt by default
    daily = s.score(rows)                        # risk 0-100, level, alert per row
    users = s.score_users(rows)                  # risk_peak, level, alert, peak_day, first_alert_day, n_days
    expl  = s.explain(rows, k=5)                 # list of dicts for shap_explanation_jsonb

What the numbers mean (manifest.json -> calibration.meaning):
  risk  = 100 x calibrated daily probability that the employee had at least one attack day in the last
          10 workdays, under the development population mix. It is NOT the probability that the employee
          is malicious.
  risk_peak = the maximum daily risk over the rows given for that user (a ranking statistic, not a
          calibrated per-employee probability; it grows with the number of days observed).
  level: Low (risk <= 60), Medium (60 < risk <= 85), High (risk > 85). alert = risk > 85.
"""

import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

EPS = 1e-7
NAN_ALLOWED_PREFIX = "channel_distribution_shift_"


class MubsirScorer:
    def __init__(self, bundle_dir=".", verify=True):
        self.dir = Path(bundle_dir)
        if verify:
            bad = verify_sums(self.dir)
            if bad:
                raise RuntimeError(f"bundle integrity check failed: {bad}")
        self.manifest = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        self.features = list(self.manifest["features"])
        cal = self.manifest["calibration"]
        self.a, self.b = float(cal["a"]), float(cal["b"])
        self.booster = lgb.Booster(model_file=str(self.dir / "lgbm_y10.txt"))
        if self.booster.feature_name() != self.features:
            raise RuntimeError("model feature order does not match manifest")
        self.version = f"{self.manifest['model']}-{self.manifest['version']}"

    # ------------------------------------------------------------ input
    def _matrix(self, rows):
        missing = [c for c in self.features if c not in rows.columns]
        if missing:
            raise ValueError(f"missing feature columns ({len(missing)}): {missing[:10]}")
        X = rows[self.features].apply(pd.to_numeric, errors="raise").astype("float32")
        nan_cols = [c for c in self.features if X[c].isna().any() and not c.startswith(NAN_ALLOWED_PREFIX)]
        if nan_cols:
            raise ValueError(f"NaN/null not allowed in: {nan_cols[:10]} (only {NAN_ALLOWED_PREFIX}* may be NaN; "
                             "never replace NaN by 0 there)")
        if np.isinf(X.to_numpy()).any():
            raise ValueError("infinite values in features")
        return X.to_numpy()

    # ------------------------------------------------------------ scoring
    def _risk(self, p):
        p = np.clip(p, EPS, 1 - EPS)
        return 100.0 / (1.0 + np.exp(-(self.a * np.log(p / (1 - p)) + self.b)))

    @staticmethod
    def level(risk):
        return np.where(risk > 85, "High", np.where(risk > 60, "Medium", "Low"))

    def score(self, rows):
        p = self.booster.predict(self._matrix(rows))
        risk = self._risk(p)
        out = pd.DataFrame({"risk": risk, "level": self.level(risk), "alert": risk > 85}, index=rows.index)
        for k in ("user", "day"):
            if k in rows.columns:
                out.insert(0, k, rows[k].to_numpy())
        out["model_version"] = self.version
        return out

    def score_users(self, rows):
        if "user" not in rows.columns or "day" not in rows.columns:
            raise ValueError("score_users needs 'user' and 'day' columns")
        s = self.score(rows).sort_values(["user", "day"])
        g = s.groupby("user")
        out = pd.DataFrame({"risk_peak": g["risk"].max(), "n_days": g.size()})
        out["peak_day"] = s.loc[g["risk"].idxmax(), ["user", "day"]].set_index("user")["day"]
        out["first_alert_day"] = s[s["alert"]].groupby("user")["day"].min()
        out["level"] = self.level(out["risk_peak"].to_numpy())
        out["alert"] = out["risk_peak"] > 85
        return out.reset_index()

    def explain(self, rows, k=5):
        """Top-k contributions per row in calibrated log-odds (a x TreeSHAP); positive raises the risk."""
        X = self._matrix(rows)
        contrib = self.booster.predict(X, pred_contrib=True)
        p = self.booster.predict(X)
        risk = self._risk(p)
        res = []
        for i in range(len(X)):
            c = contrib[i, :-1] * self.a
            top = np.argsort(-np.abs(c))[:k]
            res.append({"model_version": self.version, "risk_score": round(float(risk[i]), 2),
                        "base_value_logit": round(float(self.a * contrib[i, -1] + self.b), 4),
                        "units": "calibrated log-odds contribution (a * TreeSHAP); positive = raises risk",
                        "top_features": [{"feature": self.features[j],
                                          "value": None if np.isnan(X[i, j]) else float(X[i, j]),
                                          "contribution": round(float(c[j]), 4),
                                          "direction": "raises risk" if c[j] > 0 else "lowers risk"} for j in top]})
        return res

    def thresholds(self):
        return dict(self.manifest["thresholds"])


def verify_sums(bundle_dir):
    """Return the list of files whose SHA-256 differs from SHA256SUMS.txt (empty = OK)."""
    d = Path(bundle_dir)
    bad = []
    for line in (d / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        f = d / name.strip().lstrip("*")
        if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest() != digest:
            bad.append(name.strip())
    return bad
