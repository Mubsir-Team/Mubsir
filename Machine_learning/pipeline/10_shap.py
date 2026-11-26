"""
10 — What the frozen model relies on: TreeSHAP (LightGBM pred_contrib) by feature family and horizon.

Explanations are computed for the final model (model/lgbm_y10.txt) on:
  test  the 136 test users (frozen scoring; the model never saw them)   -> main numbers
  dev   the 859 dev users (training rows, in-sample)                    -> stability check only
Contributions are in the model's raw log-odds. Platt is affine in that space (calibrated logit = a*margin + b),
so the calibrated contributions are the same values times a (0.955); shares are unchanged.
No decision is taken from these results.

Outputs: results/10_*.csv, reports/10_shap.md, results/10_example_explanation.json (shap_explanation_jsonb format)
"""

import json
import re

import lightgbm as lgb
import numpy as np
import pandas as pd

from common import WORK, RESULTS, REPORTS, MODEL_DIR, LABELS, EVAL_META, USER_META, FINAL_SPLIT, load_dataset, md_table, write_json

man = json.loads((MODEL_DIR / "model" / "manifest.json").read_text(encoding="utf-8"))
FEATS = man["features"]
bst = lgb.Booster(model_file=str(MODEL_DIR / "model" / "lgbm_y10.txt"))


def family(c):
    if c.startswith("nr_gap_alt"):
        return "Rotation: gap-alt (nr_gap_alt_2)"
    if c.startswith("nr_"):
        return "Rotation: deviation-based (nr_*)"
    if c.startswith(("dominant_channel_switches", "multi_channel_days")) or c == "active_channel_count":
        return "Rotation/multi-channel: team (switches, multi-channel days)"
    if c.startswith(("cumulative_deviation", "combined_channel_intensity", "consecutive_elevated_days",
                     "cross_channel_deviation_days", "channel_distribution_shift")):
        return "Accumulation & shift (all channels)"
    if c.startswith(("email", "external_email", "internal_email", "E_email")):
        return "Email"
    if c.startswith(("usb", "file_", "E_usb")):
        return "USB / file"
    if c.startswith(("http", "www", "unique_domains", "E_http")):
        return "HTTP / web"
    if c.startswith(("logon", "logoff")):
        return "Logon"
    return "Other"


def horizon(c):
    m = re.search(r"_(5|10|20|30)wd$", c)
    return f"{m.group(1)} workdays" if m else "daily"


fam = pd.Series({c: family(c) for c in FEATS})
hor = pd.Series({c: horizon(c) for c in FEATS})
assert (fam != "Other").all(), fam[fam == "Other"].index.tolist()

df = load_dataset()
lab = pd.read_parquet(LABELS)
meta = pd.read_parquet(EVAL_META)
um = pd.read_csv(USER_META).set_index("user")
split = pd.read_csv(FINAL_SPLIT)
part = df["user"].astype(str).map(split.set_index("user")["part"])


def shap_for(mask):
    X = df.loc[mask, FEATS].astype("float32").to_numpy()
    c = bst.predict(X, pred_contrib=True)
    assert c.shape[1] == len(FEATS) + 1
    margin = bst.predict(X, raw_score=True)
    assert np.allclose(c.sum(axis=1), margin, atol=1e-6)        # additivity check
    return pd.DataFrame(c[:, :-1], columns=FEATS), X, c[:, -1]


res = {}
for name in ("test", "dev"):
    mask = (part == name).to_numpy()
    sv, X, base = shap_for(mask)
    y10 = lab.loc[mask, "y_10"].to_numpy()
    users = df.loc[mask, "user"].astype(str).to_numpy()
    sw = pd.Series(users).map(um["switching_level"]).to_numpy()
    absv = sv.abs()
    groups = {"all rows": np.ones(len(sv), bool), "y_10 = 1 rows": y10 == 1,
              "benign-user rows": pd.Series(users).map(um["is_attacker"]).to_numpy() == 0}
    fam_tab = pd.DataFrame({g: absv[m].mean().groupby(fam).sum() / absv[m].mean().sum() for g, m in groups.items()})
    hor_tab = pd.DataFrame({g: absv[m].mean().groupby(hor).sum() / absv[m].mean().sum() for g, m in groups.items()})
    rot_sw = pd.DataFrame({lvl: absv[(y10 == 1) & (sw == lvl)].mean().groupby(fam).sum() /
                           absv[(y10 == 1) & (sw == lvl)].mean().sum() for lvl in ("Low", "Medium", "High")})
    top = absv.mean().sort_values(ascending=False).head(15).to_frame("mean_abs_shap")
    top["share"] = top["mean_abs_shap"] / absv.mean().sum()
    top["family"] = fam[top.index]
    # direction: does a higher feature value push the risk up? (Spearman of value vs SHAP)
    top["direction"] = [("higher -> more risk" if pd.Series(X[:, FEATS.index(f)]).corr(sv[f], method="spearman") > 0
                         else "higher -> less risk") for f in top.index]
    res[name] = dict(fam=fam_tab, hor=hor_tab, rot_sw=rot_sw, top=top, sv=sv, X=X, base=base, users=users, y10=y10,
                     mask=mask)
    fam_tab.to_csv(RESULTS / f"10_{name}_share_by_family.csv")
    hor_tab.to_csv(RESULTS / f"10_{name}_share_by_horizon.csv")
    rot_sw.to_csv(RESULTS / f"10_{name}_family_share_by_switching_level.csv")
    top.to_csv(RESULTS / f"10_{name}_top15.csv")

# ------------------------------------------------------------------ example explanation (shap_explanation_jsonb format)
t = res["test"]
risk_rows = pd.read_parquet(WORK / "test" / "test_scores.parquet")
assert (risk_rows["user"].to_numpy() == t["users"]).all()
i = int(np.argmax(np.where(t["y10"] == 1, risk_rows["risk"].to_numpy(), -1)))   # highest-risk positive test row
A = man["calibration"]["a"]
contrib = t["sv"].iloc[i] * A
topk = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(5)
example = {
    "model_version": f"{man['model']}-{man['version']}",
    "risk_score": round(float(risk_rows['risk'].iloc[i]), 2),
    "base_value_logit": round(float(A * t["base"][i] + man["calibration"]["b"]), 4),
    "units": "calibrated log-odds contribution (a * TreeSHAP); positive = raises risk",
    "top_features": [{"feature": f, "value": None if np.isnan(t["X"][i, FEATS.index(f)]) else float(t["X"][i, FEATS.index(f)]),
                      "contribution": round(float(v), 4), "family": fam[f],
                      "direction": "raises risk" if v > 0 else "lowers risk"} for f, v in topk.items()],
}
write_json(RESULTS / "10_example_explanation.json", example)

# ------------------------------------------------------------------ report
fam_cmp = pd.DataFrame({"test: all rows": res["test"]["fam"]["all rows"], "test: y_10 = 1 rows": res["test"]["fam"]["y_10 = 1 rows"],
                        "dev (in-sample): all rows": res["dev"]["fam"]["all rows"]}).sort_values("test: all rows", ascending=False)
text = f"""# 10 — What the model relies on (TreeSHAP, final model)

Shares of mean |SHAP| (raw log-odds; additivity checked: contributions + base = margin on every row).
**test** = 136 unseen users (main). **dev** = the training rows (in-sample, stability check only).

## By feature family
{md_table(fam_cmp)}

## By horizon (test)
{md_table(res['test']['hor'])}

## Family share on attack-window rows (y_10 = 1) by switching level (test)
{md_table(res['test']['rot_sw'])}

## Top 15 features (test)
{md_table(res['test']['top'])}

## Example `shap_explanation_jsonb` (highest-risk positive test row)
```json
{json.dumps(example, indent=2)}
```
"""
(REPORTS / "10_shap.md").write_text(text, encoding="utf-8")
print(text)
