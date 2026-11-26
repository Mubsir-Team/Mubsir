"""
MUBSIR (new dataset) — shared paths, constants and helpers for the pipeline.

Every script in this folder imports from here. Nothing in this module reads
ground-truth columns into the predictor matrix: predictors come only from the
approved allowlist (config/training_predictor_allowlist.csv, decision == keep)
plus the 3 engineered features (config/engineered_features.csv).
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
CV_SEEDS = (42, 43, 44)          # 3 repeats of 5-fold grouped CV on dev
N_FOLDS = 5
ES_FRACTION_SPLITS = 7           # 1/7 ~ 14-15% of fold-train users held out for early stopping
HORIZONS = (5, 10, 20)           # candidate label horizons (workdays); one is selected
DEFAULT_HORIZON = 10
FPR_POINTS = (0.01, 0.05)

PIPELINE_DIR = Path(__file__).resolve().parent
MODEL_DIR = PIPELINE_DIR.parent
ROOT = MODEL_DIR.parent
NAS = ROOT / "New_Attack_Simulation" / "reports"

CONFIG = MODEL_DIR / "config"
DATA = MODEL_DIR / "data"
WORK = MODEL_DIR / "work"
RESULTS = MODEL_DIR / "results"
REPORTS = MODEL_DIR / "reports"
for _d in (WORK, RESULTS, REPORTS):
    _d.mkdir(exist_ok=True)

# Inputs (read-only)
DATASET = DATA / "model_dataset_995.parquet"
ALLOWLIST = CONFIG / "training_predictor_allowlist.csv"
ENGINEERED = CONFIG / "engineered_features.csv"
ELIGIBLE = CONFIG / "eligible_users_995.csv"
OLD_SPLIT = CONFIG / "user_split.csv"            # legacy 70/15/15 split from prepare_dataset.py
SRC_SCENARIOS = NAS / "08_Attacker_Selection_and_Scenario_Assignment" / "approved_attacker_scenarios_r62.parquet"
SRC_CANDIDATES = NAS / "03_Channel_Behavior_Audit" / "candidate_behavior_profile.parquet"
SRC_GT_USER_DAYS = NAS / "11_Ground_Truth_and_Quality_Validation" / "ground_truth_user_days_r62.parquet"

# Outputs of the pipeline
LABELS = WORK / "labels.parquet"                 # user, day, y, y_5, y_10, y_20
EVAL_META = WORK / "eval_meta.parquet"           # per-row evaluation context (never predictors)
USER_META = CONFIG / "user_meta.csv"             # per-user evaluation context (never predictors)
FINAL_SPLIT = CONFIG / "final_split.csv"         # user -> dev / test
CV_FOLDS = CONFIG / "cv_folds.csv"               # user x repeat x fold -> fit / es / valid
SELECTION_RULES = CONFIG / "selection_rules.json"

KEYS = ["user", "day"]
# Columns that must never be predictors (ground truth, scenario design, keys, legacy split)
FORBIDDEN = {
    "y", "y_5", "y_10", "y_20", "user", "day", "split", "first_logon", "last_logoff",
    "scenario_option_id", "scenario_type_id", "window_workdays", "window_start", "window_end",
    "channel_configuration", "switching_level", "stealth_level", "stealth_multiplier",
    "forced_offdays", "proposed_active_days", "set_switch_bitmask", "is_attack_day",
    "is_active_day", "is_forced_offday", "day_type", "injected_channels", "sync_type",
    "executed_channel_count", "is_attacker", "schedule_signature",
}


class CheckFailed(SystemExit):
    pass


CHECKS = []


def check(name, condition, detail=""):
    """Record a check and stop the run on the first failure."""
    CHECKS.append((name, bool(condition), str(detail)))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}  {detail}")
    if not condition:
        raise CheckFailed(f"STOPPED — check failed: {name} ({detail})")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def feature_list():
    """The 100 approved predictors, in allowlist order then engineered order."""
    allow = pd.read_csv(ALLOWLIST)
    eng = pd.read_csv(ENGINEERED)
    feats = allow.loc[allow["decision"] == "keep", "feature_name"].tolist() + eng["feature_name"].tolist()
    assert len(feats) == 100 and len(set(feats)) == 100, len(feats)
    assert not (set(feats) & FORBIDDEN), set(feats) & FORBIDDEN
    return feats


def load_dataset():
    """Model dataset sorted by (user, day). The legacy `split` column is dropped."""
    df = pd.read_parquet(DATASET).drop(columns=["split"])
    return df.sort_values(KEYS).reset_index(drop=True)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def md_table(df, index=True, floatfmt="{:.4f}"):
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in d.itertuples(index=False):
        vals = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)
