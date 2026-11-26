"""
MUBSIR — Model dataset preparation (995 eligible multi-channel users).

Reads the original Notebook 12 feature table READ-ONLY, restricts it to the
995 eligible candidate users (400 attackers + 595 benign), applies the
approved predictor allowlist (97 source columns), adds 3 engineered
same-day per-channel deviation features (config/engineered_features.csv),
builds a user-level stratified split, and writes derived artifacts inside
the Model folder only.

Any count or condition mismatch raises an error and stops the run; nothing
is silently corrected. Original dataset files are never modified.

Outputs (all inside Model/):
    config/eligible_users_995.csv          user list + attacker flag + scenario metadata (evaluation only)
    config/user_split.csv                  user -> train / val / test
    data/model_dataset_995.parquet         keys + split + y + allowlisted predictors
    reports/pre_training_verification.md   verification report

Usage:
    python prepare_dataset.py
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SEED = 42
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Decided (2026-11-23): the 19 raw 2010 baseline columns and the 4 NaN-bearing
# USB/HTTP ratio columns are excluded as predictors (`exclude_C2`). The
# baseline data stays untouched in the original dataset and is still used to
# compute the derived deviation features, including the 3 engineered ones.
N_SOURCE_PREDICTORS = 97

# Engineered same-day deviation per channel, identical to the E_c definition
# used by the nr_* and cumulative_deviation_* features:
#   E_c = max(0, Count_c - P50_c) / max(1, P95_c - P50_c), 0 when P50/P95 missing
ENGINEERED = {
    "E_email_today": ("external_email_count", "email_count_p50", "email_count_p95"),
    "E_usb_today": ("usb_copy_count", "usb_copy_p50", "usb_copy_p95"),
    "E_http_today": ("www_upload_count", "http_upload_p50", "http_upload_p95"),
}

EXPECTED = {
    "source_rows": 387_162,
    "source_cols": 143,
    "source_pos": 9_080,
    "source_neg": 378_082,
    "eligible_users": 995,
    "attackers": 400,
    "benign": 595,
    "excluded_users": 2_742,
}

KEY_COLS = ["user", "day"]
FORBIDDEN_PREDICTORS = {"y", "user", "day", "first_logon", "last_logoff"}

MODEL_DIR = Path(__file__).resolve().parent
ROOT = MODEL_DIR.parent
NAS = ROOT / "New_Attack_Simulation" / "reports"

SRC_FEATURES = NAS / "12_Feature_Engineering_and_Validation" / "behavior_features_r62_ml_final_new.parquet"
SRC_MANIFEST = NAS / "12_Feature_Engineering_and_Validation" / "export_manifest.json"
SRC_CANDIDATES = NAS / "03_Channel_Behavior_Audit" / "candidate_behavior_profile.parquet"
SRC_SCENARIOS = NAS / "08_Attacker_Selection_and_Scenario_Assignment" / "approved_attacker_scenarios_r62.parquet"
SRC_GT_USER_DAYS = NAS / "11_Ground_Truth_and_Quality_Validation" / "ground_truth_user_days_r62.parquet"

ALLOWLIST = MODEL_DIR / "config" / "training_predictor_allowlist.csv"
ENGINEERED_DICT = MODEL_DIR / "config" / "engineered_features.csv"
OUT_ELIGIBLE = MODEL_DIR / "config" / "eligible_users_995.csv"
OUT_SPLIT = MODEL_DIR / "config" / "user_split.csv"
OUT_DATASET = MODEL_DIR / "data" / "model_dataset_995.parquet"
OUT_REPORT = MODEL_DIR / "reports" / "pre_training_verification.md"

CHECKS = []


def check(name, condition, detail=""):
    """Record a check; stop the run on the first failure."""
    CHECKS.append((name, bool(condition), str(detail)))
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}  {detail}")
    if not condition:
        raise SystemExit(f"STOPPED — verification failed: {name} ({detail}). No outputs were written.")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------
# 1. Load sources (read-only) and verify their integrity
# ------------------------------------------------------------------
manifest = json.loads(SRC_MANIFEST.read_text(encoding="utf-8"))
expected_sha = next(f["sha256"] for f in manifest["files"] if f["filename"] == SRC_FEATURES.name)
source_sha = sha256(SRC_FEATURES)
check("Source parquet SHA-256 matches NB12 manifest", source_sha == expected_sha, source_sha[:16])

src = pd.read_parquet(SRC_FEATURES)
check("Source shape", src.shape == (EXPECTED["source_rows"], EXPECTED["source_cols"]), src.shape)
check("Source (user, day) unique", not src.duplicated(KEY_COLS).any())
vc = src["y"].value_counts().to_dict()
check("Source label counts", vc == {0: EXPECTED["source_neg"], 1: EXPECTED["source_pos"]}, vc)

candidates = pd.read_parquet(SRC_CANDIDATES)
scenarios = pd.read_parquet(SRC_SCENARIOS)
gt_user_days = pd.read_parquet(SRC_GT_USER_DAYS, columns=KEY_COLS)

# ------------------------------------------------------------------
# 2. Eligible population: 995 = 400 attackers + 595 benign
# ------------------------------------------------------------------
eligible = set(candidates["user"])
attackers = set(scenarios["user"])
positive_users = set(src.loc[src["y"] == 1, "user"])
all_users = set(src["user"])

check("Candidate list has no duplicate users", not candidates["user"].duplicated().any())
check("Eligible users = 995", len(eligible) == EXPECTED["eligible_users"], len(eligible))
check("Attackers = 400", len(attackers) == EXPECTED["attackers"], len(attackers))
check("All attackers are eligible", attackers <= eligible, len(attackers - eligible))
check("Benign eligible users = 595", len(eligible - attackers) == EXPECTED["benign"], len(eligible - attackers))
check("Users with y=1 are exactly the 400 attackers", positive_users == attackers)
check("All eligible users present in feature table", eligible <= all_users, len(eligible - all_users))
excluded = all_users - eligible
check("Excluded users = 2,742", len(excluded) == EXPECTED["excluded_users"], len(excluded))
excluded_pos = int(src.loc[src["user"].isin(excluded), "y"].sum())
check("Excluded users carry zero positive days", excluded_pos == 0, excluded_pos)

# ------------------------------------------------------------------
# 3. Filter to the 995 eligible users (labels untouched)
# ------------------------------------------------------------------
pop = src[src["user"].isin(eligible)].copy()
check("Filtered population has 995 users", pop["user"].nunique() == EXPECTED["eligible_users"], pop["user"].nunique())
check("No excluded user remains", not pop["user"].isin(excluded).any())
check("All 9,080 positive user-days kept", int(pop["y"].sum()) == EXPECTED["source_pos"], int(pop["y"].sum()))

gt_keys = gt_user_days.assign(day=gt_user_days["day"].astype("datetime64[ms]"), _gt=1)
lab = pop[KEY_COLS + ["y"]].assign(day=pop["day"].astype("datetime64[ms]")).merge(gt_keys, on=KEY_COLS, how="left")
check("y equals User-Day Ground Truth membership", (lab["y"] == lab["_gt"].fillna(0).astype(int)).all())

# ------------------------------------------------------------------
# 4. Predictor allowlist
# ------------------------------------------------------------------
allow = pd.read_csv(ALLOWLIST)
check("Allowlist covers all 143 columns", list(allow["feature_name"]) == list(src.columns))
check("Allowlist has no undecided columns", set(allow["decision"]) <= {"keep", "exclude", "exclude_C1", "exclude_C2"},
      sorted(set(allow["decision"])))
source_predictors = allow.loc[allow["decision"] == "keep", "feature_name"].tolist()
check("Source predictor count = 97", len(source_predictors) == N_SOURCE_PREDICTORS, len(source_predictors))
c1 = set(allow.loc[allow["decision"] == "exclude_C1", "feature_name"])
c2 = set(allow.loc[allow["decision"] == "exclude_C2", "feature_name"])
check("No C1 companion column among predictors", not (set(source_predictors) & c1))
check("No C2 baseline/ratio column among predictors", not (set(source_predictors) & c2))
check("No predictor left with baseline-missing NaN",
      not [c for c in source_predictors if c.endswith(("_p90_ratio", "_p95_ratio")) and pop[c].isna().any()])

# ------------------------------------------------------------------
# 4b. Engineered same-day per-channel deviation (E_c)
# ------------------------------------------------------------------
eng_dict = pd.read_csv(ENGINEERED_DICT)
check("Engineered feature dictionary matches code", list(eng_dict["feature_name"]) == list(ENGINEERED))
pop = pop.sort_values(KEY_COLS).reset_index(drop=True)
for name, (count_col, p50_col, p95_col) in ENGINEERED.items():
    valid = pop[p50_col].notna() & pop[p95_col].notna()
    numerator = np.maximum(0.0, pop[count_col] - pop[p50_col].fillna(0.0))
    denominator = np.maximum(1.0, (pop[p95_col] - pop[p50_col]).fillna(0.0))
    pop[name] = np.where(valid, numerator / denominator, 0.0)
    check(f"{name}: no NaN/inf and non-negative",
          np.isfinite(pop[name]).all() and (pop[name] >= 0).all())

e_total = pop[list(ENGINEERED)].sum(axis=1)
for w in (5, 10, 20, 30):
    rolled = e_total.groupby(pop["user"]).transform(lambda s: s.rolling(w, min_periods=1).sum())
    diff = float(np.abs(rolled - pop[f"cumulative_deviation_{w}wd"]).max())
    check(f"Engineered E_c reproduces cumulative_deviation_{w}wd", diff < 1e-9, f"max abs diff {diff:.1e}")

predictors = source_predictors + list(ENGINEERED)
check("Predictor count = 100", len(predictors) == N_SOURCE_PREDICTORS + len(ENGINEERED), len(predictors))
check("No forbidden column among predictors", not (set(predictors) & FORBIDDEN_PREDICTORS), set(predictors) & FORBIDDEN_PREDICTORS)
non_numeric = [c for c in predictors if not pd.api.types.is_numeric_dtype(pop[c])]
check("All predictors numeric", not non_numeric, non_numeric)
y_arr = pop["y"].to_numpy()
equal_to_target = [c for c in predictors
                   if np.array_equal(pop[c].to_numpy(), y_arr) or np.array_equal(pop[c].to_numpy(), 1 - y_arr)]
check("No predictor equals y or 1-y", not equal_to_target, equal_to_target)

# ------------------------------------------------------------------
# 5. User-level stratified split
#    Attackers: stratified by duration x stealth. Benign: one stratum.
# ------------------------------------------------------------------
sc = scenarios[["user", "window_workdays", "stealth_level", "switching_level", "channel_configuration"]].copy()
for col in ["stealth_level", "switching_level", "channel_configuration"]:
    sc[col] = sc[col].astype(str)

users = pd.DataFrame({"user": sorted(eligible)})
users["is_attacker"] = users["user"].isin(attackers).astype(int)
users = users.merge(sc, on="user", how="left")
users["stratum"] = np.where(
    users["is_attacker"] == 1,
    "A_" + users["window_workdays"].astype("Int64").astype(str) + "wd_" + users["stealth_level"],
    "benign",
)

rng = np.random.default_rng(SEED)
assignment = {}
for stratum in sorted(users["stratum"].unique()):
    members = users.loc[users["stratum"] == stratum, "user"].to_numpy(dtype=object)
    members = members[rng.permutation(len(members))]
    n = len(members)
    n_train = int(round(SPLIT_FRACTIONS["train"] * n))
    n_val = int(round((SPLIT_FRACTIONS["train"] + SPLIT_FRACTIONS["val"]) * n)) - n_train
    for i, u in enumerate(members):
        assignment[u] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")

users["split"] = users["user"].map(assignment)
check("Every eligible user assigned to exactly one split", users["split"].notna().all() and not users["user"].duplicated().any())
split_sets = {s: set(users.loc[users["split"] == s, "user"]) for s in SPLIT_FRACTIONS}
overlaps = {f"{a}&{b}": len(split_sets[a] & split_sets[b]) for a, b in [("train", "val"), ("train", "test"), ("val", "test")]}
check("Zero user overlap between splits", all(v == 0 for v in overlaps.values()), overlaps)
for s in SPLIT_FRACTIONS:
    sub = users[users["split"] == s]
    check(f"Split '{s}' contains attackers and benign users",
          sub["is_attacker"].sum() > 0 and (sub["is_attacker"] == 0).sum() > 0,
          f"attackers={int(sub['is_attacker'].sum())}, benign={int((sub['is_attacker'] == 0).sum())}")

pop["split"] = pop["user"].map(assignment)
check("Every row inherits its user's split", pop["split"].notna().all() and pop.groupby("user")["split"].nunique().max() == 1)

# ------------------------------------------------------------------
# 6. Write derived outputs (Model/ only)
# ------------------------------------------------------------------
out = pop[KEY_COLS + ["split", "y"] + predictors].sort_values(KEY_COLS).reset_index(drop=True)
check("Output rows = filtered rows", len(out) == len(pop), len(out))

eligible_out = users[["user", "is_attacker", "window_workdays", "stealth_level", "switching_level", "channel_configuration"]]
eligible_out.to_csv(OUT_ELIGIBLE, index=False)
users[["user", "is_attacker", "stratum", "split"]].to_csv(OUT_SPLIT, index=False)
out.to_parquet(OUT_DATASET, index=False)

# ------------------------------------------------------------------
# 7. Verification report
# ------------------------------------------------------------------
split_users = users.groupby(["split", "is_attacker"]).size().unstack(fill_value=0).reindex(list(SPLIT_FRACTIONS))
split_rows = out.groupby("split")["y"].agg(rows="size", positives="sum").reindex(list(SPLIT_FRACTIONS))
split_rows["positive_rate"] = (split_rows["positives"] / split_rows["rows"]).round(4)
attacker_mix = users[users["is_attacker"] == 1].pivot_table(index="stealth_level", columns="split", values="user",
                                                             aggfunc="count", fill_value=0)[list(SPLIT_FRACTIONS)]
duration_mix = users[users["is_attacker"] == 1].pivot_table(index="window_workdays", columns="split", values="user",
                                                            aggfunc="count", fill_value=0)[list(SPLIT_FRACTIONS)]


def md_table(df):
    df = df.reset_index()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


report = f"""# MUBSIR — Pre-Training Verification (995 eligible users)

Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `prepare_dataset.py` · seed={SEED}

**Result: {sum(ok for _, ok, _ in CHECKS)}/{len(CHECKS)} checks PASS.** The run stops on the first failed check, so outputs exist only when every check passes.

## Population
| Item | Value |
|---|---:|
| Eligible users | {pop['user'].nunique()} |
| Attackers | {len(attackers)} |
| Benign users | {len(eligible - attackers)} |
| Excluded users (out of scope) | {len(excluded)} |
| Rows (user-days) | {len(out):,} |
| Positive rows (y=1) | {int(out['y'].sum()):,} ({out['y'].mean():.2%}) |
| Predictors | {len(predictors)} ({len(source_predictors)} source + {len(ENGINEERED)} engineered) |

## Split (by user)
{md_table(split_users.rename(columns={0: 'benign_users', 1: 'attacker_users'}))}

{md_table(split_rows)}

Attackers per stealth level:

{md_table(attacker_mix)}

Attackers per attack duration (workdays):

{md_table(duration_mix)}

## All checks
| # | Check | Status | Detail |
|---|---|---|---|
""" + "\n".join(f"| {i} | {n} | {'PASS' if ok else 'FAIL'} | {d} |" for i, (n, ok, d) in enumerate(CHECKS, 1)) + f"""

## Outputs
- `config/eligible_users_995.csv` — evaluation metadata only; never join scenario columns into predictors.
- `config/user_split.csv`
- `data/model_dataset_995.parquet` — columns: `user`, `day`, `split`, `y` + {len(predictors)} predictors. SHA-256 `{sha256(OUT_DATASET)}`.

## Notes
- Source file `{SRC_FEATURES.name}` was read only (SHA-256 verified against the NB12 manifest).
- `user`, `day`, `split` are keys for grouping and evaluation, not predictors.
- Excluded as predictors: 8 identifiers/constants, 15 C1 companion columns, 23 C2 columns (19 raw baseline + 4 NaN ratios).
  Baseline values remain in the original data and are used inside the derived deviation features.
- Engineered features ({', '.join(ENGINEERED)}) are defined in `config/engineered_features.csv`; same-day only, never NaN.
- Remaining NaN: only `channel_distribution_shift_*` (window without any activity), kept as-is, not filled with zero.
"""
OUT_REPORT.write_text(report, encoding="utf-8")
print(f"\nDone. {len(CHECKS)} checks passed. Outputs written to {MODEL_DIR}")
