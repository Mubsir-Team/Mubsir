# MUBSIR — Pre-Training Verification (995 eligible users)

Generated: 2026-11-23 15:07 UTC by `prepare_dataset.py` · seed=42

**Result: 42/42 checks PASS.** The run stops on the first failed check, so outputs exist only when every check passes.

## Population
| Item | Value |
|---|---:|
| Eligible users | 995 |
| Attackers | 400 |
| Benign users | 595 |
| Excluded users (out of scope) | 2742 |
| Rows (user-days) | 103,395 |
| Positive rows (y=1) | 9,080 (8.78%) |
| Predictors | 100 (97 source + 3 engineered) |

## Split (by user)
| split | benign_users | attacker_users |
|---|---|---|
| train | 416 | 282 |
| val | 90 | 57 |
| test | 89 | 61 |

| split | rows | positives | positive_rate |
|---|---|---|---|
| train | 72548 | 6401 | 0.0882 |
| val | 15247 | 1271 | 0.0834 |
| test | 15600 | 1408 | 0.0903 |

Attackers per stealth level:

| stealth_level | train | val | test |
|---|---|---|---|
| High | 93 | 21 | 20 |
| Low | 95 | 17 | 21 |
| Medium | 94 | 19 | 20 |

Attackers per attack duration (workdays):

| window_workdays | train | val | test |
|---|---|---|---|
| 5.0 | 35 | 8 | 7 |
| 10.0 | 50 | 9 | 11 |
| 20.0 | 49 | 11 | 10 |
| 30.0 | 49 | 10 | 11 |
| 40.0 | 49 | 10 | 11 |
| 50.0 | 50 | 9 | 11 |

## All checks
| # | Check | Status | Detail |
|---|---|---|---|
| 1 | Source parquet SHA-256 matches NB12 manifest | PASS | 446047aa1276f154 |
| 2 | Source shape | PASS | (387162, 143) |
| 3 | Source (user, day) unique | PASS |  |
| 4 | Source label counts | PASS | {0: 378082, 1: 9080} |
| 5 | Candidate list has no duplicate users | PASS |  |
| 6 | Eligible users = 995 | PASS | 995 |
| 7 | Attackers = 400 | PASS | 400 |
| 8 | All attackers are eligible | PASS | 0 |
| 9 | Benign eligible users = 595 | PASS | 595 |
| 10 | Users with y=1 are exactly the 400 attackers | PASS |  |
| 11 | All eligible users present in feature table | PASS | 0 |
| 12 | Excluded users = 2,742 | PASS | 2742 |
| 13 | Excluded users carry zero positive days | PASS | 0 |
| 14 | Filtered population has 995 users | PASS | 995 |
| 15 | No excluded user remains | PASS |  |
| 16 | All 9,080 positive user-days kept | PASS | 9080 |
| 17 | y equals User-Day Ground Truth membership | PASS |  |
| 18 | Allowlist covers all 143 columns | PASS |  |
| 19 | Allowlist has no undecided columns | PASS | ['exclude', 'exclude_C1', 'exclude_C2', 'keep'] |
| 20 | Source predictor count = 97 | PASS | 97 |
| 21 | No C1 companion column among predictors | PASS |  |
| 22 | No C2 baseline/ratio column among predictors | PASS |  |
| 23 | No predictor left with baseline-missing NaN | PASS |  |
| 24 | Engineered feature dictionary matches code | PASS |  |
| 25 | E_email_today: no NaN/inf and non-negative | PASS |  |
| 26 | E_usb_today: no NaN/inf and non-negative | PASS |  |
| 27 | E_http_today: no NaN/inf and non-negative | PASS |  |
| 28 | Engineered E_c reproduces cumulative_deviation_5wd | PASS | max abs diff 2.3e-14 |
| 29 | Engineered E_c reproduces cumulative_deviation_10wd | PASS | max abs diff 2.8e-14 |
| 30 | Engineered E_c reproduces cumulative_deviation_20wd | PASS | max abs diff 3.9e-14 |
| 31 | Engineered E_c reproduces cumulative_deviation_30wd | PASS | max abs diff 4.3e-14 |
| 32 | Predictor count = 100 | PASS | 100 |
| 33 | No forbidden column among predictors | PASS | set() |
| 34 | All predictors numeric | PASS | [] |
| 35 | No predictor equals y or 1-y | PASS | [] |
| 36 | Every eligible user assigned to exactly one split | PASS |  |
| 37 | Zero user overlap between splits | PASS | {'train&val': 0, 'train&test': 0, 'val&test': 0} |
| 38 | Split 'train' contains attackers and benign users | PASS | attackers=282, benign=416 |
| 39 | Split 'val' contains attackers and benign users | PASS | attackers=57, benign=90 |
| 40 | Split 'test' contains attackers and benign users | PASS | attackers=61, benign=89 |
| 41 | Every row inherits its user's split | PASS |  |
| 42 | Output rows = filtered rows | PASS | 103395 |

## Outputs
- `config/eligible_users_995.csv` — evaluation metadata only; never join scenario columns into predictors.
- `config/user_split.csv`
- `data/model_dataset_995.parquet` — columns: `user`, `day`, `split`, `y` + 100 predictors. SHA-256 `1b26162ba13fd53f049e47608ea86efeb70d6ec42ba5839c9d7dda548337ddbc`.

## Notes
- Source file `behavior_features_r62_ml_final_new.parquet` was read only (SHA-256 verified against the NB12 manifest).
- `user`, `day`, `split` are keys for grouping and evaluation, not predictors.
- Excluded as predictors: 8 identifiers/constants, 15 C1 companion columns, 23 C2 columns (19 raw baseline + 4 NaN ratios).
  Baseline values remain in the original data and are used inside the derived deviation features.
- Engineered features (E_email_today, E_usb_today, E_http_today) are defined in `config/engineered_features.csv`; same-day only, never NaN.
- Remaining NaN: only `channel_distribution_shift_*` (window without any activity), kept as-is, not filled with zero.
