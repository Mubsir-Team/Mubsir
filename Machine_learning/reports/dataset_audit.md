# MUBSIR: Deep Audit Summary (New_Attack_Simulation)

> **About this file.** This is an English translation of the dataset audit written in Arabic on 2026-11-23/24, before
> any model was trained. The Arabic original is kept locally (`archive/audit_summary.md`). Section numbers are unchanged:
> `pipeline/02_make_split.py` cites §8.1 and §8.4.
>
> The six summary images of the original were Arabic screenshots of the simulation reports. They are not reproduced
> here. Their numbers appear in the text of §2.1–2.2.
>
> The `train / val / test` split named in this audit is the **preliminary 70/15/15 split** made by
> `prepare_dataset.py` (`config/user_split.csv`). It was used only for the decisions below. The final split is the
> locked test (13.7%) plus grouped 5×3 CV on dev (`reports/02_split.md`). The 147 `val` users of the preliminary split
> informed the decisions in §8.1 and §8.4, so they were kept out of the final test set.

**Audit date:** 2026-11-23. **Scope:** read-only. No file of the dataset was modified or deleted.

**Tools:** Python 3.14, pandas 3.0.3, pyarrow 24.

**Method:**
- Independent check of the Parquet and CSV files.
- Code review of notebooks 00–12.
- Comparison with the original user-day file before injection (`Data_engineering/data/processed/user_day_r62_v1.parquet`).
- Comparison with the original features (`Data_engineering/data/features/behavior_features_r62_v1.parquet`).

> Note on the inputs:
> - There are **13** notebooks (00 to 12) and 13 report folders, not 12.
> - The README is named `README_Model_Handoff.md`.
> - Report folder 12 has no `notebook_summary.md`. It has `feature_methodology_and_decisions.md` and
>   `final_feature_qa_report.md` instead.

---

## 0. Executive summary

| Item | Result |
|---|---|
| File integrity (shape, duplicates, inf, NaN) | ✅ Sound. NaN appears only in 16 documented columns |
| Columns vs. data dictionary | ✅ 143/143 by name, order and NaN count (one type difference in `day`: ms vs s) |
| `y` vs. ground truth | ✅ Exact match (9,080 = 9,080, zero differences) |
| 400 attackers and their distribution | ✅ Matches the reports exactly |
| Off-days | ✅ All 1,670 are **`y=0`** and are absent from the ground truth (correct by design) |
| Temporal leakage in rolling and baseline features | ✅ None (empirical check + code review) |
| Direct target leakage | ✅ No column leaks the target |
| **Simulation artefacts and selection bias** | ⚠️ Two critical issues (C1, C2) → ✅ **decided**: 15 columns excluded, and the model works on the 995 candidates only (§7 and §8) |

> **Quick plain-language reading:** go straight to §9 at the end of this file.

---

## 1. Schema and integrity

| File | Shape | Key duplicates | Row duplicates | inf | NaN |
|---|---|---|---|---|---|
| `user_day_injected_counts_r62.parquet` | 1,394,010 × 34 | 0 (`user,day`) | 0 | 0 | `first_logon` only: 881 |
| `behavior_features_r62_ml_final_new.parquet` | **387,162 × 143** | 0 (`user,day`) | 0 | 0 | 16 columns (3,904,805 values) |
| `ground_truth_user_days_r62.parquet` | 9,080 × 42 | 0 | 0 | 0 | 0 |
| `ground_truth_channel_days_r62.parquet` | 10,124 × 27 | 0 (`user,day,channel`) | 0 | 0 | 0 |

**Final feature file:**
- Types: 115 float64, 21 int64, 2 int32, 1 int8 (`y`), 3 datetime (`day`, `first_logon`, `last_logoff`), 1 string (`user`).
- 3,737 users, from 2011-01-03 to 2011-06-01.

**Columns with NaN (all intended and documented):**
- `usb_copy_{p50,p90,p95,mad,to_p90_ratio,to_p95_ratio}`: 312,211 rows. These are users with no USB activity at all in 2010.
- `http_upload_{p50,p90,p95,mad,to_p90_ratio,to_p95_ratio}`: 328,257 rows.
- `channel_distribution_shift_{5,10,20,30}wd`: 32,420 / 16,262 / 7,506 / 5,809 rows. NaN means the window has no activity.

**Match with the dictionary `feature_dictionary_r62_final.csv`:**
- Names and order (`column_order` 1 to 143) match.
- `null_rows` and the `nullable` column match the actual data in every column.
- The only difference: `day` is stored as `datetime64[ms]`, while the dictionary says `datetime64[s]`. The README explains this.

**Constant columns:** `is_weekend` (always 0) and `unique_usb_extensions` (always 0).

---

## 2. Attack distribution and labelling logic

### 2.1 Attackers

Checked against `approved_attacker_scenarios_r62.parquet` and the ground truth.

- **400 distinct attackers.** All are in the feature table, and all have `y=1` on at least one day.
- Scenario summary from the simulation reports:
  - 261 two-channel scenarios (65.25%);
  - 139 three-channel scenarios (34.75%);
  - 9,080 scheduled active attack user-days.
- **Channels:** E+U 117 · E+H 117 · U+H 27 · E+U+H 139.
- **Switching:** Low 133 · Medium 134 · High 133.
- **Stealth:**
  - High (factor 0.25): 134 attackers;
  - Medium (0.50): 133;
  - Low (0.75): 133.
  - The mapping from level to factor is 1:1, with no errors.
- **Duration:** 5 days for 50 attackers; 70 attackers each for 10, 20, 30, 40 and 50 days.

> Every number in the six summary images of the simulation reports was checked against the files, and all are correct.
> Each dimension is balanced on its own, but crossings of two dimensions are not (see the balance note below).

- **Off-days by duration:** 5 days → 0, 10 → 90, 20 → 185, 30 → 325, 40 → 465, 50 → 605. Total **1,670**.
- **Balance note:** some cells of the cross-distribution are small. For example, 5-day duration × Medium switching has
  **only 3 attackers**, and U+H × High has 5. This matters when reporting performance per subgroup.

### 2.2 Actual execution (ground truth)

- **9,080 attack days:** 8,073 single-channel, 970 two-channel, 37 three-channel. The primary channel switched 3,116 times.
- **10,124 channel-day injections, 29,007 events in total:**
  - EMAIL: 6,703 (17,791 events);
  - USB: 1,935 (4,920);
  - HTTP: 1,486 (6,296).
- The injection formula `I = max(1, ceil(m × P95))` matches **100%** of ground-truth rows.
  `count_after = count_before + count_added` always holds.

### 2.3 The critical point: how are off-days labelled?

- **`y = 0` on all 1,670 off-days**, and all of them are in the feature table.
- `ground_truth_user_days_r62.parquet` **contains no off-day**: all 9,080 rows have `is_active_day=True` and
  `is_forced_offday=False`. The labelling is therefore **consistent** with the ground truth and with the definition in
  the reports: `y=1` only on a day with an actual injection.
- All 10,750 days of the attack windows are in the feature table.
- An attacker's non-attacked days are all `y=0`, 32,562 rows in total:
  - 13,255 before the window;
  - 1,670 off-days;
  - 17,637 after the window.
- ⚠️ These days are "hard negatives": their rolling features are very close to attack days (see M1).

### 2.4 Class balance

| | Count | Share |
|---|---:|---:|
| `y=0` | 378,082 | 97.655% |
| `y=1` | 9,080 | **2.345%** (1 : 41.6) |

- **By month,** the positive rate varies: January 1.77%, February 3.04%, March 3.02%, April 2.58%, May 1.30%. June has
  only 7 rows, with a 0% rate.
- **By weekday,** it is almost constant (2.33%–2.35%), so there is no artificial weekday pattern.

---

## 3. Leakage audit

### 3.1 Temporal leakage: ✅ none

1. **Independent empirical check.**
   - The 83 original columns of the final file were compared with their pre-injection version (`behavior_features_r62_v1`).
   - Result: **zero differences** on every row of benign users, and on every attacker row before the first attack day.
   - If any rolling feature used future data, the injection would show up before it started.
2. **Injection vs. the original source.**
   - Only 8 fields changed, in only 9,080 rows, all of them ground-truth days.
   - The per-field sums of the differences match the ground truth per channel (17,791 / 4,920 / 6,296).
3. **Code review of NB12.**
   - Windows are computed as `left=max(0,t-w+1), right=t+1` with prefix sums.
   - `rolling(window, min_periods=1)` is applied within each user, after sorting by `user, day`.
   - The `handoff` feature uses only the previous day.
   - There is no `center=True`, `bfill` or `shift(-k)`.
4. **The baseline is fixed from 2010 data.**
   - P95 was recomputed from the original 2010 data, on normal days with a logon and activity in the channel.
   - It matched 100% of `email_count_p95` (3,737 users), `usb_copy_p95` (721) and `http_upload_p95` (567).
   - A P95 computed from 2011 data matches only 24%–63% of the values, so nothing leaks from the attack period into
     the baseline.
   - Every baseline column is constant per user.

### 3.2 Direct target leakage: ✅ none

The feature file has no column from the ground truth, the schedule or the scenario.

The strongest single features are:
- by AUC: `cumulative_deviation_10wd`, AUC 0.971;
- by AP: `cumulative_deviation_5wd`, AP 0.674.

This is an intended behavioural signal, not leakage. It does mean the task is relatively easy, especially at Low stealth.

### 3.3 Simulation artefacts and selection bias: ⚠️ important

See C1, C2 and M2 in §5.

---

## 4. Code vs. reports

| Item | Status |
|---|---|
| Summary numbers (400, 9,080, 10,124, 29,007, 1,670, 970/37, 3,116) | ✅ Match what was computed from the files |
| QA files (CSV) in 00 and 07–11 | ✅ All cases PASS (00: 13, 07: 33, 08: 13+15+16, 09: 47, 10: 7+5+43+6, 11: 20+58+35+27) |
| SHA-256 manifest in NB12 | ⚠️ The Parquet matches. The three text files (dictionary, methodology, QA report) do not match in this copy. **The cause is line endings only:** converting `\r\n` to `\r\r\n` restores the original hash exactly. The content is unchanged |
| Saved execution state of NB12 | ⚠️ Cell 12.9.1 holds an `AssertionError`: a guard against overwriting existing files, raised because the cell was re-run after the export. Cells 12.9.2–12.9.4 have no execution count. The post-export check cell (12.9.5) succeeded: 32/32, and 4/4 SHA matches |
| NB09 | ⚠️ The first 9.4 cell holds `RuntimeError: Joint DP contradicts approved zero-synchronization switching evidence`. A second cell replaced it and succeeded. Execution counts are not sequential (4 → 6 → 14 → 15 → 17 → 20) |
| NB02 | ⚠️ A `PermissionError` during export, fixed in cell 6.2. The output file exists |
| Dictionary: window wording | ⚠️ In 30 rows, `time_scope` says "full window required", but the actual policy is `min_periods=1`, documented in the `effective_window_policy` column and in the README. The values are indeed partial: there is no NaN in any 20/30wd window |
| Dictionary: `column_role` | Note: `day_of_week` and `is_weekend` are labelled "Behavioral Feature", although they are calendar fields |
| NB04 | Explicitly recommends keeping multi-channel benign users to reduce shortcut learning. This was done: 595 benign candidates |

---

## 5. Issue classification

### 🔴 Critical: must be decided before training

**C1: dilution artefact.**

The injection raised the count fields (`usb_copy_count`, `www_upload_count`, `external_email_count`, …) but **did not
change the companion fields**:
- `unique_usb_files`, `unique_upload_domains`, `unique_external_recipients`, `unique_external_domains`;
- `external_email_bytes`, `total_sent_email_bytes`;
- `attachment_count`, `emails_with_attachments`, `from_removable_count`.

The result is unrealistic ratios on attack days. Within attackers only (same users, so no selection bias):

| Ratio | Median y=1 | Median y=0 | Separation |
|---|---:|---:|---:|
| `unique_usb_files / usb_copy_count` | 0.50 | 1.00 | AUC ≈ 0.85 |
| `unique_upload_domains / www_upload_count` | 0.50 | 1.00 | AUC ≈ 0.85 |
| `unique_external_recipients / external_email_count` | 0.83 | 1.67 | AUC ≈ 0.85 |
| `external_email_bytes / external_email_count` | 33.5K | 54.1K | AUC ≈ 0.66 |

Tree models (XGBoost/LightGBM) would easily learn "the count rose but the unique files did not". In reality, leaking more
files raises the number of unique files and the bytes.

**Approved decision:**
- These companion fields and the byte-derived columns are excluded from the predictors (15 columns, see §6.4).
- The injection will not be redone.
- A version with these fields may be trained as an ablation, only to document the size of the artefact.

**C2: population selection shortcut.**

Attackers were chosen only from **995 multi-channel candidate users** with a USB or HTTP baseline. Therefore:
- **100% of positives** come from users with a non-empty `usb_copy_p95` or `http_upload_p95` (26.7% of rows). The other
  2,742 users (73% of rows) have zero positives.
- The NaN pattern of the baseline columns (`*_p95`, `*_to_p95_ratio`, and others) reveals whether a user is a
  "candidate". An isnull indicator has AUC ≈ 0.76.
- A simple model was tried with features that the injection **never changes**: baseline presence, today's channel
  activity and multi-activity days, with a by-user split.
  - It reached **AP 0.17 vs. 0.024 at random (7.1× lift)**, without any real attack signal.
  - Within the 995 the lift drops to about 1.9×.

**Approved decision (2026-11-23):**
- The model's scope is **the 995 candidate users only** (400 attackers, 595 benign), for training, validation, test
  and every metric.
- The other 2,742 users are outside the project: they are not treated as benign samples, they enter no experiment,
  and there is no secondary evaluation on the full population.
- This is a deliberate definition of the project's scope (detecting multi-channel exfiltration), not only a fix for the
  bias.
- The pre-training check is in §8.

**C3: the split must be grouped by user.**

- The injection amount is **constant per user-channel**: `count_added` has a single value per user × channel.
- Attack windows span several months (January to May).
- A random row split, or a purely temporal one, would put the same attacker with the same injection amount in both
  train and test. That is identity leakage, and it inflates results.

### 🟠 Medium

- **M1: off-days and post-window days are hard negatives.**
  - Median `cumulative_deviation_5wd`: attack days 5.50, off-days 4.25, after the window 1.50, before the window 1.00,
    benign users 0.50.
  - `y=0` is correct at day level, but rolling features will give "reasonable" false positives on these days.
  - **Fix:** keep `y=0`, but report the FP rate on off-days separately. Add a window- and user-level evaluation: was the
    attack detected, and when?
- **M2: injection happens only on days with normal activity in the same channel.**
  - `count_before>0` holds in 100% of cases. So the binary activity features (`*_active_today`, `active_*_days`,
    `multi_channel_days`) are not changed by the injection, but they are linked to the target through the choice of days.
  - **38.1% of rows** have no activity in any of the three channels, and all of them are `y=0`. These easy negatives
    inflate ROC-AUC.
  - **Fix:** use PR-AUC, and add a report on active days only.
- **M3: uneven cross-distribution.**
  - Example: 5 days × Medium switching has only 3 attackers.
  - Results for small subgroups will be unstable. Report confidence intervals, or merge small groups.
- **M4: reproducibility.**
  - The saved execution states of NB09 and NB12 are not a clean end-to-end run.
  - NB02 contains paths from another machine (`C:\Users\hp\...`).
  - A clean re-run before final delivery is preferable.
- **M5: the manifest.**
  - The SHA failure of the text files is caused by line endings only.
  - Fix the hash after normalising line endings, or add `.gitattributes` (`*.csv binary`).
- **M6: `user`, `day`, `first_logon` and `last_logoff`.**
  - These must not be used as predictors as they are. If wanted, only the logon and logoff hour could be extracted
    from `first_logon` and `last_logoff`.
  - The positive rate varies by month (1.3%–3.0%), which also rules out the date or month as a feature.

### 🟢 Good findings

- No NaN outside the documented columns, no inf, no duplicates. The dictionary matches 143/143.
- `y` matches the ground truth exactly, and day-level labelling is correct: an attacker's non-attacked days are 0.
- The injection was verified independently against the original source: only 8 fields, only 9,080 days, and the
  totals match.
- No temporal leakage: 0 differences before the attack in 83 columns. The baseline comes from 2010 only and was not
  recomputed.
- No weekday pattern. 595 multi-channel benign users were kept to reduce the shortcut.
- Missing values carry meaning ("no history in this channel"). Keep them as NaN with XGBoost/LightGBM; do not turn
  them into zero.

---

## 6. Proposed split and preparation strategy

### 6.1 Main split: by user, with stratification

- Use `StratifiedGroupKFold`, or a manual split **by `user`**, with every row of a user in one split.
- Rolling features are computed within each user, so there is no historical leakage between splits.
- Proposed ratios: 70/15/15 of **the 995 users only**, stratified separately for two groups:
  - attackers (400): about 280/60/60, balanced by duration × stealth, and as far as possible by switching × channels;
  - benign candidates (595): about 417/89/89.
- Prefer 5-fold group CV on train+val to choose hyperparameters, then one test set that stays locked until the end.

*(Later superseded by the final split in `reports/02_split.md`.)*

### 6.2 Optional secondary evaluation: temporal

- A purely temporal split is not recommended as the main split, because one attack window spreads over several months.
- A robustness test can be added: test users only, evaluated on later months (e.g. April–May). Remember that May has
  fewer positives.

### 6.3 Evaluation population and metrics

- **The only evaluation population:** the 995 candidates (103,395 rows, 8.78% positives, 1 : 10.4). There is no
  evaluation on the full population.
- **Metrics:**
  - PR-AUC;
  - Precision/Recall/F1 at a threshold chosen on val;
  - Recall at a fixed FPR;
  - confusion matrix.
- **Breakdowns:**
  - results by stealth, duration, switching and channel configuration;
  - FP rate on off-days and after the window.
- **User and window level:** was the attacker detected, and how many days until the first alert?

### 6.4 Predictor allowlist

The full list of all 143 columns, with the decision and reason for each, is in
[`config/training_predictor_allowlist.csv`](../config/training_predictor_allowlist.csv):

| Decision | Count | Columns |
|---|---:|---|
| `keep` | **97** | Predictors taken from the original file |
| `exclude` | 8 | `y`, `user`, `day`, `first_logon`, `last_logoff`, `is_weekend`, `unique_usb_extensions`, `day_of_week` |
| `exclude_C1` | 15 | **9 companion fields** not changed by the injection: `unique_usb_files`, `unique_upload_domains`, `unique_external_recipients`, `unique_external_domains`, `external_email_bytes`, `total_sent_email_bytes`, `attachment_count`, `emails_with_attachments`, `from_removable_count`. **6 byte-derived columns:** `email_bytes_above_p95`, `email_bytes_to_p95_ratio`, `email_bytes_p50/p90/p95/mad` |
| `exclude_C2` | 23 | **19 raw baseline columns:** `baseline_workdays`, `*_active_days`, `*_active_rate`, `email_count_p50/p90/p95/mad`, `usb_copy_p50/p90/p95/mad`, `http_upload_p50/p90/p95/mad`. **4 ratios with NaN:** `usb_copy_to_p90/p95_ratio`, `http_upload_to_p90/p95_ratio` |
| **New features** | **3** | `E_email_today`, `E_usb_today`, `E_http_today`, defined in [`config/engineered_features.csv`](../config/engineered_features.csv) |
| **Total** | **100** | 97 from the original file + 3 new |

- **C1 dependencies:** the dictionary confirms that none of the 30 or 27 engineered features depends on the fields
  excluded under C1. The only columns derived from them are the six listed above.
- **Baseline columns:** they are excluded only as direct inputs. Their values remain in the original file and are used
  inside 53 of the 97 deviation features, and inside the three new features (§8.2).
- **Remaining NaN:** after this decision, no NaN caused by a missing baseline remains. The only NaN left is in
  `channel_distribution_shift_*`, and it means "a window without any activity".

### 6.5 Processing

- Fit all imputation, scaling and feature selection on train only.
- **NaN:** use the native support in LightGBM/XGBoost, or add a missing indicator plus imputation. Do not turn values
  into zero.
- **Imbalance:**
  - use `scale_pos_weight ≈ 10` (the negative-to-positive ratio within the 995) or class weights;
  - **no SMOTE** on temporal data;
  - tune the threshold on val.
  - *(Later decision: no reweighting by default. `scale_pos_weight` was tested only as a pre-registered ablation and
    was not adopted; see `reports/06_ablations.md`.)*
- Save the list of columns used and the split (the user list of each split) for reproducibility.

---

## 7. Final verdict

The dataset is **structurally sound, consistently labelled and free of temporal leakage**. But it is **not ready to be
trained on "as is"** until three decisions are made:

1. **(C1) ✅ Decided (2026-11-23).** The non-injected companion fields (15 columns) are excluded from the predictors, and
   the injection will not be redone. Details in §6.4 and in `training_predictor_allowlist.csv`.
2. **(C2) ✅ Fully decided (2026-11-23).** The model works on the 995 candidates only. 19 raw baseline columns and 4
   ratios with NaN were excluded as inputs. Three daily per-channel deviation features were added in their place
   (§8.1 and §8.2).
3. **(C3) ✅ Implemented.** A by-user split with stratification, saved in `config/user_split.csv`.

**The original file `behavior_features_r62_ml_final_new.parquet` stays unchanged.** Its SHA-256 was checked and did not
change after the run. The filtering to the 995 and the allowlist are applied by `prepare_dataset.py` in the `Model`
folder.

### Files in the `Model` folder

| File | Description |
|---|---|
| `prepare_dataset.py` | Reads the original file only, filters to the 995, applies the allowlist, computes the three new features, splits the data, and stops at any count mismatch (42 checks) |
| `config/training_predictor_allowlist.csv` | Decision for each of the 143 columns |
| `config/engineered_features.csv` | Definition of the three new features |
| `config/eligible_users_995.csv` | The 995 users, attacker flag and scenario properties (evaluation only, not model inputs) |
| `config/user_split.csv` | Each user → train, val or test (the preliminary split) |
| `data/model_dataset_995.parquet` | 103,395 × 104: `user`, `day`, `split`, `y` and 100 predictors |
| `reports/pre_training_verification.md` | Verification report: 42/42 PASS |

To re-run: `python prepare_dataset.py` from inside the `Model` folder. The result is deterministic (seed=42).

---

## 8. Pre-training verification (C2: the 995 population)

`prepare_dataset.py` ran this check. The full report is `reports/pre_training_verification.md` (42/42 PASS). The
official source of the candidate list is `reports/03_Channel_Behavior_Audit/candidate_behavior_profile.parquet`.

| Requirement | Result |
|---|---|
| Eligible users | ✅ **995** (no duplicates, all present in the feature file) |
| Attackers | ✅ **400**, all within the 995 (zero attackers outside) |
| Benign | ✅ **595** |
| Population rows | 103,395 rows: `y=1` 9,080, `y=0` 94,315, positives 8.78% |
| Excluded users | 2,742 users with **zero** positives, so no attack day is lost |
| No user overlap between train/val/test | ✅ 70/15/15 split, seed=42, attackers stratified by duration × stealth. Train: 282 attackers + 416 benign (72,548 rows). Val: 57 + 90 (15,247). Test: 61 + 89 (15,600). Overlap is **zero** for all three pairs; each user is in exactly one split |
| Allowlist applied | ✅ All 100 columns (97 + 3 new) present. None of them is `y`, `user`, `day`, a date, or any C1 or C2 column |
| Target leakage | ✅ None. The strongest feature within the 995 is `cumulative_deviation_5wd` (AUC 0.942), an expected behavioural signal |

**Eligibility definition:**
- The 995 list matches **exactly** the users with a USB or HTTP baseline in 2010. So eligibility can be decided from
  historical data alone, without looking at the attack period.
- Actual candidate combinations: EU 428, EUH 291, EH 276.

**C2 follow-up within the 995 (the four ratios with NaN):**

| Column | NaN, attackers | NaN, benign | Row-level AUC of an isnull indicator |
|---|---:|---:|---:|
| `usb_copy_to_p90/p95_ratio` | 29.0% | 26.6% | 0.512 (almost no signal) |
| `http_upload_to_p90/p95_ratio` | 29.2% | **52.3%** | 0.442 (weak signal: 0.56 as a separator) |

- After filtering, the strong NaN-pattern signal (AUC 0.76 on the full population) **almost disappears** for USB, and a
  weak effect remains for HTTP. The reason: 52% of benign users have no HTTP baseline, against 29% of attackers.
- For attackers, the NaN follows directly from the channel configuration: E+H → USB empty in 99%; E+U → HTTP empty
  in 100%.
- No attacker was injected in USB or HTTP without a baseline in that channel.

### 8.1 Review of the C2 columns

This review was an ablation on val only, with a linear logistic-regression model. Test was not touched.

**The 19 columns (raw baseline) reveal the attacker from his profile:**

| Column | Ability to separate attacker from benign (user-level AUC) |
|---|---:|
| `email_active_rate` / `email_active_days` | 0.79 |
| `email_count_p90` | 0.76 |
| `http_upload_p90` | 0.66 |
| `usb_*` and `baseline_workdays` | about 0.50 |

| Experiment | AP | User-level AUC |
|---|---:|---:|
| 101 columns (before the decision) | 0.885 | 0.965 |
| 101 + the 19 columns | 0.900 | 0.965 |
| The 19 columns alone | 0.166 | **0.837** |
| Without the four ratios | 0.885 | 0.963 |
| NaN pattern of the four ratios alone | 0.094 (random 0.083) | 0.601 |

**Decision:** exclude the 19 columns and the four ratios. The 19 columns tell the model "who the person is", not "what
he did today". The four ratios add no performance.

### 8.2 Do the remaining features suffice to measure deviation?

- **53 of the 97 are baseline-derived:**
  - `*_above_p95`, `email_count_to_p95/p90_ratio`;
  - `cumulative_deviation_*`, `combined_channel_intensity_*`, `dominant_channel_switches_*`;
  - `channel_distribution_shift_*`, `consecutive_elevated_days_*`, `cross_channel_deviation_days_*`;
  - all 24 `nr_*` features.

  They compare the user with his own normal behaviour, which is the intent. None of them is constant per user.
- **The gap found:**
  - USB and HTTP had no continuous daily per-channel deviation, only the binary flag `*_above_p95`.
  - That flag catches 47% of USB-only attack days and 63% of HTTP-only ones. At High stealth it falls to 21% and 25%.
  - The deviation `E_c > 0` appears on 96% of them.
- **The applied fix:**
  - `E_email_today`, `E_usb_today` and `E_http_today` were added, with the same `E_c` definition used in the `nr_*`
    features. They are 0 instead of NaN when there is no baseline.
  - The script checks that their sums over the windows equal `cumulative_deviation_{5,10,20,30}wd` exactly
    (difference below 1e-13).
- **Effect on val:** overall AP 0.885 → 0.887; USB-only attacks 0.505 → 0.521.
- ⚠️ **Note for training:** the model is clearly weaker on USB-only attacks (AP ≈ 0.52) and HTTP-only attacks (≈ 0.65)
  than on email attacks (≈ 0.87). Results must be reported per channel.

### 8.3 Missing values and the meaning of zero: decision (2026-11-24)

- **`channel_distribution_shift_*` stays NaN.**
  - NaN occurs only when `external_email_count_{w}wd + usb_copy_count_{w}wd + http_upload_count_{w}wd = 0`, that is, a
    window with no activity, in all four windows.
  - No attack day is among these rows. Shares: 2.09% (5wd), 0.71% (10wd), 0.48% (20wd), 0.46% (30wd).
  - The NaN carries a legitimate behavioural meaning (inactivity). It has nothing to do with baseline presence or with
    selection bias.
  - It was not turned into 0, so that it does not mix with "distribution identical to history" (exactly 0 in 2–6 rows
    per window).
  - In training it is left to the native support of LightGBM/XGBoost. If a model that does not accept NaN is used,
    handling is fitted on train only.
- **`E_*_today = 0` when there is no baseline.**
  - In the 995 data there is no activity in a channel without a baseline (0 rows for USB and HTTP).
  - So zero does not mix "normal" with "no history".
  - No "no history" flag was added, because it would bring back the C2 pattern without any behavioural information.
- **Known limit:** the first use of a channel the employee never used before was not simulated, and the model cannot
  learn it. In real use it needs a separate alert rule.
- **Open gap:** the baseline has no feature that describes the employee's 2010 habit of switching between channels.
  The switching features therefore measure "how much he switches now", not "whether he switches more than usual".
  Analysed in §8.4.

### 8.4 2010 switching baseline analysis

(2026-11-24. Analysis only; nothing was added to the data.)

**Method:**
- A function was written that recomputes `dominant_channel_switches_*` and `nr_alt_*` on 2011 data. It matched the
  existing features **exactly** (0 differences).
- The same function was applied to the 2010 workdays of each of the 995 users (250,882 days, at least 250 per user).
  This gives each user's "normal switching rate".
- 207 users have fewer than 10 consecutive deviation-day pairs in 2010, so their `nr_alt` estimate is weak.

**Results:**

| Question | Result |
|---|---|
| Does normal 2010 switching differ between attackers and benign users? | **Yes.** User-level AUC: `h_alt_mean_20` 0.74, `h_switch_mean_20` 0.72. Attackers already switched more before the attack. The switching level assigned in their scenario follows their nature: median 7.49 for High, 6.03 for Low, 5.12 for benign |
| Does the attack raise switching (by most active channel) above the employee's habit? | **Hardly.** Only 36% of attackers switch more inside the attack window than their 2010 mean. Exceeding the historical P95 happens on 3.1% of attack days vs. 3.5% of normal days. AUC of `dominant_channel_switches_20wd` within attackers = 0.52 |
| Does the attack raise alternation (by most deviating channel, `nr_alt`)? | **Yes.** 74% of attackers are above their history. At High switching: 0.49 inside the window, against 0.33 in 2010 and 0.29 outside the window |
| Drift between the two years | Switching in 2011 outside attack windows is lower than in 2010 for everyone, so comparing with 2010 mostly gives negative values |

**Comparison on val** (linear logistic regression; test not touched):

| Features | AP | User-level AUC |
|---|---:|---:|
| The current 100 | 0.887 | 0.963 |
| + 3 relative alternation features (`nr_alt` − 2010 habit) | 0.892 | 0.959 |
| + 7 relative features (alternation and switching) | 0.894 | 0.959 |
| 2010 switching profile **alone** | 0.136 | **0.798** |

**Proposed decision: do not add them now.**
1. The gain is small (+0.005 to +0.007 AP), within the noise margin with 147 users in val.
2. The relative features let the model infer the employee's own switching profile (raw − relative = 2010 value). That
   profile alone reveals the attacker (AUC 0.80), a shortcut of the same kind as C2.
3. The drift between 2010 and 2011 biases the comparison.

**What this means for MUBSIR:**
- The "switching" signal in this data lies in **deviation-based alternation** (`nr_alt_*`, `nr_gap_alt_2_*`,
  `nr_handoff_*`), which is already among the 100 features.
- `dominant_channel_switches_*` carries almost no signal at day level.
- When reporting results by `switching_level`, note that the assigned switching level is linked to the employee's
  behaviour before the attack.
- **Alternative to try later with LightGBM:** compare alternation with the employee's recent history in 2011 itself
  (a moving earlier window). This avoids the year drift but needs the same shortcut check.

*(Later tested in the pre-registered study `studies/switching_baseline/`, using the switching rate rather than counts:
no meaningful added value.)*

---

## 9. Plain-language explanation: read this first

**1. What is this data?**
A daily record per employee: how many external emails he sent, how many files he copied to USB, and how many files he
uploaded to the internet. Each row = one employee on one workday.

**2. How were the attacks made?**
400 employees were chosen and made into simulated "attackers". On specific days, slightly more events were added for
them, for example a few extra emails or files. The amounts were kept small so they would not be easy to spot; this is
"stealth". The channel was varied from day to day; this is "switching".

**3. What does `y` mean?**
- `y=1`: an attack was actually added on this day.
- `y=0`: nothing was added on this day, even if the employee is an attacker.
- Off-days (the attacker's rest days inside the attack period) are `y=0`, because no exfiltration happened on them.

**4. Is the data sound?**
Yes. There are no duplicate rows and no strange values, the labels are 100% correct, and the model cannot "see the
future".

**5. What was the first problem (C1), and how was it solved?**
When extra files were added, the "number of files" rose but the "number of distinct files" and the bytes did not
change. That does not happen in reality, and the model would have detected the attack from this mistake rather than
from behaviour.
**Fix:** these 15 columns were removed from the model's inputs.

**6. What was the second problem (C2), and how was it solved?**
All attackers were chosen from 995 employees who use more than one channel. If the other 2,742 were kept, the model
would learn "this employee is from the special group" instead of "this employee is exfiltrating".
**Fix:** work on the 995 only. This is the project's goal anyway: detecting exfiltration by people who use more than one
channel.

**7. What was the third problem (C3)?**
If days of the same attacker were put in both training and test, the model would memorise him.
**Fix:** each employee goes entirely into one group: train, val or test.

**8. What exactly will be trained?**
- **Data:** 995 employees and 103,395 days, of which 9,080 are attack days (about 9%).
- **Inputs:** 100 columns: 97 from the original file, plus 3 new features that measure "how much each channel rose
  today above this employee's normal".
- **Split:** 70% of employees for training, 15% for tuning (val), 15% for the final test. *(Later replaced by the final
  split in `reports/02_split.md`.)*

**9. How is success measured?**
Not by accuracy: 91% of days are normal, so a model that says "everything is normal" would score 91%. Instead:
- **PR-AUC.**
- **Precision:** of the alerts, how many are correct?
- **Recall:** of the attacks, how many were detected?

Results are shown per stealth level and per duration, and the evaluation checks whether each attacker was detected and
after how many days.

**10. What was left before starting?**
- ✅ All decisions (C1, C2, C3) were made.
  - The raw baseline columns were removed from the inputs, because they revealed "who the employee is" rather than
    "what he did".
  - They are still used to compute the deviation features.
- ✅ The preparation files are ready in the `Model` folder. The training-ready data is in
  `data/model_dataset_995.parquet`, and the original files were not touched.
- Next step: install the training libraries (scikit-learn and LightGBM), then build the first model.
