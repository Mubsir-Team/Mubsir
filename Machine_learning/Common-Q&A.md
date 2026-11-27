# MUBSIR Model v2: Common Q&A

Short answers first. Click **Details** for the full explanation, the numbers and where they come from.

**Words used in this document**
- **Dev users:** the 859 employees used to build and check the model.
- **Test users:** the 136 employees kept aside until the very end.
- **Benign:** an employee who was not given a simulated attack.
- **Workday:** a weekday on which the employee logged on.

---

## 1. Dataset and features

### 1. Why were 97 of the 143 delivered features kept, and why were 3 new ones added?

46 columns were removed because they were identifiers, constant, or would let the model "cheat". The 3 new features
fill a gap: a daily, per-channel measure of how far the employee is above their own normal level.

<details><summary>Details</summary>

The delivered feature file has **143 columns**.

| Decision | Columns | Why |
|---:|---|---|
| **Kept** | 97 | Behavioural features that describe what the employee did |
| Removed | 8 | Identifiers, dates and constant columns: user, day, logon and logoff times, weekday, weekend flag (always 0), USB extensions (always 0), and the label itself |
| Removed | 15 | **Simulation flaw.** When attacks were injected, the counts went up (more files, more uploads, more e-mails), but the matching fields did not (number of distinct files, distinct domains, distinct recipients, bytes). A real leak raises both. A model could detect "count rose but distinct files did not", which only exists in the simulation. Within attackers alone, these ratios separated attack days with AUC ≈ 0.85 |
| Removed | 23 | **Profile instead of behaviour.** 19 columns describe the employee's 2010 habits (for example their usual e-mail volume). Alone, they identified attackers with AUC 0.84 *without looking at any activity*, because of how attackers were chosen in the simulation. The other 4 are ratios that are often empty and added no performance |
| **Added** | 3 | `E_email_today`, `E_usb_today`, `E_http_today`: how far today's count is above the employee's own normal (median), scaled by their normal range |

**Why the 3 new features:**
- Before them, USB and HTTP only had a yes/no flag "above the 95th percentile".
- That flag caught 47% of USB-only attack days and 63% of HTTP-only ones, falling to 21% and 25% at high stealth.
- A continuous deviation is above zero on 96% of those days.

**Total:** 97 + 3 = **100 features**.

The removed baseline values are still used *inside* 53 of the deviation features: they compare today with the
employee's own normal. They are just not given to the model directly.

Source: `reports/dataset_audit.md` (sections 5, 6.4, 8.1, 8.2), `config/training_predictor_allowlist.csv`.
</details>

---

## 2. Data split

### 2. What are the training and test proportions?

859 employees (86.3%) for development and 136 (13.7%) for the final test. Each employee is in only one part.

<details><summary>Details</summary>

| | Development | Test | Total |
|---|---:|---:|---:|
| Employees | 859 (86.3%) | 136 (13.7%) | 995 |
| Attackers | 348 | 52 | 400 |
| Benign | 511 | 84 | 595 |
| Share of attackers | 40.5% | 38.2% | 40.2% |
| Rows (employee × workday) | 89,471 | 13,924 | 103,395 |

- The split is **by employee**: all of an employee's days are in the same part.
- Inside development, the data is split again 5 ways, 3 times (cross-validation). All model choices were made this way.
- The test set was opened **once**, at the very end.

Source: `reports/02_split.md`.
</details>

### 3. Why 86.3% / 13.7% and not exactly 85% / 15%?

The target was 15%, but three rules made exactly 15% impossible. 13.7% was the closest valid split.

<details><summary>Details</summary>

1. **Attackers with the same attack schedule stay together.**
   - Some attackers share an identical schedule (400 attackers form 318 groups).
   - If one were in training and another in test, the model would already "know" that schedule, and the test result
     would look better than reality.
   - A group goes into the test set whole, or not at all.
2. **147 employees could not be in the test set.**
   - Before the final split, the data was split 70/15/15 once.
   - The 147 employees of that first validation part were used to decide which features to keep (question 1).
   - Testing on them would favour the features chosen with their data. They stayed in development.
3. **The test set must contain every kind of attack in similar proportions:** every stealth level, duration, rotation
   level and channel combination.

300 random seeds were tried. The one whose test set was closest to 15% in **every** category was kept, judged on
composition only, never on model scores. The rule written beforehand accepted 13% to 17%.

Source: `reports/02_split.md`, `pipeline/02_make_split.py`.
</details>

### 4. Why 85/15 rather than 80/20 or 90/10?

It balances two needs: a test set large enough to give a stable result, and as much data as possible for training.

<details><summary>Details</summary>

The table assumes the development detection rate (90.6%):

| Test share | Attackers in test | Left for development | Width of the 95% range for detection | One attacker changes the result by |
|---|---:|---:|---:|---:|
| 10% | 40 | 895 | 20.9 points | 2.5 points |
| **15%** | **60** | **846** | **16.7 points** | **1.7 points** |
| 20% | 80 | 796 | 14.3 points | 1.2 points |
| 30% | 120 | 697 | 11.1 points | 0.8 points |

- **Up to 15%:** going from 10% to 15% improves precision clearly.
- **Beyond 15%:** the gain slows down. 30% narrows the range by only 5.6 more points but removes 149 employees from
  training.
- **The real test result:** 48 of 52 attackers. Its 95% range is 81.5%–97.9%, which is precise enough to show that
  detection is high.
</details>

### 5. Why cross-validation instead of a fixed validation set?

A single validation set gives a result that depends on luck: which employees happen to be in it. Cross-validation is
about 6 times more stable and uses every development employee.

<details><summary>Details</summary>

**Cross-validation:** the 859 development employees are split into 5 parts. The model is trained on 4 parts and
checked on the 5th, five times, so every employee is checked once. This is repeated 3 times with different splits.

Detection at threshold 85:

| | One validation part (~172 employees) | 5 parts combined |
|---|---|---|
| Attackers detected | from **85.9%** to **94.2%** (8.3-point spread) | from **89.9%** to **91.4%** (1.4-point spread) |
| Benign employees alarmed | from **1.0%** to **6.9%** | 3.1% on average |

With one fixed validation set, the reported result could differ by 8 points just by chance.
</details>

### 6. What statistical evidence supports the split?

Three checks:
- the test set is large enough (question 4);
- it matches the development data in every category;
- keeping it aside cost almost nothing (question 7).

<details><summary>Details</summary>

**Test vs development composition** (chi-square test; a high p-value means no difference):

| Category | Test share per category | p |
|---|---|---:|
| Stealth level | 12.7%–13.5% | 0.97 |
| Attack duration | 8.0%–14.3% | 0.92 |
| Rotation level | 12.0%–13.5% | 0.92 |
| Channel combination | 11.1%–14.4% | 0.86 |
| Benign employees' channels | 13.7%–14.4% | 0.99 |
| Attacker vs benign | 13.0%–14.1% | 0.68 |

Share of positive days: development 13.54%, test 13.28%.

The test set is a faithful small copy of the data, so its result reflects real performance.
</details>

### 7. How much detection does keeping the test set aside cost?

About **half a point**. Training on 86% instead of 100% of the employees (the same ratio as 859 / 995) lowered
detection by 0.57 points. That is well below the 2-point limit set in advance.

<details><summary>Details</summary>

A pre-registered study retrained the model on growing shares of the training employees, always checked on the same
employees:

| Share of training data | 30% | 45% | 60% | 75% | **86%** | **100%** |
|---|---:|---:|---:|---:|---:|---:|
| Attackers detected (3% false alarms) | 86.1% | 88.3% | 89.3% | 89.4% | **90.4%** | **91.0%** |

- **Cost of keeping the test aside:** +0.57 points (90% range +0.05 to +1.15). At 1% false alarms the cost is
  +1.2 points: real, but still below 2 points.
- **More data still helps, slowly:** +1.7 points from 60% to 100%.
- **Projection (a fitted curve, not a measurement):** adding the 136 test employees to the final model would have
  added about +0.3 points.

Source: `studies/learning_curve/REPORT.md`.
</details>

---

## 3. The model

### 8. What question does the model answer?

> **"Did this employee exfiltrate data on at least one day in the last 10 workdays?"**

The answer is a number from 0 to 100, computed every day for every employee. Above 85, an alert is raised.

<details><summary>Details</summary>

The model does **not** answer these questions:
- **"Is this employee a bad person?"** It judges recent behaviour, not the person.
- **"Did they exfiltrate today exactly?"** It looks at a 10-workday window.
- **"What did they exfiltrate?"** It sees activity counts and deviations, not file contents.
</details>

### 9. Why LightGBM and not another algorithm?

Five algorithms were compared under a rule written before training. LightGBM was the default, and no other algorithm
beat it by the required margin. It is also the fastest of the strong ones.

<details><summary>Details</summary>

Development cross-validation, averaged over the 5, 10 and 20-day windows:

| Algorithm | Detection at 1% false alarms (daily) | PR-AUC | Seconds per training |
|---|---:|---:|---:|
| CatBoost | 0.863 | 0.943 | ~5.7 |
| **LightGBM** | **0.860** | **0.943** | **~1.9** |
| XGBoost | 0.857 | 0.942 | ~5.4 |
| Random forest | 0.803 | 0.910 | ~3.7 |
| Decision tree | 0.744 | 0.877 | ~1.9 |

- **The rule written beforehand:** replace LightGBM only if a challenger is at least +0.005 better in detection and not
  worse in PR-AUC.
- **CatBoost** was +0.004 better, below the margin, so LightGBM stayed. CatBoost is reported as a strong alternative.
- **Hyperparameter tuning** was also tried, with nested cross-validation. It did **not** help: 0.8185 tuned vs 0.8209
  default. The small apparent gain inside tuning was selection bias, so the default settings were kept.

Source: `reports/04_cv_compare.md`, `reports/05_tune.md`, `config/selection_rules.json`.
</details>

### 10. Why one model instead of three?

The first version used three models (5, 10 and 20-day windows) and took the highest score. Version 2 uses one model,
because the three-model setup added almost nothing and broke the meaning of the score.

<details><summary>Details</summary>

1. **No real gain:** the three models together caught **one** more attacker, which is within noise.
2. **The score keeps its meaning:** one model gives a calibrated probability. The highest of three probabilities is
   inflated and is no longer a probability.
3. **It fits the database:** there is one score column.
4. **Simpler to run and explain:** one file, one score, one explanation.
</details>

### 11. Why does the model look at the last 10 workdays, and what were the results for 5, 10 and 20?

10 was the default in the rule written beforehand. Neither 5 nor 20 beat it by the required margin. 20 was worse on
every measure, and 5 was about the same.

<details><summary>Details</summary>

LightGBM, development cross-validation:

| Window | User AUC | Attackers detected (1% of benign employees alarmed) | Median days to detect |
|---|---:|---:|---:|
| 5 days | **0.988** | 86.4% | 3 |
| **10 days** ✅ | 0.987 | **87.1%** | 3 |
| 20 days | 0.985 | 85.1% | 4 |

Source: `reports/04_cv_compare.md`.
</details>

### 12. Can I write "the last h workdays" instead of "the last 10 workdays"?

Yes, when describing the **method**, as long as you say what h is. When describing the **final model**, write 10.

<details><summary>Details</summary>

- **The method is general:** windows of h = 5, 10 and 20 were tested, and h = 10 was chosen by a rule written
  beforehand.
- **The final model was trained for 10 days only:** its score answers the 10-day question. Changing h needs retraining.

Suggested wording:

> "The model answers: did the employee exfiltrate data on any day in the last *h* workdays, with *h* = 10 in the final
> model?"
</details>

### 13. How does the model detect exfiltration while it is happening?

It compares each day with the employee's own normal behaviour, adds up the small increases over several days and
channels, and turns all of this into one score.

<details><summary>Details</summary>

1. **Compare with normal.** For each channel (external e-mail, USB copy, web upload), it measures how far today's
   activity is above the employee's 2010 normal.
2. **Accumulate.** A careful attacker raises activity only slightly each day. The model looks at the last 5, 10, 20 and
   30 workdays:
   - How much has the deviation built up?
   - How many days in a row was activity high?
   - Was more than one channel involved?
3. **Score.** The 100 features go into LightGBM, which outputs a score from 0 to 100, with the top 5 reasons (SHAP).

**Why it works:** one small increase looks normal, but the accumulation over days and channels does not. The single
most important feature is the accumulated deviation over the last 10 workdays.

**Speed of detection:** most attackers are caught about 3 workdays after the attack starts (median 3, at threshold 85).

Source: `reports/10_shap.md`, `reports/11_latency.md`.
</details>

### 14. Why can't the model predict future exfiltration?

It only sees the past and today, and before an attack starts there is nothing in the behaviour to learn from. MUBSIR
**detects** exfiltration; it does not **predict** it.

<details><summary>Details</summary>

1. **By design, it never sees the future.**
   - Every feature uses only today and earlier days.
   - In real use, tomorrow's data does not exist yet.
   - The data audit checked this and found no use of future data.
2. **No signal before the attack.**
   - Before the first attack day, attackers behave exactly like everyone else: the audit found zero differences in
     their data.
   - So there is no pattern that says "this employee will leak next week".
3. **Profiling was removed on purpose.** Some 2010 habits did separate attackers, but only because of how attackers
   were chosen in the simulation. Using them would judge **who someone is**, not **what they did**. That is unfair, and
   it would not hold in reality.
</details>

---

## 4. The score and the thresholds

### 15. What exactly does the 0–100 score represent?

> **The calibrated probability (×100) that the employee exfiltrated data on at least one day in the last 10 workdays.**

A score of 85 means roughly an 85% chance. Suggested name: **Exfiltration Likelihood Score**.

<details><summary>Details</summary>

**Calibration.** The raw model output is adjusted so that the number matches reality (Platt scaling, fitted on
development cross-validation only). On development data, predicted and observed rates are close:

| Score range | Average predicted | Actually positive |
|---|---:|---:|
| 0–10 | 1.1% | 1.1% |
| 40–50 | 45.0% | 43.0% |
| 60–70 | 65.1% | 60.5% |
| 80–90 | 85.5% | 85.1% |
| 90–100 | 98.5% | 98.9% |

**What it is not:**
- **Not the probability that the employee is an attacker:** it is about the last 10 workdays.
- **Not a raw "deviation score."** Deviation is what the features measure; the score is the model's judgement of
  whether that deviation looks like exfiltration.
  - The two are strongly related (correlation 0.78), but the score is clearly better at detection: at 1% of benign
    employees alarmed, the score catches 87.4% of attackers, while raw deviation alone catches 73.0%.
  - A large deviation spread evenly can stay low, and a small one that builds up over several days and channels can
    score high.

**One limit:** the probability is correct for the attacker share in the development data (40%). In a real company
attackers are much rarer, so read the score as a ranking and alert level, not as an exact real-world probability.

Source: `reports/07_calibrate_fit.md`, `model/manifest.json`.
</details>

### 16. Why are the thresholds 85 and 60?

They are fixed project decisions agreed by the team before training. They were not tuned on the data, so the results
are not "tailored" to the thresholds.

<details><summary>Details</summary>

| Level | Score | Action |
|---|---|---|
| Low | 60 or below | none |
| Medium | above 60, up to 85 | watch |
| High | above 85 | **alert** |

- **Where they come from:** version 1 of the project defined these three levels for the database and the dashboard.
  Version 2 kept them, and froze them before the test set was opened.
- **What they mean:** because the score is calibrated, 85 ≈ an 85% chance (almost certain, so alert), and 60 ≈ more
  likely than not (worth watching).

| | At 85 | At 60 |
|---|---:|---:|
| Development: attackers detected | 90.6% | 95.4% |
| Development: benign employees alarmed | 3.1% | 10.7% |
| Test: attackers detected | 48 of 52 | 51 of 52 |
| Test: false alarms | 1 of 84 | 6 of 84 |

60 catches more attackers, but gives about 3 times more false alarms, so alerts are raised at 85.
</details>

### 17. Why does the medium level start at 60?

There is no documented reason for exactly 60; it was a team decision. Data analysis shows that 60 is a reasonable
middle point, and that any value between about 55 and 70 could be defended.

<details><summary>Details</summary>

Development data only, for different medium thresholds:

| Medium starts at | Attackers reaching it | Benign employees reaching it (of 511) | Attackers seen only in medium (never reach 85) | Days earlier than the alert |
|---:|---:|---:|---:|---:|
| 50 | 96.3% | 73 (14.4%) | 20 | 2.0 |
| 55 | 96.0% | 64 (12.5%) | 19 | 1.9 |
| **60** | **95.4%** | **55 (10.7%)** | **17** | **1.7** |
| 65 | 94.2% | 45 (8.9%) | 12 | 1.5 |
| 70 | 93.6% | 39 (7.7%) | 10 | 1.2 |
| 85 (alert) | 90.6% | 16 (3.1%) | — | — |

- **No sharp break at 60:** each 5-point step trades about 1% of attackers for about 9 benign employees.
- **What medium at 60 adds:** 17 attackers (about 5%) who never reach 85, warning about 1.7 workdays earlier, and about
  70% of the days in the medium level belong to a real attack window.
- **The cost:** about 11% of benign employees reach medium at some point. That is why medium means "watch", not "alert".
</details>

---

## 5. Results

### 18. What are the final test results?

On the 136 test employees, at the alert threshold of 85: **48 of 52 attackers detected, and 1 false alarm out of 84
benign employees.**

<details><summary>Details</summary>

| Measure | Value | Meaning |
|---|---:|---|
| **Recall** | **92.3%** (48/52) | Of all attackers, how many were caught |
| **Specificity** | **98.8%** (83/84) | Of all benign employees, how many were left alone |
| **Precision** | **98.0%** (48/49) | Of all alerts, how many were real attackers |
| **F1-score** | **95.0%** | Balance of precision and recall |
| **Accuracy** | **96.3%** (131/136) | Share of employees classified correctly |
| **User ROC-AUC** | **98.9%** | How well the model ranks attackers above benign employees, at any threshold |

**Caution:** the test set has 38% attackers, far more than a real company. In real use, precision, F1 and accuracy
will be lower; recall and specificity do not depend on that share.

Source: `reports/13_user_classification.md`, `reports/08_test_evaluation.md`.
</details>

### 19. What is the confusion matrix at threshold 85?

|  | Predicted attacker | Predicted benign |
|---|---:|---:|
| **Actual attacker (52)** | **48** caught | **4** missed |
| **Actual benign (84)** | **1** false alarm | **83** correct |

<details><summary>Details</summary>

- **Correct:** 48 + 83 = **131** of 136.
- **Errors:** 4 missed attackers + 1 false alarm = **5**.
- **Who is missed:** the stricter count only credits alerts inside the attack window (47 of 52). On that count, all 5
  misses are **short attacks**: 3 of the 4 five-day attacks and 2 of the 10 ten-day attacks. 3 of the 5 are high
  stealth. Every attack of 20 days or longer was caught. Source: `reports/09_breakdown.md`.
- Every measure in question 18 is computed from this table.
</details>

### 20. What is the difference between results per employee and results per day, and which belongs in the report?

Per employee asks "was this attacker caught?". Per day asks "was each attack-window day flagged?". The **per-employee**
result is the main one, because an alert is about a person. Per-day results are supporting detail.

<details><summary>Details</summary>

Test set, threshold 85:

| | Per employee | Per day |
|---|---:|---:|
| Recall | 92.3% | 73.6% |
| Precision | 98.0% | 98.8% |
| Specificity | 98.8% | 99.9% |
| Accuracy | 96.3% | 96.4% |

**Why per-day recall is lower:** the first days of an attack are hard to catch, because the deviation has not built up
yet. One alert is enough to catch the attacker, even if the earliest days score low.

**For the report:** lead with the per-employee results, and add the per-day results as detail on how early and how
consistently the model flags attack days.

Source: `reports/13_user_classification.md`, `results/14_row_threshold_metrics.csv`.
</details>

### 21. How do we know the model is not overfitting or underfitting?

- **Not overfitting:** results on the never-seen test set match the development results almost exactly.
- **Not underfitting:** the model beats simpler methods clearly, and the learning curve shows only small gains from
  more data.

<details><summary>Details</summary>

**Development vs test** (overfitting would show as much worse test results):

| Measure | Development (cross-validated) | Test (seen once) |
|---|---:|---:|
| PR-AUC (daily) | 0.944 | 0.955 |
| User ROC-AUC | 0.987 | 0.989 |
| Attackers caught at 85 (inside the attack window) | 90.6% | 90.4% |
| Median days to detect | 3 | 3 |

**Guards against overfitting:**
- **Early stopping:** training stops when results on separate employees stop improving.
- **No tuning:** tuning was rejected because its gain was selection bias (question 9).
- **Frozen decisions:** every threshold and rule was fixed before the test set was opened.

**Against underfitting:**
- **Simpler methods lose clearly:** the model beats a decision tree (0.877 PR-AUC) and a random forest (0.910), and
  beats raw deviation alone (73% vs 87% detection, question 15).
- **More data helps only a little:** going from 60% to 100% of the data adds 1.7 points (question 7). The model is
  close to what this data can give.
</details>

### 22. How is the processing-time requirement (PR-01) addressed?

The model itself is very fast: it scores all employees for one day in about **12 milliseconds**, or about 0.3 seconds
with explanations. The full processing time, including log aggregation, feature extraction and the database, will be
measured once those parts run in the system.

<details><summary>Details</summary>

> PR-01: "The system shall complete scheduled log processing and detection within a defined processing window under
> the expected workload."

**Model scoring only**, measured on development data (laptop, 16 CPU cores):

| Task | Rows | Time |
|---|---:|---:|
| Load the model and verify its integrity | — | 0.08 s |
| Score one busy day | 859 employees | 11.5 ms |
| Score and explain (top 5 reasons) one busy day | 859 employees | 0.32 s |
| Score the whole development period | 89,471 rows | 0.38 s |

**Not yet measured:**
- reading the raw logs;
- daily aggregation;
- feature extraction from the database;
- writing the results back.

These steps run in the production system and will be timed after integration. The final PR-01 figure will come from
that measurement.
</details>

---

## 6. Channel rotation

**Rotation** means the attacker moves between channels (e-mail, USB, web upload) from one day to the next, so that no
single channel looks unusual.

### 23. How was channel rotation simulated? Does using two channels on the same day count as a rotation?

Each attacker has a rotation level: low, medium or high, meaning how often the main channel changes between attack
days. Using two channels on the same day is **not** a rotation; it is counted separately as simultaneous use.

<details><summary>Details</summary>

- **Rotation** = the main channel of one attack day differs from the main channel of the previous attack day.
- **Share of attack days where the main channel changes:**
  - low: 10–25%;
  - medium: 30–45%;
  - high: 50–65%.
- **5-day attacks:** low, medium and high mean exactly 1, 2 and 3 changes.
- **Which channels:** attacks were injected only into channels the employee already used normally, within their
  assigned channel combination. No new channel was added to create rotation.
- **Two or three channels on the same day** are recorded as simultaneous days: 970 days with two channels, and 37
  with three.
</details>

### 24. Do dedicated rotation features improve detection?

No. Rotation information is already captured by the overall deviation across channels. Adding new dedicated rotation
features changed detection by −0.8 points, which is no improvement.

<details><summary>Details</summary>

**Studies:**
- **Rotation study:** MUBSIR catches more rotating attackers than separate per-channel monitors: +2.9 points at 3%
  false alarms, and +8.1 points for high-rotation attackers.
  - But a simple model given all channels together, without any rotation features, did just as well.
  - So the advantage comes from **looking at all channels together**, not from rotation features.
- **Controlled rotation study:** the same activity was arranged as rotating vs single-channel blocks, and new
  rotation features were tested.
  - Change: −0.8 points (95% range −2.8 to +1.2), so no added signal.

Source: `studies/rotation/REPORT.md`, `studies/controlled_rotation/REPORT.md`.
</details>

### 25. Does comparing rotation with the employee's normal habit help?

No. Features that measure "is this employee rotating more than usual?" (based on their 2010 rotation rate) changed
detection by −0.5 points, which is no improvement.

<details><summary>Details</summary>

- **Habit measure:** each employee's normal rotation **rate** in 2010. The rate was used rather than the raw count,
  because the count alone could identify attackers.
- **Result:** −0.5 points (95% range −1.3 to +0.4), with no benefit in any subgroup.
- **Why:** in the simulation, attackers' rotation during the attack is close to their own normal habit. Their 2010
  rotation already followed their assigned level. So "more than usual" carries little signal here.

Source: `studies/switching_baseline/REPORT.md`.
</details>

### 26. Does giving rotation features more weight in training help?

No, it hurts. Forcing the model to rely more on rotation features lowered detection, by up to 2.8 points at the
highest weight. Removing them completely changed nothing, which confirms they are redundant.

<details><summary>Details</summary>

Attackers detected at 3% false alarms:

| Rotation features | Detected |
|---|---:|
| Removed (weight 0) | 91.2% |
| **Normal (original model)** | **91.0%** |
| Weight ×2 | 90.0% |
| Weight ×5 | 90.4% |
| Weight ×10 | 88.2% (−2.8 points, 95% range −4.3 to −1.4) |

- **Hardest hit:** rotating attacks dropped from 83% to 60% at weight ×10.
- **Share of the model's decisions:** in the original model, rotation features account for about 6%.

Source: `studies/switching_weight/REPORT.md`.
</details>

### 27. Why did rotation features alone catch rotating attacks less often than single-channel attacks?

Because the model learned that very frequent alternation is *not* typical of attacks in this data. Strong rotation
therefore pushes the score down, while single-channel blocks build up in a way the model links to attacks.

<details><summary>Details</summary>

A model given **only** rotation features caught **15.5%** of rotating attacks but **37.2%** of single-channel block
attacks.

**Why:**
- **Extreme alternation is rare on attack days:** days with very high alternation are 8.2% attack days, against 13.5%
  overall.
  - So the model learned that extreme alternation lowers the score.
  - For the main alternation feature, its contribution was **negative** for rotating attacks (−0.22) and **positive**
    for block attacks (+0.41).
- **Overall effect:** the total push from rotation features was larger for block attacks (2.41) than for rotating
  attacks (1.75).

This is a descriptive analysis, done after the study, and it is not a formal test. It explains why rotation features
alone are weak, and why the full model relies on accumulated deviation instead.

Source: `studies/switching_weight/posthoc_switching_only.py`, `studies/switching_weight/REPORT.md`.
</details>

---

## 7. Dashboard

### 28. Can the dashboard show three probabilities for different periods?

Not with the current model: it produces one probability, for the last 10 workdays. Showing three probabilities would
need two more models.

<details><summary>Details</summary>

**Option 1 (recommended, no new training):** show the same score over different periods:
- the score today;
- the highest score in the last 5 workdays;
- the highest score in the last 20 workdays;
- a daily trend chart.

Call these "highest score", not "probability". The highest value over a longer period rises naturally, even for
benign employees.

**Option 2:** train two more models (5 and 20 days), calibrate and validate them, and add two database columns. This
is what version 1 did. It was dropped because it caught only one extra attacker (question 10).
</details>

### 29. What exactly should the dashboard show?

An overview page (open alerts and the employee list), and an employee page with:
- the score and level;
- the recent highest scores;
- a trend chart;
- the top 5 reasons;
- channel activity against normal;
- the alert status.

<details><summary>Details</summary>

**Overview page**
- **Open alerts:** employees above 85, split into critical (above 92.63) and high.
- **Employee list:** sorted by today's score, with name, department, score, level and alert status.
- **Distribution:** how many employees are low, medium and high.

**Employee page**
1. **Today's score** (0–100), its level (low ≤ 60 · medium 60–85 · high > 85), and one fixed line: *"Probability
   that the employee exfiltrated data on any day in the last 10 workdays."*
2. **Highest score** in the last 5 workdays and in the last 20 workdays.
3. **Trend chart** of the last 30 workdays, with a dashed line at 85.
4. **Top 5 reasons** (from the stored SHAP explanation), in plain words with an up or down arrow. Example: "↑ More
   external e-mails than usual over the last 10 days."
5. **Channel activity table:** for external e-mail, USB copy and web upload, show today's count, the normal (2010
   median) and the normal upper limit (2010 95th percentile), and flag any value above the limit.
6. **Alert and review:** status, priority, analyst and notes.

**Do not show:**
- three probabilities for 5, 10 and 20 days;
- the words "employee risk" or "probability of being an attacker";
- the raw score before calibration.
</details>

### 30. Can today's score equal the highest score of the last 10 workdays?

Yes, when today's score is the highest of those days. Otherwise the highest score is larger. The highest score is
always greater than or equal to today's score.

<details><summary>Details</summary>

| Day | 1 | 2 | 3 | … | 9 | 10 (today) |
|---|---:|---:|---:|---|---:|---:|
| Score | 40 | 55 | 91 | … | 70 | 65 |

Here today's score is **65**, and the highest score of the last 10 workdays is **91**.

**Why the dashboard uses 5 and 20 instead of 10:**
- Each daily score already looks back over its own 10 workdays.
- So "highest of the last 10 workdays" actually reflects behaviour up to 19 workdays back, and it is easy to confuse
  with today's score.
- Using 5 and 20 avoids that confusion.
</details>

---

## 8. Integrity and security

### 31. Was the test set ever used for decisions?

No. Every choice (features, algorithm, window, calibration and thresholds) was made on development data. The test set
was opened once, at the end, after a 24-point audit.

<details><summary>Details</summary>

- **Rules first:** the selection rules were written and hashed before any training. Changes were recorded before use.
- **Audit before the test:** a 24/24 PASS check was run before the test set was opened. The model, dataset, split and
  rules were frozen with their hashes (`results/pretest/freeze.json`).
- **Opened once:** the test set was opened on 2026-11-24, and its scores were saved once. No decision was changed
  after seeing them.
- **Later studies used development data only:** all six research studies were run after the model was frozen, each
  pre-registered.

Source: `reports/08_test_evaluation.md`, `audit/pretest_audit.py`.
</details>

### 32. How is the trained model protected and verified before use (SR-03)?

Every model file has a recorded SHA-256 fingerprint. The model refuses to load if any file has changed, and a
standalone check can verify the files at any time.

<details><summary>Details</summary>

> SR-03: "The system shall protect trained ML models against unauthorized modification and verify their integrity
> before deployment or use."

| Control | What it does |
|---|---|
| `model/SHA256SUMS.txt` | Fingerprint of every file in the model bundle |
| `model/verify_bundle.py` | Standalone check (standard library only): `python verify_bundle.py` prints `8/8 files OK`, or names the changed file |
| Check at load time | The scorer refuses to load if any fingerprint differs |
| `model/manifest.json` | Records the fingerprints of the training dataset, the selection rules and the model file |
| `results/pretest/freeze.json` | Freezes the model, dataset, split, rules and thresholds before the test |
| `.gitattributes` (`* -text`) | Stops Git from changing line endings, so fingerprints stay valid on every computer |
| Git history | Every change to the published model is recorded, with author and date |

The copy of the model used by the production system has its own fingerprint file, and it is verified the same way.

**Outside the model work:** server and folder permissions, which decide who may replace the model, are part of the
production deployment.
</details>

### 33. How are the logs, datasets, labels and results protected (SR-10)?

The original data is only read and never changed. Every study checked the fingerprints of all project files before
and after, with no change in any of them. Results are written only to new folders.

<details><summary>Details</summary>

> SR-10: "The system shall protect activity logs, behavioral records, model-development datasets, ground-truth labels,
> and historical detection results against unauthorized modification or deletion."

**What the model work delivers:**
- **The original simulation data is read only.** Its fingerprint was checked after the data was prepared and had not
  changed.
- **Labels are checked against the ground truth:** 9,080 of 9,080 attack days match.
- **Before/after fingerprints for every study:** each study fingerprinted every project file before and after (350 to
  560 files, depending on the date). All six studies show **0 files changed, 0 removed**.
- **Each study writes only inside its own folder.** Earlier results are never overwritten.
- **The test scores** are saved once and never rewritten.
- **The published repository** keeps a full history, so any modification or deletion is visible.

**Belongs to the production system and database:**
- access permissions on the log and result tables;
- backups;
- audit logging of who changed what (the `system_audit_logs` table).

These are part of the database design, not of the model files.
</details>
