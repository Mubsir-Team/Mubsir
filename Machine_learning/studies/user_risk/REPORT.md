# MUBSIR — User risk aggregation & alert threshold validation study

**Date:** 2026-11-25 · **Data:** development (dev) users only, out-of-fold (OOF) predictions · **Test set:** not opened, not used.
**Scripts:** `run_study.py` (all numbers), `figures.py` (all figures), `hash_artifacts.py` (integrity). **Parameters:** `params.json`.
**Results:** `results/policy_comparison.csv`, `results/duration_bias_analysis.csv`, `results/persistence_policies.csv`,
`results/operational_alert_volume.csv`, `results/policy_by_duration_stealth.csv`, plus supporting files. **Figures:** `figures/fig1…fig6`.

---

## 0. Answer in brief

1. **Observation-duration bias is real.** In a controlled experiment that truncates the same benign users' history, the share of benign employees whose maximum daily risk exceeds 85 rises from **0.33% over 10 workdays to 3.07% over the full ~105-workday period** (about 9×). Days are strongly dependent: an independence model predicts 9.3% for the full period.
2. **The aggregation method is not the problem.** At an equal benign false-positive rate, every alternative tested (top-3/top-5 mean, 90th/95th percentile, EWMA maximum, exposure-adjusted maximum) has the same recall as the plain maximum: all paired differences have confidence intervals that include 0. None improves short or high-stealth detection; the maximum is best or tied for those.
3. **Persistence rules do not pay off.** They sit on the same recall–FPR curve as simply raising the single-day threshold. They add about one workday of median latency and lose more short attacks (5-day attacks: 46% window detection for "2 of 3 days > 85" vs about 51% for a single-day threshold at the same FPR).
4. **85 is defensible but not uniquely justified.** MCC, F1 and balanced accuracy are flat between 70 and 92.63 (differences inside the CIs). The threshold is a workload choice. 85 corresponds almost exactly to a **3% chance that a benign employee is alerted at least once over ~5 months** (the empirical exposure-adjusted threshold at α = 3% for 105 workdays is 85.5).
5. **Recommendation:** keep the model's daily risk and the single-day alert rule `risk > 85` provisionally. Separate three things that `risk_peak` currently mixes:
   - (a) the daily score;
   - (b) alert generation as daily events, grouped into incidents and counted per employee-month;
   - (c) an employee-level summary for ranking that is comparable across employees: a fixed trailing window, or an exposure-adjusted tail probability.

   Do not adopt persistence, top-k, percentile or EWMA alerting. Treat the exposure-adjusted maximum as a promising candidate that needs independent validation before use.

---

## 1. Safeguards and setup

- **Source of `risk_peak`:**
  - `model/mubsir_infer.py` → `MubsirScorer.score_users`: `risk_peak = max(daily risk over the rows given)`, `alert = risk_peak > 85`.
  - The daily risk is `100·σ(a·logit(p) + b)` with the frozen Platt coefficients (a = 0.9549, b = 0.0760).
- **Data used:**
  - Stage A LightGBM y_10 OOF scores (`work/oof/lightgbm_h10.parquet`): 3 repeats, 859 dev users (348 attackers, 511 benign), 89,471 user-days.
  - These are mapped through the frozen calibration, exactly as in `07_calibrate_fit.py`.
  - Every dev day is scored once per repeat by a model that never saw that user.
- **Not used:** the final model's in-sample dev scores (optimistic) and anything from the test set. Test rows are dropped at load, and the script asserts that no test user remains.
- **Aggregation over repeats:**
  - Each user-level metric is computed per repeat, then averaged. Confusion counts are therefore repeat means and can be fractional.
  - 95% CIs: 1,000 bootstrap resamples of users (stratified attacker/benign), with the metric averaged over the 3 repeats inside each resample.
  - Paired differences use the same resamples.
- **Iso-FPR comparisons:** the threshold for every method is cross-fitted. For each repeat and validation fold, it is set from the scores of benign users in the other four folds, so no user's own score sets its threshold.
- **Integrity:** SHA-256 of all 349 existing project files was recorded before and after the study (§10). Nothing frozen was modified.
- **Label reminder:**
  - `y_10 = 1` means an injected attack day occurred on the current day or one of the previous 9 workdays. It is not an attack on that exact day.
  - User-level ground truth here is `is_attacker`: at least one injected attack day.
- **Definitions used throughout:**
  - **User-level alert classification:** the employee crosses the rule at least once anywhere in the observation period.
  - **Attack-window detection:** the rule fires on a day inside the attacker's own attack window.
  - **Latency:** workdays from the first injected day to the first in-window alert.

---

## 2. Baseline: `risk_peak` with fixed thresholds (dev OOF)

| Threshold | TP | FN | FP | TN | Recall (95% CI) | Specificity | FPR (95% CI) | Precision | F1 | F2 | Bal. acc. | MCC (95% CI) |
|---|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---|
| 70 | 328.7 | 19.3 | 39.3 | 471.7 | 0.944 | 0.923 | 0.077 | 0.893 | 0.918 | 0.934 | 0.934 | 0.861 |
| 75 | 324.3 | 23.7 | 30.3 | 480.7 | 0.932 | 0.941 | 0.059 | 0.915 | 0.923 | 0.928 | 0.936 | 0.870 |
| 80 | 321.0 | 27.0 | 22.3 | 488.7 | 0.922 (0.896–0.947) | 0.956 | 0.044 (0.028–0.059) | 0.935 | 0.929 | 0.925 | 0.939 | 0.881 (0.851–0.908) |
| **85 (current)** | **317.0** | **31.0** | **15.7** | **495.3** | **0.911 (0.883–0.937)** | **0.969** | **0.031 (0.018–0.044)** | **0.953** | **0.931** | **0.919** | **0.940** | **0.887 (0.858–0.914)** |
| 90 | 310.3 | 37.7 | 10.0 | 501.0 | 0.892 (0.859–0.920) | 0.980 | 0.020 (0.009–0.031) | 0.969 | 0.929 | 0.906 | 0.936 | 0.885 (0.855–0.912) |
| 92.63 | 305.0 | 43.0 | 5.3 | 505.7 | 0.876 (0.841–0.906) | 0.990 | 0.010 (0.004–0.018) | 0.983 | 0.927 | 0.896 | 0.933 | 0.885 (0.856–0.912) |

Further detail at the current threshold of 85:
- **Employees alerted:** 332.7 unique employees (38.7 per 100), of whom 15.7 benign (3.07 per 100 benign) and 317.0 attackers. 31.0 attackers are missed.
- **Attack-window detection:** 90.6%. About 1.7 attackers are alerted only outside their window (counted as TP at user level).
- **Paired differences vs 85:**
  - At 90: recall −0.019 (−0.030 to −0.011), FPR −0.011 (−0.018 to −0.006), MCC difference CI −0.012 to +0.009.
  - At 80: recall +0.011 (+0.005 to +0.018), FPR +0.013 (+0.007 to +0.020).
- Precision, F1, F2 and MCC depend on the dev prevalence (40.5% attackers) and do not transfer to production.
- This reproduces the earlier dev figures exactly (15.7 benign users at 85, 90.6% window detection, median latency 3 workdays).

---

## 3. Observation-duration bias

### 3.1 Natural variation in dev is too small to test the question
| Subset | n | min | P1 | P5 | P25 | median | mean | P75 | P90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| All | 859 | 7 | 43 | 105 | 105 | 105 | 104.2 | 105 | 106 | 108 |
| Benign | 511 | 7 | 50 | 105 | 105 | 105 | 104.2 | 105 | 106 | 107 |
| Attackers | 348 | 28 | 55 | 105 | 105 | 105 | 104.1 | 105 | 106 | 108 |

- Groups: ≤30 workdays (5 benign / 3 attackers), 31–60 (1/1), 61–90 (3/4), 91–104 (2/2), ≥105 (500/338).
- None of the 11 short-history benign users crossed 85, but their CIs span 0–52% or more (Fig. 1, left).
- Spearman ρ(n_days, risk_peak) is 0.010 for benign users (−0.08 to 0.09) and 0.060 for attackers (−0.04 to 0.16).
- A logistic regression `alert85 ~ log(n_days) + attacker` gives a positive log(n_days) coefficient (4.25; CI 2.8–20.1). It rests on 21 short-history users with almost no alerts, so it is not reliable evidence.
- **The cross-section cannot answer the question.** The controlled experiment below does.

### 3.2 Controlled exposure: the same benign users, truncated to H workdays
H = the first H workdays of each benign user's period. Users with fewer than H days are excluded and counted.

| H (workdays) | Benign users (excluded) | Max > 80 | **Max > 85 (95% CI)** | Max > 90 | Any H-window max > 85 | Independence prediction (85) | Benign max: P50 / P95 / P99 |
|---|---|---:|---:|---:|---:|---:|---|
| 10 | 510 (1) | 0.46% | **0.33% (0.00–0.85)** | 0.13% | 0.38% | 0.92% | 1.2 / 33.5 / 68.9 |
| 20 | 509 (2) | 0.65% | **0.52% (0.07–1.24)** | 0.26% | 0.66% | 1.84% | 2.0 / 44.1 / 75.2 |
| 30 | 507 (4) | 1.18% | **0.85% (0.20–1.71)** | 0.53% | 0.93% | 2.74% | 2.9 / 53.8 / 81.9 |
| 60 | 505 (6) | 1.98% | **1.45% (0.53–2.44)** | 1.12% | 1.65% | 5.41% | 3.9 / 62.2 / 90.7 |
| 90 | 502 (9) | 3.65% | **2.39% (1.26–3.59)** | 1.66% | 2.68% | 8.00% | 6.8 / 75.3 / 92.4 |
| full (~105) | 511 (0) | 4.37% | **3.07% (1.89–4.50)** | 1.96% | 3.07% | 9.27% | 7.5 / 77.6 / 92.9 |

- **The same threshold produces very different benign alert rates as monitoring lengthens.** At 85, the rate rises about 9× from 10 to ~105 workdays (Fig. 1 right, Fig. 2). The pattern holds at 80 and 90, and for any-start windows as well as first-H windows.
- The rise is **far slower than independence would predict**: benign exceedances cluster in a few users (alerted benign users have about 3 alert days each). Daily scores are strongly dependent, so the chance of ever crossing 85 cannot be derived from a daily rate.
- **Per unit of time** the benign alert rate is not constant either. In the mixed-duration experiment, users alerted per 100 benign employee-months at 85 were 1.35, 0.69, 0.62, 0.40 and 0.69 for H = 10, 20, 30, 60 and 90. So neither "per employee" nor "per employee-month" makes the fixed-threshold maximum duration-invariant.
- **Attackers:** the probability of an alert within H workdays of the first injected day at 85 is 85.3% (H = 10), 89.8% (20), 90.5% (30), 91.0% (60) and 91.1% (90). Almost all detection happens in the first 20 workdays after the attack starts.

### 3.3 Mixed-duration population (20 random draws; each user gets one H ∈ {10, 20, 30, 60, 90})
- Benign window start: uniform. Attacker window start: uniform among windows that contain the first injected day.
- Exposure-adjusted thresholds are cross-fitted from benign users of the other folds.

| Policy | Benign FPR at H = 10 / 30 / 90 | FPR ratio H90/H10 | Overall benign FPR | Overall attacker recall | High-stealth recall | 5-day-attack recall |
|---|---|---:|---:|---:|---:|---:|
| max > 85 (current) | 0.64% / 0.88% / 2.95% | **4.6** | 1.26% | 81.9% | 62.4% | 63.9% |
| max > 90 | 0.37% / 0.58% / 2.07% | 5.6 | 0.85% | 78.9% | 56.8% | 56.1% |
| ≥ 2 of 3 days > 85 | 0.34% / 0.33% / 1.54% | 4.6 | 0.59% | 75.1% | 52.5% | 48.1% |
| top-3 mean > 85 | 0.35% / 0.38% / 1.64% | 4.6 | 0.64% | 74.3% | 52.8% | 48.3% |
| EWMA(h=5) > 80 | 0.19% / 0.23% / 0.16% | 0.8 | 0.16% | 45.1% | 26.5% | 13.5% |
| **exposure-adjusted, α = 1%** | 1.20% / 0.96% / 1.31% | **1.09** | **1.04%** | **83.5%** | 63.7% | 64.9% |
| exposure-adjusted, α = 3% | 3.22% / 2.73% / 3.43% | 1.07 | 2.97% | 89.7% | 76.8% | 79.7% |

- **Only the exposure-adjusted maximum holds the per-employee false-alarm probability constant across monitoring lengths.**
- At α = 1% it catches more attackers than the current rule (83.5% vs 81.9%) at a lower overall benign FPR (1.04% vs 1.26%). It does this by lowering the bar for short histories and raising it for long ones.
- Its thresholds T_α(H), for H = 10 / 20 / 30 / 60 / 90 / 105 workdays:

| α | 10 | 20 | 30 | 60 | 90 | 105 |
|---|---:|---:|---:|---:|---:|---:|
| 1% | 70.9 | 79.2 | 83.6 | 90.3 | 92.5 | 93.6 |
| 3% | 42.3 | 57.0 | 65.1 | 76.2 | 82.6 | 85.5 |

  So **a peak of about 71 over 10 workdays is as unusual for a benign employee as a peak of about 94 over 105 workdays.**
- **The trade-off:** holding the per-employee rate constant increases per-month workload for short-history employees (α = 3%: 6.8 benign alerts per 100 benign employee-months at H = 10 vs 0.8 at H = 60–90).
- **Estimation risk:** the α = 1% tail rests on about 5 exceedances among about 500 reference users per horizon. It is not a calibrated probability.

---

## 4. Alternative aggregation methods (daily scores unchanged)

### 4.1 Same fixed threshold (dev, full period)
| Aggregation (threshold) | Recall | FPR | MCC | 5-day attackers alerted | High-stealth alerted | Note |
|---|---:|---:|---:|---:|---:|---|
| max (85), current | 0.911 | 0.031 | 0.887 | 0.645 | 0.764 | |
| max over first 30 / 60 days (85) | 0.425 / 0.677 | 0.008 / 0.014 | 0.536 / 0.722 | | | misses attacks outside the window |
| max over last 30 / 60 days (85) | 0.420 / 0.702 | 0.016 / 0.021 | 0.516 / 0.732 | | | misses attacks outside the window |
| top-3 mean (85) / top-5 mean (85) | 0.884 / 0.871 | 0.016 / 0.011 | 0.884 / 0.880 | 0.522 / 0.464 | | |
| 95th percentile (60) / (85) | 0.900 / 0.830 | 0.020 / 0.006 | 0.893 / 0.854 | 0.565 / 0.370 | | |
| EWMA h = 5 (60) / (85) | 0.860 / 0.659 | 0.009 / 0.000 | 0.874 / 0.731 | 0.457 / 0.087 | | median latency 8 / 16 workdays |
| exposure-adjusted max, α = 3% / 1% | 0.908 / 0.868 | 0.025 / 0.009 | 0.891 / 0.880 | 0.645 / 0.514 | | |

At a fixed threshold, methods differ mainly in how strict they are. So the fair comparison is at equal FPR.

### 4.2 Equal benign FPR (cross-fitted iso-FPR thresholds)
| Method | Recall @ 3% FPR (Δ vs current, 95% CI) | Recall @ 1% FPR | 5-day attackers alerted @ 3% |
|---|---|---:|---:|
| max (full period) | 0.912 (+0.001, −0.005 to +0.007) | 0.876 | **0.652** |
| top-3 mean | 0.907 (−0.004, −0.012 to +0.004) | 0.875 | 0.601 |
| top-5 mean | 0.908 | 0.867 | 0.601 |
| 95th percentile | 0.909 | 0.869 | 0.601 |
| 90th percentile | 0.894 | 0.857 | 0.522 |
| EWMA h = 5 max | 0.908 (−0.003, −0.014 to +0.009) | 0.867 | 0.609 |
| EWMA h = 10 max | 0.907 | 0.874 | 0.594 |
| exposure-adjusted max (α = 3% / 1%) | 0.908 at FPR 2.5% | 0.868 at FPR 0.9% | 0.645 |
| max over first / last 60 days | 0.728 / 0.719 | 0.661 / 0.677 | 0.652 / 0.210 |

- **No aggregation beats the maximum at equal FPR.** All whole-period alternatives lie on the same recall–FPR curve (Fig. 3).
- Averaging methods (top-k, percentiles, EWMA) **lose short attacks** (5-day: 0.60 vs 0.65 at 3% FPR), because a 5-day attack yields only a few high days. They also discard the "which day" information that SHAP explanations and latency depend on.
- **Top-k:** no meaningful improvement.
- **Percentile:** ignoring isolated peaks removes some benign noise, but costs sensitivity to brief attacks at the same FPR.
- **EWMA:** with a fixed threshold it is slow (median latency 8–35 workdays) and nearly blind to 5- and 10-day attacks. At equal FPR it ties the maximum, so temporal decay buys nothing here.
- **Fixed-window snapshots** (first/last 30 or 60 days) are unsuitable for alerting: they cannot see attacks outside the window.

---

## 5. Persistence policies (`results/persistence_policies.csv`)

| Policy | Recall | FPR | Precision | Benign alerted | Window det. | Latency P50 / P75 / P90 | 5-day window det. | High-stealth window det. | Missed attackers |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1 day > 80 | 0.922 | 0.044 | 0.935 | 22.3 | 0.916 | 2 / 4 / 7 | 0.645 | 0.775 | 27.0 |
| **1 day > 85 (current)** | **0.911** | **0.031** | **0.953** | **15.7** | **0.906** | **3 / 5 / 8** | **0.616** | **0.749** | **31.0** |
| 1 day > 90 | 0.892 | 0.020 | 0.969 | 10.0 | 0.886 | 3 / 6 / 9 | 0.536 | 0.707 | 37.7 |
| 1 day > 92.63 | 0.876 | 0.010 | 0.983 | 5.3 | 0.871 | 3 / 6 / 9.2 | 0.493 | 0.670 | 43.0 |
| ≥ 2 of 3 > 80 | 0.898 | 0.021 | 0.967 | 10.7 | 0.884 | 4 / 6 / 9 | 0.514 | 0.698 | 35.7 |
| ≥ 2 of 3 > 85 | 0.880 | 0.015 | 0.976 | 7.7 | 0.872 | 4 / 6 / 9 | 0.464 | 0.672 | 41.7 |
| ≥ 3 of 5 > 85 | 0.868 | 0.011 | 0.982 | 5.7 | 0.852 | 5 / 8 / 11 | 0.406 | 0.644 | 46.0 |
| ≥ 2 of 3 > 90 | 0.865 | 0.010 | 0.984 | 5.0 | 0.855 | 4 / 7 / 11 | 0.435 | 0.641 | 47.0 |
| ≥ 2 of 3 > 85, or 1 day > 95 | 0.881 | 0.015 | 0.976 | 7.7 | 0.875 | 3 / 6 / 9 | 0.478 | 0.681 | 41.3 |

- **Against the current rule,** "≥ 2 of 3 days > 85":
  - halves benign alerts: FPR −0.016 (−0.023 to −0.008);
  - costs recall −0.031 (−0.046 to −0.018);
  - costs 1 workday of median latency;
  - costs 15 points of 5-day window detection (0.616 → 0.464).
- **Against raising the single-day threshold to the same FPR** (about 1.5%, between 90 and 92.63: recall ≈ 0.884, 5-day ≈ 0.51), persistence is **no better on recall and worse on short attacks**, with longer latency.
- The **escalation rule** (any single day > 95) restores one day of median latency but almost no detection. The 95 and 98 cut-offs were evaluated on dev, and neither changes FPR.
- Persistence also produces more attackers alerted only outside their window (3.0–5.3 vs 1.7), because confirmation can arrive after a short window has ended.
- **Conclusion:** the false-positive reduction does not justify the extra delay. The same reduction is available more cheaply by moving the single-day threshold.

---

## 6. Operational alert volume (`results/operational_alert_volume.csv`)

**Counting units:**
- Exposure: dev benign users contribute 2,535 employee-months (1 employee-month = 21 workdays); all users contribute 4,261.
- An **alert event** is one day on which the rule holds.
- An **incident** groups alert days for the same employee; a new incident starts after at least 5 quiet workdays.

| Policy | Unique employees alerted / 100 | Benign alerted / 100 benign | Benign alert events / 100 benign emp-months | **Benign incidents / 100 benign emp-months** | Events per alerted employee (mean / median) | Incidents per alerted employee | Events per alerted benign employee |
|---|---:|---:|---:|---:|---|---:|---:|
| 1 day > 80 | 40.0 | 4.37 | 2.83 | 1.05 | 26.9 / 26 | 1.15 | 3.2 |
| **1 day > 85** | **38.7** | **3.07** | **1.95** | **0.68** | **26.6 / 25** | **1.12** | **3.2** |
| 1 day > 90 | 37.3 | 1.96 | 1.05 | 0.43 | 26.0 / 25 | 1.11 | 2.7 |
| 1 day > 92.63 | 36.1 | 1.04 | 0.54 | 0.22 | 25.6 / 24 | 1.11 | 2.6 |
| ≥ 2 of 3 > 85 | 36.6 | 1.50 | 1.45 | 0.30 | 28.0 / 26 | 1.07 | 4.8 |
| ≥ 3 of 5 > 85 | 35.8 | 1.11 | 1.24 | 0.22 | 28.4 / 27 | 1.05 | 5.5 |
| EWMA h = 5 > 60 | 35.4 | 0.91 | 1.62 | 0.20 | 29.7 / 28 | 1.02 | 8.8 |

- **Repeated daily alerts must not be counted as separate investigations.** An alerted attacker generates about 25 alert days but about 1.1 incidents. An alerted benign employee generates about 3 alert days in about 1 incident.
- **Workload of the current rule on dev:** about **0.7 false incidents per 100 benign employee-months.** For 1,000 benign employees that is roughly 7 false incidents per month (Fig. 4).
- The "employees alerted per 100" column is dominated by attackers (40% of dev). It is not a production estimate.
- The benign per-employee-month column is the transferable one, assuming production benign behaviour resembles this simulation.
- **Duration effect on volume:** see §3.2–3.3. Per-employee alert rates grow with monitoring length, and per-month rates are not constant.

---

## 7. Fair comparison of candidate policies

| Family | Best representative | Recall | FPR | Window det. | Latency P50 | 5-day | High stealth | Duration-robust (per-employee FPR ratio H90/H10) | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Current max | 1 day > 85 | 0.911 | 0.031 | 0.906 | 3 | 0.616 | 0.749 | 4.6 | keep provisionally |
| Max, other threshold | 1 day > 90 | 0.892 | 0.020 | 0.886 | 3 | 0.536 | 0.707 | 5.6 | valid lower-workload option |
| Fixed window (snapshot) | last 60 d @ 3% FPR | 0.719 | 0.033 | — | — | 0.210 | — | n/a | reject for alerting |
| Top-k | top-3 @ 3% FPR | 0.907 | 0.031 | — | — | 0.601 | — | 4.6 | no gain |
| Percentile | p95 @ 3% FPR | 0.909 | 0.032 | — | — | 0.601 | — | — | no gain |
| EWMA | EWMA h = 5 @ 3% FPR | 0.908 | 0.033 | — | — | 0.609 | — | 0.8 at a fixed threshold, at the cost of recall | no gain; slow at fixed thresholds |
| Exposure-adjusted | α = 3% | 0.908 | 0.025 | — | — | 0.645 | — | **1.07** | promising; needs validation |
| Persistence | ≥ 2 of 3 > 85 | 0.880 | 0.015 | 0.872 | 4 | 0.464 | 0.672 | 4.6 | reject |

- Recall, FPR, 5-day and high-stealth figures are user-level unless marked "window det.".
- "—" means the method has no daily alert stream, so latency and window detection are undefined for it.
- All rows use the same 859 users. Paired 95% CIs are in `policy_comparison.csv` (`d_*` columns).
- No MCC difference against the current rule excludes 0.
- **No policy is declared superior on F1.** F1 ranges only 0.918–0.933 across the credible candidates.

---

## 8. Answers to the study questions

1. **Is `risk_peak` statistically appropriate when monitoring durations differ?**
   - Not as a comparable employee-level score. Its benign distribution shifts with monitoring length (§3.2), so equal values do not mean equal unusualness.
   - It is appropriate as the record that "the alert rule fired at least once", which is what `alert` means.
   - It is not a calibrated probability that the employee is malicious.
2. **How much does monitoring duration affect the chance of crossing 85 among benign users?**
   - 0.33% over 10 workdays, 0.85% over 30, 1.45% over 60, 2.39% over 90, and 3.07% over ~105, within the same users.
   - The rise is sublinear relative to independence because exceedances cluster in a few employees.
3. **Is a fixed observation window better?**
   - As an alert rule, no: a snapshot window misses attacks outside it (recall 0.42–0.73).
   - As the definition of the employee-level summary score (the maximum over the last N workdays, recomputed daily), yes. It makes employees comparable, and with daily evaluation it detects the same attacks, because alerting stays daily.
4. **Does top-k aggregation improve meaningfully over the maximum?** No. Recall is equal at equal FPR, and short attacks are slightly worse.
5. **Does temporal decay give a better balance?** No. EWMA ties the maximum at equal FPR, is slow and misses short attacks at fixed thresholds, and adds a parameter.
6. **Can the exposure-adjusted maximum control the duration bias without losing sensitivity?**
   - In the controlled mixed-duration experiment, yes: the per-employee FPR is flat across 10–90 workdays (ratio 1.07–1.09 vs 4.6), and recall is higher than the fixed 85 rule at lower overall FPR.
   - Caveats: its thresholds come from ~500 benign users of one simulated period, the 1% tail rests on about 5 exceedances, and it raises the per-month workload for short-history employees.
   - It is not a calibrated probability and has not been validated outside dev.
7. **Does requiring repeated exceedances reduce false positives enough to justify the delay?** No. The same FPR reduction comes from a higher single-day threshold, with equal recall, better short-attack detection and lower latency.
8. **Which policy gives the best operational balance for an SOC?**
   - The single-day rule `risk > T`, with alert days grouped into incidents (5-workday cool-down) and workload reported per employee-month.
   - T is a capacity choice: 85 gives about 0.7 false incidents per 100 benign employee-months; 90 about 0.4 (recall −1.9 points, 5-day attacks −8 points).
9. **Is 85 still defensible?**
   - Yes, as a provisional operating point: it lies in a flat region of MCC, F1 and balanced accuracy, and corresponds to about a 3% per-employee benign false-alarm probability over ~5 months.
   - The evidence does not show 85 is optimal. The choice between 80, 85 and 90 is a recall-versus-workload decision for the SOC, not a statistical result.
10. **Retain, replace, or separate?**
    - **Separate.** Keep the daily score and the provisional single-day alert rule.
    - Replace the use of whole-period `risk_peak` as an employee-level comparison score with a fixed trailing-window maximum, or an exposure-adjusted tail probability once validated.
    - Report workload per employee-month, and count incidents, not alert days.

---

## 9. Recommendation and limitations

**Recommendation (provisional; no frozen rule is changed retrospectively):**
1. Keep the frozen model, calibration and the registered alert rule `risk > 85` (single day) for the current release.
2. Document that `alert` means "crossed 85 at least once", and that per-employee alert probability depends on the monitoring length.
3. Add incident grouping (new incident after at least 5 quiet workdays) and report SOC workload as incidents per 100 employee-months.
4. For employee ranking and dashboards, use the maximum daily risk over the **last 30 workdays**. Do not use the whole period.
5. Do not adopt persistence, top-k, percentile or EWMA aggregation.
6. Register the exposure-adjusted maximum (α = 1–3%) as a candidate. Validate it on new data with genuinely varied monitoring lengths before any adoption.

**Additional data needed before changing the policy:**
- A monitored population with naturally varying tenure (new joiners, leave).
- Several more months of benign history, to estimate tails at α ≤ 1% (thousands of benign employee-periods).
- Realistic attacks that are not sized in P95 units.
- An independent period, to test temporal transfer.
- SOC capacity figures, to set T from workload rather than from the data.

**Limitations:**
- Everything is simulated and dev-only; nothing here was validated on test or production.
- About 99 policy variants were evaluated on the same dev OOF scores, so small differences may be noise: only paired CIs that exclude 0 are treated as differences.
- The controlled exposure experiment truncates real histories. The first workdays of 2011 use partial rolling windows (slightly lower risk); the sliding-window version gives the same picture.
- Production prevalence is far below 40%, so precision-type metrics are not transferable.

---

## 10. Integrity statement

- SHA-256 of all 349 pre-existing project files (outside `study_user_risk/` and the virtual environment) was recorded before the study (`hashes_before.json`) and after it (`hashes_after.json`). The comparison is in `hashes_comparison.json`.
- **No file was changed, removed or added outside this folder.** That covers the model (`model/lgbm_y10.txt`), manifest, calibration, thresholds, dataset, split, selection rules, OOF predictions, the frozen test scores (`work/test/test_scores.parquet`) and every existing report.
- The test set was not read by any script of this study.
