# Controlled channel rotation study — Phase 0 audit and proposed design

**Date:** 2026-11-26. **Status:** audit only. Nothing has been trained, no data has been generated, and no frozen file has been touched.
**Sources read:**
- `study_rotation/` (REPORT, PREREGISTRATION, run_experiment.py)
- NB12 feature code (§12.5.2 Clear-30, §12.6.1 rotation features)
- `prepare_dataset.py` (`E_*_today`)
- Simulation reports NB05/07/09/10
- `audit_summary.md` §8.4 (2010 switching audit)
- `config/` (split, folds, user_meta)
- Pre-injection user-day counts (`Data_engineering/data/processed/user_day_r62_v1.parquet`) and 2010 baselines.

---

## 1. What each existing feature actually measures

All deviation features use the per-channel daily deviation `E_c(t) = max(0, x_c(t) − P50_c) / max(1, P95_c − P50_c)`.

The **Clear-30** features require P50 > 0, P95 > 0 and P95 ≥ P50 for a channel to count. The **nr_*** features only require the baselines to be present. Channels are Email (external), USB (copies) and HTTP (uploads). Windows are trailing partial windows of w workdays.

| Feature | Exact definition (NB12) | What it measures |
|---|---|---|
| `cumulative_deviation_w` | Σ over the window of Σ_c E_c(t) | **aggregated deviation**: total excess, blind to channel and order |
| `combined_channel_intensity_w` | mean over the window of mean_c (x_c / P95_c) | aggregated raw intensity, including normal activity |
| `consecutive_elevated_days_w` | longest run of days with any E_c > 0 | persistence of any deviation |
| `nr_stealth_acc_w` | Σ of E_c on channel-days with E_c > 0 and x_c ≤ P95_c | aggregated sub-P95 deviation |
| `nr_m_w` | number of channels with Σ_window E_c > 0 | **diversity**: how many channels deviate (no order) |
| `nr_entropy_w` | normalised entropy of the per-channel shares of window deviation | **distribution** of deviation over channels (no order) |
| `nr_excl_days_w` / `nr_overlap_days_w` | days with exactly 1 / at least 2 deviating channels | **distribution in time** (same-day overlap; no order) |
| `cross_channel_deviation_days_w` | days with at least 2 channels with E_c > 0 | same-day multi-channel deviation (≈ overlap days, Clear validity rule) |
| `channel_distribution_shift_w` | total-variation distance between the window's raw-activity channel shares and the 2010 historical shares | **distribution vs personal history**, raw activity (no order) |
| `active_channel_count`, `multi_channel_days_10wd` | channels with any activity today; days with at least 2 active channels | raw multi-channel **activity**, not deviation |
| `dominant_channel_switches_w` | changes of the **raw-volume** dominant channel (argmax of x_c / P95_c) between consecutive observed days; ties and inactive days skipped | **transitions**, but of normal activity dominance, not of anomalous activity |
| `nr_alt_w` | share of adjacent deviation-day pairs whose dominant **deviation** channel differs (a quiet day breaks adjacency) | **transitions of anomalous activity** (normalised) |
| `nr_gap_alt_2_w` | the same, bridging at most 2 quiet days | transitions of anomalous activity |
| `nr_handoff_w` | Σ over days of channels newly deviating on a day after any deviation on the previous day | **onset transitions** (order-aware, but also counts adding a channel) |

**Classification used in this study:**
- **Aggregated deviation:** `cumulative_deviation`, `combined_channel_intensity`, `consecutive_elevated_days`, `nr_stealth_acc`.
- **Distributed deviation / diversity (no order):** `nr_m`, `nr_entropy`, `nr_excl_days`, `nr_overlap_days`, `cross_channel_deviation_days`, `channel_distribution_shift`, `active_channel_count`, `multi_channel_days_10wd`.
- **Transition / order:** `nr_alt`, `nr_gap_alt_2`, `nr_handoff`, `dominant_channel_switches`.
  - Only `nr_alt` and `nr_gap_alt_2` measure transitions of **anomalous** activity purely.
  - `nr_handoff` mixes onset and order.
  - `dominant_channel_switches` follows **normal** activity, and the earlier audit found it carries no day-level signal (within-attacker AUC 0.52).
- **Missing from the current feature set:**
  - channel **revisitation** (A→B→A);
  - **sequence regularity** (e.g. a repeated E→U→H cycle);
  - transition counts normalised by the employee's own recent norm.

  The existing `nr_alt` cannot distinguish "E,E,U,U,H,H" (block) from "E,U,H,E,U,H" (rotation) once normalised over many pairs. It gives 2/5 vs 5/5, which it does distinguish; but it cannot distinguish rotation from random channel noise.

## 2. How the simulation generated switching (NB05/07/09/10)

- **Injection only on naturally active channels.** A channel can be injected on a day only if the employee used it naturally that day (`count_before > 0`; NB05 target-channel rule). Off-days are the days with no available channel.
- **Primary-channel switching** is defined over consecutive active days: Low 10–25%, Medium 30–45%, High 50–65% of transitions.
  - Five-day windows use exact counts of 1, 2 and 3.
  - Feasible schedules were found by dynamic programming **over the employee's natural channel availability**.
  - A rotation pattern therefore follows the employee's own habits.
- **Same-day multichannel** days are optional, taken from natural co-activity.
- **Amount:** `I = max(1, ceil(m × P95_c))` per injected channel-day, constant per user-channel (m = stealth multiplier 0.25 / 0.50 / 0.75).
  - Deviation added per day is therefore channel-specific: ≈ m·P95_c / (P95_c − P50_c).
  - **Switching to a different channel changes the deviation and volume injected.** Totals are not matched across switching levels.
- **Attacker selection and switching level** follow natural behaviour. Audit §8.4 found that 2010 switching alone separates attackers from benign users (user-level AUC 0.72–0.74), and that assigned switching level tracks the employee's natural switching (median 7.49 High vs 6.03 Low vs 5.12 benign).
- **Channel configuration** is constrained by available baselines. U+H is rare (24 dev attackers), and three-channel attacks are only possible for users active on all three channels.

## 3. Can the existing data isolate switching? **No.**

In the existing simulation, switching level co-varies with:
- the employee's natural switching habit (the selection shortcut);
- the channels used, and therefore the deviation per day (P95-dependent);
- the number of channels;
- which days are injected (natural availability).

Total injected volume and total deviation are not held constant. The previous rotation-evasion result (+8.1 points for high switching) is therefore **observational and confounded**. Subgroup comparisons on the existing data cannot support a causal statement about switching, and none will be made from them.

## 4. Can a controlled experiment be built without modifying original data? **Yes, with one documented relaxation.**

**Ingredients available (read-only):**
- pre-injection 2011 daily counts for every user (`user_day_r62_v1.parquet`);
- the fixed 2010 baselines;
- the complete NB12 feature code.

A counterfactual copy can be built by injecting new, fully specified attack schedules into the pre-injection history of selected host users, then recomputing all 100 features. Nothing original is overwritten.

**Constraint found:** under the original rule (inject only into a channel the employee naturally used that day), matched variants must share the same days, so every channel must be naturally active on every attack day.

| Hosts: dev users with valid baselines on all 3 channels | 253 (109 benign, 144 attackers' pre-injection histories) |
|---|---|
| Days per host with all 3 channels naturally active | median **3**, P75 7, P90 13.8 |
| Hosts with ≥ 5 / ≥ 10 / ≥ 20 such days | 89 / **39** / 8 |
| Natural active days per host (Email / USB / HTTP), median | 70 / 23 / 24 of about 105 |

Under the strict rule, a matched 10-day experiment has only about **39 hosts (10 benign)**, on unrepresentative high-activity days. That is underpowered.

**Relaxation proposed:** inject on any workday with logon, regardless of natural channel activity. All 253 hosts become usable. The cost is a documented assumption:
- the counterfactual attacker may use a channel on a day they would not have used it;
- this changes per-channel "active today/active days" features differently across variants (a channel-activation side effect);
- the original simulation never did this, so it is a shift from the training distribution, affecting all models equally.

The strict-rule version on the 39 hosts is kept as a sensitivity analysis.

**Deviation matching:**
- Injected counts are set per channel-day so that the deviation added is exactly δ:
  `added_c = ceil-free integer δ·(P95_c − P50_c) + max(0, P50_c − x_before)`.
- This is exact when the denominator is an integer. That holds for 94% (Email), 77% (USB) and 79% (HTTP) of hosts.
- For the rest, the achieved deviation is recorded and the residual mismatch reported. If the mismatch exceeds 0.01 E units per day, the host is excluded, and the exclusion is reported.
- **Raw volume cannot be matched at the same time** as deviation, because channels have different scales. Deviation is the matched quantity, since every model reasons in it. Raw injected counts are reported per variant.

## 5. Proposed controlled design (for approval; not yet registered)

**Gate 0 (stop if it fails):**
- Re-implement the full 100-feature pipeline (NB12 §12.4 + §12.5.2 + §12.6.1 + `E_*_today`) as a standalone module.
- It must reproduce `behavior_features_r62_ml_final_new.parquet` for the 995 users with **0 differences** (tolerance 1e-9) before any counterfactual is generated.
- This module is also the missing production feature extractor.

**Counterfactual set (non-destructive copy in `study_controlled_rotation/data/`):**
- **Hosts:** the 253 dev users above, using their **pre-injection** history. The attackers' original attacks are removed, so every host starts clean. The original hosts remain benign references.
- **Window:** one per host, L attack workdays starting at a seeded random day with ≥ 30 prior workdays. L ∈ {10, 20}.
- **Daily deviation budget δ:** δ ∈ {0.5, 1.0} E units, taken from the simulation's own High- and Medium-stealth medians (audit part 2).
- **Variants per host,** all with identical attack days and identical total deviation L·δ:

| Variant | Daily channel(s) | Channels used | Switches | Purpose |
|---|---|---|---|---|
| FIX-E / FIX-U / FIX-H | the same channel every day | 1 | 0 | single-channel reference |
| BLK | E for L/3 days, then U, then H | 3 | 2 | multi-channel, not rotating |
| ROT | E, U, H, E, U, H, … | 3 | L − 1 | rotation |
| SIM | E, U and H each δ/3 every day | 3 | 0 | distributed, simultaneous |

- **ROT vs BLK** isolates order and switching: same days, same per-channel totals, same total deviation, same channels.
- **BLK vs FIX** isolates using more channels.
- **SIM vs BLK** isolates same-day distribution.

**Models,** all LightGBM defaults, y_10, trained on the **original** dev simulation with the same 15 outer folds and seeds. Each counterfactual variant of a host is scored only by fold models for which that host is in the validation fold.

| Arm | Features |
|---|---|
| A (reference) | frozen final model, 100 features (in-sample for host users' benign days; reference only) |
| A-cv | the same 100 features, refit per fold (identical to the Stage A procedure) |
| B | per-channel only (50) |
| C | B + aggregated deviation (4 families) |
| C+ | C + distributed/diversity features (8 families) |
| D | C+ + existing transition features (`nr_alt`, `nr_gap_alt_2`, `nr_handoff`, `dominant_channel_switches`) + new sequence features (revisits, transition entropy, cycle regularity) + personal-norm switching (Exp. A) |

**New features:**
- All are causal (data ≤ t) and defined mathematically in the pre-registration.
- **Personal-norm switching** uses the employee's own **trailing 2011 history** (e.g. workdays t−70 … t−11), not 2010.
- It must pass a shortcut audit before use: the user-level AUC of the baseline component alone must be < 0.60 on dev pre-window days. Otherwise it is dropped and the drop is reported.

**Primary hypothesis (proposed):**
- Rotation carries independent signal if switching-aware D detects more ROT variants than C+ at the same benign FPR, **and** that gain is larger for ROT than for BLK.
- Primary metric: `Δ = [D − C+](ROT) − [D − C+](BLK)`, attack-window detection at cross-fitted iso-FPR 3%.
- Pairing is within host, and CIs come from 2,000 host-cluster bootstraps.
- Secondary comparisons use Holm correction.

**Secondary training regime:** retrain C+ and D on counterfactual training hosts, with fold-disjoint hosts and benign high-switchers included. This asks whether rotation is learnable at all when the training data varies it independently.

**Outcome mapping:** the four outcomes of the brief (independent switching signal / distributed deviation only / neither / inconclusive). Inconclusive is reported if Gate 0 fails, if fewer than 150 hosts survive matching, or if CIs cannot exclude effects of ±5 points.

## 6. Limitations already known
- Counterfactuals are synthetic and sized in the model's own deviation units. They test what the model can detect, not real-world attacker behaviour.
- The relaxed day rule introduces channel-activation side effects. The strict-rule sensitivity analysis on 39 hosts is underpowered.
- Models trained on the original simulation learned from confounded rotation. A null result in the primary regime may mean "not learned" rather than "not useful". The secondary regime addresses this.
- Hosts include attackers' pre-injection histories. Those users were selected for high natural switching, which inflates benign-looking switching among hosts. Within-host pairing removes this from the primary contrast.

## 7. Decisions needed before pre-registration
1. Accept the relaxed day rule (253 hosts) as primary, with the strict rule (39 hosts) as sensitivity?
2. Accept deviation (E units) as the matched quantity rather than raw volume?
3. Include the secondary training regime (training on counterfactuals)?
4. Accept Gate 0 (building and validating the full feature re-implementation) as a prerequisite? It is the largest piece of work and doubles as the production feature extractor.
