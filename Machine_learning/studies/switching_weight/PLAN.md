# Research Study: Up-weighting Switching Features in MUBSIR Training — analysis plan (for review; NOT yet pre-registered)

**Status:** draft for the owner's review, 2026-11-26.
- Nothing has been trained on project data.
- No attack data is generated, simulated, injected or modified.
- No frozen artefact is changed.
- Integrity baseline: `hashes_before.json` covers 506 pre-existing `Model/` files, closed studies included.

## 1. Question and hypothesis
**Question:** if MUBSIR is forced, during training, to rely more on its channel-switching features, does detection change?

**Hypothesis (owner's, H-red):** detection stays the same, because the switching information is redundant with the
volume / aggregated-deviation features.

The alternatives are also tested:
- **Harm:** relying on a weaker signal lowers detection.
- **Help:** the information was under-used.

**Background (why this is a real question):**
- The 13 switching features hold about 6% of mean |SHAP| on dev data, 3.8% on attack days.
- Removing them changes dev detection by only +0.3 points at 3% and +1.0 at 1% (read-only check, 2026-11-26).
- Weighting them **up** has not been tested.

## 2. Switching features (the 13 existing transition features; no new features)
- `nr_alt_{5,10,20}wd`
- `nr_gap_alt_2_{5,10,20}wd`
- `nr_handoff_{5,10,20}wd`
- `dominant_channel_switches_{5,10,20,30}wd`

These are the same 13 features as the "transition" group of `study_controlled_rotation/`.

## 3. How the weight is raised
- **Mechanism:** LightGBM's `feature_contri` (alias `feature_penalty`). The split gain of feature i is multiplied by its
  factor, so the trees prefer the up-weighted features whenever they are close to the best split.
- **Delivery:** the parameter is passed through the frozen `models.fit_predict(..., params={"feature_contri": ...})`,
  which forwards extra parameters to LightGBM unchanged. No pipeline code is modified.
- **Synthetic check (no project data):** a noisy, redundant copy of a strong feature was given factor 1 → 2 → 5 → 10.
  - Its SHAP share rose 3.8% → 5.1% → 10.9% → 16.2%.
  - AUC moved 0.970 → 0.969 → 0.964 → 0.960.
  - So the knob works, and on purely redundant information it costs a little accuracy.

## 4. Arms (identical except the weight; LightGBM defaults, label y_10, Stage A folds, early stopping and seeds 42 + repeat)
| Arm | Factor on the 13 switching features | Factor on the other 87 | Role |
|---|---:|---:|---|
| **A0** | 1 | 1 | the frozen MUBSIR configuration (must reproduce the Stage A OOF exactly) |
| **W2** | 2 | 1 | weighted |
| **W5** | 5 | 1 | weighted |
| **W10** | 10 | 1 | weighted |
| W0 | 0 (switching features never used) | 1 | descriptive anchor (= removing them) |
| SW-only | the 13 switching features alone | — | descriptive: how much the switching signal detects on its own |

## 5. Data and populations (existing only)
- **P1 (primary):** the existing Dev simulation, all 859 dev users (348 attackers, 511 benign), the existing 5×3 grouped
  folds, the same setup as Stage A. Test users are never used.
- **P2 (secondary):** the existing matched H146 cases of `study_controlled_rotation/` (145 hosts; ROT, BLK, FIX, SIM),
  read-only. Each case is scored by the fold model whose validation fold contains its host.

Inputs: `model_dataset_995.parquet` (dev rows), `labels.parquet`, `cv_folds.csv`, `user_meta.csv`, frozen
`pipeline/*.py`, and the closed study's `counterfactual_cases.parquet`. All are read-only.

## 6. Gates (after pre-registration, before any estimand)
- **G1 — reproduction.** A0 reproduces the Stage A OOF (`work/oof/lightgbm_h10.parquet`) exactly. This proves that only
  the weight differs between arms.
- **G2 — manipulation check.**
  - Compute the share of mean |SHAP| of the 13 switching features on OOF validation rows (fold models, repeat 0).
  - It must rise monotonically A0 < W2 < W5 < W10, and W10 must be ≥ 2 × A0.
  - If not, the weight did not change the model's reliance, and the outcome is inconclusive.
- **G3 — integrity.** The pipeline, folds, dataset and labels are identical to `hashes_before.json`.

## 7. Evaluation
- **Thresholds:** cross-fitted iso-FPR from the whole-period peak OOF scores of the 511 benign dev users in the other
  folds, as in Stage A and the closed studies. 3% is primary, 1% secondary.
- **Detection:** an attacker is detected if at least one `y_10 = 1` day scores above the threshold. Det is averaged over
  the 3 repeats.
- **Uncertainty:** 2,000 bootstrap resamples of users, stratified by attacker vs benign; P2 resamples hosts. The rng seed
  is fixed at pre-registration.

## 8. Estimands
**Primary (P1, iso-FPR 3%):** Δ_w = Det(W_w) − Det(A0), for w = 2, 5 and 10.

**Secondary (Holm):**

| ID | Estimand |
|---|---|
| S1 | Δ₅ at iso-FPR 1% |
| S2 | Δ₅ among High-switching minus Low-switching attackers |
| S3 | P2: [Det(W5) − Det(A0)](ROT) − [Det(W5) − Det(A0)](BLK) |
| S4 | row-level PR-AUC(W5) − PR-AUC(A0) |

**Descriptive:**
- the dose-response curve W0 → A0 → W2 → W5 → W10 (detection and SHAP share);
- SW-only detection;
- per stealth level and channel count;
- **which benign users are alarmed:** does weighting move false alarms toward habitual switchers? This uses the 2010
  habit from `study_switching_baseline/` (read-only), descriptive only.

## 9. Decision rules (applied in order)
| Outcome | Condition |
|---|---|
| **4 — inconclusive (gate)** | G1, G2 or G3 fails |
| **2 — weighting harms detection** | for at least one w, the 95% CI of Δ_w lies entirely below 0 **and** Δ_w ≤ −2 points |
| **1 — same detection (redundancy confirmed)** | for **all three** w, the 90% CI of Δ_w lies within (−2, +2) points (equivalence; TOST at α = 0.05 per side, intersection–union over the three weights, so no multiplicity adjustment) |
| **3 — weighting helps** | for at least one w, the 95% CI (Holm over the three w) lies entirely above 0 **and** Δ_w ≥ +2 points |
| **4 — inconclusive** | none of the above |

Statistically significant changes smaller than 2 points are reported, but they do not change the outcome. A 2-point change
is the smallest one considered practically meaningful.

**Expected precision:** in earlier studies, paired detection differences on the 348 dev attackers had CIs of about
±1 point. So the ±2-point equivalence margin is attainable.

**Adoption:** no result modifies the frozen MUBSIR.

## 10. Questions for the owner before freezing
1. **Weights:** 2, 5 and 10 (recommended)? Or other values?
2. **Equivalence margin:** ±2 points at iso-FPR 3% (recommended)?
3. **Anchor arms:** include the descriptive W0 (switching removed) and SW-only arms (recommended)?
