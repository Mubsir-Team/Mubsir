# Up-weighting switching features in MUBSIR training — pre-registration (frozen protocol)

**Written:** 2026-11-26, after `PLAN.md` was approved by the owner with its recommended choices: weights 2, 5 and 10;
equivalence margin ±2 points; anchor arms W0 and SW-only. This is **before** any training on project data.

This file is hashed. The hash is stored in `PREREGISTRATION.sha256` and checked by every later script. Any change must be a
dated amendment with a new hash. **No rule may be changed after results are seen. Null and inconclusive results are valid
outcomes.**

## 1. Hypothesis
- **H-red (owner's hypothesis):** forcing MUBSIR to rely more on its channel-switching features during training leaves
  user-level detection unchanged, because the switching information is redundant with the volume and aggregated-deviation
  features.
- **Alternatives:** up-weighting harms detection, or up-weighting helps detection.

## 2. Constraints
- No attack data is generated, simulated, injected or modified.
- Only existing dev data and the closed study's existing cases are used, read-only.
- Test users, test rows and test scores are never used.
- No frozen artefact is modified.
- All outputs go inside `study_switching_weight/`.
- SHA-256 of the 506 pre-existing `Model/` files is checked before and after.

## 3. Switching features (13, existing)
- `nr_alt_{5,10,20}wd`
- `nr_gap_alt_2_{5,10,20}wd`
- `nr_handoff_{5,10,20}wd`
- `dominant_channel_switches_{5,10,20,30}wd`

That is, every manifest feature whose name starts with `nr_alt`, `nr_gap_alt_2`, `nr_handoff` or
`dominant_channel_switches`.

## 4. Arms
All arms use LightGBM defaults, label `y_10`, the Stage A folds (`config/cv_folds.csv`, 5 × 3), early stopping on the es
users, and seed 42 + repeat.

| Arm | Features | `feature_contri` |
|---|---|---|
| **A0** | 100 manifest features | 1 for all (the frozen configuration) |
| **W2** | 100 | 2 for the 13 switching features, 1 for the other 87 |
| **W5** | 100 | 5 / 1 |
| **W10** | 100 | 10 / 1 |
| W0 (descriptive) | 100 | 0 for the 13 switching features (they cannot be split on), 1 for the others |
| SW-only (descriptive) | the 13 switching features only | 1 |

**Implementation**
- The models are trained with `models._lgbm(seed, models.MAX_TREES, params)` and exactly the fit and early-stopping call
  of the frozen `models.fit_predict` (lightgbm branch), with `params = {"feature_contri": [...]}` in manifest order.
- A local replica of that call is used only because the fitted model must be returned for SHAP. G1 proves that it is
  identical.

## 5. Populations
- **P1 (primary):** all 859 dev users (348 attackers, 511 benign).
- **P2 (secondary):** the existing H146 cases of `study_controlled_rotation/data/counterfactual_cases.parquet` (145 hosts;
  ROT, BLK, FIX ×3, SIM; L ∈ {6, 12}), read-only. Each case is scored, in each repeat, by the fold model whose validation
  fold contains its host.

## 6. Gates (after this freeze, before any estimand)
- **G1 — reproduction.** The A0 OOF scores on the dev rows equal `work/oof/lightgbm_h10.parquet` exactly (all 3 repeats).
  If not, the study stops (outcome 4).
- **G2 — manipulation check.**
  - The **switching share** is the mean |TreeSHAP| of the 13 switching features divided by the mean |TreeSHAP| summed over
    all 100 features. It is computed on the OOF validation rows of the 5 repeat-0 fold models of each arm, pooled, with
    `pred_contrib` at the early-stopping iteration.
  - The share must satisfy A0 < W2 < W5 < W10, and share(W10) ≥ 2 × share(A0).
  - If not, the outcome is 4 (inconclusive: the weight did not change reliance). Estimands are still reported
    descriptively.
- **G3 — integrity.** `pipeline/*.py`, `config/cv_folds.csv`, `config/selection_rules.json`, `data/model_dataset_995.parquet`
  and `work/labels.parquet` are identical to `hashes_before.json`. If not, the study stops.

## 7. Evaluation
**Thresholds**
- For arm a, repeat r and fold k: `metrics.threshold_at_fpr(peaks, f)`.
- The peaks are the whole-period peak OOF scores of the 511 benign dev users not in fold k.
- f = 0.03 is primary; f = 0.01 is secondary.

**Detection**
- **P1 attacker:** at least one row with `y_10 = 1` scores above the threshold of its validation fold.
- **P2 case:** at least one row in [first attack day, last attack day + 9] scores above the threshold of its host's fold.
- Det = share detected, averaged over the 3 repeats.

**Bootstrap**
- 2,000 resamples with `numpy.random.default_rng(2026)`: first the P1 user resamples, then the P2 host resamples.
- Resampling is stratified by attacker vs benign; in P2 a host's cases are resampled together.
- Estimands are recomputed in each resample.
- CIs: the 95% CI is the 2.5–97.5 percentile range, and the 90% CI is the 5–95 percentile range.
- p = min(1, 2 · min(share ≤ 0, share ≥ 0)).

## 8. Estimands
**Primary (P1, iso-FPR 3%):** Δ_w = Det(W_w) − Det(A0), for w ∈ {2, 5, 10}.

**Secondary (Holm–Bonferroni across S1–S4, α = 0.05):**

| ID | Estimand |
|---|---|
| S1 | Δ₅ at iso-FPR 1% |
| S2 | Δ₅ among High-switching attackers minus Δ₅ among Low-switching attackers (`switching_level`) |
| S3 | P2: [Det(W5) − Det(A0)](ROT) − [Det(W5) − Det(A0)](BLK), pooled over L |
| S4 | row-level PR-AUC(W5) − PR-AUC(A0) on P1 OOF rows. PR-AUC is computed per repeat and averaged; the bootstrap weights rows by the resampled user counts |

**Descriptive (no tests):**
- the dose-response W0 → A0 → W2 → W5 → W10 in Det (3% and 1%), PR-AUC and switching share;
- SW-only detection;
- Det by switching level, stealth level and channel count;
- P2 by variant;
- unique detections (in ≥ 2 of 3 repeats);
- realised held-out benign FPR;
- **benign alarms vs habit:** among benign users whose whole-period peak exceeds the 3% threshold, the share in the
  high-habit tertile of `study_switching_baseline/results/habit_2010_dev_users.csv` (eligible users only; read-only).

## 9. Decision rules (applied in this order)
| Outcome | Condition |
|---|---|
| **4 — inconclusive (gate)** | G1 or G3 fails (stop), or G2 fails |
| **2 — up-weighting harms detection** | for at least one w, the 95% CI of Δ_w lies entirely below 0 **and** Δ_w ≤ −0.02 |
| **1 — same detection (redundancy confirmed)** | for all three w, the 90% CI of Δ_w lies within (−0.02, +0.02). This is TOST equivalence, α = 0.05 per side; intersection–union over the three weights, so no adjustment is needed |
| **3 — up-weighting helps detection** | for at least one w, Δ_w ≥ +0.02 and its Holm-adjusted p (over the three w) < 0.05 with a 95% CI entirely above 0 |
| **4 — inconclusive** | none of the above |

- Statistically significant changes smaller than 2 points are reported, but they do not change the outcome. A 2-point
  change is the smallest one considered practically meaningful.
- **Adoption:** no result modifies the frozen MUBSIR.
