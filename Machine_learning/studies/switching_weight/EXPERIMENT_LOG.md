# Experiment log — switching-weight study

| Time (local) | Step | Command / action | Result |
|---|---|---|---|
| 2026-11-26 08:53 | Integrity baseline | python hash_artifacts.py before (copied from study_switching_baseline; closed studies included) | 506 pre-existing Model/ files hashed |
| 2026-11-26 08:53 | Mechanism check (synthetic data only, no project data) | LightGBM 4.7.0 feature_contri on a redundant noisy copy of a strong feature, factors 1/2/5/10 | SHAP share 3.8% -> 5.1% -> 10.9% -> 16.2%; AUC 0.970 -> 0.960: knob works; feature_contri is forwarded unchanged by the frozen models.fit_predict(params=...) |
| 2026-11-26 08:53 | Plan | PLAN.md written (draft, not frozen) | stopped for owner review; nothing trained on project data |
| 2026-11-26 08:54 | Owner approval | "Freeze as recommended" on PLAN.md | weights 2/5/10; equivalence margin +-2 pts at iso-FPR 3%; anchor arms W0 and SW-only |
| 2026-11-26 08:54 | Pre-registration frozen | sha256sum PREREGISTRATION.md > PREREGISTRATION.sha256 | 6809b403cd0a15557b61f0f2c9b38627b30f1a971b7ba5bdaba968a10eea5dd6. Bootstrap rng 2026. Nothing trained on project data |
| 2026-11-26 09:00 | Gates + training | python train.py (LightGBM defaults, seed 42 + repeat, Stage A folds; local replica of fit_predict returning the model) | **ALL GATES PASS** (results/gates.json). G3: pipeline, folds, rules, dataset, labels, closed-study cases identical to hashes_before. G1: A0 OOF == Stage A OOF exactly (max diff 0). G2: switching SHAP share W0 0.000, A0 0.058, W2 0.194, W5 0.315, W10 0.380 (monotone; W10 = 6.6 x A0). 6 arms x 15 folds; P1 859 users and P2 145-host cases scored OOF; 247 s; scores/ |
| 2026-11-26 09:00 | Integrity check | python hash_artifacts.py after | 506 pre-existing files unchanged, 0 removed, 0 added outside the study. No estimand computed yet |
| 2026-11-26 09:01 | Implementation note (before the analysis run) | analyze.py written; choices not fixed by the protocol: raw-probability scale; S4 exact step-wise weighted AP checked equal to sklearn, per repeat averaged; Outcome-3 Holm over the three w uses each Delta_w's two-sided bootstrap p | - |
| 2026-11-26 09:05 | Registered analysis | python analyze.py | Det @3%: W0 0.912, A0 0.910, W2 0.900, W5 0.904, W10 0.882, SW-only 0.503. Delta W2 -0.0096 [-0.0201, 0.0000]; W5 -0.0057 [-0.0172, +0.0057]; W10 -0.0278 [-0.0431, -0.0144] (Holm < 0.005). S1 -0.0125 (Holm 0.07); S2 -0.0230 (Holm 0.17); S3 -0.0851 [-0.1161, -0.0540] (Holm < 0.005); S4 PR-AUC -0.0103 (Holm < 0.005). **Outcome 2 - up-weighting harms detection** (W10 >= 2-pt drop with CI < 0; W2/W5 inside +-2) |
| 2026-11-26 09:05 | Descriptive check after the analysis | case detection by L (A0 vs W5 vs W10) | losses at both lengths: ROT L6 0.678 -> 0.386, L12 0.984 -> 0.821; BLK L6 0.692 -> 0.503, L12 0.986 -> 0.945 (reported in REPORT §8, no test) |
| 2026-11-26 09:05 | Figures, inputs manifest, README, run_all.ps1, REPORT.md | python figures.py; python inputs_manifest.py | 2 figures; 15 external inputs hashed, none changed |
| 2026-11-26 09:05 | Final integrity check | python hash_artifacts.py after | 506 pre-existing files unchanged (closed studies included), 0 removed, 0 added outside the study; PREREGISTRATION.md hash unchanged |
| 2026-11-26 09:25 | Post-hoc diagnostic (descriptive, after the analysis, no test) | python posthoc_switching_only.py (refit of repeat-0 SW-only fold models; reproduces stored case scores exactly) | Why SW-only detects ROT (0.155) less than BLK (0.372): total switching log-odds in span BLK +2.41 vs ROT +1.75. ROT's extreme alternation (nr_alt_20wd 0.83 vs BLK 0.37) gets negative SHAP (-0.22 vs +0.41); dominant_channel_switches_30wd 14.2 vs 10.2 gets -0.28 vs -0.04; nr_handoff favours ROT (+1.56 vs +0.86) but does not compensate. results/posthoc_switching_only_*.csv |
| 2026-11-26 09:25 | Post-hoc check (descriptive) | read-only count on dev rows | attack rows: nr_alt_20wd median 0.40, p95 0.80; rows with nr_alt_20wd >= 0.75 are 8.2% attack rows vs 13.5% overall (57% benign-user rows): in the training simulation very high alternation is LESS typical of attacks |

## 2026-11-26 — Folder move (no re-run, no result changed)
- The folder moved from `Model/study_switching_weight/` to `Model/studies/switching_weight/` for the MUBSIR Model v2 repository layout.
- Only path lines changed: `MODEL_DIR = HERE.parent.parent`, paths to sibling studies, `run_all.ps1` → `..\..\.venv`.
- No score, result, figure, report, pre-registration or hash file changed. Old folder names in the text above are kept as records.
- Details and the before/after hash check are in `Model/RESTRUCTURE.md`.
