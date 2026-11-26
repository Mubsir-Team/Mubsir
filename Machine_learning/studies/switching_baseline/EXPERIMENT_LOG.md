# Experiment log — switching-baseline study

| Time (local) | Step | Command / action | Result |
|---|---|---|---|
| 2026-11-26 06:05 | Integrity baseline | python hash_artifacts.py before (closed studies included in the hashed set) | 448 pre-existing Model/ files hashed (hashes_before.json) |
| 2026-11-26 06:05 | Library copy | cp ../study_controlled_rotation/rot_features.py . | byte-identical (sha256 e15d7740c96bf764...); used for the audit only |
| 2026-11-26 06:05 | Phase 0 audit | python audit_2010.py (read-only; no model; test never read) | 2010 baseline definition reproduced exactly (weekday & logon>0; linear quantiles over active days); habit measurable for ~85% of hosts (22/146 with < 20 linked pairs); habit stability rho 0.37; shortcut: switching rate AUC 0.58 but count-based P95 AUC 0.74 (attackers more active in 2010); ROT exceeds own 2010 P95 for 88-100% of hosts in every habit tertile. AUDIT.md written; stopped for owner review before pre-registration |
| 2026-11-26 06:14 | Owner decision | review of AUDIT.md | No attack generation, simulation, injection or modification; use only 2010 activity, existing pre-attack activity, existing Dev simulation and the closed study's existing cases; habit as a rate. AUDIT §6-7 marked superseded |
| 2026-11-26 06:14 | Population check (read-only) | python population_check.py | dev 859 users: 714 habit-eligible (>= 20 linked pairs, >= 2 baseline channels): 347/348 attackers, 367/511 benign; habit-rate AUC 0.586; closed-study eligible hosts H146 123/145, R253 97/104, S8 36/36 |
| 2026-11-26 06:14 | Revised plan | PLAN.md written (draft, not frozen) | stopped for owner review; nothing trained |
| 2026-11-26 06:17 | Owner approval | "Freeze as recommended" on PLAN.md | population = 714 habit-eligible users; primary Delta1 = Det(H) - Det(A) at iso-FPR 3%, Outcome-1 margin +2 pts, no-value band +-3 pts; >= 20 linked pairs |
| 2026-11-26 06:17 | Pre-registration frozen | sha256sum PREREGISTRATION.md > PREREGISTRATION.sha256 | 55f15d44a978c58db6a45b73c9f077de042ede430cff31dac78abaafe4da6bef. Frozen constants: m0 = 0.4444444444; tertile cuts 0.3868524 / 0.5; bootstrap rng 2026; habit file sha adc39b4b... Nothing trained; no 2011 switching feature computed yet |
| 2026-11-26 06:22 | Gates G1-G4 + feature computation | python gates.py | **ALL PASS** (results/gates.json): G1 toy sequences, causality, r0 reproduced exactly (max diff 0); G2 shortcut AUC: R 0.513-0.525, H 0.553-0.568 (all < 0.60; 689 users with pre-attack rows); G3 min 67 benign / 68 attackers per validation fold, no group overlap; G4 pipeline, folds, rules, dataset, labels identical to hashes_before. Features: data/features_p1.parquet (714 users), data/features_p2.parquet (123 hosts, existing H146 cases, read-only source) |
| 2026-11-26 06:22 | Training | python train.py (LightGBM defaults, seed 42 + repeat, existing folds restricted to P1) | arms A 100 / R 104 / H 104; 15 folds x 3 arms; every P1 user and P2 case scored OOF in each repeat; 85 s; scores/ |
| 2026-11-26 06:22 | Integrity check | python hash_artifacts.py after | 448 pre-existing files unchanged (closed studies included), 0 removed, 0 added outside the study. No estimand computed yet |
| 2026-11-26 06:24 | Implementation note (before the analysis run) | analyze.py written; choices not fixed by the protocol, fixed before any estimand: raw-probability scale; one rng default_rng(2026) drawing first P1 user resamples then P2 host resamples; S7 PR-AUC per repeat averaged over repeats, bootstrap-weighted exact step-wise AP (checked equal to sklearn); 'Delta2 CI includes 0' = ci_low <= 0 <= ci_high | - |
| 2026-11-26 06:27 | Registered analysis | python analyze.py | Det @3%: A 0.899, R 0.899, H 0.894. Primary Delta1 = -0.0048 [-0.0125, +0.0038]; key Delta2 = -0.0048 [-0.0115, +0.0019]. S1-S6 n.s. after Holm; S7 PR-AUC H-A = -0.0012 [-0.0019, -0.0006], Holm p < 0.005 (negative, negligible). **Outcome 3 - no meaningful added value** (both CIs within +-3 pts). Held-out benign FPR 3.0-3.8% (3%), 1.0-1.3% (1%) |
| 2026-11-26 06:27 | Figures, inputs manifest, README, run_all.ps1, REPORT.md | python figures.py; python inputs_manifest.py | 2 figures; 15 external inputs hashed, none changed |
| 2026-11-26 06:27 | Final integrity check | python hash_artifacts.py after | 448 pre-existing files unchanged (closed studies included), 0 removed, 0 added outside the study; PREREGISTRATION.md hash 55f15d44a978c58d... unchanged |

## 2026-11-26 — Folder move (no re-run, no result changed)
- The folder moved from `Model/study_switching_baseline/` to `Model/studies/switching_baseline/` for the MUBSIR Model v2 repository layout.
- Only path lines changed: `MODEL_DIR = HERE.parent.parent`, paths to sibling studies, `run_all.ps1` → `..\..\.venv`.
- No score, result, figure, report, pre-registration or hash file changed. Old folder names in the text above are kept as records.
- Details and the before/after hash check are in `Model/RESTRUCTURE.md`.
