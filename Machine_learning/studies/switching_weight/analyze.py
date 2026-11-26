"""
Registered analysis of the switching-weight study (PREREGISTRATION.md §7-§9). Reads scores/, labels via the dev order of
scores, dev metadata and (descriptive only) the 2010 habit file of study_switching_baseline. Never reads test data.
Writes results/analysis_*.

Implementation details fixed before the first run (logged in EXPERIMENT_LOG.md):
- thresholds on the raw probability scale (monotone in risk);
- S4 PR-AUC: exact step-wise weighted average precision (checked equal to sklearn), per repeat, averaged over repeats;
- "Holm p over the three w" for Outcome 3 uses the two-sided bootstrap p of each Delta_w.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent.parent  # studies/<name>/ -> Model/ (moved 2026-11-26)
sys.path.insert(0, str(MODEL_DIR / "pipeline"))
from metrics import threshold_at_fpr  # noqa: E402

PRE = (HERE / "PREREGISTRATION.sha256").read_text().strip()
assert hashlib.sha256((HERE / "PREREGISTRATION.md").read_bytes()).hexdigest() == PRE, "pre-registration changed"
gates = json.loads((HERE / "results" / "gates.json").read_text())
assert gates["g1_passed"] and gates["g3_passed"], "hard gate failed -> study stopped"
RES, SC = HERE / "results", HERE / "scores"
t0 = time.time()
NB, ALPHA, FPRS, REPS = 2000, 0.05, (0.03, 0.01), (0, 1, 2)
ARMS = ["W0", "A0", "W2", "W5", "W10", "SW-only"]
WS = ["W2", "W5", "W10"]

um = pd.read_csv(MODEL_DIR / "config" / "user_meta.csv").set_index("user")
folds = pd.read_csv(MODEL_DIR / "config" / "cv_folds.csv")
valid = folds[folds.role == "valid"]
fold_of = {r: dict(zip(g.user, g.fold)) for r, g in valid.groupby("repeat")}
S1 = {a: pd.read_parquet(SC / f"p1_{a}.parquet") for a in ARMS}
S2 = {a: pd.read_parquet(SC / f"p2_{a}.parquet") for a in ARMS}
base = S1["A0"]
users = sorted(base.user.unique())
benign = np.array([u for u in users if um.loc[u, "is_attacker"] == 0])
attackers = np.array([u for u in users if um.loc[u, "is_attacker"] == 1])
assert len(users) == 859 and len(benign) == 511 and len(attackers) == 348

# ------------------------------------------------------------------ thresholds
THR, CHK, ALARMED = {}, [], {}
for a in ARMS:
    for r in REPS:
        pk = S1[a].groupby("user")[f"rep{r}"].max().loc[benign]
        fk = pd.Series({u: fold_of[r][u] for u in benign})
        for f in FPRS:
            al = []
            for k in sorted(fk.unique()):
                other, own = pk[(fk != k).to_numpy()], pk[(fk == k).to_numpy()]
                t = threshold_at_fpr(other.to_numpy(), f)
                THR[(a, r, k, f)] = t
                CHK.append(dict(arm=a, repeat=r, fold=k, fpr=f, heldout_fpr=float((own > t).mean())))
                al += list(own.index[own > t])
            ALARMED[(a, r, f)] = al
chk = pd.DataFrame(CHK)
chk.to_csv(RES / "analysis_threshold_check.csv", index=False)

# ------------------------------------------------------------------ P1 detection
att_rows = base[(base.y_10 == 1) & base.user.isin(attackers)]
DET1 = {}
for a in ARMS:
    sc = S1[a].loc[att_rows.index]
    for f in FPRS:
        m = np.zeros((len(REPS), len(attackers)))
        for r in REPS:
            t = np.array([THR[(a, r, kk, f)] for kk in att_rows["user"].map(fold_of[r])])
            hit = pd.Series(sc[f"rep{r}"].to_numpy() > t, index=att_rows.index)
            m[r] = hit.groupby(att_rows["user"]).any().reindex(attackers).to_numpy()
        DET1[(a, f)] = m
pd.DataFrame({f"{a}|{f}": DET1[(a, f)].mean(axis=0) for a in ARMS for f in FPRS}, index=attackers) \
    .to_csv(RES / "analysis_p1_attacker_detection.csv")

# ------------------------------------------------------------------ P2 detection
keys2 = S2["A0"][["case", "host", "variant", "L", "in_span"]]
span = keys2[keys2.in_span]
DET2 = {}
for a in ARMS:
    for f in FPRS:
        cols = []
        for r in REPS:
            t = np.array([THR[(a, r, kk, f)] for kk in span["host"].map(fold_of[r])])
            cols.append(pd.Series(S2[a].loc[span.index, f"rep{r}"].to_numpy() > t, index=span.index).groupby(span["case"]).any())
        DET2[(a, f)] = pd.concat(cols, axis=1).mean(axis=1)
cmeta = keys2.drop_duplicates("case").set_index("case")[["host", "variant", "L"]]

# ------------------------------------------------------------------ bootstrap
rng = np.random.default_rng(2026)


def strat_weights(units, is_att):
    W = np.zeros((NB, len(units)))
    for s in (0, 1):
        idx = np.flatnonzero(is_att == s)
        dr = rng.integers(0, len(idx), size=(NB, len(idx)))
        for b in range(NB):
            np.add.at(W[b], idx[dr[b]], 1)
    return W


W1 = strat_weights(np.array(users), um.loc[users, "is_attacker"].to_numpy())
pos1 = {u: i for i, u in enumerate(users)}
Watt = W1[:, [pos1[u] for u in attackers]]
hosts2 = np.array(sorted(cmeta.host.unique()))
W2b = strat_weights(hosts2, um.loc[hosts2, "is_attacker"].to_numpy())


def det1(a, f, mask=None):
    v = DET1[(a, f)].mean(axis=0)
    m = np.ones(len(attackers), bool) if mask is None else mask
    return v[m].mean(), (Watt[:, m] @ v[m]) / Watt[:, m].sum(axis=1)


def det2(a, f, variant):
    c = cmeta[cmeta.variant == variant]
    g = pd.DataFrame({"d": DET2[(a, f)].loc[c.index], "h": c.host}).groupby("h")["d"].agg(["sum", "count"]) \
        .reindex(hosts2, fill_value=0)
    return g["sum"].sum() / g["count"].sum(), (W2b @ g["sum"].to_numpy()) / (W2b @ g["count"].to_numpy())


def diff(x, y):
    return x[0] - y[0], x[1] - y[1]


def summ(name, est, desc):
    p, b = est
    lo, hi = np.percentile(b, [2.5, 97.5])
    lo90, hi90 = np.percentile(b, [5, 95])
    pv = min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))
    return dict(id=name, estimand=desc, estimate=float(p), ci_low=float(lo), ci_high=float(hi), ci90_low=float(lo90),
                ci90_high=float(hi90), p_two_sided=float(pv))


sw = um.loc[attackers, "switching_level"].to_numpy()
prim = [summ(f"PRIMARY_{w}", diff(det1(w, 0.03), det1("A0", 0.03)), f"Delta = Det({w}) - Det(A0), iso-FPR 3%") for w in WS]
order = np.argsort([p["p_two_sided"] for p in prim])
run = 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (len(prim) - rank) * prim[i]["p_two_sided"]))
    prim[i]["p_holm_over_w"] = run

sec = [summ("S1", diff(det1("W5", 0.01), det1("A0", 0.01)), "Delta_5 at iso-FPR 1%"),
       summ("S2", diff(diff(det1("W5", 0.03, sw == "High"), det1("A0", 0.03, sw == "High")),
                       diff(det1("W5", 0.03, sw == "Low"), det1("A0", 0.03, sw == "Low"))),
            "Delta_5 High-switching minus Low-switching attackers"),
       summ("S3", diff(diff(det2("W5", 0.03, "ROT"), det2("A0", 0.03, "ROT")), diff(det2("W5", 0.03, "BLK"), det2("A0", 0.03, "BLK"))),
            "P2: [W5-A0](ROT) - [W5-A0](BLK)")]


def ap_batch(y, s, user_idx, Wb):
    o = np.argsort(-s, kind="mergesort")
    ys, ss, ui = y[o], s[o], user_idx[o]
    last = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss) - 1]
    w = Wb[:, ui]
    tp = np.cumsum(w * ys, axis=1)[:, last]
    fp = np.cumsum(w * (1 - ys), axis=1)[:, last]
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = tp / tp[:, -1:]
    return (np.diff(np.c_[np.zeros(len(Wb)), rec], axis=1) * prec).sum(axis=1)


y1 = base["y_10"].to_numpy().astype(float)
uidx = base["user"].map(pos1).to_numpy()
assert np.isclose(ap_batch(y1, S1["A0"]["rep0"].to_numpy(), uidx, np.ones((1, len(users))))[0],
                  average_precision_score(y1, S1["A0"]["rep0"].to_numpy())), "AP implementation mismatch"
pr_pt = {a: float(np.mean([average_precision_score(y1, S1[a][f"rep{r}"].to_numpy()) for r in REPS])) for a in ARMS}
pr_b = {}
for a in ("A0", "W5"):
    acc = np.zeros(NB)
    for r in REPS:
        s = S1[a][f"rep{r}"].to_numpy()
        for b0 in range(0, NB, 100):
            acc[b0:b0 + 100] += ap_batch(y1, s, uidx, W1[b0:b0 + 100])
    pr_b[a] = acc / len(REPS)
    print(f"PR-AUC bootstrap {a} done {time.time() - t0:.0f}s", flush=True)
sec.append(summ("S4", (pr_pt["W5"] - pr_pt["A0"], pr_b["W5"] - pr_b["A0"]), "row-level PR-AUC(W5) - PR-AUC(A0)"))
order = np.argsort([s["p_two_sided"] for s in sec])
run = 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (len(sec) - rank) * sec[i]["p_two_sided"]))
    sec[i]["p_holm"] = run
for s in sec:
    s["significant_holm"] = bool(s["p_holm"] < ALPHA)

# ------------------------------------------------------------------ outcome (§9, in order)
g2_fail = not gates["g2_passed"]
harm = any(p["ci_high"] < 0 and p["estimate"] <= -0.02 for p in prim)
equiv = all(-0.02 < p["ci90_low"] and p["ci90_high"] < 0.02 for p in prim)
helps = any(p["estimate"] >= 0.02 and p["p_holm_over_w"] < ALPHA and p["ci_low"] > 0 for p in prim)
outcome = 4 if g2_fail else 2 if harm else 1 if equiv else 3 if helps else 4
labels = {1: "Outcome 1 - same detection (redundancy confirmed)", 2: "Outcome 2 - up-weighting harms detection",
          3: "Outcome 3 - up-weighting helps detection", 4: "Outcome 4 - inconclusive"}

# ------------------------------------------------------------------ descriptives
rows = []
groups = {"all": np.ones(len(attackers), bool)}
for lv in ("Low", "Medium", "High"):
    groups[f"switching_{lv}"] = sw == lv
for st in ("Low", "Medium", "High"):
    groups[f"stealth_{st}"] = um.loc[attackers, "stealth_level"].to_numpy() == st
for cc in (2, 3):
    groups[f"channels_{cc}"] = um.loc[attackers, "channel_count"].to_numpy() == cc
for f in FPRS:
    for gname, m in groups.items():
        for a in ARMS:
            p, b = det1(a, f, m)
            d = diff(det1(a, f, m), det1("A0", f, m))
            rows.append(dict(fpr=f, group=gname, attackers=int(m.sum()), arm=a, det=p, ci_low=np.percentile(b, 2.5),
                             ci_high=np.percentile(b, 97.5), minus_A0=d[0], minus_A0_ci_low=np.percentile(d[1], 2.5),
                             minus_A0_ci_high=np.percentile(d[1], 97.5)))
pd.DataFrame(rows).to_csv(RES / "analysis_p1_detection_table.csv", index=False)
p2rows = [dict(fpr=f, variant=v, arm=a, cases=int((cmeta.variant == v).sum()), det=float(det2(a, f, v)[0]))
          for f in FPRS for v in sorted(cmeta.variant.unique()) for a in ARMS]
pd.DataFrame(p2rows).to_csv(RES / "analysis_p2_detection_table.csv", index=False)
uq = []
for f in FPRS:
    for w in WS + ["W0"]:
        x, y_ = DET1[(w, f)], DET1[("A0", f)]
        uq.append(dict(fpr=f, comparison=f"{w} vs A0", only_weighted=int((((x == 1) & (y_ == 0)).sum(axis=0) >= 2).sum()),
                       only_A0=int((((y_ == 1) & (x == 0)).sum(axis=0) >= 2).sum())))
pd.DataFrame(uq).to_csv(RES / "analysis_unique_detections.csv", index=False)
habit = pd.read_csv(MODEL_DIR / "studies" / "switching_baseline" / "results" / "habit_2010_dev_users.csv").set_index("user")
elig_b = habit[habit.eligible & (habit.is_attacker == 0)]
hi_tert = set(elig_b.index[elig_b.rate > 0.5])
ba = []
for a in ARMS:
    for f in FPRS:
        al = [set(ALARMED[(a, r, f)]) & set(elig_b.index) for r in REPS]
        ba.append(dict(arm=a, fpr=f, alarmed_benign_mean=np.mean([len(ALARMED[(a, r, f)]) for r in REPS]),
                       alarmed_eligible_mean=np.mean([len(s) for s in al]),
                       share_high_habit=np.mean([len(s & hi_tert) / len(s) if s else np.nan for s in al]),
                       base_share_high_habit=len(hi_tert) / len(elig_b)))
pd.DataFrame(ba).to_csv(RES / "analysis_benign_alarms_vs_habit.csv", index=False)

out = dict(preregistration_sha256=PRE, gates=dict(g1=gates["g1_passed"], g2=gates["g2_passed"], g3=gates["g3_passed"],
                                                 switching_shap_share=gates["switching_shap_share"]),
           det_P1_3pct={a: float(det1(a, 0.03)[0]) for a in ARMS}, det_P1_1pct={a: float(det1(a, 0.01)[0]) for a in ARMS},
           pr_auc=pr_pt, det_P2_3pct={f"{a}|{v}": float(det2(a, 0.03, v)[0]) for a in ARMS for v in ("ROT", "BLK")},
           primary=prim, secondary=sec, outcome_conditions=dict(g2_fail=g2_fail, harm=harm, equivalence=equiv, helps=helps),
           outcome=outcome, outcome_label=labels[outcome],
           heldout_benign_fpr=chk.groupby(["arm", "fpr"])["heldout_fpr"].mean().reset_index().to_dict("records"),
           seconds=round(time.time() - t0))
(RES / "analysis_primary.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
pd.DataFrame(prim + sec).to_csv(RES / "analysis_estimands.csv", index=False)

print("\nDet P1 @3%:", {k: round(v, 4) for k, v in out["det_P1_3pct"].items()})
print("Det P1 @1%:", {k: round(v, 4) for k, v in out["det_P1_1pct"].items()})
print("PR-AUC:", {k: round(v, 4) for k, v in pr_pt.items()})
for s in prim + sec:
    print(f"{s['id']:11s} {s['estimate']:+.4f} 95%[{s['ci_low']:+.4f},{s['ci_high']:+.4f}] 90%[{s['ci90_low']:+.4f},{s['ci90_high']:+.4f}] "
          f"p={s['p_two_sided']:.3f}" + (f" holm={s['p_holm']:.3f}" if "p_holm" in s else f" holm_w={s['p_holm_over_w']:.3f}"))
print("conditions:", out["outcome_conditions"], "->", labels[outcome])
print(chk.groupby(["arm", "fpr"])["heldout_fpr"].mean().round(4).unstack().to_string())
print("done", round(time.time() - t0), "s")
