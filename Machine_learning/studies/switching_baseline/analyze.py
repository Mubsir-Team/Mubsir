"""
Registered analysis of the switching-baseline study (PREREGISTRATION.md §9-§11). Reads scores/, data/features_p2.parquet
(keys only), the habit file and dev metadata; never reads test data. Writes results/analysis_*.

Implementation details fixed before the first run (not specified in the protocol; logged in EXPERIMENT_LOG.md):
- thresholds on the raw probability scale (monotone in risk);
- one rng = default_rng(2026): first the 2,000 stratified P1 user draws, then the 2,000 stratified P2 host draws;
- S7 PR-AUC: computed per repeat on the P1 OOF rows and averaged over repeats (as Det); weighted by the bootstrap user
  counts (exact step-wise average precision, identical to sklearn's average_precision_score with sample weights);
- "Delta2 CI includes 0" = ci_low <= 0 <= ci_high.
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
assert gates["hard_gates_passed"]
RES, SC = HERE / "results", HERE / "scores"
t0 = time.time()
NB, ALPHA, FPRS, ARMS, REPS = 2000, 0.05, (0.03, 0.01), ("A", "R", "H"), (0, 1, 2)
LOW_CUT, HIGH_CUT = 0.3868524, 0.5

habit = pd.read_csv(RES / "habit_2010_dev_users.csv").set_index("user")
um = pd.read_csv(MODEL_DIR / "config" / "user_meta.csv").set_index("user")
folds = pd.read_csv(MODEL_DIR / "config" / "cv_folds.csv")
P1 = sorted(habit.index[habit.eligible])
valid = folds[(folds.role == "valid") & folds.user.isin(P1)]
fold_of = {r: dict(zip(g.user, g.fold)) for r, g in valid.groupby("repeat")}
benign = np.array([u for u in P1 if um.loc[u, "is_attacker"] == 0])
attackers = np.array([u for u in P1 if um.loc[u, "is_attacker"] == 1])
assert len(benign) == 367 and len(attackers) == 347

S1 = {a: pd.read_parquet(SC / f"p1_{a}.parquet") for a in ARMS}
S2 = {a: pd.read_parquet(SC / f"p2_{a}.parquet") for a in ARMS}
base1 = S1["A"]
base1["pos"] = base1.groupby("user").cumcount()

# ------------------------------------------------------------------ thresholds (cross-fitted, benign P1 users of other folds)
THR, CHK = {}, []
for a in ARMS:
    for r in REPS:
        pk = S1[a].groupby("user")[f"rep{r}"].max().loc[benign]
        fk = pd.Series({u: fold_of[r][u] for u in benign})
        for k in sorted(fk.unique()):
            other, own = pk[(fk != k).to_numpy()], pk[(fk == k).to_numpy()]
            for f in FPRS:
                t = threshold_at_fpr(other.to_numpy(), f)
                THR[(a, r, k, f)] = t
                CHK.append(dict(arm=a, repeat=r, fold=k, fpr=f, n_ref=len(other), heldout_fpr=float((own > t).mean())))
chk = pd.DataFrame(CHK)
chk.to_csv(RES / "analysis_threshold_check.csv", index=False)

# ------------------------------------------------------------------ P1 detection (attackers: any y_10 = 1 row above threshold)
first_y = base1[base1.y == 1].groupby("user")["pos"].min()
att_rows = base1[(base1.y_10 == 1) & base1.user.isin(attackers)]
DET1, LAT1 = {}, []
for a in ARMS:
    sc = S1[a].loc[att_rows.index]
    for f in FPRS:
        m = np.zeros((len(REPS), len(attackers)))
        for r in REPS:
            k = att_rows["user"].map(fold_of[r]).to_numpy()
            t = np.array([THR[(a, r, kk, f)] for kk in k])
            hit = sc[f"rep{r}"].to_numpy() > t
            d = att_rows.assign(hit=hit)
            g = d.groupby("user")["hit"].any().reindex(attackers)
            m[r] = g.to_numpy()
            fd = d[d.hit].groupby("user")["pos"].min()
            for u, p in fd.items():
                LAT1.append(dict(arm=a, fpr=f, repeat=r, user=u, latency=int(p - first_y[u])))
        DET1[(a, f)] = m
det_case = pd.DataFrame({f"{a}|{f}": DET1[(a, f)].mean(axis=0) for a in ARMS for f in FPRS}, index=attackers)
det_case.to_csv(RES / "analysis_p1_attacker_detection.csv")

# ------------------------------------------------------------------ P2 detection (existing cases)
keys2 = S2["A"][["case", "host", "variant", "L", "in_span"]]
span = keys2[keys2.in_span]
DET2 = {}
for a in ARMS:
    for f in FPRS:
        cols = []
        for r in REPS:
            k = span["host"].map(fold_of[r]).to_numpy()
            t = np.array([THR[(a, r, kk, f)] for kk in k])
            hit = S2[a].loc[span.index, f"rep{r}"].to_numpy() > t
            cols.append(pd.Series(hit, index=span.index).groupby(span["case"]).any())
        DET2[(a, f)] = pd.concat(cols, axis=1).mean(axis=1)
cmeta = keys2.drop_duplicates("case").set_index("case")[["host", "variant", "L"]]
cmeta["host_attacker"] = cmeta["host"].map(um["is_attacker"]).astype(int)

# ------------------------------------------------------------------ bootstrap draws
rng = np.random.default_rng(2026)


def strat_weights(units, is_att):
    W = np.zeros((NB, len(units)))
    for s in (0, 1):
        idx = np.flatnonzero(is_att == s)
        dr = rng.integers(0, len(idx), size=(NB, len(idx)))
        for b in range(NB):
            np.add.at(W[b], idx[dr[b]], 1)
    return W


W1 = strat_weights(np.array(P1), um.loc[P1, "is_attacker"].to_numpy())
pos1 = {u: i for i, u in enumerate(P1)}
Watt = W1[:, [pos1[u] for u in attackers]]
hosts2 = np.array(sorted(cmeta.host.unique()))
W2 = strat_weights(hosts2, um.loc[hosts2, "is_attacker"].to_numpy())
pos2 = {h: i for i, h in enumerate(hosts2)}


def det1(a, f, mask=None):
    v = DET1[(a, f)].mean(axis=0)
    m = np.ones(len(attackers), bool) if mask is None else mask
    return v[m].mean(), (Watt[:, m] @ v[m]) / Watt[:, m].sum(axis=1)


def det2(a, f, variant):
    s = DET2[(a, f)]
    c = cmeta[cmeta.variant == variant]
    g = pd.DataFrame({"d": s.loc[c.index], "h": c.host}).groupby("h")["d"].agg(["sum", "count"]).reindex(hosts2, fill_value=0)
    return g["sum"].sum() / g["count"].sum(), (W2 @ g["sum"].to_numpy()) / (W2 @ g["count"].to_numpy())


def diff(x, y):
    return x[0] - y[0], x[1] - y[1]


def summ(name, est, desc):
    p, b = est
    lo, hi = np.percentile(b, [2.5, 97.5])
    pv = min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))
    return dict(id=name, estimand=desc, estimate=float(p), ci_low=float(lo), ci_high=float(hi), p_two_sided=float(pv))


r0 = habit.loc[attackers, "rate"].to_numpy()
low, high = r0 <= LOW_CUT, r0 > HIGH_CUT
sw = um.loc[attackers, "switching_level"].to_numpy()

d1 = diff(det1("H", 0.03), det1("A", 0.03))
d2 = diff(det1("H", 0.03), det1("R", 0.03))
primary = summ("PRIMARY", d1, "Delta1 = Det(H) - Det(A), P1 attackers, iso-FPR 3%")
key = summ("KEY", d2, "Delta2 = Det(H) - Det(R), P1 attackers, iso-FPR 3%")

sec = [summ("S1", diff(det1("H", 0.01), det1("A", 0.01)), "Delta1 at iso-FPR 1%"),
       summ("S2", d2, "Delta2 at iso-FPR 3%")]
d2l = diff(det1("H", 0.03, low), det1("R", 0.03, low))
d2h = diff(det1("H", 0.03, high), det1("R", 0.03, high))
sec.append(summ("S3", diff(d2l, d2h), f"Delta2 low-habit ({low.sum()}) minus high-habit ({high.sum()}) attackers"))
d1H = diff(det1("H", 0.03, sw == "High"), det1("A", 0.03, sw == "High"))
d1L = diff(det1("H", 0.03, sw == "Low"), det1("A", 0.03, sw == "Low"))
sec.append(summ("S4", diff(d1H, d1L), "Delta1 High-switching minus Low-switching attackers"))
sec.append(summ("S5", diff(diff(det2("H", 0.03, "ROT"), det2("A", 0.03, "ROT")), diff(det2("H", 0.03, "BLK"), det2("A", 0.03, "BLK"))),
                "P2: [H-A](ROT) - [H-A](BLK)"))
sec.append(summ("S6", diff(diff(det2("H", 0.03, "ROT"), det2("R", 0.03, "ROT")), diff(det2("H", 0.03, "BLK"), det2("R", 0.03, "BLK"))),
                "P2: [H-R](ROT) - [H-R](BLK)"))


# S7: weighted average precision, exact step-wise, per repeat, averaged over repeats
def ap_batch(y, s, user_idx, Wb):
    o = np.argsort(-s, kind="mergesort")
    ys, ss, ui = y[o], s[o], user_idx[o]
    last = np.r_[np.flatnonzero(np.diff(ss) != 0), len(ss) - 1]     # end of each tied-score group
    w = Wb[:, ui]                                                     # (b, rows)
    tp = np.cumsum(w * ys, axis=1)[:, last]
    fp = np.cumsum(w * (1 - ys), axis=1)[:, last]
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = tp / tp[:, -1:]
    return (np.diff(np.c_[np.zeros(len(Wb)), rec], axis=1) * prec).sum(axis=1)


y1 = base1["y_10"].to_numpy().astype(float)
uidx = base1["user"].map(pos1).to_numpy()
assert np.isclose(ap_batch(y1, S1["A"]["rep0"].to_numpy(), uidx, np.ones((1, len(P1))))[0],
                  average_precision_score(y1, S1["A"]["rep0"].to_numpy())), "AP implementation mismatch"
pr_pt, pr_b = {}, {}
for a in ("A", "H"):
    pr_pt[a] = np.mean([average_precision_score(y1, S1[a][f"rep{r}"].to_numpy()) for r in REPS])
    acc = np.zeros(NB)
    for r in REPS:
        s = S1[a][f"rep{r}"].to_numpy()
        for b0 in range(0, NB, 100):
            acc[b0:b0 + 100] += ap_batch(y1, s, uidx, W1[b0:b0 + 100])
    pr_b[a] = acc / len(REPS)
    print(f"PR-AUC bootstrap {a} done {time.time() - t0:.0f}s", flush=True)
sec.append(summ("S7", (pr_pt["H"] - pr_pt["A"], pr_b["H"] - pr_b["A"]), "row-level PR-AUC(H) - PR-AUC(A), pooled P1 OOF"))

order = np.argsort([s["p_two_sided"] for s in sec])
run = 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (len(sec) - rank) * sec[i]["p_two_sided"]))
    sec[i]["p_holm"] = run
for s in sec:
    s["significant_holm"] = bool(s["p_holm"] < ALPHA)

# ------------------------------------------------------------------ outcome (§11, in order)
gate_fail = (not gates["hard_gates_passed"]) or bool(gates["g2_h_failed"])
o1 = primary["estimate"] >= 0.02 and primary["ci_low"] > 0 and key["estimate"] > 0 and key["ci_low"] > 0
o2 = primary["ci_low"] > 0 and key["ci_low"] <= 0 <= key["ci_high"]
o3 = (-0.03 < primary["ci_low"] and primary["ci_high"] < 0.03) and (-0.03 < key["ci_low"] and key["ci_high"] < 0.03)
outcome = 4 if gate_fail else 1 if o1 else 2 if o2 else 3 if o3 else 4
labels = {1: "Outcome 1 - personal switching habit adds information",
          2: "Outcome 2 - switching rate helps, the personal reference does not",
          3: "Outcome 3 - no meaningful added value", 4: "Outcome 4 - inconclusive"}

# ------------------------------------------------------------------ descriptives
rows = []
groups = {"all": np.ones(len(attackers), bool), "habit_low": low, "habit_mid": ~low & ~high, "habit_high": high}
for lv in ("Low", "Medium", "High"):
    groups[f"switching_{lv}"] = sw == lv
for cc in (2, 3):
    groups[f"channels_{cc}"] = um.loc[attackers, "channel_count"].to_numpy() == cc
for st in ("Low", "Medium", "High"):
    groups[f"stealth_{st}"] = um.loc[attackers, "stealth_level"].to_numpy() == st
for f in FPRS:
    for gname, m in groups.items():
        rec = dict(fpr=f, group=gname, attackers=int(m.sum()))
        for a in ARMS:
            p, b = det1(a, f, m)
            rec[f"det_{a}"], rec[f"ci_{a}"] = p, f"[{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]"
        for a1, a0 in (("H", "A"), ("H", "R"), ("R", "A")):
            p, b = diff(det1(a1, f, m), det1(a0, f, m))
            rec[f"{a1}-{a0}"], rec[f"{a1}-{a0}_ci"] = p, f"[{np.percentile(b, 2.5):+.3f}, {np.percentile(b, 97.5):+.3f}]"
        rows.append(rec)
pd.DataFrame(rows).to_csv(RES / "analysis_p1_detection_table.csv", index=False)

rows = []
h_t = pd.Series(np.where(habit.loc[cmeta.host, "rate"] <= LOW_CUT, "low", np.where(habit.loc[cmeta.host, "rate"] > HIGH_CUT, "high", "mid")),
                index=cmeta.index)
for f in FPRS:
    for v in sorted(cmeta.variant.unique()):
        for t in ("all", "low", "mid", "high"):
            c = cmeta[(cmeta.variant == v) & ((h_t == t) if t != "all" else True)]
            rec = dict(fpr=f, variant=v, habit_tertile=t, cases=len(c), hosts=c.host.nunique())
            for a in ARMS:
                rec[f"det_{a}"] = float(DET2[(a, f)].loc[c.index].mean())
            rows.append(rec)
pd.DataFrame(rows).to_csv(RES / "analysis_p2_detection_table.csv", index=False)

lat = pd.DataFrame(LAT1)
lat.groupby(["arm", "fpr"])["latency"].describe().to_csv(RES / "analysis_latency.csv")
uq = []
for f in FPRS:
    for a1, a0 in (("H", "A"), ("H", "R")):
        x, y_ = DET1[(a1, f)], DET1[(a0, f)]
        uq.append(dict(fpr=f, comparison=f"{a1} vs {a0}", only_first=int((((x == 1) & (y_ == 0)).sum(axis=0) >= 2).sum()),
                       only_second=int((((y_ == 1) & (x == 0)).sum(axis=0) >= 2).sum())))
pd.DataFrame(uq).to_csv(RES / "analysis_unique_detections.csv", index=False)
rma = summ("desc_R-A", diff(det1("R", 0.03), det1("A", 0.03)), "Det(R) - Det(A), iso-FPR 3% (descriptive)")

out = dict(preregistration_sha256=PRE, populations=dict(P1_users=len(P1), P1_attackers=len(attackers), P1_benign=len(benign),
                                                       P2_hosts=len(hosts2), P2_cases=int(len(cmeta))),
           det_P1_3pct={a: float(det1(a, 0.03)[0]) for a in ARMS}, det_P1_1pct={a: float(det1(a, 0.01)[0]) for a in ARMS},
           det_P2_3pct={f"{a}|{v}": float(det2(a, 0.03, v)[0]) for a in ARMS for v in ("ROT", "BLK")},
           pr_auc={a: float(v) for a, v in pr_pt.items()},
           primary=primary, key_secondary=key, secondary=sec, r_minus_a=rma,
           outcome_conditions=dict(gate_fail=gate_fail, o1=bool(o1), o2=bool(o2), o3=bool(o3)),
           outcome=outcome, outcome_label=labels[outcome],
           heldout_benign_fpr=chk.groupby(["arm", "fpr"])["heldout_fpr"].mean().reset_index().to_dict("records"),
           seconds=round(time.time() - t0))
(RES / "analysis_primary.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
pd.DataFrame([primary, key, rma] + sec).to_csv(RES / "analysis_estimands.csv", index=False)

print("\nDet P1 @3%:", {k: round(v, 4) for k, v in out["det_P1_3pct"].items()}, " @1%:", {k: round(v, 4) for k, v in out["det_P1_1pct"].items()})
print("Det P2 @3%:", {k: round(v, 4) for k, v in out["det_P2_3pct"].items()})
for s in [primary, key, rma] + sec:
    print(f"{s['id']:8s} {s['estimate']:+.4f} CI [{s['ci_low']:+.4f}, {s['ci_high']:+.4f}] p={s['p_two_sided']:.3f}"
          + (f" holm={s['p_holm']:.3f}" if "p_holm" in s else ""))
print("conditions:", out["outcome_conditions"], "->", labels[outcome])
print(chk.groupby(["arm", "fpr"])["heldout_fpr"].mean().round(4).to_string())
print("done", round(time.time() - t0), "s")
