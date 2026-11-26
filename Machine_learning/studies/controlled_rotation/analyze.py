"""
Registered analysis of the controlled rotation study (PREREGISTRATION.md §1, §6, §7, §8). Reads only scores/, data/ and the
dev fold assignment; never reads test data or test scores. Writes results/analysis_*.

Implementation choices fixed here BEFORE the first run (not specified in the protocol; logged in EXPERIMENT_LOG.md):
- scores are compared on the model's raw probability scale (monotone in risk, so thresholds and detections are identical);
- bootstrap RNG: numpy default_rng(2026) per host set; one shared set of 2,000 draws per host set, so all estimands on the
  same hosts use the same resamples;
- "S5 positive after Holm" = point estimate > 0 and Holm-adjusted p < 0.05; if Outcomes 2 and 3 both apply, Outcome 2 is
  reported (listed first in §7) and the overlap is flagged;
- ceiling/floor (Outcome 4) is checked before Outcomes 1-3;
- p = min(1, 2 * min(share of bootstrap values <= 0, share >= 0)).
- A-frozen (descriptive): the frozen bundle scored on H146 cases at its fixed operating points (risk > 85, and the frozen
  dev 1% user-FPR point 92.632). It is in-sample for the hosts' own benign days and enters no test.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent.parent  # studies/<name>/ -> Model/ (moved 2026-11-26)
sys.path.insert(0, str(MODEL_DIR / "pipeline"))
from metrics import threshold_at_fpr  # noqa: E402

assert hashlib.sha256((HERE / "PREREGISTRATION.md").read_bytes()).hexdigest() == \
    (HERE / "PREREGISTRATION.sha256").read_text().strip(), "pre-registration changed"
SC, RES = HERE / "scores", HERE / "results"
t0 = time.time()
NB, ALPHA, FPRS = 2000, 0.05, (0.03, 0.01)
ARMS1 = ["B", "C", "C+", "A-cv", "D"]
ARMS2 = ["C+", "D"]

# ------------------------------------------------------------------ dev users, folds
umeta = pd.read_csv(MODEL_DIR / "config" / "user_meta.csv").set_index("user")
folds = pd.read_csv(MODEL_DIR / "config" / "cv_folds.csv")
valid = folds[folds.role == "valid"]
fold_of = {r: dict(zip(g.user, g.fold)) for r, g in valid.groupby("repeat")}
row_user = pd.read_parquet(SC / "orig_users.parquet")["user"].astype(str).to_numpy()
dev_users = np.unique(row_user)
benign = np.array([u for u in dev_users if umeta.loc[u, "is_attacker"] == 0])
assert len(dev_users) == 859 and len(benign) == 511, (len(dev_users), len(benign))


def peaks(orig_scores):
    """Whole-period peak score per benign dev user (index = user)."""
    s = pd.Series(orig_scores, index=row_user)
    return s.groupby(level=0).max().loc[benign]


def thresholds(orig, reps):
    """thr[(r, k, f)] from the peaks of the 511 benign users in the OTHER folds of repeat r; plus realised held-out FPR."""
    thr, chk = {}, []
    for r in reps:
        pk = peaks(orig[f"rep{r}"].to_numpy())
        fk = pd.Series({u: fold_of[r][u] for u in benign})
        for k in sorted(fk.unique()):
            other, own = pk[fk[pk.index] != k], pk[fk[pk.index] == k]
            for f in FPRS:
                t = threshold_at_fpr(other.to_numpy(), f)
                thr[(r, k, f)] = t
                chk.append(dict(repeat=r, fold=k, fpr=f, n_ref=len(other), n_heldout=len(own),
                                heldout_fpr=float((own > t).mean())))
    return thr, chk


# ------------------------------------------------------------------ case-level detection
def detect(cases, thr, reps):
    """Per case and repeat: detected (any in-span day > threshold of the host's validation fold) and latency."""
    out = []
    span = cases[cases.in_span]
    for r in reps:
        k = span["host"].map(fold_of[r]).to_numpy()
        for f in FPRS:
            t = np.array([thr[(r, kk, f)] for kk in k])
            hit = span[f"rep{r}"].to_numpy() > t
            d = span.assign(hit=hit, lat=np.where(hit, span["row_index"] - span["first_attack_index"], np.inf))
            g = d.groupby("case").agg(detected=("hit", "any"), latency=("lat", "min"))
            g["repeat"], g["fpr"] = r, f
            out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)


meta_cols = ["case", "host", "cohort", "variant", "L"]
DET, CHK = [], []
for regime, arms, reps in ((1, ARMS1, (0, 1, 2)), (2, ARMS2, (0,))):
    for a in arms:
        orig = pd.read_parquet(SC / f"orig_regime{regime}_{a}.parquet")
        cases = pd.read_parquet(SC / f"cases_regime{regime}_{a}.parquet")
        if regime == 2:
            cases = cases[cases.cohort == "H146"]
        assert not cases[[f"rep{r}" for r in reps]].isna().any().any()
        thr, chk = thresholds(orig, reps)
        CHK += [dict(regime=regime, arm=a, **c) for c in chk]
        d = detect(cases, thr, reps)
        d = d.merge(cases[meta_cols].drop_duplicates("case"), on="case", how="left")
        d["regime"], d["arm"] = regime, a
        DET.append(d)
    print(f"regime {regime} detection done {time.time() - t0:.0f}s", flush=True)
DET = pd.concat(DET, ignore_index=True)
DET["host_attacker"] = DET["host"].map(umeta["is_attacker"]).astype(int)
DET["vgroup"] = np.where(DET["variant"].str.startswith("FIX"), "FIX", DET["variant"])
DET.to_csv(RES / "analysis_case_detection.csv", index=False)
chk = pd.DataFrame(CHK)
chk.to_csv(RES / "analysis_threshold_check.csv", index=False)

# mean over repeats per case (Det is averaged over repeats; linear, so equal to averaging per-repeat Det)
CD = (DET.groupby(["regime", "arm", "fpr", "case", "host", "host_attacker", "cohort", "variant", "vgroup", "L"])
      ["detected"].mean().reset_index())


# ------------------------------------------------------------------ bootstrap machinery
class HostSet:
    def __init__(self, hosts, seed=2026):
        self.hosts = np.array(sorted(hosts))
        self.pos = {h: i for i, h in enumerate(self.hosts)}
        att = umeta.loc[self.hosts, "is_attacker"].to_numpy()
        rng = np.random.default_rng(seed)
        W = np.zeros((NB, len(self.hosts)))
        for s in (0, 1):
            idx = np.flatnonzero(att == s)
            draws = rng.integers(0, len(idx), size=(NB, len(idx)))
            for b in range(NB):
                np.add.at(W[b], idx[draws[b]], 1)
        self.W = W
        self.n_att, self.n_ben = int(att.sum()), int((att == 0).sum())

    def det(self, sel):
        """Point estimate and bootstrap vector of Det over the selected case rows (mean over cases)."""
        S = np.zeros(len(self.hosts))
        N = np.zeros(len(self.hosts))
        g = sel.groupby("host")["detected"].agg(["sum", "count"])
        ix = [self.pos[h] for h in g.index]
        S[ix], N[ix] = g["sum"].to_numpy(), g["count"].to_numpy()
        return S.sum() / N.sum(), (self.W @ S) / (self.W @ N)


def sel(regime, arm, f, vgroup, cohorts=("H146",), L=None):
    m = (CD.regime == regime) & (CD.arm == arm) & (CD.fpr == f) & (CD.vgroup == vgroup) & CD.cohort.isin(cohorts)
    if L is not None:
        m &= CD.L == L
    return CD[m]


def summarize(name, point, boot, desc):
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    return dict(id=name, estimand=desc, estimate=float(point), ci_low=float(lo), ci_high=float(hi), p_two_sided=float(p))


def interaction(hs, regime, arm1, arm0, f, v1="ROT", v0="BLK", cohorts=("H146",), L=None):
    """[Det(arm1,v1) - Det(arm0,v1)] - [Det(arm1,v0) - Det(arm0,v0)], plus its components."""
    d = {(a, v): hs.det(sel(regime, a, f, v, cohorts, L)) for a in (arm1, arm0) for v in (v1, v0)}
    e1 = d[(arm1, v1)][0] - d[(arm0, v1)][0]
    e0 = d[(arm1, v0)][0] - d[(arm0, v0)][0]
    b1 = d[(arm1, v1)][1] - d[(arm0, v1)][1]
    b0 = d[(arm1, v0)][1] - d[(arm0, v0)][1]
    return e1 - e0, b1 - b0, d, (e1, b1), (e0, b0)


H146 = HostSet(CD.loc[CD.cohort == "H146", "host"].unique())
ALL249 = HostSet(CD.loc[CD.cohort.isin(["H146", "R253"]), "host"].unique())
S8H = HostSet(CD.loc[CD.cohort == "S8", "host"].unique())
n_h146 = len(H146.hosts)

# ------------------------------------------------------------------ primary
dP, bP, compP, rotP, blkP = interaction(H146, 1, "D", "C+", 0.03)
primary = summarize("PRIMARY", dP, bP, "[D-C+](ROT) - [D-C+](BLK), iso-FPR 3%, regime 1, H146")
s2 = summarize("S2", rotP[0], rotP[1], "Det(D,ROT) - Det(C+,ROT), iso-FPR 3%")
blk_diff = summarize("desc_BLK", blkP[0], blkP[1], "Det(D,BLK) - Det(C+,BLK), iso-FPR 3% (descriptive)")

# ------------------------------------------------------------------ secondaries
sec = []
d1, b1, *_ = interaction(H146, 1, "D", "C+", 0.01)
sec.append(summarize("S1", d1, b1, "primary interaction at iso-FPR 1%"))
sec.append(s2)
d3, b3, *_ = interaction(H146, 1, "A-cv", "C+", 0.03)
sec.append(summarize("S3", d3, b3, "[A-cv-C+](ROT) - [A-cv-C+](BLK), iso-FPR 3%"))
d4, b4, *_ = interaction(H146, 2, "D", "C+", 0.03)
sec.append(summarize("S4", d4, b4, "primary interaction under regime 2 (repeat 0, trained with counterfactual data)"))
d5, b5, comp5, *_ = interaction(H146, 1, "C+", "C", 0.03, v1="BLK", v0="FIX")
s5 = summarize("S5", d5, b5, "[C+-C](BLK) - [C+-C](FIX pooled), iso-FPR 3%")
sec.append(s5)
for L in (6, 12):
    d6, b6, *_ = interaction(H146, 1, "D", "C+", 0.03, L=L)
    sec.append(summarize(f"S6_L{L}", d6, b6, f"primary interaction, L = {L} only"))
d7, b7, *_ = interaction(ALL249, 1, "D", "C+", 0.03, cohorts=("H146", "R253"))
sec.append(summarize("S7", d7, b7, f"primary interaction, relaxed cohort ({len(ALL249.hosts)} hosts: H146 + approximate R253)"))
d8, b8, *_ = interaction(S8H, 1, "D", "C+", 0.03, cohorts=("S8",))
sec.append(summarize("S8", d8, b8, f"primary interaction, strict natural-activity cohort ({len(S8H.hosts)} hosts)"))

# Holm across the 9 secondary tests
order = np.argsort([s["p_two_sided"] for s in sec])
m, running = len(sec), 0.0
for rank, i in enumerate(order):
    adj = min(1.0, (m - rank) * sec[i]["p_two_sided"])
    running = max(running, adj)
    sec[i]["p_holm"] = running
for s in sec:
    s["significant_holm"] = bool(s["p_holm"] < ALPHA)

# ------------------------------------------------------------------ decision rules (§7)
det_cp_rot, det_cp_blk = compP[("C+", "ROT")][0], compP[("C+", "BLK")][0]
ceiling = (det_cp_rot >= 0.97 and det_cp_blk >= 0.97) or (det_cp_rot <= 0.03 and det_cp_blk <= 0.03)
gate0 = json.loads((RES / "gate0.json").read_text())["passed"]
c1 = dict(delta_ge_0_02=dP >= 0.02, delta_ci_above_0=primary["ci_low"] > 0,
          rot_gain_positive_ci_above_0=rotP[0] > 0 and s2["ci_low"] > 0, no_ceiling_floor=not ceiling)
primary_pass = all(c1.values())
s5r = next(s for s in sec if s["id"] == "S5")
o2 = (not primary_pass) and s5r["estimate"] > 0 and s5r["significant_holm"]
o3 = (not primary_pass) and (-0.05 < primary["ci_low"] and primary["ci_high"] < 0.05) and \
     (-0.05 < s5r["ci_low"] and s5r["ci_high"] < 0.05)
if not gate0 or n_h146 < 120 or ceiling:
    outcome = 4
elif primary_pass:
    outcome = 1
elif o2:
    outcome = 2
elif o3:
    outcome = 3
else:
    outcome = 4
labels = {1: "Outcome 1 - independent switching signal demonstrated",
          2: "Outcome 2 - distributed deviation matters, explicit switching does not",
          3: "Outcome 3 - neither adds measurable value",
          4: "Outcome 4 - inconclusive"}

# ------------------------------------------------------------------ descriptive tables
rows = []
for (regime, arm, f, cohort, variant), g in CD.groupby(["regime", "arm", "fpr", "cohort", "variant"]):
    hs = {"H146": H146, "R253": ALL249, "S8": S8H}[cohort]
    for L in (None, 6, 12):
        gg = g if L is None else g[g.L == L]
        if not len(gg):
            continue
        pt, bt = hs.det(gg)
        lo, hi = np.percentile(bt, [2.5, 97.5])
        rows.append(dict(regime=regime, arm=arm, fpr=f, cohort=cohort, variant=variant, L="pooled" if L is None else L,
                         cases=len(gg), det=pt, ci_low=lo, ci_high=hi,
                         det_benign_hosts=gg.loc[gg.host_attacker == 0, "detected"].mean(),
                         det_attacker_hosts=gg.loc[gg.host_attacker == 1, "detected"].mean()))
dt = pd.DataFrame(rows)
dt.to_csv(RES / "analysis_detection_table.csv", index=False)

# ladder of adjacent-arm gains on ROT vs BLK (descriptive)
ladder = []
for f in FPRS:
    for a1, a0 in (("C", "B"), ("C+", "C"), ("A-cv", "C+"), ("D", "A-cv"), ("D", "C+")):
        dd, bb, _, r_, b_ = interaction(H146, 1, a1, a0, f)
        ladder.append(dict(fpr=f, step=f"{a0} -> {a1}", gain_ROT=r_[0], gain_BLK=b_[0], interaction=dd,
                           ci_low=np.percentile(bb, 2.5), ci_high=np.percentile(bb, 97.5)))
pd.DataFrame(ladder).to_csv(RES / "analysis_arm_ladder.csv", index=False)

# latency (detected case-repeats, regime 1, H146)
lat = (DET[(DET.regime == 1) & (DET.cohort == "H146") & DET.detected]
       .groupby(["arm", "fpr", "variant", "L"])["latency"]
       .agg(n="count", median="median", mean="mean", p75=lambda s: s.quantile(0.75)).reset_index())
lat.to_csv(RES / "analysis_latency.csv", index=False)

# unique detections: ROT (and BLK for reference) cases detected by D but not C+ in >= 2 of 3 repeats, and reverse
uniq = []
for f in FPRS:
    for v in ("ROT", "BLK"):
        g = DET[(DET.regime == 1) & (DET.cohort == "H146") & (DET.variant == v) & (DET.fpr == f) & DET.arm.isin(["D", "C+"])]
        p = g.pivot_table(index=["case", "repeat"], columns="arm", values="detected").reset_index()
        p["D_only"] = (p["D"] == 1) & (p["C+"] == 0)
        p["Cp_only"] = (p["C+"] == 1) & (p["D"] == 0)
        c = p.groupby("case")[["D_only", "Cp_only"]].sum()
        uniq.append(dict(fpr=f, variant=v, cases=len(c), D_not_Cplus=int((c.D_only >= 2).sum()),
                         Cplus_not_D=int((c.Cp_only >= 2).sum())))
pd.DataFrame(uniq).to_csv(RES / "analysis_unique_detections.csv", index=False)

# A-frozen reference (descriptive; in-sample on the hosts' benign days)
sys.path.insert(0, str(MODEL_DIR / "model"))
from mubsir_infer import MubsirScorer  # noqa: E402
scorer = MubsirScorer(MODEL_DIR / "model", verify=True)
cases = pd.read_parquet(HERE / "data" / "counterfactual_cases.parquet")
cases = cases[cases.cohort == "H146"]
risk = scorer.score(cases[scorer.features])["risk"].to_numpy()
cs = cases.assign(risk=risk)[lambda d: d.in_span]
af = []
for (v, L), g in cs.groupby(["variant", "L"]):
    pk = g.groupby("case")["risk"].max()
    af.append(dict(variant=v, L=L, cases=len(pk), det_risk_gt_85=float((pk > 85).mean()),
                   det_risk_gt_92_632=float((pk > 92.63198568310092).mean()), median_peak_risk=float(pk.median())))
pd.DataFrame(af).to_csv(RES / "analysis_a_frozen.csv", index=False)

# ------------------------------------------------------------------ save
out = dict(
    preregistration_sha256=(HERE / "PREREGISTRATION.sha256").read_text().strip(),
    hosts=dict(H146=n_h146, H146_benign=H146.n_ben, H146_attacker=H146.n_att, relaxed=len(ALL249.hosts), S8=len(S8H.hosts)),
    bootstrap=dict(resamples=NB, unit="host, stratified benign/attacker", seed=2026),
    primary=primary, primary_components={f"{a}|{v}": float(x[0]) for (a, v), x in compP.items()},
    rot_gain=s2, blk_gain=blk_diff,
    outcome1_conditions={k: bool(v) for k, v in c1.items()}, ceiling_or_floor=bool(ceiling),
    secondary=sec, s5_components={f"{a}|{v}": float(x[0]) for (a, v), x in comp5.items()},
    outcome2_applies=bool(o2), outcome3_applies=bool(o3), outcome=outcome, outcome_label=labels[outcome],
    heldout_benign_fpr=chk.groupby(["regime", "arm", "fpr"])["heldout_fpr"].mean().reset_index().to_dict("records"),
    seconds=round(time.time() - t0))
(RES / "analysis_primary.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
pd.DataFrame([primary, blk_diff] + sec).to_csv(RES / "analysis_estimands.csv", index=False)

print(f"\nPRIMARY  Delta = {dP:+.4f}  CI [{primary['ci_low']:+.4f}, {primary['ci_high']:+.4f}]  p={primary['p_two_sided']:.3f}")
for k, v in compP.items():
    print(f"   Det{k} = {v[0]:.4f}")
print(f"   D-C+ on ROT {rotP[0]:+.4f}  on BLK {blkP[0]:+.4f}")
for s in sec:
    print(f"{s['id']:7s} {s['estimate']:+.4f}  CI [{s['ci_low']:+.4f}, {s['ci_high']:+.4f}]  p={s['p_two_sided']:.3f}  "
          f"holm={s['p_holm']:.3f}")
print("outcome-1 conditions:", c1, "ceiling:", ceiling)
print("o2:", o2, "o3:", o3, "->", labels[outcome])
print("held-out benign FPR (mean):")
print(chk.groupby(["regime", "arm", "fpr"])["heldout_fpr"].mean().round(4).to_string())
print("done", round(time.time() - t0), "s")
