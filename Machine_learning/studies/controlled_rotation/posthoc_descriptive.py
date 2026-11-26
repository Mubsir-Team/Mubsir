"""
POST-HOC, DESCRIPTIVE ONLY (written after the registered analysis was seen; not part of PREREGISTRATION.md; no test, no
p-value, no effect on the outcome). Decomposes the registered S5 interaction and the arm ladder into per-variant arm
differences with 95% host-bootstrap CIs (same bootstrap design as analyze.py: 2,000 stratified host resamples, rng 2026).
Reads results/analysis_case_detection.csv; writes results/posthoc_arm_differences.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
NB = 2000
det = pd.read_csv(RES / "analysis_case_detection.csv")
det = det[(det.regime == 1) & (det.cohort == "H146")]
cd = det.groupby(["arm", "fpr", "case", "host", "host_attacker", "variant", "vgroup"])["detected"].mean().reset_index()

hosts = np.array(sorted(cd.host.unique()))
att = cd.drop_duplicates("host").set_index("host").loc[hosts, "host_attacker"].to_numpy()
rng = np.random.default_rng(2026)
W = np.zeros((NB, len(hosts)))
for s in (0, 1):
    idx = np.flatnonzero(att == s)
    dr = rng.integers(0, len(idx), size=(NB, len(idx)))
    for b in range(NB):
        np.add.at(W[b], idx[dr[b]], 1)


def det_vec(g):
    s = g.groupby("host")["detected"].agg(["sum", "count"]).reindex(hosts, fill_value=0)
    S, N = s["sum"].to_numpy(), s["count"].to_numpy()
    return S.sum() / N.sum(), (W @ S) / (W @ N)


rows = []
pairs = [("C", "B"), ("C+", "C"), ("A-cv", "C"), ("D", "C"), ("A-cv", "C+"), ("D", "C+")]
for f in (0.03, 0.01):
    for vcol, v in [("variant", x) for x in ("ROT", "BLK", "FIX-EMAIL", "FIX-USB", "FIX-HTTP", "SIM")] + [("vgroup", "FIX")]:
        for a1, a0 in pairs:
            p1, b1 = det_vec(cd[(cd.arm == a1) & (cd.fpr == f) & (cd[vcol] == v)])
            p0, b0 = det_vec(cd[(cd.arm == a0) & (cd.fpr == f) & (cd[vcol] == v)])
            lo, hi = np.percentile(b1 - b0, [2.5, 97.5])
            rows.append(dict(fpr=f, variant=v if vcol == "variant" else "FIX (pooled)", comparison=f"{a1} - {a0}",
                             det_a1=p1, det_a0=p0, difference=p1 - p0, ci_low=lo, ci_high=hi))
out = pd.DataFrame(rows)
out.to_csv(RES / "posthoc_arm_differences.csv", index=False)
pd.set_option("display.width", 200)
print(out[out.fpr == 0.03].pivot_table(index="variant", columns="comparison", values="difference").round(3))
print(out[(out.fpr == 0.03) & out.comparison.isin(["C+ - C", "A-cv - C"])].round(3).to_string(index=False))
