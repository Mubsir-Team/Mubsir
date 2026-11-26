"""
New switching features of the controlled rotation study, exactly as defined in PREREGISTRATION.md §5.1 and §5.2.
Input: E (n x 3) daily per-channel deviation of ONE user (columns Email, USB, HTTP; contract rule, = E_*_today).
All features at row t use rows <= t only.
"""

import numpy as np

W = (10, 20)
NAMES = [f"rot_{k}_{w}" for w in W for k in ("changes", "revisits", "trans_entropy", "cycle")]
NORM_NAMES = ["sw_excess_rate_10", "sw_excess_count_10"]


def _linked_stats(dev_idx, dom):
    """For an ordered array of deviation-day indices: linked flags (gap <= 2 quiet days) and change flags per pair."""
    if len(dev_idx) < 2:
        return np.zeros(0, bool), np.zeros(0, bool)
    linked = (np.diff(dev_idx) - 1) <= 2
    change = linked & (dom[dev_idx[1:]] != dom[dev_idx[:-1]])
    return linked, change


def rotation_features(E):
    E = np.asarray(E, dtype="float64")
    n = len(E)
    dev = E.sum(axis=1) > 0
    dom = np.full(n, -1, dtype=int)
    dom[dev] = E[dev].argmax(axis=1)          # ties: Email < USB < HTTP (first maximum)
    out = {k: np.zeros(n) for k in NAMES + NORM_NAMES}
    for t in range(n):
        for w in W:
            lo = max(0, t - w + 1)
            d = np.flatnonzero(dev[lo:t + 1]) + lo
            linked, change = _linked_stats(d, dom)
            changes = int(change.sum())
            # revisits: dom(d_i) seen earlier in the window but not at d_(i-1)
            rev, seen = 0, set()
            for i, di in enumerate(d):
                if i >= 1 and dom[di] in seen and dom[di] != dom[d[i - 1]]:
                    rev += 1
                seen.add(dom[di])
            # entropy over the 6 ordered change types
            ent = 0.0
            if changes >= 2:
                types = [(dom[d[i]], dom[d[i + 1]]) for i in range(len(d) - 1) if change[i]]
                _, cnt = np.unique(np.array(types), axis=0, return_counts=True)
                p = cnt / cnt.sum()
                ent = float(-(p * np.log(p)).sum() / np.log(6))
            # cycle: three consecutive deviation days, pairwise-distinct dominants, both pairs linked
            cyc = 0
            for i in range(2, len(d)):
                a, b, c = dom[d[i - 2]], dom[d[i - 1]], dom[d[i]]
                if linked[i - 2] and linked[i - 1] and len({a, b, c}) == 3:
                    cyc += 1
            out[f"rot_changes_{w}"][t] = changes
            out[f"rot_revisits_{w}"][t] = rev
            out[f"rot_trans_entropy_{w}"][t] = ent
            out[f"rot_cycle_{w}"][t] = cyc / max(1, changes - 1)
        # personal norm: base [t-70, t-11], current [t-9, t]
        def rate(a, b):
            if b < a:
                return 0.0, 0
            d = np.flatnonzero(dev[max(0, a):b + 1]) + max(0, a)
            linked, change = _linked_stats(d, dom)
            pairs = int(linked.sum())
            return (change.sum() / pairs if pairs > 0 else 0.0), pairs
        base_rate, _ = rate(t - 70, t - 11)
        cur_rate, cur_pairs = rate(t - 9, t)
        out["sw_excess_rate_10"][t] = cur_rate - base_rate
        out["sw_excess_count_10"][t] = out["rot_changes_10"][t] - base_rate * cur_pairs
        out.setdefault("_base_rate", np.zeros(n))[t] = base_rate
    return out
