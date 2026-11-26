"""
Habit-relative switching features, exactly as defined in PREREGISTRATION.md §4 and §6.

Input E: (n x 3) daily per-channel deviation of ONE user or case (Email, USB, HTTP), rows in workday order.
All features at row t use rows <= t only. r0 (personal 2010 rate) and m0 (population reference) are never features.
"""

import numpy as np

from rot_features import _linked_stats   # linked pair = <= 2 quiet workdays; change = linked pair with different dominant

W = (10, 20)
M0 = 0.4444444444                        # frozen population reference (PREREGISTRATION §4)
R_NAMES = [f"{k}_{w}" for w in W for k in ("rate_dev_pop", "excess_pop")]
H_NAMES = [f"{k}_{w}" for w in W for k in ("rate_dev_hab", "excess_hab")]


def dominant(E):
    E = np.asarray(E, dtype="float64")
    dev = E.sum(axis=1) > 0
    dom = np.full(len(E), -1, dtype=int)
    dom[dev] = E[dev].argmax(axis=1)              # ties: first maximum = Email < USB < HTTP
    return dev, dom


def habit_rate(E):
    """r0 over a whole period: (rate, linked pairs, changes)."""
    dev, dom = dominant(E)
    linked, change = _linked_stats(np.flatnonzero(dev), dom)
    p, c = int(linked.sum()), int(change.sum())
    return (c / p if p else np.nan), p, c


def window_counts(E):
    """pairs_w, changes_w per row for w in W (pairs with both days inside [t-w+1, t]; partial windows at the start)."""
    dev, dom = dominant(E)
    n = len(E)
    out = {}
    for w in W:
        pairs, changes = np.zeros(n), np.zeros(n)
        for t in range(n):
            lo = max(0, t - w + 1)
            d = np.flatnonzero(dev[lo:t + 1]) + lo
            linked, change = _linked_stats(d, dom)
            pairs[t], changes[t] = linked.sum(), change.sum()
        out[w] = (pairs, changes)
    return out


def switching_features(E, r0, m0=M0):
    """The 8 study features (4 R, 4 H) for one user/case."""
    wc = window_counts(E)
    f = {}
    for w, (p, c) in wc.items():
        rate = np.divide(c, p, out=np.zeros_like(c), where=p > 0)
        f[f"rate_dev_pop_{w}"] = np.where(p > 0, rate - m0, 0.0)
        f[f"excess_pop_{w}"] = c - m0 * p
        f[f"rate_dev_hab_{w}"] = np.where(p > 0, rate - r0, 0.0)
        f[f"excess_hab_{w}"] = c - r0 * p
        f[f"_pairs_{w}"], f[f"_changes_{w}"] = p, c
    return f
