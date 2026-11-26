"""
Standalone re-implementation of the 100 MUBSIR model features (Gate 0 of the controlled rotation study).

Reproduces, from daily counts + fixed 2010 baselines:
  NB12 §12.4  original daily / rolling features (2011-only partial windows)
  NB12 §12.5.2 Clear-30 features (cumulative_deviation, combined_channel_intensity, dominant_channel_switches,
               channel_distribution_shift, consecutive_elevated_days, cross_channel_deviation_days, 20/30wd counts)
  NB12 §12.6.1 rotation-27 features (nr_*; email_count_to_p90_ratio)
  prepare_dataset.py  E_email_today, E_usb_today, E_http_today

Input: one row per (user, eligible 2011 workday), sorted by user then day, holding the raw daily count columns and the
user's fixed baseline columns (P50/P90/P95, active rates, mean-active values). Every feature at day t uses rows <= t of
the same user only. Output: the 100 features in the order of model/manifest.json.

Also provides `inject(...)`, the NB10 count mapping used to build counterfactual attacks.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

CHANNELS = [("external_email_count", "email_count", "email_active_rate", "email_count_mean_active"),
            ("usb_copy_count", "usb_copy", "usb_active_rate", "usb_copy_mean_active"),
            ("www_upload_count", "http_upload", "http_upload_active_rate", "http_upload_mean_active")]
RAW_DAILY = ["email_event_count", "email_send_count", "email_view_count", "external_email_count",
             "internal_email_count", "file_event_count", "file_copy_count", "file_open_count", "file_write_count",
             "file_delete_count", "usb_copy_count", "http_event_count", "www_visit_count", "www_download_count",
             "www_upload_count", "unique_domains", "logon_count", "logoff_count"]
# NB10: an injected channel-day adds the same count to the channel field and its event-total fields
INJECTION_FIELDS = {"EMAIL": ["external_email_count", "email_send_count", "email_event_count"],
                    "USB": ["usb_copy_count", "file_copy_count", "file_event_count"],
                    "HTTP": ["www_upload_count", "http_event_count"]}
CLEAR_W = (5, 10, 20, 30)
NR_W = (5, 10, 20)


def trailing_sum(v, w):
    """Causal partial-window sum over rows [t-w+1, t] (NB12 rolling_sum_1252)."""
    v = np.asarray(v, dtype="float64")
    n = len(v)
    ends = np.arange(1, n + 1)
    prefix = np.r_[0.0, np.cumsum(v)]
    return prefix[ends] - prefix[np.maximum(0, ends - w)]


def longest_run(flags, w):
    flags = np.asarray(flags, dtype=bool)
    win = sliding_window_view(np.pad(flags, (w - 1, 0), constant_values=False), w)
    cur = np.zeros(len(flags), dtype="int64")
    best = np.zeros(len(flags), dtype="int64")
    for c in range(w):
        cur = np.where(win[:, c], cur + 1, 0)
        best = np.maximum(best, cur)
    return best.astype("float64")


def dominant_switches(labels, w):
    labels = np.asarray(labels, dtype="int64")
    n = len(labels)
    out = np.zeros(n)
    idx = np.flatnonzero(labels >= 0)
    if len(idx) < 2:
        return out
    tr = np.r_[0, np.cumsum((labels[idx][1:] != labels[idx][:-1]).astype("int64"))]
    for d in range(n):
        lo = np.searchsorted(idx, max(0, d - w + 1), side="left")
        hi = np.searchsorted(idx, d, side="right")
        if hi - lo >= 2:
            out[d] = tr[hi - 1] - tr[lo]
    return out


def _user_features(x, p50, p90, p95, rate, mean_active, out, s):
    """Fill the Clear-30 and rotation-27 features of one user (rows s .. s+n-1 of the output arrays)."""
    n = len(x)
    sl = slice(s, s + n)
    # ---------- Clear-30 (NB12 §12.5.2)
    valid_dev = np.isfinite(p50) & np.isfinite(p95) & (p50 > 0) & (p95 > 0) & (p95 >= p50)
    Ec = np.zeros((n, 3))
    for c in range(3):
        if valid_dev[c]:
            Ec[:, c] = np.maximum(0.0, x[:, c] - p50[c]) / max(1.0, p95[c] - p50[c])
    daily_dev = Ec.sum(axis=1)
    elevated = (Ec > 0).any(axis=1)
    cross_elev = (Ec > 0).sum(axis=1) >= 2
    valid_p95 = np.isfinite(p95) & (p95 > 0)
    norm = np.zeros((n, 3))
    for c in range(3):
        if valid_p95[c]:
            norm[:, c] = x[:, c] / p95[c]
    k = int(valid_p95.sum())
    intensity = norm.sum(axis=1) / k if k > 0 else np.full(n, np.nan)
    dmax = norm.max(axis=1)
    ties = np.isclose(norm, dmax[:, None], rtol=1e-9, atol=1e-12)
    uniq = (ties.sum(axis=1) == 1) & (dmax > 0)
    dom_raw = np.full(n, -1, dtype="int64")
    dom_raw[uniq] = norm[uniq].argmax(axis=1)
    hw = np.zeros(3)
    for c in range(3):
        if valid_p95[c]:
            hw[c] = rate[c] * mean_active[c] / p95[c]
    hist = hw / hw.sum() if hw.sum() > 0 else None
    for w in CLEAR_W:
        out[f"cumulative_deviation_{w}wd"][sl] = trailing_sum(daily_dev, w)
        if k > 0:
            out[f"combined_channel_intensity_{w}wd"][sl] = trailing_sum(intensity, w) / np.minimum(np.arange(1, n + 1), w)
        out[f"dominant_channel_switches_{w}wd"][sl] = dominant_switches(dom_raw, w)
        obs = np.column_stack([trailing_sum(norm[:, c], w) for c in range(3)])
        tot = obs.sum(axis=1)
        shift = np.full(n, np.nan)
        if hist is not None:
            dfd = tot > 0
            shift[dfd] = np.clip(0.5 * np.abs(obs[dfd] / tot[dfd, None] - hist).sum(axis=1), 0.0, 1.0)
        out[f"channel_distribution_shift_{w}wd"][sl] = shift
        out[f"consecutive_elevated_days_{w}wd"][sl] = longest_run(elevated, w)
        out[f"cross_channel_deviation_days_{w}wd"][sl] = trailing_sum(cross_elev.astype(float), w)
        if w in (20, 30):
            for name, c in (("external_email_count", 0), ("usb_copy_count", 1), ("http_upload_count", 2)):
                out[f"{name}_{w}wd"][sl] = trailing_sum(x[:, c], w)
    # ---------- rotation-27 (NB12 §12.6.1): E with the contract rule (baselines present; P50 = 0 allowed)
    E = np.zeros((n, 3))
    for c in range(3):
        if np.isfinite(p50[c]) and np.isfinite(p95[c]):
            E[:, c] = np.maximum(0.0, x[:, c] - p50[c]) / max(1.0, p95[c] - p50[c])
    flags = E > 0
    dev_day = flags.any(axis=1)
    dom = np.full(n, -1, dtype=np.int8)
    dom[dev_day] = E[dev_day].argmax(axis=1)
    nd = flags.sum(axis=1)
    excl, overlap = (nd == 1).astype(float), (nd >= 2).astype(float)
    prev = np.vstack([np.zeros((1, 3), dtype=bool), flags[:-1]])
    handoff = np.where(prev.any(axis=1), (flags & ~prev).sum(axis=1), 0).astype(float)
    stealth = np.zeros(n)
    for c in range(3):
        if np.isfinite(p95[c]):
            stealth += np.where((E[:, c] > 0) & (x[:, c] <= p95[c]), E[:, c], 0.0)
    pairs = np.zeros(n)
    changes = np.zeros(n)
    if n > 1:
        adj = dev_day[1:] & dev_day[:-1]
        pairs[1:] = adj
        changes[1:] = adj & (dom[1:] != dom[:-1])
    Ep = np.vstack([np.zeros((1, 3)), np.cumsum(E, axis=0)])
    P = {k_: np.r_[0.0, np.cumsum(v)] for k_, v in (("excl", excl), ("ov", overlap), ("ho", handoff), ("st", stealth),
                                                     ("pa", pairs), ("ch", changes))}
    for w in NR_W:
        m_, ent, alt, gap = (np.zeros(n) for _ in range(4))
        for t in range(n):
            lo, hi = max(0, t - w + 1), t + 1
            S_c = np.maximum(Ep[hi] - Ep[lo], 0.0)
            S = float(S_c.sum())
            m_[t] = float(np.count_nonzero(S_c > 1e-12))
            if S > 1e-12:
                pr = S_c[S_c > 1e-12] / S
                ent[t] = float(np.clip(-float(np.sum(pr * np.log(pr)) / np.log(3.0)), 0.0, 1.0))
            pc = P["pa"][hi] - P["pa"][lo + 1]
            cc = P["ch"][hi] - P["ch"][lo + 1]
            alt[t] = cc / pc if pc > 0 else 0.0
            di = np.flatnonzero(dev_day[lo:hi]) + lo
            ep = cp_ = 0
            for j in range(1, len(di)):
                if di[j] - di[j - 1] - 1 <= 2:
                    ep += 1
                    if dom[di[j]] != dom[di[j - 1]]:
                        cp_ += 1
            gap[t] = cp_ / ep if ep > 0 else 0.0
        out[f"nr_m_{w}wd"][sl] = m_
        out[f"nr_entropy_{w}wd"][sl] = ent
        out[f"nr_excl_days_{w}wd"][sl] = trailing_sum(excl, w)
        out[f"nr_overlap_days_{w}wd"][sl] = trailing_sum(overlap, w)
        out[f"nr_alt_{w}wd"][sl] = alt
        out[f"nr_handoff_{w}wd"][sl] = trailing_sum(handoff, w)
        out[f"nr_stealth_acc_{w}wd"][sl] = trailing_sum(stealth, w)
        out[f"nr_gap_alt_2_{w}wd"][sl] = gap
    if np.isfinite(p90[0]) and p90[0] > 0:
        out["email_count_to_p90_ratio"][sl] = x[:, 0] / p90[0]
    # ---------- E_*_today (prepare_dataset.py): same contract rule, 0 when the baseline is missing
    for c, name in enumerate(("E_email_today", "E_usb_today", "E_http_today")):
        out[name][sl] = E[:, c]


def compute_features(rows, feature_order):
    """rows: DataFrame sorted by (user, day) with raw count + baseline columns. Returns DataFrame[user, day, *features]."""
    df = rows.reset_index(drop=True)
    assert df[["user", "day"]].equals(df.sort_values(["user", "day"])[["user", "day"]].reset_index(drop=True))
    n = len(df)
    out = {f: np.full(n, np.nan) for f in feature_order}
    # ---------- original features (NB12 §12.4)
    for f in RAW_DAILY:
        out[f] = df[f].to_numpy(dtype="float64")
    act = {"email": (df["external_email_count"] > 0).astype(float).to_numpy(),
           "usb": (df["usb_copy_count"] > 0).astype(float).to_numpy(),
           "http": (df["www_upload_count"] > 0).astype(float).to_numpy()}
    out["email_active_today"], out["usb_active_today"], out["http_upload_active_today"] = act["email"], act["usb"], act["http"]
    out["active_channel_count"] = act["email"] + act["usb"] + act["http"]
    for prefix, cf, p95f in (("email_count", "external_email_count", "email_count_p95"),
                             ("usb_copy", "usb_copy_count", "usb_copy_p95"),
                             ("http_upload", "www_upload_count", "http_upload_p95")):
        out[f"{prefix}_above_p95"] = (df[p95f].notna() & df[cf].gt(df[p95f])).astype(float).to_numpy()
    out["email_count_to_p95_ratio"] = (df["external_email_count"] / df["email_count_p95"].where(df["email_count_p95"].gt(0))).to_numpy()
    starts = np.r_[0, np.flatnonzero(df["user"].to_numpy()[1:] != df["user"].to_numpy()[:-1]) + 1, n]
    rs = [(starts[i], starts[i + 1]) for i in range(len(starts) - 1)]

    def per_user_trailing(v, w):
        res = np.empty(n)
        for a, b in rs:
            res[a:b] = trailing_sum(v[a:b], w)
        return res

    for pre, key in (("email_active_days", "email"), ("usb_active_days", "usb"), ("http_active_days", "http")):
        for w in (5, 10, 20):
            out[f"{pre}_{w}wd"] = per_user_trailing(act[key], w)
    for pre, cf in (("external_email_count", "external_email_count"), ("usb_copy_count", "usb_copy_count"),
                    ("http_upload_count", "www_upload_count")):
        for w in (5, 10):
            out[f"{pre}_{w}wd"] = per_user_trailing(df[cf].to_numpy(dtype="float64"), w)
    out["multi_channel_days_10wd"] = per_user_trailing((out["active_channel_count"] >= 2).astype(float), 10)
    # ---------- per-user Clear-30, rotation-27, E_today
    X = df[[c[0] for c in CHANNELS]].to_numpy(dtype="float64")
    for a, b in rs:
        u = df.iloc[a]
        g = lambda suf: np.array([u[f"{c[1]}_{suf}"] for c in CHANNELS], dtype="float64")
        rate = np.array([u[c[2]] for c in CHANNELS], dtype="float64")
        mean_a = np.array([u[c[3]] for c in CHANNELS], dtype="float64")
        _user_features(X[a:b], g("p50"), g("p90"), g("p95"), rate, mean_a, out, a)
    res = pd.DataFrame({"user": df["user"].to_numpy(), "day": df["day"].to_numpy()})
    return pd.concat([res, pd.DataFrame({f: out[f] for f in feature_order})], axis=1)


def inject(rows, user, day, channel, added):
    """Add `added` events of `channel` (EMAIL/USB/HTTP) on (user, day) using the NB10 field mapping (in place)."""
    m = (rows["user"] == user) & (rows["day"] == day)
    assert m.sum() == 1, (user, day)
    for f in INJECTION_FIELDS[channel]:
        rows.loc[m, f] = rows.loc[m, f] + added
