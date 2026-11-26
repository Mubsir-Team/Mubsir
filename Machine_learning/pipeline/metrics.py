"""
MUBSIR (new dataset) — evaluation metrics.

Row level (threshold-free): PR-AUC, ROC-AUC on the model's own label y_h.
Row level at fixed FPR: recall at the score where a given share of BENIGN-USER
rows alarm (reference negatives = rows of users who never attack).

User level, comparable across label horizons:
  - user AUC: peak score per user, attackers vs benign users;
  - in-window detection at a fixed benign-USER FPR: the threshold is set so that
    a given share of benign users ever alarm (peak > thr); an attacker counts as
    detected if any row inside its attack window (attack or forced off-day) is
    above the threshold;
  - latency: workdays from the first injected day to the first in-window alarm
    (negative = alarm on a forced off-day that opens the window).
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def threshold_at_fpr(neg_scores, fpr):
    """Smallest-alarm threshold t such that share(neg > t) <= fpr."""
    neg = np.sort(np.asarray(neg_scores, dtype=float))[::-1]
    k = int(np.floor(fpr * len(neg)))
    return neg[min(k, len(neg) - 1)]


def recall_at_fpr(pos_scores, neg_scores, fpr):
    t = threshold_at_fpr(neg_scores, fpr)
    return float(np.mean(np.asarray(pos_scores) > t)), float(t)


def row_metrics(y, score, benign_user_mask, prefix=""):
    """y: label array; benign_user_mask: rows of never-attacking users."""
    out = {
        f"{prefix}pr_auc": float(average_precision_score(y, score)),
        f"{prefix}roc_auc": float(roc_auc_score(y, score)),
        f"{prefix}pos_rate": float(np.mean(y)),
    }
    neg = score[benign_user_mask]
    for f in (0.01, 0.05):
        r, _ = recall_at_fpr(score[y == 1], neg, f)
        out[f"{prefix}recall@{int(f * 100)}%fpr"] = r
    return out


def user_metrics(frame, fprs=(0.01, 0.05)):
    """frame columns: user, is_attacker, in_window (bool), wd_from_first_attack (int, in-window rows), score."""
    peak = frame.groupby("user").agg(score=("score", "max"), is_attacker=("is_attacker", "first"))
    out = {"user_auc": float(roc_auc_score(peak["is_attacker"], peak["score"])),
           "n_attackers": int(peak["is_attacker"].sum()),
           "n_benign": int((peak["is_attacker"] == 0).sum())}
    benign_peaks = peak.loc[peak["is_attacker"] == 0, "score"].to_numpy()
    win = frame[frame["in_window"]]
    for f in fprs:
        t = threshold_at_fpr(benign_peaks, f)
        tag = f"{int(f * 100)}%ufpr"
        alarms = win[win["score"] > t]
        first = alarms.groupby("user")["wd_from_first_attack"].min()
        out[f"user_recall@{tag}"] = float(len(first) / out["n_attackers"])
        out[f"latency_median@{tag}"] = float(first.median()) if len(first) else np.nan
        out[f"latency_mean@{tag}"] = float(first.mean()) if len(first) else np.nan
        out[f"benign_users_alarmed@{tag}"] = int((benign_peaks > t).sum())
    return out


# ---------------------------------------------------------------- Stage B (selection_rules.json stage_B.definitions)
DURATIONS = (5, 10, 20, 30, 40, 50)


def duration_macro_row_pr_auc(y, score, row_duration, benign_row):
    """Mean over attack durations d of AP(y) on rows of all benign users + attackers with duration d.

    y: y_10 per row; row_duration: window_workdays of the row's user (NaN for benign users);
    benign_row: True for rows of never-attacking users. Used identically by Optuna's inner
    objective, the tuned arm and the default arm.
    """
    vals = []
    for d in DURATIONS:
        m = benign_row | (row_duration == d)
        vals.append(average_precision_score(y[m], score[m]))
    return float(np.mean(vals)), vals


def user_peaks(users, score):
    """Max score per user over all of the user's rows (dict-like Series indexed by user)."""
    import pandas as pd
    return pd.Series(score).groupby(np.asarray(users)).max()


def frozen_threshold(inner_users, inner_scores, benign_users, fpr=0.01):
    """Threshold from inner out-of-fold peaks of the benign outer-train users only."""
    peaks = user_peaks(inner_users, inner_scores)
    neg = peaks.loc[sorted(set(peaks.index) & set(benign_users))].to_numpy()
    return threshold_at_fpr(neg, fpr), len(neg)


def detect_with_threshold(users, score, in_window, attacker_users, benign_users, t):
    """Detected attackers (any in-window row > t) and false-positive benign users (peak > t)."""
    import pandas as pd
    f = pd.DataFrame({"user": np.asarray(users), "s": score, "w": np.asarray(in_window)})
    hit = set(f.loc[f["w"] & (f["s"] > t), "user"])
    detected = sorted(hit & set(attacker_users))
    peaks = f.groupby("user")["s"].max()
    fp = int((peaks.loc[sorted(set(peaks.index) & set(benign_users))] > t).sum())
    return detected, fp
