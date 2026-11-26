"""
MUBSIR (new dataset) — the 5 algorithms with the default configurations carried
over from the previous project (PROJECT_HANDOVER.md §3.1). No class weights and
no resampling (handover §3.3).

Boosting models (LightGBM, XGBoost, CatBoost) use early stopping on a held-out,
schedule-grouped user subset (`es`) with a PR-AUC metric (30 rounds, <= 800 trees).
DecisionTree and RandomForest need no early stopping, so they are trained on the
fit + es users together (the same users the boosting models see in total).
"""

import numpy as np
import pandas as pd

ALGOS = ("lightgbm", "xgboost", "catboost", "random_forest", "decision_tree")
BOOSTED = {"lightgbm", "xgboost", "catboost"}
MAX_TREES = 800
ES_ROUNDS = 30
import os

# Thread count per model. Verified (Stage B pre-run check): LightGBM predictions are identical
# for 4, 5 and 8 threads (deterministic=True, force_col_wise=True), so parallel jobs may lower it.
N_JOBS = int(os.environ.get("MUBSIR_NJOBS", "8"))

LGBM_DEFAULTS = dict(
    learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=1.0,
)


def _lgbm(seed, n_estimators, params=None):
    import lightgbm as lgb
    p = dict(LGBM_DEFAULTS, **(params or {}))
    return lgb.LGBMClassifier(
        n_estimators=n_estimators, objective="binary", metric="average_precision",
        random_state=seed, n_jobs=N_JOBS, deterministic=True, force_col_wise=True, verbose=-1, **p,
    )


def _xgb(seed, n_estimators, early_stopping):
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=n_estimators, learning_rate=0.05, max_depth=6, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist", eval_metric="aucpr",
        early_stopping_rounds=ES_ROUNDS if early_stopping else None,
        random_state=seed, n_jobs=N_JOBS,
    )


def _cat(seed, n_estimators, early_stopping):
    from catboost import CatBoostClassifier
    kw = dict(od_type="Iter", od_wait=ES_ROUNDS, use_best_model=True) if early_stopping else {}
    return CatBoostClassifier(
        iterations=n_estimators, learning_rate=0.08, depth=6, eval_metric="PRAUC",
        random_seed=seed, thread_count=N_JOBS, verbose=False, allow_writing_files=False, **kw,
    )


def _rf(seed):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=150, min_samples_leaf=20, max_features="sqrt", max_samples=0.5,
        random_state=seed, n_jobs=N_JOBS,
    )


def _dt(seed):
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, random_state=seed)


def fit_predict(algo, X_fit, y_fit, X_es, y_es, X_valid, seed, params=None):
    """Train on fit users (early-stop on es users) and score valid rows.

    Returns (scores for X_valid, number of trees used or None).
    """
    if algo == "lightgbm":
        m = _lgbm(seed, MAX_TREES, params)
        import lightgbm as lgb
        m.fit(X_fit, y_fit, eval_set=[(X_es, y_es)],
              callbacks=[lgb.early_stopping(ES_ROUNDS, first_metric_only=True, verbose=False)])
        n = int(m.best_iteration_) or MAX_TREES
        return m.predict_proba(X_valid, num_iteration=n)[:, 1], n
    if algo == "xgboost":
        m = _xgb(seed, MAX_TREES, early_stopping=True)
        m.fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
        n = int(m.best_iteration) + 1
        return m.predict_proba(X_valid, iteration_range=(0, n))[:, 1], n
    if algo == "catboost":
        m = _cat(seed, MAX_TREES, early_stopping=True)
        m.fit(X_fit, y_fit, eval_set=(X_es, y_es))
        n = int(m.get_best_iteration()) + 1
        return m.predict_proba(X_valid)[:, 1], n
    X_all = pd.concat([X_fit, X_es])
    y_all = np.concatenate([y_fit, y_es])
    m = _rf(seed) if algo == "random_forest" else _dt(seed)
    m.fit(X_all, y_all)
    return m.predict_proba(X_valid)[:, 1], None


def fit_final(algo, X, y, n_trees, seed, params=None):
    """Refit on all development users with a fixed tree count (mean of CV best iterations)."""
    if algo == "lightgbm":
        m = _lgbm(seed, n_trees, params)
    elif algo == "xgboost":
        m = _xgb(seed, n_trees, early_stopping=False)
    elif algo == "catboost":
        m = _cat(seed, n_trees, early_stopping=False)
    elif algo == "random_forest":
        m = _rf(seed)
    else:
        m = _dt(seed)
    m.fit(X, y)
    return m
