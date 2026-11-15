"""GBDT CTR ranker. Uses LightGBM when installed (pip install recsys[lgb]),
otherwise scikit-learn's HistGradientBoosting — same model family."""

from __future__ import annotations

import itertools

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from .features import CATEGORICAL, FEATURES


class Ranker:
    def __init__(self, model, backend: str):
        self.model, self.backend = model, backend

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        if self.backend == "lightgbm":
            return self.model.predict(X)
        Xc = X.copy()
        Xc[:, CATEGORICAL] = np.maximum(Xc[:, CATEGORICAL], 0)
        return self.model.predict_proba(Xc)[:, 1]


def gauc(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> float:
    """Impression-weighted per-user AUC (users with both classes only)."""
    num = den = 0.0
    order = np.argsort(groups, kind="stable")
    y, p, g = y[order], p[order], groups[order]
    bounds = np.flatnonzero(np.r_[True, g[1:] != g[:-1], True])
    for a, b in itertools.pairwise(bounds):
        yy = y[a:b]
        if 0 < yy.sum() < len(yy):
            num += roc_auc_score(yy, p[a:b]) * (b - a)
            den += b - a
    return num / den if den else float("nan")


def train_ranker(X: np.ndarray, y: np.ndarray, groups: np.ndarray, valid_mask: np.ndarray) -> tuple[Ranker, dict]:
    tr, va = ~valid_mask, valid_mask
    try:
        import lightgbm as lgb

        model = lgb.train(
            dict(
                objective="binary",
                learning_rate=0.05,
                num_leaves=31,
                min_data_in_leaf=50,
                feature_fraction=0.9,
                bagging_fraction=0.8,
                bagging_freq=1,
                verbose=-1,
            ),
            lgb.Dataset(X[tr], y[tr], feature_name=FEATURES, categorical_feature=CATEGORICAL),
            num_boost_round=400,
            valid_sets=[lgb.Dataset(X[va], y[va], categorical_feature=CATEGORICAL)],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        ranker = Ranker(model, "lightgbm")
    except (ImportError, OSError):
        from sklearn.ensemble import HistGradientBoostingClassifier

        cat_mask = np.zeros(X.shape[1], bool)
        cat_mask[CATEGORICAL] = True
        Xc = X.copy()
        Xc[:, CATEGORICAL] = np.maximum(Xc[:, CATEGORICAL], 0)
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=400,
            max_leaf_nodes=31,
            min_samples_leaf=50,
            categorical_features=cat_mask,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            random_state=0,
        )
        model.fit(Xc[tr], y[tr])
        ranker = Ranker(model, "sklearn-hgb")
    p = ranker.predict(X[va])
    metrics = dict(
        backend=ranker.backend,
        train_rows=int(tr.sum()),
        valid_rows=int(va.sum()),
        pos_rate=float(y.mean()),
        auc=float(roc_auc_score(y[va], p)),
        gauc=float(gauc(y[va], p, groups[va])),
        logloss=float(log_loss(y[va], p)),
    )
    return ranker, metrics
