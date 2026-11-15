"""GBDT CTR ranker. Uses LightGBM when installed (pip install recsys[lgb]),
otherwise scikit-learn's HistGradientBoosting — same model family."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

from .features import CATEGORICAL, FEATURES


class Ranker:
    """GBDT CTR model. When the model was exported to ONNX (sklearn backend + onnxruntime
    installed), serving uses ONNX Runtime; the sklearn model stays as the reference."""

    def __init__(self, model, backend: str, onnx_model: bytes | None = None):
        self.model, self.backend, self.onnx_model = model, backend, onnx_model
        self._scorer: Any = None

    def __getstate__(self) -> dict:
        d = self.__dict__.copy()
        d["_scorer"] = None  # ONNX Runtime sessions are not picklable; rebuilt lazily
        return d

    def __setstate__(self, d: dict) -> None:
        self.__dict__.update({"onnx_model": None, **d})
        self._scorer = None

    @property
    def serving_backend(self) -> str:
        return "onnxruntime" if self._onnx() is not None else self.backend

    def _onnx(self) -> Any:
        if self._scorer is None and self.onnx_model is not None:
            try:
                from .onnx_export import OnnxScorer

                self._scorer = OnnxScorer(self.onnx_model)
            except ImportError:
                self.onnx_model = None  # onnxruntime not installed here: stay on sklearn
        return self._scorer

    def predict(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        if self.backend == "lightgbm":
            return self.model.predict(X)
        Xc = X.copy()
        Xc[:, CATEGORICAL] = np.maximum(Xc[:, CATEGORICAL], 0)
        scorer = self._onnx()
        if scorer is not None:
            return scorer(Xc).astype(np.float64)
        return self.model.predict_proba(Xc)[:, 1]

    def predict_reference(self, X: np.ndarray) -> np.ndarray:
        """Always the original (non-ONNX) model."""
        scorer, self._scorer, onnx_model, self.onnx_model = self._scorer, None, self.onnx_model, None
        try:
            return self.predict(X)
        finally:
            self._scorer, self.onnx_model = scorer, onnx_model


def export_onnx(ranker: Ranker, X_check: np.ndarray, tol: float = 1e-5) -> dict:
    """Export a sklearn ranker to ONNX and verify parity on `X_check`; on success the
    ranker serves through ONNX Runtime. Returns a report (never raises)."""
    if ranker.backend != "sklearn-hgb":
        return {"onnx": "skipped", "reason": f"backend {ranker.backend}"}
    try:
        from .onnx_export import OnnxScorer, hgb_to_onnx

        blob = hgb_to_onnx(ranker.model).SerializeToString()
        Xc = X_check.copy()
        Xc[:, CATEGORICAL] = np.maximum(Xc[:, CATEGORICAL], 0)
        diff = float(np.abs(OnnxScorer(blob)(Xc) - ranker.model.predict_proba(Xc)[:, 1]).max()) if len(Xc) else 0.0
    except ImportError as e:
        return {"onnx": "skipped", "reason": f"{e.name} not installed"}
    except Exception as e:  # exporter bug: keep serving with sklearn
        return {"onnx": "failed", "reason": repr(e)}
    if diff > tol:
        return {"onnx": "failed", "reason": f"parity max|diff|={diff:.2e} > {tol:g}"}
    ranker.onnx_model, ranker._scorer = blob, None
    return {"onnx": "ok", "onnx_bytes": len(blob), "onnx_max_abs_diff": diff, "onnx_check_rows": len(Xc)}


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
    onnx_report = export_onnx(ranker, X[va][:20000])
    metrics = dict(
        backend=ranker.backend,
        train_rows=int(tr.sum()),
        valid_rows=int(va.sum()),
        pos_rate=float(y.mean()),
        auc=float(roc_auc_score(y[va], p)),
        gauc=float(gauc(y[va], p, groups[va])),
        logloss=float(log_loss(y[va], p)),
        **onnx_report,
        serving_backend=ranker.serving_backend,
    )
    return ranker, metrics
