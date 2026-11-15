from __future__ import annotations

import pickle

import numpy as np
import onnx
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from rec_server.features import CATEGORICAL, FEATURES
from rec_server.onnx_export import OnnxScorer, hgb_to_onnx
from rec_server.rank import Ranker, export_onnx


def _data(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURES))).astype(np.float32)
    X[:, CATEGORICAL[0]] = rng.integers(0, 12, n)
    X[:, CATEGORICAL[1]] = rng.integers(0, 2, n)
    logit = X[:, 0] + np.isin(X[:, CATEGORICAL[0]], [1, 4, 7]) * 1.5 - X[:, CATEGORICAL[1]] + 0.5 * X[:, 3]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


@pytest.fixture(scope="module")
def model():
    X, y = _data()
    mask = np.zeros(X.shape[1], bool)
    mask[CATEGORICAL] = True
    return HistGradientBoostingClassifier(max_iter=60, categorical_features=mask, random_state=0).fit(X, y), X


def test_onnx_matches_sklearn_exactly(model):
    m, X = model
    proto = hgb_to_onnx(m)
    onnx.checker.check_model(proto)
    s = OnnxScorer(proto.SerializeToString())
    assert np.abs(s(X) - m.predict_proba(X)[:, 1]).max() < 1e-6


def test_onnx_unknown_categories_and_missing_follow_sklearn(model):
    m, X = model
    s = OnnxScorer(hgb_to_onnx(m).SerializeToString())
    Z = X[:500].copy()
    Z[:100, CATEGORICAL[0]] = 99  # never seen in training -> sklearn treats as missing
    Z[100:200, CATEGORICAL[1]] = 5
    Z[200:300, 0] = np.nan  # numeric missing
    Z[300:400, CATEGORICAL[0]] = np.nan
    assert np.abs(s(Z) - m.predict_proba(Z)[:, 1]).max() < 1e-6


def test_ranker_serves_via_onnx_and_survives_pickle(model):
    m, X = model
    r = Ranker(m, "sklearn-hgb")
    rep = export_onnx(r, X[:1000])
    assert rep["onnx"] == "ok" and r.serving_backend == "onnxruntime"
    r2 = pickle.loads(pickle.dumps(r))
    p = r2.predict(X[:300])
    assert np.abs(p - r2.predict_reference(X[:300])).max() < 1e-6
    assert r2.serving_backend == "onnxruntime"


def test_old_pickles_without_onnx_still_work(model):
    m, X = model
    r = Ranker(m, "sklearn-hgb")
    state = r.__dict__.copy()
    del state["onnx_model"]
    r2 = Ranker.__new__(Ranker)
    r2.__setstate__(state)
    assert r2.serving_backend == "sklearn-hgb" and len(r2.predict(X[:10])) == 10
