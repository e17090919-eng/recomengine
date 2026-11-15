"""Export the GBDT ranker to ONNX (ai.onnx.ml TreeEnsembleRegressor + Sigmoid).

Why our own exporter: skl2onnx cannot convert HistGradientBoosting models with
categorical splits (and breaks on recent sklearn/onnx versions). The model is small
enough to translate node by node, which also makes parity exact:

- numeric split `x <= t` (float64 t) -> BRANCH_LEQ with the largest float32 <= t, so the
  float32 feature row takes the same branch as in sklearn;
- categorical split -> chain of BRANCH_EQ nodes over the raw category values, with the
  smaller branch duplicated; unknown values follow sklearn's "treat as missing" rule;
- sklearn's ColumnTransformer (categorical columns moved first and ordinal-encoded) is
  undone, so the ONNX model takes the original feature matrix.

The ONNX file is the language-neutral artifact (Java / C++ / ONNX Runtime serving);
`OnnxScorer` serves it from Python.
"""

from __future__ import annotations

from typing import Any

import numpy as np

FLOAT32_MIN = np.float32(-np.inf)


def _floor32(t: float) -> float:
    f = np.float32(t)
    if float(f) > t:
        f = np.nextafter(f, FLOAT32_MIN, dtype=np.float32)
    return float(f)


def _feature_maps(model: Any) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """transformed column -> original column; categorical transformed column -> raw values by code."""
    n = model.n_features_in_
    pre = getattr(model, "_preprocessor", None)
    if pre is None:
        return np.arange(n), {}
    is_cat = np.asarray(model.is_categorical_, bool)
    order = np.r_[np.flatnonzero(is_cat), np.flatnonzero(~is_cat)]
    enc = pre.named_transformers_["encoder"]
    cats = {}
    for j, values in enumerate(enc.categories_):
        v = np.asarray(values, dtype=float)
        cats[j] = v[~np.isnan(v)]
    return order, cats


def _bits(bitset_row: np.ndarray) -> set[int]:
    out = set()
    for w, word in enumerate(bitset_row.astype(np.uint64)):
        for b in range(32):
            if int(word) >> b & 1:
                out.add(w * 32 + b)
    return out


def _export_tree(tree_id: int, pred: Any, order: np.ndarray, cats: dict[int, np.ndarray], a: dict[str, list]) -> None:
    """Append one sklearn TreePredictor to the TreeEnsemble attribute lists `a`."""
    nodes, left_bits = pred.nodes, pred.raw_left_cat_bitsets
    counter = [0]

    def new_id() -> int:
        counter[0] += 1
        return counter[0] - 1

    def branch(nid: int, feat: int, mode: str, value: float) -> int:
        """Append a node record (pre-order: ONNX Runtime takes a tree's first node as its root);
        children ids are filled in by `link` once they exist."""
        a["nodes_treeids"].append(tree_id)
        a["nodes_nodeids"].append(nid)
        a["nodes_featureids"].append(int(feat))
        a["nodes_modes"].append(mode)
        a["nodes_values"].append(float(value))
        a["nodes_truenodeids"].append(0)
        a["nodes_falsenodeids"].append(0)
        a["nodes_missing_value_tracks_true"].append(0)
        return len(a["nodes_nodeids"]) - 1

    def link(pos: int, t: int, f: int, missing_true: bool = False) -> None:
        a["nodes_truenodeids"][pos] = t
        a["nodes_falsenodeids"][pos] = f
        a["nodes_missing_value_tracks_true"][pos] = int(missing_true)

    def emit(k: int) -> int:
        nd = nodes[k]
        nid = new_id()
        if nd["is_leaf"]:
            branch(nid, 0, "LEAF", 0.0)
            a["target_treeids"].append(tree_id)
            a["target_nodeids"].append(nid)
            a["target_ids"].append(0)
            a["target_weights"].append(float(nd["value"]))
            return nid
        j = int(nd["feature_idx"])
        feat = int(order[j])
        md_left = bool(nd["missing_go_to_left"])
        if not nd["is_categorical"]:
            pos = branch(nid, feat, "BRANCH_LEQ", _floor32(float(nd["num_threshold"])))
            left = emit(int(nd["left"]))
            link(pos, left, emit(int(nd["right"])), md_left)
            return nid
        raw = cats[j]
        left_codes = _bits(left_bits[int(nd["bitset_idx"])])
        L = [raw[c] for c in sorted(left_codes) if c < len(raw)]
        R = [raw[c] for c in range(len(raw)) if c not in left_codes]
        # values in `chain` go to `hit`; everything else (incl. unknown / missing) goes to `miss`
        if md_left:
            chain, hit, miss = R, int(nd["right"]), int(nd["left"])
        else:
            chain, hit, miss = L, int(nd["left"]), int(nd["right"])
        if not chain:  # no known value reaches `hit`: always-true test on +inf
            pos = branch(nid, feat, "BRANCH_LEQ", float(np.float32(np.inf)))
            sub = emit(miss)
            link(pos, sub, sub, True)
            return nid
        cid = nid
        for n, v in enumerate(chain):
            pos = branch(cid, feat, "BRANCH_EQ", float(np.float32(v)))
            hit_id = emit(hit)  # duplicated per member: tree nodes have exactly one parent
            nxt = new_id() if n + 1 < len(chain) else emit(miss)
            link(pos, hit_id, nxt)
            cid = nxt
        return nid

    emit(0)


def hgb_to_onnx(model: Any) -> Any:
    """HistGradientBoostingClassifier (binary) -> onnx.ModelProto; input 'X' float32 [N, F], output 'p' [N, 1]."""
    from onnx import TensorProto, helper

    if getattr(model, "n_trees_per_iteration_", 1) != 1:
        raise ValueError("only binary classifiers are supported")
    order, cats = _feature_maps(model)
    a: dict[str, list] = {
        k: []
        for k in (
            "nodes_treeids",
            "nodes_nodeids",
            "nodes_featureids",
            "nodes_modes",
            "nodes_values",
            "nodes_truenodeids",
            "nodes_falsenodeids",
            "nodes_missing_value_tracks_true",
            "target_treeids",
            "target_nodeids",
            "target_ids",
            "target_weights",
        )
    }

    for tree_id, (pred,) in enumerate(model._predictors):
        _export_tree(tree_id, pred, order, cats, a)

    base = float(np.ravel(model._baseline_prediction)[0])
    trees = helper.make_node(
        "TreeEnsembleRegressor",
        ["X"],
        ["raw"],
        domain="ai.onnx.ml",
        n_targets=1,
        aggregate_function="SUM",
        base_values=[base],
        post_transform="NONE",
        **a,  # type: ignore[arg-type]
    )
    sig = helper.make_node("Sigmoid", ["raw"], ["p"])
    graph = helper.make_graph(
        [trees, sig],
        "recsys_ranker",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, int(model.n_features_in_)])],
        [helper.make_tensor_value_info("p", TensorProto.FLOAT, [None, 1])],
    )
    m = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("ai.onnx.ml", 3)],
        producer_name="recsys",
    )
    m.ir_version = 9
    return m


class OnnxScorer:
    """ONNX Runtime session over an exported ranker (single-threaded: serving parallelism
    comes from workers, not intra-op threads)."""

    def __init__(self, model_bytes: bytes):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(model_bytes, so, providers=["CPUExecutionProvider"])

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.sess.run(["p"], {"X": np.ascontiguousarray(X, dtype=np.float32)})[0][:, 0]
