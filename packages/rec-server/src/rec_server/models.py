"""Offline-trained artifacts: Swing item-to-item similarity, item embeddings,
and per-market hot / category / brand lists."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds

from .data import EVENT_WEIGHTS, Catalog, ItemStats, Tables


def swing(
    user_items: sparse.csr_matrix, alpha: float = 1.0, max_users: int = 300, topk: int = 50, seed: int = 0
) -> sparse.csr_matrix:
    """Swing similarity (Alibaba): sim(i,j) = sum_{u<v in U_i∩U_j} w_u w_v / (alpha + |I_u ∩ I_v|).

    For each item i we take its users U_i (sampled to max_users), compute the user-user
    overlap O = M_U M_U^T, W = w w^T / (alpha + O) (zero diagonal), and then
    sim(i, ·) = 0.5 * colsum((W @ M_U) ∘ M_U). Vectorised per item; fine for 1e5 items
    on a batch cluster, seconds for the demo.
    """
    rng = np.random.default_rng(seed)
    M = (user_items > 0).astype(np.float32).tocsr()
    n_items = M.shape[1]
    deg = np.asarray(M.sum(1)).ravel()
    uw = 1.0 / np.sqrt(np.maximum(deg, 1.0)) ** 0.5  # dampen very active users
    Mc = M.tocsc()
    rows, cols, vals = [], [], []
    for i in range(n_items):
        users = Mc.indices[Mc.indptr[i] : Mc.indptr[i + 1]]
        if len(users) < 2:
            continue
        if len(users) > max_users:
            users = rng.choice(users, max_users, replace=False)
        Mu = M[users]
        overlap = (Mu @ Mu.T).toarray()
        w = uw[users]
        W = np.outer(w, w) / (alpha + overlap)
        np.fill_diagonal(W, 0.0)
        sim = 0.5 * np.asarray(Mu.multiply(sparse.csr_matrix(W) @ Mu).sum(0)).ravel()
        sim[i] = 0.0
        nz = np.flatnonzero(sim > 0)
        if len(nz) == 0:
            continue
        top = nz[np.argsort(-sim[nz])[:topk]]
        rows.extend([i] * len(top))
        cols.extend(top.tolist())
        vals.extend(sim[top].tolist())
    S = sparse.csr_matrix((vals, (rows, cols)), shape=(n_items, n_items), dtype=np.float32)
    # row-normalise so different seeds contribute comparably
    mx = np.asarray(S.max(1).todense()).ravel()
    S = sparse.diags(1.0 / np.maximum(mx, 1e-9)) @ S
    return S.tocsr()


def item_embeddings(
    user_items: sparse.csr_matrix,
    catalog: Catalog,
    dim: int = 32,
    w_cf: float = 0.7,
    w_cat: float = 0.6,
    w_brand: float = 0.5,
    w_price: float = 0.3,
) -> np.ndarray:
    """Hybrid item vectors (a stand-in for a two-tower model): truncated SVD of the
    log-scaled engagement matrix, concatenated with content one-hots (category, brand)
    and price position. Items with no engagement get the CF part from their brand
    (else category) mean, so cold-start items are still retrievable."""
    X = user_items.astype(np.float32).tocsr().copy()
    X.data = np.log1p(X.data)
    k = min(dim, min(X.shape) - 1)
    _, s, vt = svds(X, k=k, random_state=0)
    V = (vt.T * np.sqrt(s)).astype(np.float32)
    has = np.asarray((X > 0).sum(0)).ravel() > 0
    for key in (catalog.brand, catalog.cat):
        missing = ~has & (np.linalg.norm(V, axis=1) == 0)
        if not missing.any():
            break
        df = pd.DataFrame({"k": key[has]})
        for g, ix in df.groupby("k").groups.items():
            members = np.flatnonzero(has)[np.asarray(ix)]
            tgt = np.flatnonzero(missing & (key == g))
            if len(tgt):
                V[tgt] = V[members].mean(0)
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)
    cat_oh = np.eye(len(catalog.cat_names), dtype=np.float32)[catalog.cat]
    brand_oh = np.eye(len(catalog.brand_names), dtype=np.float32)[catalog.brand]
    price = np.stack([np.cos(np.pi * catalog.price_pct), np.sin(np.pi * catalog.price_pct)], 1).astype(np.float32)
    H = np.hstack([w_cf * V, w_cat * cat_oh, w_brand * brand_oh, w_price * price])
    return H / np.maximum(np.linalg.norm(H, axis=1, keepdims=True), 1e-9)


@dataclass
class Models:
    swing: sparse.csr_matrix  # click/cart co-engagement
    buy_swing: sparse.csr_matrix  # cart/purchase co-occurrence ("bought together")
    item_vec: np.ndarray
    hot: dict[str, np.ndarray]  # market -> item idx sorted by hotness
    cat_lists: dict[str, dict[int, np.ndarray]]  # market -> cat -> sorted idx
    brand_lists: dict[str, dict[int, np.ndarray]]

    @classmethod
    def train(cls, t: Tables, catalog: Catalog, stats: ItemStats, cutoff_ts: float) -> Models:
        b = t.behaviors[(t.behaviors["ts"] < cutoff_ts) & (t.behaviors["event_type"] != "impression")]
        uix = pd.Index(b["user_id"].unique())
        rows = uix.get_indexer(b["user_id"])
        cols = b["item_id"].map(catalog.id2idx).to_numpy()
        w = b["event_type"].map(EVENT_WEIGHTS).fillna(0).to_numpy()
        shape = (len(uix), catalog.n)
        UI = sparse.csr_matrix((w, (rows, cols)), shape=shape)
        UI.sum_duplicates()
        buy = b["event_type"].isin(["add_cart", "purchase"]).to_numpy()
        UB = sparse.csr_matrix((np.ones(buy.sum()), (rows[buy], cols[buy])), shape=shape)

        hot, cat_lists, brand_lists = {}, {}, {}
        for m in catalog.markets:
            sold = ~np.isnan(catalog.price[m])
            score = stats.hot[m] + 1e-6 * stats.ctr  # tie-break by prior CTR
            order = np.argsort(-score)
            order = order[sold[order]]
            hot[m] = order
            cat_lists[m] = {c: order[catalog.cat[order] == c] for c in range(len(catalog.cat_names))}
            brand_lists[m] = {g: order[catalog.brand[order] == g] for g in range(len(catalog.brand_names))}
        return cls(
            swing=swing(UI),
            buy_swing=swing(UB, alpha=0.5),
            item_vec=item_embeddings(UI, catalog),
            hot=hot,
            cat_lists=cat_lists,
            brand_lists=brand_lists,
        )
