"""Multi-channel recall. Each channel returns (item_idx, score) restricted to the
market's available items; the orchestrator merges channels by quota and weight."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .data import DAY, Catalog
from .features import UserCtx, swing_affinity
from .models import Models


@dataclass
class RecallInput:
    ctx: UserCtx
    avail: np.ndarray  # bool mask of items sellable in this market now
    parent: int | None = None
    cart: list[int] | None = None


Result = tuple[np.ndarray, np.ndarray]
EMPTY: Result = (np.zeros(0, int), np.zeros(0))


def _topk(scores: np.ndarray, mask: np.ndarray, k: int) -> Result:
    s = np.where(mask & (scores > 0), scores, -np.inf)
    k = min(k, int(np.isfinite(s).sum()))
    if k <= 0:
        return EMPTY
    ix = np.argpartition(-s, k - 1)[:k]
    ix = ix[np.argsort(-s[ix])]
    return ix, s[ix]


def _from_list(order: np.ndarray, avail: np.ndarray, k: int) -> Result:
    order = order[avail[order]][:k]
    return order, 1.0 / np.log2(np.arange(len(order)) + 2)


def hot(m: Models, c: Catalog, r: RecallInput, k: int) -> Result:
    return _from_list(m.hot.get(r.ctx.market, np.zeros(0, int)), r.avail, k)


def interest(m: Models, c: Catalog, r: RecallInput, k: int) -> Result:
    """Top categories/brands of the user (decayed weights) -> hottest items in them, quota ∝ weight."""
    ctx = r.ctx
    if ctx.engaged <= 0:
        return EMPTY
    out_i, out_s = [], []
    for weights, lists in ((ctx.cat_w, m.cat_lists[ctx.market]), (ctx.brand_w, m.brand_lists[ctx.market])):
        top = np.argsort(-weights)[:3]
        top = top[weights[top] > 0]
        if len(top) == 0:
            continue
        share = weights[top] / weights[top].sum()
        for g, sh in zip(top, share):
            ix, sc = _from_list(lists.get(int(g), np.zeros(0, int)), r.avail, max(1, int(k / 2 * sh)))
            out_i.append(ix)
            out_s.append(sc * sh)
    if not out_i:
        return EMPTY
    return np.concatenate(out_i), np.concatenate(out_s)


def retarget(m: Models, c: Catalog, r: RecallInput, k: int) -> Result:
    """Recently engaged but not purchased; carted items first."""
    ctx = r.ctx
    cart = [i for i in ctx.cart if r.avail[i]]
    rest = [i for i in ctx.recent if r.avail[i] and i not in ctx.purchased and i not in set(cart)]
    ix = np.array((cart + rest)[:k], dtype=int)
    return ix, 1.0 / np.log2(np.arange(len(ix)) + 2)


def swing_recall(m: Models, c: Catalog, r: RecallInput, k: int) -> Result:
    if len(r.ctx.recent) == 0:
        return EMPTY
    all_items = np.arange(c.n)
    return _topk(swing_affinity(m, r.ctx, all_items), r.avail, k)


def embedding(m: Models, c: Catalog, r: RecallInput, k: int) -> Result:
    if r.ctx.user_vec is None:
        return EMPTY
    return _topk(m.item_vec @ r.ctx.user_vec, r.avail, k)


def new_items(m: Models, c: Catalog, r: RecallInput, k: int, days: float = 14) -> Result:
    fresh = r.avail & ((r.ctx.now - c.publish_ts) <= days * DAY)
    score = 0.05 + (r.ctx.cat_w[c.cat] if len(r.ctx.cat_w) else 0.0)
    return _topk(np.where(fresh, score, 0.0), fresh, k)


def parent_swing(m: Models, c: Catalog, r: RecallInput, k: int) -> Result:
    if r.parent is None:
        return EMPTY
    return _topk(np.asarray(m.swing[r.parent].todense()).ravel(), r.avail, k)


def parent_embedding(m: Models, c: Catalog, r: RecallInput, k: int) -> Result:
    if r.parent is None:
        return EMPTY
    s = m.item_vec @ m.item_vec[r.parent]
    s[r.parent] = 0
    return _topk(s, r.avail, k)


def parent_brand(m: Models, c: Catalog, r: RecallInput, k: int) -> Result:
    if r.parent is None:
        return EMPTY
    return _from_list(m.brand_lists[r.ctx.market].get(int(c.brand[r.parent]), np.zeros(0, int)), r.avail, k)


def parent_category(m: Models, c: Catalog, r: RecallInput, k: int) -> Result:
    if r.parent is None:
        return EMPTY
    return _from_list(m.cat_lists[r.ctx.market].get(int(c.cat[r.parent]), np.zeros(0, int)), r.avail, k)


def bought_together(m: Models, c: Catalog, r: RecallInput, k: int) -> Result:
    seeds = list(r.cart or []) or ([r.parent] if r.parent is not None else []) or list(r.ctx.cart)
    if not seeds:
        return EMPTY
    s = np.asarray(m.buy_swing[seeds].sum(0)).ravel() + 0.3 * np.asarray(m.swing[seeds].sum(0)).ravel()
    s[seeds] = 0
    return _topk(s, r.avail, k)


def repurchase(m: Models, c: Catalog, r: RecallInput, k: int, early: float = 0.7, late: float = 2.5) -> Result:
    """Consumables the shopper is due to buy again: purchased items whose category has a
    learned repurchase cycle and whose last purchase was `early`..`late` cycles ago. Scores
    peak at exactly one cycle; the same item ranks above same-brand alternatives."""
    cyc = m.repurchase_days
    if not len(cyc) or not r.ctx.purchased:
        return EMPTY
    score = np.zeros(c.n)
    for i, ts in r.ctx.purchased.items():
        days = cyc[c.cat[i]]
        if not np.isfinite(days):
            continue
        x = (r.ctx.now - ts) / DAY / days
        if not early <= x <= late:
            continue
        s = 1.0 / (1.0 + abs(np.log(x)))
        score[i] = max(score[i], s)
        alts = m.brand_lists[r.ctx.market].get(int(c.brand[i]), np.zeros(0, int))
        alts = alts[(c.cat[alts] == c.cat[i]) & r.avail[alts] & (alts != i)][:3]
        score[alts] = np.maximum(score[alts], 0.5 * s)
    return _topk(score, r.avail, k)


RECALLS: dict[str, Callable[..., Result]] = {
    "hot": hot,
    "interest": interest,
    "retarget": retarget,
    "swing": swing_recall,
    "embedding": embedding,
    "new_items": new_items,
    "parent_swing": parent_swing,
    "parent_embedding": parent_embedding,
    "parent_brand": parent_brand,
    "parent_category": parent_category,
    "bought_together": bought_together,
    "repurchase": repurchase,
}


def merge(results: dict[str, Result], weights: dict[str, float]) -> tuple[np.ndarray, np.ndarray, dict[int, list[str]]]:
    """Rank-based fusion: each channel contributes weight / log2(rank+2)."""
    score: dict[int, float] = {}
    src: dict[int, list[str]] = {}
    for name, (ix, _) in results.items():
        w = weights.get(name, 1.0)
        for rank, i in enumerate(ix.tolist()):
            score[i] = score.get(i, 0.0) + w / np.log2(rank + 2)
            if name not in src.setdefault(i, []):
                src[i].append(name)
    if not score:
        return np.zeros(0, int), np.zeros(0), {}
    items = np.fromiter(score.keys(), dtype=int)
    scores = np.fromiter(score.values(), dtype=float)
    o = np.argsort(-scores)
    return items[o], scores[o], src
