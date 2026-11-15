"""Business rules: filters (keep-masks) and re-ranking (boost, dispersion).
Market-specific behaviour (allowance, purchase limits) is driven by config."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .data import DAY, Catalog
from .features import UserCtx


@dataclass
class RuleCtx:
    ctx: UserCtx
    catalog: Catalog
    market_cfg: dict
    avail: np.ndarray
    parent: int | None = None
    cart: list[int] = field(default_factory=list)
    filter_items: set[int] = field(default_factory=set)
    filter_spec: dict = field(default_factory=dict)  # request context.filter
    issued: set[int] = field(default_factory=set)  # cross-request dedup
    allowance_remaining: float | None = None


Filter = Callable[[np.ndarray, RuleCtx], np.ndarray]


def item_pool(c: np.ndarray, r: RuleCtx) -> np.ndarray:
    return r.avail[c]


def request_filter(c: np.ndarray, r: RuleCtx) -> np.ndarray:
    keep = ~np.isin(c, list(r.filter_items)) if r.filter_items else np.ones(len(c), bool)
    cat = r.catalog
    sa = r.filter_spec.get("stringArrayFeature", {}) or {}
    if "category" in sa:
        want = {cat.cat_names.index(x) for x in sa["category"].get("values", []) if x in cat.cat_names}
        keep &= np.isin(cat.cat[c], list(want))
    if "brand" in sa:
        want = {cat.brand_names.index(x) for x in sa["brand"].get("values", []) if x in cat.brand_names}
        keep &= np.isin(cat.brand[c], list(want))
    num = {**(r.filter_spec.get("floatFeature") or {}), **(r.filter_spec.get("doubleFeature") or {})}
    price = cat.price[r.ctx.market][c]
    if "price_min" in num:
        keep &= price >= num["price_min"]
    if "price_max" in num:
        keep &= price <= num["price_max"]
    return keep


def purchased(c: np.ndarray, r: RuleCtx, days: float = 90) -> np.ndarray:
    recent = [i for i, ts in r.ctx.purchased.items() if r.ctx.now - ts <= days * DAY]
    return ~np.isin(c, recent)


def allowance(c: np.ndarray, r: RuleCtx) -> np.ndarray:
    """Hide items the shopper cannot buy within their remaining duty-free allowance."""
    cap = float(r.market_cfg.get("allowance", 0) or 0)
    if cap <= 0 and r.allowance_remaining is None:
        return np.ones(len(c), bool)
    remaining = r.allowance_remaining if r.allowance_remaining is not None else cap - r.ctx.spend
    return r.catalog.price[r.ctx.market][c] <= max(remaining, 0.0)


def category_limit(c: np.ndarray, r: RuleCtx, window_days: float = 7) -> np.ndarray:
    limits = r.market_cfg.get("category_limits", {}) or {}
    if not limits:
        return np.ones(len(c), bool)
    cat = r.catalog
    bought: dict[int, int] = {}
    for i, ts in r.ctx.purchased.items():
        if r.ctx.now - ts <= window_days * DAY:
            bought[int(cat.cat[i])] = bought.get(int(cat.cat[i]), 0) + 1
    blocked = [
        cat.cat_names.index(k)
        for k, lim in limits.items()
        if k in cat.cat_names and bought.get(cat.cat_names.index(k), 0) >= lim
    ]
    return ~np.isin(cat.cat[c], blocked)


def issued_dedup(c: np.ndarray, r: RuleCtx) -> np.ndarray:
    return ~np.isin(c, list(r.issued)) if r.issued else np.ones(len(c), bool)


def exclude_parent(c: np.ndarray, r: RuleCtx) -> np.ndarray:
    return c != r.parent if r.parent is not None else np.ones(len(c), bool)


def in_cart(c: np.ndarray, r: RuleCtx) -> np.ndarray:
    return ~np.isin(c, list(r.cart) + list(r.ctx.cart))


FILTERS: dict[str, Filter] = {
    "item_pool": item_pool,
    "request_filter": request_filter,
    "purchased": purchased,
    "allowance": allowance,
    "category_limit": category_limit,
    "issued_dedup": issued_dedup,
    "exclude_parent": exclude_parent,
    "in_cart": in_cart,
}


def boost_new(c: np.ndarray, score: np.ndarray, catalog: Catalog, now: float, days: float, weight: float) -> np.ndarray:
    fresh = (now - catalog.publish_ts[c]) <= days * DAY
    return score * np.where(fresh, 1.0 + weight, 1.0)


def disperse(order: np.ndarray, catalog: Catalog, rules: list[dict], size: int) -> np.ndarray:
    """Greedy sliding-window dispersion: at most `max` items sharing `field` within any
    `window` consecutive slots. Rules earlier in the list have higher priority; if no
    candidate satisfies all rules, the lowest-priority rules are relaxed first."""
    if not rules:
        return order[:size]
    keys = {"brand": catalog.brand, "category": catalog.cat}
    pending = deque(order.tolist())
    out: list[int] = []
    while pending and len(out) < size:
        pick = None
        for n_rules in range(len(rules), -1, -1):
            active = rules[:n_rules]
            for j, i in enumerate(pending):
                if all(
                    sum(keys[rl["field"]][x] == keys[rl["field"]][i] for x in out[-(rl["window"] - 1) :]) < rl["max"]
                    for rl in active
                    if rl["window"] > 1
                ):
                    pick = j
                    break
            if pick is not None:
                break
        assert pick is not None  # with zero active rules the first pending item always qualifies
        i = pending[pick]
        del pending[pick]
        out.append(i)
    return np.array(out, dtype=int)
