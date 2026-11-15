"""Online user context and ranking features. The same code builds training
samples offline and serves online, so there is no train/serve skew."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import DAY, EVENT_WEIGHTS, Catalog, ItemStats, Profile, UserStore
from .models import Models

FEATURES = [
    "item_ctr",
    "item_log_clicks",
    "item_log_purchases",
    "item_log_price",
    "item_age_days",
    "item_cat",
    "user_cat_share",
    "user_brand_share",
    "user_log_engaged",
    "user_member_level",
    "price_gap",
    "emb_dot",
    "swing_aff",
    "market",
    "item_is_new",
    "user_dislike_brand",
    "item_price_pct",
    "item_market_ctr",
    "item_market_hot",
]
CATEGORICAL = [FEATURES.index("item_cat"), FEATURES.index("market")]


@dataclass
class UserCtx:
    user_id: str
    market: str
    profile: Profile
    now: float
    recent: np.ndarray = field(default_factory=lambda: np.zeros(0, int))  # distinct engaged items, newest first
    recent_w: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cat_w: np.ndarray = field(default_factory=lambda: np.zeros(0))  # category weights (sum to 1)
    brand_w: np.ndarray = field(default_factory=lambda: np.zeros(0))
    engaged: float = 0.0
    mean_price_pct: float = float("nan")
    purchased: dict[int, float] = field(default_factory=dict)
    cart: list[int] = field(default_factory=list)
    disliked_brands: set[int] = field(default_factory=set)
    user_vec: np.ndarray | None = None
    spend: float = 0.0


def build_user_ctx(
    store: UserStore,
    catalog: Catalog,
    models: Models,
    user_id: str,
    market: str | None,
    now: float,
    default_market: str,
    half_life_days: float = 7.0,
) -> UserCtx:
    prof_, history = store.view(user_id)
    prof = prof_ or Profile(user_id=user_id, market=market or default_market)
    mkt = market or prof.market or default_market
    ctx = UserCtx(user_id=user_id, market=mkt, profile=prof, now=now, spend=prof.allowance_used)
    nc, nb = len(catalog.cat_names), len(catalog.brand_names)
    ctx.cat_w, ctx.brand_w = np.zeros(nc), np.zeros(nb)
    hist = [h for h in history if h[0] <= now]
    if not hist:
        return ctx
    item_w: dict[int, float] = {}
    last_seen: dict[int, float] = {}
    in_cart: dict[int, float] = {}
    pcts, logw = [], []
    for ts, i, et in hist:
        if et == "dislike":
            ctx.disliked_brands.add(int(catalog.brand[i]))
            continue
        w = EVENT_WEIGHTS[et] * 0.5 ** ((now - ts) / (half_life_days * DAY))
        item_w[i] = item_w.get(i, 0.0) + w
        last_seen[i] = max(last_seen.get(i, 0.0), ts)
        ctx.cat_w[catalog.cat[i]] += w
        ctx.brand_w[catalog.brand[i]] += w
        pcts.append(catalog.price_pct[i])
        logw.append(w)
        if et == "add_cart":
            in_cart[i] = ts
        elif et == "purchase":
            ctx.purchased[i] = ts
            in_cart.pop(i, None)
    ctx.engaged = float(sum(item_w.values()))
    if ctx.engaged > 0:
        ctx.cat_w /= ctx.engaged
        ctx.brand_w /= ctx.engaged
        ctx.mean_price_pct = float(np.average(pcts, weights=logw))
    order = sorted(last_seen, key=lambda i: -last_seen[i])[:30]
    ctx.recent = np.array(order, dtype=int)
    ctx.recent_w = np.array([item_w[i] for i in order])
    ctx.cart = sorted(in_cart, key=lambda i: -in_cart[i])
    v = models.item_vec[ctx.recent]
    uv = (ctx.recent_w[:, None] * v).sum(0)
    n = np.linalg.norm(uv)
    ctx.user_vec = uv / n if n > 0 else None
    return ctx


def swing_affinity(models: Models, ctx: UserCtx, cand: np.ndarray, n_seeds: int = 20) -> np.ndarray:
    if len(ctx.recent) == 0 or len(cand) == 0:
        return np.zeros(len(cand))
    seeds = ctx.recent[:n_seeds]
    w = ctx.recent_w[:n_seeds]
    aff = models.swing[seeds].T @ w
    return np.asarray(aff).ravel()[cand]


def build_features(
    ctx: UserCtx, cand: np.ndarray, catalog: Catalog, stats: ItemStats, models: Models, market_codes: dict[str, int]
) -> np.ndarray:
    n = len(cand)
    X = np.zeros((n, len(FEATURES)), dtype=np.float32)
    if n == 0:
        return X
    lp = np.log(catalog.price_usd[cand])
    age = (ctx.now - catalog.publish_ts[cand]) / DAY
    X[:, 0] = stats.ctr[cand]
    X[:, 1] = np.log1p(stats.clicks[cand])
    X[:, 2] = np.log1p(stats.purchases[cand])
    X[:, 3] = lp
    X[:, 4] = age
    X[:, 5] = catalog.cat[cand]
    X[:, 6] = ctx.cat_w[catalog.cat[cand]]
    X[:, 7] = ctx.brand_w[catalog.brand[cand]]
    X[:, 8] = np.log1p(ctx.engaged)
    X[:, 9] = ctx.profile.member_level
    pp = catalog.price_pct[cand]
    X[:, 10] = 0.0 if np.isnan(ctx.mean_price_pct) else np.abs(pp - ctx.mean_price_pct)
    X[:, 11] = models.item_vec[cand] @ ctx.user_vec if ctx.user_vec is not None else 0.0
    X[:, 12] = swing_affinity(models, ctx, cand)
    X[:, 13] = market_codes.get(ctx.market, -1)
    X[:, 14] = (age <= 14).astype(np.float32)
    X[:, 15] = np.isin(catalog.brand[cand], list(ctx.disliked_brands)) if ctx.disliked_brands else 0.0
    X[:, 16] = pp
    X[:, 17] = stats.market_ctr[ctx.market][cand] if ctx.market in stats.market_ctr else stats.ctr[cand]
    X[:, 18] = np.log1p(stats.hot[ctx.market][cand]) if ctx.market in stats.hot else 0.0
    return X
