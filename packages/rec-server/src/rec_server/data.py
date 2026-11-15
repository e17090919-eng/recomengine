"""Tables, catalog and user state. Storage is in-memory here; the interfaces
(UserStore.add_event / history, Catalog.available) are what Redis-backed
implementations will provide in production."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from recsys_core.config import load_config  # re-exported for callers
from recsys_core.lake import Lake

__all__ = ["DAY", "EVENT_WEIGHTS", "Catalog", "Clock", "ItemStats", "Profile", "Tables", "UserStore", "load_config"]

DAY = 86400
EVENT_WEIGHTS = {"click": 1.0, "favorite": 2.0, "add_cart": 3.0, "purchase": 4.0}
ENGAGE_EVENTS = set(EVENT_WEIGHTS)


@dataclass
class Tables:
    """In-memory view of the canonical lake used by batch training."""

    users: pd.DataFrame
    items: pd.DataFrame
    item_market: pd.DataFrame
    behaviors: pd.DataFrame
    categories: pd.DataFrame
    meta: dict

    @classmethod
    def load(cls, lake_dir: Path | str) -> Tables:
        lake = Lake(lake_dir)
        items = lake.read("item")
        b = lake.read("behavior")
        known = b["item_id"].isin(items["item_id"])  # unknown SKUs are reported by the quality gate
        b = b[known].reset_index(drop=True)
        b["request_id"] = b["request_id"].fillna("0")
        b["parent_item_id"] = b["parent_item_id"].fillna("")
        b["paid_amount"] = b["paid_amount"].fillna(0.0)
        u = lake.read("user")
        for c, d in (("country", ""), ("language", "en"), ("member_level", 0), ("allowance_used", 0.0)):
            u[c] = u[c].fillna(d)
        meta = {**lake.meta(), "dropped_unknown_item_events": int((~known).sum())}
        return cls(
            users=u,
            items=items,
            item_market=lake.read("item_market"),
            behaviors=b,
            categories=lake.read("category"),
            meta=meta,
        )

    @property
    def start_ts(self) -> int:
        if "start_ts" in self.meta:
            return int(self.meta["start_ts"])
        return int(self.behaviors["ts"].min() // DAY * DAY)

    @property
    def end_ts(self) -> int:
        if "start_ts" in self.meta and "n_days" in self.meta:
            return int(self.meta["start_ts"] + self.meta["n_days"] * DAY)
        return int(self.behaviors["ts"].max()) + 1


# ----------------------------------------------------------------- catalog
@dataclass
class Catalog:
    item_ids: np.ndarray
    id2idx: dict[str, int]
    cat: np.ndarray
    brand: np.ndarray
    cat_names: list[str]
    brand_names: list[str]
    price_usd: np.ndarray
    price_pct: np.ndarray  # price percentile within its category (0..1)
    publish_ts: np.ndarray
    titles: dict[str, list[str]]
    image: list[str]
    markets: list[str]
    price: dict[str, np.ndarray]  # local price, NaN = not sold in market
    stock: dict[str, np.ndarray]
    currency: dict[str, str]
    cat_meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.item_ids)

    def available(self, market: str, now: float) -> np.ndarray:
        if market not in self.price:
            return np.zeros(self.n, dtype=bool)
        return ~np.isnan(self.price[market]) & (self.stock[market] > 0) & (self.publish_ts <= now)

    def idx(self, item_id: str) -> int | None:
        return self.id2idx.get(item_id)

    def update_market(self, item_id: str, market: str, price: float | None = None, stock: int | None = None) -> bool:
        i = self.idx(item_id)
        if i is None or market not in self.price:
            return False
        if price is not None:
            self.price[market][i] = price
        if stock is not None:
            self.stock[market][i] = stock
        return True

    def describe(self, i: int, market: str, lang: str = "en") -> dict:
        lang = lang if lang in self.titles else "en"
        cat = self.cat_names[self.cat[i]]
        cm = self.cat_meta.get(cat, {})
        p = self.price.get(market, np.full(self.n, np.nan))[i]
        return dict(
            id=str(self.item_ids[i]),
            title=self.titles[lang][i],
            brand=self.brand_names[self.brand[i]],
            category=cat,
            category_label=cm.get(lang, cat) if lang != "en" else cat,
            image=self.image[i],
            price=None if np.isnan(p) else float(p),
            currency=self.currency.get(market),
            publish_ts=int(self.publish_ts[i]),
        )

    @classmethod
    def build(cls, t: Tables) -> Catalog:
        it = t.items.reset_index(drop=True)
        cat_names = sorted(it["category"].unique())
        brand_names = sorted(it["brand"].unique())
        cmap = {c: i for i, c in enumerate(cat_names)}
        bmap = {b: i for i, b in enumerate(brand_names)}
        ids = it["item_id"].to_numpy()
        id2idx = {k: i for i, k in enumerate(ids)}
        n = len(ids)
        markets = sorted(t.item_market["market"].unique())
        price = {m: np.full(n, np.nan) for m in markets}
        stock = {m: np.zeros(n, dtype=np.int64) for m in markets}
        currency: dict[str, str] = {}
        for m, g in t.item_market.groupby("market"):
            ix = g["item_id"].map(id2idx).to_numpy()
            price[m][ix] = g["price"].to_numpy()
            stock[m][ix] = g["stock"].to_numpy()
            currency[str(m)] = str(g["currency"].iloc[0])
        price_pct = it.groupby("category")["price_usd"].rank(pct=True).to_numpy()
        tcols = [c for c in it.columns if c.startswith("title_")]
        fallback = it[tcols].bfill(axis=1).iloc[:, 0].fillna(it["item_id"]) if tcols else it["item_id"]
        titles = {c.removeprefix("title_"): it[c].fillna(fallback).tolist() for c in tcols}
        titles.setdefault("en", fallback.tolist())
        return cls(
            item_ids=ids,
            id2idx=id2idx,
            cat=it["category"].map(cmap).to_numpy(),
            brand=it["brand"].map(bmap).to_numpy(),
            cat_names=cat_names,
            brand_names=brand_names,
            price_usd=it["price_usd"].to_numpy(dtype=float),
            price_pct=price_pct,
            publish_ts=it["publish_ts"].to_numpy(),
            titles=titles,
            image=it["image"].fillna("").tolist() if "image" in it else [""] * n,
            markets=markets,
            price=price,
            stock=stock,
            currency=currency,
            cat_meta=_category_meta(t.categories),
        )


def _category_meta(df: pd.DataFrame) -> dict:
    """category -> {lang: label, 'image': ...} from the optional category table."""
    out: dict[str, dict] = {}
    for r in df.to_dict("records"):
        out[str(r["category"])] = {
            str(k).removeprefix("label_"): v for k, v in r.items() if str(k).startswith("label_") and v
        }
        if r.get("image"):
            out[r["category"]]["image"] = r["image"]
    return out


# ----------------------------------------------------------------- users
@dataclass
class Profile:
    user_id: str
    market: str
    country: str = ""
    language: str = "en"
    member_level: int = 0
    allowance_used: float = 0.0


class UserStore:
    """Per-user profile + engagement history (+ recent impressions)."""

    def __init__(self, max_events: int = 500, max_impressions: int = 300):
        self.profiles: dict[str, Profile] = {}
        self.events: dict[str, deque] = defaultdict(partial(deque, maxlen=max_events))
        self.impressions: dict[str, deque] = defaultdict(partial(deque, maxlen=max_impressions))

    def upsert_profile(self, p: Profile) -> None:
        self.profiles[p.user_id] = p

    def add_event(self, user_id: str, ts: float, item_idx: int, event_type: str, paid: float = 0.0) -> None:
        if event_type == "impression":
            self.impressions[user_id].append((ts, item_idx))
        elif event_type in ENGAGE_EVENTS or event_type == "dislike":
            self.events[user_id].append((ts, item_idx, event_type))
            if event_type == "purchase" and user_id in self.profiles:
                self.profiles[user_id].allowance_used += paid

    def history(self, user_id: str) -> list[tuple]:
        return list(self.events.get(user_id, ()))

    def recent_impressions(self, user_id: str) -> list[tuple]:
        return list(self.impressions.get(user_id, ()))

    @classmethod
    def build(cls, t: Tables, catalog: Catalog, cutoff_ts: float) -> UserStore:
        s = cls()
        for r in t.users.to_dict("records"):
            s.upsert_profile(
                Profile(
                    str(r["user_id"]),
                    str(r["market"]),
                    str(r["country"]),
                    str(r["language"]),
                    int(r["member_level"]),
                    float(r["allowance_used"]),
                )
            )
        b = t.behaviors[t.behaviors["ts"] < cutoff_ts]
        idx = b["item_id"].map(catalog.id2idx)
        for uid, ts, ix, et, paid in zip(
            b["user_id"].to_numpy(),
            b["ts"].to_numpy(),
            idx.to_numpy(),
            b["event_type"].to_numpy(),
            b["paid_amount"].to_numpy(),
        ):
            if ix == ix:  # not NaN
                s.add_event(uid, float(ts), int(ix), et, float(paid))
        return s


# ----------------------------------------------------------------- item stats
@dataclass
class ItemStats:
    impressions: np.ndarray
    clicks: np.ndarray
    carts: np.ndarray
    purchases: np.ndarray
    hot: dict[str, np.ndarray]  # market -> time-decayed weighted engagement
    market_ctr: dict[str, np.ndarray]

    @property
    def ctr(self) -> np.ndarray:
        return (self.clicks + 1.0) / (self.impressions + 20.0)  # Beta prior ~5%

    @classmethod
    def build(cls, t: Tables, catalog: Catalog, cutoff_ts: float, half_life_days: float = 3.0) -> ItemStats:
        b = t.behaviors[(t.behaviors["ts"] < cutoff_ts)]
        ix = b["item_id"].map(catalog.id2idx).fillna(-1).astype(int).to_numpy()
        ok = ix >= 0
        b, ix = b[ok], ix[ok]
        et = b["event_type"].to_numpy()
        n = catalog.n

        def count(e: str) -> np.ndarray:
            return np.bincount(ix[et == e], minlength=n).astype(float)

        w = b["event_type"].map(EVENT_WEIGHTS).fillna(0).to_numpy()
        decay = 0.5 ** ((cutoff_ts - b["ts"].to_numpy()) / (half_life_days * DAY))
        hot, mctr = {}, {}
        mk = b["market"].to_numpy()
        for m in catalog.markets:
            sel = mk == m
            hot[m] = np.bincount(ix[sel], weights=(w * decay)[sel], minlength=n)
            imp = np.bincount(ix[sel & (et == "impression")], minlength=n)
            clk = np.bincount(ix[sel & (et == "click")], minlength=n)
            mctr[m] = (clk + 1.0) / (imp + 20.0)
        return cls(count("impression"), count("click"), count("add_cart"), count("purchase"), hot, mctr)


class Clock:
    """Simulation-aware clock: data time + wall time elapsed since start (live), or frozen."""

    def __init__(self, base_ts: float, live: bool = False):
        self.base_ts, self.live, self._t0 = base_ts, live, time.time()

    def now(self) -> float:
        return self.base_ts + (time.time() - self._t0 if self.live else 0.0)
