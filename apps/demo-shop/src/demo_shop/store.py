"""The shop's own data: catalog, shoppers, order history, carts.

A real retailer already has all of this in its own systems; here it is read from the
simulated lake. The shop never imports engine code — recommendations and behavior
tracking go through the public API (see app.py)."""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

DAY = 86400.0
CONSUMABLES = ("skincare", "makeup", "health", "confectionery", "liquor")
LUXURY = ("watches", "jewelry", "handbags")


@dataclass
class Order:
    id: str
    ts: float
    market: str
    lines: list[dict[str, Any]]  # {item_id, qty, price}
    total: float
    live: bool = False  # placed during the demo (vs. history from the lake)


@dataclass
class Session:
    cart: dict[str, int] = field(default_factory=dict)
    orders: list[Order] = field(default_factory=list)


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return df.to_dict("records")  # type: ignore[return-value]


class ShopData:
    def __init__(self, lake: Path):
        items = pd.read_parquet(lake / "item.parquet")
        im = pd.read_parquet(lake / "item_market.parquet")
        cats = pd.read_parquet(lake / "category.parquet")
        users = pd.read_parquet(lake / "user.parquet")
        b = pd.read_parquet(
            lake / "behavior", columns=["user_id", "item_id", "event_type", "ts", "paid_amount", "market"]
        )
        # interactions with known products only (the raw export has a few orphan rows)
        b = b[(b["event_type"] != "impression") & b["item_id"].isin(items["item_id"])].copy()
        b["category"] = b["item_id"].map(items.set_index("item_id")["category"])

        self.categories = [
            {"id": r["category"], "emoji": r["image"], "zh": r["label_zh"], "en": str(r["label_en"]).capitalize()}
            for r in _records(cats)
        ]
        self.items: dict[str, dict[str, Any]] = {
            r["item_id"]: {
                "id": r["item_id"],
                "category": r["category"],
                "brand": r["brand"],
                "emoji": r["image"],
                "title": {"zh": r["title_zh"], "en": r["title_en"]},
                "price": {},
                "stock": {},
            }
            for r in _records(items)
        }
        self.currency: dict[str, str] = {}
        for r in _records(im):
            it = self.items[r["item_id"]]
            it["price"][r["market"]] = float(r["price"])
            it["stock"][r["market"]] = int(r["stock"])
            self.currency[r["market"]] = r["currency"]
        self.users = users.set_index("user_id")
        self.data_now = float(b["ts"].max())
        self._t0 = time.time()

        # the "current approach" we compare against: best sellers of the last 7 days
        recent = b[b["ts"] >= self.data_now - 7 * DAY]
        self.bestsellers: dict[str, list[str]] = {}
        self.by_category: dict[str, dict[str, list[str]]] = {}
        for m in self.currency:
            score = (
                recent[recent["market"] == m]
                .assign(w=lambda d: d["event_type"].map({"purchase": 5, "add_cart": 2, "click": 1}).fillna(0))
                .groupby("item_id")["w"]
                .sum()
                .sort_values(ascending=False)
            )
            avail = [i for i in score.index if self.items[i]["stock"].get(m, 0) > 0]
            self.bestsellers[m] = avail[:60]
            rank = {i: n for n, i in enumerate(avail)}
            self.by_category[m] = {}
            for c in self.categories:
                ids = [i for i, it in self.items.items() if it["category"] == c["id"] and m in it["price"]]
                self.by_category[m][c["id"]] = sorted(ids, key=lambda i: rank.get(i, 10**9))

        # order history: one order per user per market per day
        pur = b[b["event_type"] == "purchase"].copy()
        pur["day"] = (pur["ts"] // DAY).astype(int)
        self.history: dict[str, list[Order]] = {}
        for key, g in pur.groupby(["user_id", "market", "day"]):
            uid, m = str(key[0]), str(key[1])
            lines = [{"item_id": r["item_id"], "qty": 1, "price": float(r["paid_amount"] or 0)} for r in _records(g)]
            total = sum(x["price"] for x in lines)
            o = Order(f"H{int(g['ts'].iloc[0])}{str(uid)[-4:]}", float(g["ts"].max()), str(m), lines, total)
            self.history.setdefault(str(uid), []).append(o)
        self.engaged = b
        self.sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._oid = itertools.count(1)

    # ------------------------------------------------------------------ clock
    def now(self) -> float:
        """Data time: the end of the simulated history plus wall time since start (same
        convention as the engine's clock)."""
        return self.data_now + (time.time() - self._t0)

    def reset(self) -> None:
        with self._lock:
            self.sessions.clear()
            self._t0 = time.time()

    # ------------------------------------------------------------------ shoppers
    def session(self, uid: str) -> Session:
        with self._lock:
            return self.sessions.setdefault(uid, Session())

    def profile(self, uid: str) -> dict[str, Any]:
        if uid in self.users.index:
            u: Any = self.users.loc[uid]
            return {
                "market": str(u["market"]),
                "language": str(u["language"]),
                "member_level": int(u["member_level"]),
                "allowance_used": float(u["allowance_used"]),
            }
        return {"market": None, "language": None, "member_level": 0, "allowance_used": 0.0}

    def orders(self, uid: str) -> list[Order]:
        live = self.sessions.get(uid, Session()).orders
        return sorted(self.history.get(uid, []) + live, key=lambda o: -o.ts)

    def spent(self, uid: str, market: str) -> float:
        """Duty-free allowance used this year: the shop's own records + orders placed now."""
        live = self.sessions.get(uid, Session()).orders
        return self.profile(uid)["allowance_used"] + sum(o.total for o in live if o.market == market)

    def place_order(self, uid: str, market: str) -> Order | None:
        s = self.session(uid)
        with self._lock:
            lines = [
                {"item_id": i, "qty": q, "price": self.items[i]["price"][market]}
                for i, q in s.cart.items()
                if market in self.items[i]["price"]
            ]
            if not lines:
                return None
            total = sum(x["price"] * x["qty"] for x in lines)
            o = Order(f"D{next(self._oid):05d}", self.now(), market, lines, total, live=True)
            s.orders.append(o)
            s.cart.clear()
        return o

    # ------------------------------------------------------------------ personas
    def personas(self) -> list[dict[str, Any]]:
        """Shoppers with a clear story, picked from the data (deterministic)."""
        b = self.engaged
        n = b.groupby("user_id").size()
        active = n[n >= 15].index
        bm = b[b["user_id"].isin(active)]

        def top_by(market: str, cats: tuple[str, ...], extra: pd.Index | None = None) -> str | None:
            d = bm[bm["market"] == market]
            if extra is not None:
                d = d[d["user_id"].isin(extra)]
            if d.empty:
                return None
            share = d.assign(hit=d["category"].isin(cats)).groupby("user_id")["hit"].agg(["mean", "size"])
            share = share[share["size"] >= 15]
            score = share["mean"] * share["size"].clip(upper=60) ** 0.5
            return str(score.idxmax()) if len(score) else None

        out: list[dict[str, Any]] = [
            {
                "id": "visitor-001",
                "market": "CN-HAINAN",
                "language": "zh",
                "label": {"zh": "新访客", "en": "New visitor"},
                "hint": {
                    "zh": "没有任何记录：先看到热门，点几个商品后首页马上变",
                    "en": "No history: sees best sellers, then the feed adapts after a few clicks",
                },
            }
        ]
        u = self.users
        low_allowance = u[(u["allowance_used"] >= 80000) & (u["allowance_used"] <= 95000)].index
        picks = [
            ("CN-HAINAN", ("skincare", "makeup"), None, {"zh": "护肤达人", "en": "Skincare lover"},
             {"zh": "海南用户，常买护肤彩妆", "en": "Hainan shopper, mostly skincare & makeup"}),
            ("CN-HAINAN", LUXURY, low_allowance, {"zh": "额度快用完的高端客", "en": "Luxury, allowance almost used"},
             {"zh": "免税额度所剩不多：搭配推荐会换成买得起的", "en": "Allowance nearly used: add-ons adapt"}),
            ("NZ-AKL", ("liquor", "wine"), None, {"zh": "烈酒爱好者", "en": "Spirits buyer"},
             {"zh": "奥克兰机场用户，常买烈酒葡萄酒", "en": "Auckland shopper, spirits & wine"}),
        ]  # fmt: skip
        for market, cats, extra, label, hint in picks:
            uid = top_by(market, cats, extra)
            if uid:
                p = self.profile(uid)
                out.append({"id": uid, "market": market, "language": p["language"], "label": label, "hint": hint})
        return out

    def restock_candidates(self, limit: int = 150) -> list[str]:
        """Users whose last consumable purchase was 1-6 weeks ago (probed against the engine)."""
        p = self.engaged[(self.engaged["event_type"] == "purchase") & self.engaged["category"].isin(CONSUMABLES)]
        last = p.groupby("user_id")["ts"].max()
        age = (self.data_now - last) / DAY
        cnt = p.groupby("user_id").size()
        ok = cnt[(age >= 7) & (age <= 45)].sort_values(ascending=False)
        return [str(u) for u in ok.index[:limit]]

    # ------------------------------------------------------------------ presentation
    def card(self, item_id: str, market: str, lang: str) -> dict[str, Any] | None:
        it = self.items.get(item_id)
        if it is None:
            return None
        cat = next((c for c in self.categories if c["id"] == it["category"]), None)
        return {
            "id": item_id,
            "title": it["title"].get(lang) or it["title"]["en"],
            "brand": it["brand"],
            "emoji": it["emoji"],
            "category": it["category"],
            "category_label": (cat or {}).get(lang, it["category"]),
            "price": it["price"].get(market),
            "currency": self.currency.get(market),
            "stock": it["stock"].get(market, 0),
        }
