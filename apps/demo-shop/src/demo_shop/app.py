"""Demo storefront.

Plays the customer: its backend calls the recommendation engine only through the public
predict / data APIs via the Python SDK (`recsys_client`), with signing, timeouts, fallback
lists, request ids and SPM positions exactly as a real integration would. The browser talks
to this backend, never to the engine. Engine debug helpers (`/demo/*`) are used only for the
"explain" overlay and the demo reset.

    demo-shop --engine http://127.0.0.1:8000 --lake lake --port 8100
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from recsys_client import RecsysClient

from .store import ShopData

log = logging.getLogger("demo-shop")
WEB = Path(__file__).parent / "web"

# The shop's own page codes, agreed with the engine team (configs/engine.toml binds them to slots).
APP = "demo"
SPM = {
    "home_feed": "demo$##$home$##$feed",
    "item_related": "demo$##$item$##$related",
    "cart": "demo$##$cart$##$bundle",
    "reorder": "demo$##$orders$##$reorder",
}
SPM_CATEGORY = "demo$##$category$##$list"
SPM_CHECKOUT = "demo$##$cart$##$checkout$##$0"
# Business facts the shop knows itself (they also live in the engine config).
MARKETS = {
    "CN-HAINAN": {"name": {"zh": "海南离岛免税", "en": "Hainan Duty Free"}, "lang": "zh", "allowance": 100000.0},
    "NZ-AKL": {"name": {"zh": "奥克兰机场免税", "en": "Auckland Airport Duty Free"}, "lang": "en", "allowance": 0.0},
}


class CartChange(BaseModel):
    uid: str
    market: str
    item_id: str
    qty: int = 1  # +n to add, -n to remove, 0 to drop the line
    rid: str | None = None
    spm: str | None = None


class Checkout(BaseModel):
    uid: str
    market: str


class Impressions(BaseModel):
    uid: str
    market: str
    request_id: str
    items: list[dict[str, str]]  # {id, spm}


def _decode(trans: str) -> dict[str, Any]:
    try:
        return json.loads(base64.urlsafe_b64decode(trans + "=" * (-len(trans) % 4)))
    except Exception:
        return {}


def create_app(engine_url: str, lake: Path, http: httpx.Client | None = None, probe: bool = True) -> FastAPI:
    """`http`: inject the HTTP client used to reach the engine (tests); `probe`: look for the
    restock persona in the background."""
    data = ShopData(lake)
    token = os.environ.get("RECSYS_TENANT_TOKEN")
    rec = RecsysClient(engine_url, os.environ.get("RECSYS_APP_ID", APP), token=token, timeout=1.0, retries=1, http=http)
    debug = http or httpx.Client(base_url=engine_url, timeout=5.0)
    state: dict[str, Any] = {"restock": None}
    app = FastAPI(title="demo shop")

    # ---------------------------------------------------------------- engine calls
    def recommend(slot: str, uid: str, market: str, lang: str, exp: str | None = None, **kw: Any) -> dict[str, Any]:
        t0 = time.perf_counter()
        res = rec.predict(
            uid,
            SPM[slot] + "$##$0",
            market=market,
            experiment={"rank_layer": exp} if exp else None,
            fallback_ids=data.bestsellers.get(market, [])[:20],  # the SDK serves these if the engine is down
            **kw,
        )
        ms = (time.perf_counter() - t0) * 1000
        items: list[dict[str, Any]] = []
        exps: dict[str, str] = {}
        for it in res.items:
            card = data.card(it.id, market, lang)
            if card is None or card["price"] is None:
                continue
            meta = _decode(it.trans_data)
            exps = meta.get("exp", exps)
            card.update(rank=it.rank, score=it.score, sources=meta.get("src", []), spm=f"{SPM[slot]}$##${it.rank - 1}")
            items.append(card)
        return {
            "slot": slot,
            "request_id": res.request_id,
            "items": items,
            "ms": round(ms, 1),
            "degraded": res.degraded,
            "fallback": res.fallback,
            "error": res.error,
            "experiments": exps,
        }

    def track(uid: str, market: str, rows: list[dict[str, Any]]) -> None:
        rows = [{"user_id": uid, "market": market, **r} for r in rows]
        try:
            r = rec.write("behavior", rows)
            if r.errors:
                log.warning("tracking rejected: %s", r.errors[:3])
        except Exception as e:  # tracking must never break the page
            log.warning("tracking failed: %r", e)

    def probe_restock() -> None:
        """Find a shopper with consumables due again, then reset so probing leaves no trace."""
        for _ in range(120):
            try:
                if debug.get("/healthz").status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        best: tuple[int, str] | None = None
        for uid in data.restock_candidates():
            market = data.profile(uid)["market"]
            try:
                res = rec.predict(uid, SPM["reorder"] + "$##$0", market=market, timeout=2.0)
            except Exception:
                continue
            n = sum(1 for it in res.items if "repurchase" in _decode(it.trans_data).get("src", []))
            if n and (best is None or n > best[0]):
                best = (n, uid)
        if best:
            p = data.profile(best[1])
            state["restock"] = {
                "id": best[1],
                "market": p["market"],
                "language": p["language"],
                "label": {"zh": "该补货了", "en": "Time to restock"},
                "hint": {
                    "zh": "常买的消耗品快用完了：订单页会提醒补货",
                    "en": "Consumables bought weeks ago are due again: see the orders page",
                },
            }
        with contextlib.suppress(httpx.HTTPError):
            debug.post("/demo/reset")
        log.info("restock persona: %s", best)

    if probe:
        threading.Thread(target=probe_restock, daemon=True).start()

    # ---------------------------------------------------------------- pages
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB / "index.html")

    @app.get("/api/bootstrap")
    def bootstrap() -> dict[str, Any]:
        personas = data.personas()
        if state["restock"]:
            personas.append(state["restock"])
        try:
            meta = debug.get("/demo/meta").json()
        except Exception:
            meta = {}
        return {
            "markets": {m: {**v, "currency": data.currency.get(m)} for m, v in MARKETS.items()},
            "categories": data.categories,
            "personas": personas,
            "experiments": meta.get("experiments", {}),
            "engine": engine_url,
        }

    @app.get("/api/home")
    def home(uid: str, market: str, lang: str = "zh", exp: str | None = None) -> dict[str, Any]:
        return recommend("home_feed", uid, market, lang, exp)

    @app.get("/api/compare")
    def compare(uid: str, market: str, lang: str = "zh", exp: str | None = None) -> dict[str, Any]:
        best = [data.card(i, market, lang) for i in data.bestsellers.get(market, [])[:20]]
        return {"incumbent": [c for c in best if c], "engine": recommend("home_feed", uid, market, lang, exp)}

    @app.get("/api/category/{cat}")
    def category(cat: str, market: str, lang: str = "zh") -> dict[str, Any]:
        ids = data.by_category.get(market, {}).get(cat)
        if ids is None:
            raise HTTPException(404, "unknown category")
        cards = [data.card(i, market, lang) for i in ids[:48]]
        for n, c in enumerate(cards):
            if c:
                c["spm"] = f"{SPM_CATEGORY}$##${n}"
        return {"items": [c for c in cards if c]}

    @app.get("/api/item/{item_id}")
    def item(
        item_id: str, uid: str, market: str, lang: str = "zh", rid: str | None = None, spm: str | None = None
    ) -> dict[str, Any]:
        card = data.card(item_id, market, lang)
        if card is None:
            raise HTTPException(404, "unknown item")
        # opening a product page = a click on whatever list it came from
        ev: dict[str, Any] = {"item_id": item_id, "event_type": "click"}
        if rid:
            ev["request_id"] = rid
        if spm:
            ev["spm"] = spm
        track(uid, market, [ev])
        related = recommend("item_related", uid, market, lang, parent_item_id=item_id)
        return {"item": card, "related": related}

    @app.post("/api/impressions")
    def impressions(body: Impressions) -> dict[str, Any]:
        rows = [
            {"item_id": x["id"], "event_type": "impression", "request_id": body.request_id, "spm": x.get("spm")}
            for x in body.items
        ]
        track(body.uid, body.market, rows)
        return {"ok": True}

    def cart_view(uid: str, market: str, lang: str, exp: str | None = None) -> dict[str, Any]:
        s = data.session(uid)
        lines = []
        for i, q in s.cart.items():
            c = data.card(i, market, lang)
            if c and c["price"] is not None:
                lines.append({**c, "qty": q, "line_total": c["price"] * q})
        total = sum(x["line_total"] for x in lines)
        cap = float(MARKETS.get(market, {}).get("allowance", 0.0))  # type: ignore[arg-type]
        out: dict[str, Any] = {"lines": lines, "total": total, "allowance": None}
        kw: dict[str, Any] = {"cart_ids": [x["id"] for x in lines]}
        if cap:
            used = data.spent(uid, market)
            left = max(cap - used - total, 0.0)
            out["allowance"] = {"cap": cap, "used": used, "in_cart": total, "left": left}
            kw["allowance_remaining"] = left
        out["bundle"] = recommend("cart", uid, market, lang, exp, **kw)
        return out

    @app.get("/api/cart")
    def cart(uid: str, market: str, lang: str = "zh") -> dict[str, Any]:
        return cart_view(uid, market, lang)

    @app.post("/api/cart")
    def cart_change(body: CartChange, lang: str = "zh") -> dict[str, Any]:
        if body.item_id not in data.items:
            raise HTTPException(404, "unknown item")
        s = data.session(body.uid)
        q = s.cart.get(body.item_id, 0)
        q = 0 if body.qty == 0 else max(q + body.qty, 0)
        if q:
            s.cart[body.item_id] = q
        else:
            s.cart.pop(body.item_id, None)
        if body.qty > 0:
            ev: dict[str, Any] = {"item_id": body.item_id, "event_type": "add_cart"}
            if body.rid:
                ev["request_id"] = body.rid
            if body.spm:
                ev["spm"] = body.spm
            track(body.uid, body.market, [ev])
        return {"count": sum(s.cart.values())}

    @app.post("/api/checkout")
    def checkout(body: Checkout) -> dict[str, Any]:
        o = data.place_order(body.uid, body.market)
        if o is None:
            raise HTTPException(400, "cart is empty")
        # no request id at checkout: the engine attributes purchases to the last click (<= 72 h)
        rows: list[dict[str, Any]] = [
            dict(item_id=x["item_id"], event_type="purchase", spm=SPM_CHECKOUT, paid_amount=x["price"] * x["qty"])
            for x in o.lines
        ]
        track(body.uid, body.market, rows)
        return {"order": o.id, "total": o.total}

    @app.get("/api/orders")
    def orders(uid: str, market: str, lang: str = "zh") -> dict[str, Any]:
        out = []
        for o in data.orders(uid)[:12]:
            lines = [
                {**(data.card(x["item_id"], o.market, lang) or {}), "qty": x["qty"], "paid": x["price"]}
                for x in o.lines
            ]
            out.append({"id": o.id, "ts": o.ts, "market": o.market, "total": o.total, "live": o.live, "lines": lines,
                        "currency": data.currency.get(o.market)})  # fmt: skip
        return {"now": data.now(), "orders": out, "reorder": recommend("reorder", uid, market, lang)}

    @app.get("/api/meta")
    def meta(uid: str) -> dict[str, Any]:
        s = data.sessions.get(uid)
        return {"cart_count": sum(s.cart.values()) if s else 0}

    # ---------------------------------------------------------------- engine debug helpers (explain overlay)
    @app.get("/api/profile")
    def profile(uid: str, market: str, lang: str = "zh") -> dict[str, Any]:
        try:
            r = debug.get(f"/demo/profile/{uid}", params={"market": market, "lang": lang})
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            raise HTTPException(502, f"engine: {e}") from None

    @app.post("/api/reset")
    def reset() -> dict[str, Any]:
        try:
            r = debug.post("/demo/reset")
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(502, f"engine reset failed: {e} (start the engine with RECSYS_DEMO=1)") from None
        data.reset()
        return {"ok": True}

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="demo storefront for the recommendation engine")
    ap.add_argument("--engine", default=os.environ.get("RECSYS_ENGINE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--lake", default="lake", help="where the shop reads its own catalog / users / orders")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8100)
    a = ap.parse_args()
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    uvicorn.run(create_app(a.engine, Path(a.lake)), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
