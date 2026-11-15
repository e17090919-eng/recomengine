"""HTTP API.

/predict/api/{app_id}/default   recommendation (request/response shape compatible with the
                                Volcengine predict API, so clients can switch either way)
/predict/api/{app_id}/callback  report what was actually shown (dedup / attribution)
/data/api/{app_id}/{table}/write  real-time data ingestion (behaviors, item price/stock, users)
/demo/*                         helpers for the demo UI (enriched items + pipeline trace)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi import Request as HTTPRequest
from fastapi.responses import FileResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from recsys_core.config import load_config
from recsys_core.signing import verify

from .data import ENGAGE_EVENTS, Profile
from .engine import Engine, Recommendation, Request, SlotNotFound, trans_data
from .pipeline import load_engine

WEB = Path(__file__).parent / "web"

REQUESTS = Counter("rec_requests_total", "Recommendation requests", ["slot", "code"])
LATENCY = Histogram(
    "rec_latency_seconds",
    "Engine latency per request",
    ["slot"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0),
)
STAGE = Histogram(
    "rec_stage_seconds",
    "Engine latency per stage",
    ["stage"],
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1),
)
EMPTY = Counter("rec_empty_results_total", "Requests that returned no items", ["slot"])
EVENTS = Counter("ingest_events_total", "Rows written via the data API", ["table", "status"])


def observe(rec: Recommendation) -> None:
    LATENCY.labels(rec.slot).observe(rec.trace["total_ms"] / 1000)
    for stage, ms in rec.trace["timings_ms"].items():
        STAGE.labels(stage).observe(ms / 1000)
    if not rec.items:
        EMPTY.labels(rec.slot).inc()


# ------------------------------------------------------------------ schemas
class Device(BaseModel):
    deviceId: str | None = None
    platform: str | None = None


class PUser(BaseModel):
    uid: str
    device: Device | None = None
    age: str | None = None
    gender: str | None = None
    country: str | None = None
    province: str | None = None
    city: str | None = None


class Typed(BaseModel):
    stringFeature: dict[str, str] = Field(default_factory=dict)
    intFeature: dict[str, int] = Field(default_factory=dict)
    floatFeature: dict[str, float] = Field(default_factory=dict)
    doubleFeature: dict[str, float] = Field(default_factory=dict)
    stringArrayFeature: dict[str, dict[str, list[str]]] = Field(default_factory=dict)


class Context(BaseModel):
    spm: str
    feature: Typed = Field(default_factory=Typed)
    filter: Typed = Field(default_factory=Typed)
    extra: dict[str, str] = Field(default_factory=dict)


class ItemRef(BaseModel):
    id: str


class PredictRequest(BaseModel):
    user: PUser
    context: Context
    candidateItems: list[ItemRef] = Field(default_factory=list)
    parentItem: ItemRef | None = None
    filterItems: list[ItemRef] = Field(default_factory=list)
    size: int | None = None


class CallbackRequest(BaseModel):
    uid: str
    request_id: str | None = None
    spm: str | None = None
    items: list[ItemRef]


class DemoRecommend(BaseModel):
    user_id: str
    scene: str = "home_feed"
    market: str | None = None
    lang: str = "en"
    parent_item_id: str | None = None
    cart_item_ids: list[str] = Field(default_factory=list)
    size: int | None = None
    variant: str | None = None  # force A/B variant for the demo


# ------------------------------------------------------------------ helpers
def _to_request(r: PredictRequest) -> Request:
    f = r.context.feature
    market = f.stringFeature.get("market")
    cart = f.stringArrayFeature.get("cart_items", {}).get("values", [])
    allowance = f.doubleFeature.get("allowance_remaining", f.floatFeature.get("allowance_remaining"))
    force = {k.removeprefix("exp."): v for k, v in r.context.extra.items() if k.startswith("exp.")} or None
    return Request(
        user_id=r.user.uid,
        spm=r.context.spm,
        market=market,
        size=r.size,
        parent_id=r.parentItem.id if r.parentItem else None,
        candidate_ids=[x.id for x in r.candidateItems],
        filter_ids=[x.id for x in r.filterItems],
        cart_ids=cart,
        filter_spec=r.context.filter.model_dump(),
        allowance_remaining=allowance,
        force_variants=force,
    )


def verify_signature(
    cfg: dict,
    body: bytes,
    tenant_id: str | None,
    ts: str | None,
    nonce: str | None,
    signature: str | None,
    seen: dict[str, float],
) -> None:
    auth = cfg.get("auth", {})
    if not auth.get("enabled"):
        return
    if not all([tenant_id, ts, nonce, signature]) or tenant_id != auth.get("tenant_id"):
        raise HTTPException(401, "missing or invalid tenant headers")
    assert tenant_id and ts and nonce and signature
    if abs(time.time() - float(ts)) > 300:
        raise HTTPException(401, "timestamp expired")
    if nonce in seen:
        raise HTTPException(401, "replayed nonce")
    if not verify(auth["token"], body, tenant_id, ts, nonce, signature):
        raise HTTPException(401, "bad signature")
    seen[nonce] = time.time()


def create_app(artifact: Path | str, config_path: str | None = None, engine: Engine | None = None) -> FastAPI:
    cfg = load_config(config_path)
    eng = engine or load_engine(Path(artifact), cfg, live=True)
    app = FastAPI(title="recsys demo", version="0.1.0")
    nonces: dict[str, float] = {}
    app.state.engine = eng

    def resp(code: int, message: str, rid: str = "", items: list | None = None) -> dict:
        return {
            "code": code,
            "message": message,
            "success": code == 200,
            "request_id": rid,
            "value": {"items": items or []},
        }

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True, "items": eng.catalog.n, "users": len(eng.users.profiles)}

    @app.post("/predict/api/{app_id}/default")
    async def predict(
        app_id: str,
        body: PredictRequest,
        http: HTTPRequest,
        tenant_id: str | None = Header(None, alias="Tenant-Id"),
        tenant_ts: str | None = Header(None, alias="Tenant-Ts"),
        tenant_nonce: str | None = Header(None, alias="Tenant-Nonce"),
        tenant_sig: str | None = Header(None, alias="Tenant-Signature"),
    ) -> dict:
        try:
            verify_signature(cfg, await http.body(), tenant_id, tenant_ts, tenant_nonce, tenant_sig, nonces)
        except HTTPException as e:
            REQUESTS.labels("-", "401").inc()
            return resp(401, e.detail)
        if app_id != cfg["app"]["app_id"]:
            REQUESTS.labels("-", "404").inc()
            return resp(404, f"unknown app_id {app_id}")
        try:
            rec = eng.recommend(_to_request(body))
        except SlotNotFound:
            REQUESTS.labels("-", "404").inc()
            return resp(404, f"spm {body.context.spm} is not bound to a slot")
        REQUESTS.labels(rec.slot, "200").inc()
        observe(rec)
        items = [
            {"id": it["id"], "rank": n + 1, "transData": trans_data(rec, it), "extra": {"score": f"{it['score']:.6f}"}}
            for n, it in enumerate(rec.items)
        ]
        return resp(200, "ok", rec.request_id, items)

    @app.post("/predict/api/{app_id}/callback")
    def callback(app_id: str, body: CallbackRequest) -> dict:
        idx = [i for i in (eng.catalog.idx(x.id) for x in body.items) if i is not None]
        eng.record_issued(body.uid, idx, eng.clock.now())
        return {"code": 200, "message": "ok", "success": True}

    @app.post("/data/api/{app_id}/{table}/write")
    def write(app_id: str, table: str, rows: list[dict[str, Any]]) -> dict:
        errors = []
        now = eng.clock.now()
        for n, r in enumerate(rows):
            try:
                if table == "behavior":
                    i = eng.catalog.idx(r["item_id"])
                    if i is None:
                        raise ValueError(f"unknown item {r['item_id']}")
                    et = r["event_type"]
                    if et not in ENGAGE_EVENTS | {"impression", "dislike"}:
                        raise ValueError(f"unregistered event_type {et}")
                    eng.users.add_event(
                        r["user_id"], float(r.get("ts") or now), i, et, float(r.get("paid_amount") or 0)
                    )
                elif table == "item":
                    if not eng.catalog.update_market(r["item_id"], r["market"], r.get("price"), r.get("stock")):
                        raise ValueError("unknown item or market")
                elif table == "user":
                    old = eng.users.profiles.get(r["user_id"])
                    base = old.__dict__ if old else {}
                    eng.users.upsert_profile(Profile(**{**base, **r}))
                else:
                    raise ValueError(f"unknown table {table}")
            except (KeyError, ValueError, TypeError) as e:
                errors.append({"index": n, "message": str(e)})
        EVENTS.labels(table, "ok").inc(len(rows) - len(errors))
        EVENTS.labels(table, "error").inc(len(errors))
        return {"code": 200 if not errors else 206, "message": "ok", "success": not errors, "errors": errors}

    # ---------------------------------------------------------------- demo helpers
    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(WEB / "index.html")

    @app.get("/demo/meta")
    def meta() -> dict:
        return {
            "markets": {
                m: {
                    "name": c.get("name", {}),
                    "currency": c.get("currency"),
                    "languages": c.get("languages", []),
                    "allowance": c.get("allowance", 0),
                }
                for m, c in cfg["markets"].items()
            },
            "slots": {k: v["spm"] for k, v in cfg["slots"].items()},
            "experiments": {k: [v["name"] for v in e["variants"]] for k, e in cfg.get("experiments", {}).items()},
            "ranker": eng.info.get("ranker", {}),
        }

    @app.get("/demo/users")
    def users(market: str, limit: int = 30) -> list[dict]:
        out = []
        for uid, p in eng.users.profiles.items():
            if p.market != market:
                continue
            h = eng.users.history(uid)
            if len(h) < 4:
                continue
            out.append(
                dict(user_id=uid, country=p.country, language=p.language, member_level=p.member_level, events=len(h))
            )
            if len(out) >= limit:
                break
        out.append(dict(user_id="guest-" + market.lower(), country="", language="", member_level=0, events=0))
        return out

    @app.get("/demo/profile/{user_id}")
    def profile(user_id: str, market: str, lang: str = "en") -> dict:
        from .features import build_user_ctx

        ctx = build_user_ctx(
            eng.users, eng.catalog, eng.models, user_id, market, eng.clock.now(), cfg["app"]["default_market"]
        )
        cat = eng.catalog
        tops = [(cat.cat_names[c], float(ctx.cat_w[c])) for c in ctx.cat_w.argsort()[::-1][:4] if ctx.cat_w[c] > 0]
        brands = [
            (cat.brand_names[b], float(ctx.brand_w[b])) for b in ctx.brand_w.argsort()[::-1][:4] if ctx.brand_w[b] > 0
        ]
        allowance = cfg["markets"].get(ctx.market, {}).get("allowance", 0)
        return dict(
            user_id=user_id,
            market=ctx.market,
            country=ctx.profile.country,
            language=ctx.profile.language,
            member_level=ctx.profile.member_level,
            categories=[
                dict(name=n, label=cat.cat_meta.get(n, {}).get(lang, n) if lang != "en" else n, weight=w)
                for n, w in tops
            ],
            brands=[dict(name=n, weight=w) for n, w in brands],
            allowance=allowance,
            allowance_remaining=max(allowance - ctx.spend, 0) if allowance else None,
            recent=[cat.describe(int(i), ctx.market, lang) for i in ctx.recent[:8]],
            cart=[cat.describe(int(i), ctx.market, lang) for i in ctx.cart[:8]],
        )

    @app.post("/demo/recommend")
    def demo_recommend(body: DemoRecommend) -> dict:
        spm = cfg["slots"][body.scene]["spm"] + "$##$0"
        force = {"rank_layer": body.variant} if body.variant else None
        rec: Recommendation = eng.recommend(
            Request(
                user_id=body.user_id,
                spm=spm,
                market=body.market,
                size=body.size,
                parent_id=body.parent_item_id,
                cart_ids=body.cart_item_ids,
                force_variants=force,
            )
        )
        items = []
        for n, it in enumerate(rec.items):
            d = eng.catalog.describe(it["idx"], rec.market, body.lang)
            d.update(rank=n + 1, score=it["score"], sources=it["sources"])
            items.append(d)
        return {"request_id": rec.request_id, "now": eng.clock.now(), "items": items, "trace": rec.trace}

    return app
