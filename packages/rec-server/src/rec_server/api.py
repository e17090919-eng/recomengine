"""HTTP API.

/predict/api/{app_id}/default   recommendation (request/response shape compatible with the
                                Volcengine predict API, so clients can switch either way)
/predict/api/{app_id}/callback  report what was actually shown (dedup / attribution)
/data/api/{app_id}/{table}/write  real-time ingestion (behavior / item price+stock / user profile)
/data/api/{app_id}/done         "data for <table, date> is complete" marker for the daily batch
/demo/*                         helpers for the demo UI (enriched items + pipeline trace)
"""

from __future__ import annotations

import inspect
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi import Request as HTTPRequest
from fastapi.responses import FileResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from recsys_core.config import RESTART_SECTIONS, hot_sections, load_config
from recsys_core.signing import verify

from .admin import ConfigError, ConfigStore, ConfigWatcher, check, diff
from .bus import Bus, make_bus
from .engine import Engine, Recommendation, Request, SlotNotFound, trans_data
from .ingest import Ingestor
from .pipeline import load_engine
from .recall import RECALLS
from .rules import FILTERS

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
DEGRADED = Counter("rec_degraded_total", "Requests served with a degraded pipeline", ["slot", "reason"])


def observe(rec: Recommendation) -> None:
    LATENCY.labels(rec.slot).observe(rec.trace["total_ms"] / 1000)
    for stage, ms in rec.trace["timings_ms"].items():
        STAGE.labels(stage).observe(ms / 1000)
    if not rec.items:
        EMPTY.labels(rec.slot).inc()
    for d in rec.trace.get("degraded", []):
        DEGRADED.labels(rec.slot, d.split(":")[0]).inc()


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


class DraftRequest(BaseModel):
    toml: str | None = None  # hot sections as TOML text ...
    config: dict[str, Any] | None = None  # ... or as JSON (structured editor)
    comment: str = ""
    author: str = "console"


class PreviewRequest(DraftRequest):
    user_id: str
    scene: str = "home_feed"
    market: str | None = None
    lang: str = "en"
    parent_item_id: str | None = None
    size: int | None = None


class RollbackRequest(BaseModel):
    version: int
    author: str = "console"


class DoneRequest(BaseModel):
    table: str = "behavior"
    date: str = Field(pattern=r"^\d{8}$")  # YYYYMMDD, customer-local business date


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


def create_app(
    artifact: Path | str,
    config_path: str | None = None,
    engine: Engine | None = None,
    bus: Bus | None = None,
    admin_store: str | None = None,
) -> FastAPI:
    """`engine` given (tests, simulations): the console is off unless `admin_store` is given."""
    cfg = load_config(config_path)
    eng = engine or load_engine(Path(artifact), cfg, live=True)
    if bus is not None:
        eng.bus = bus
    elif engine is None:
        eng.bus = make_bus(cfg)
    ingestor = Ingestor(eng)
    watcher: ConfigWatcher | None = None
    store: ConfigStore | None = None
    if cfg.get("admin", {}).get("enabled") and (engine is None or admin_store):
        store = ConfigStore(admin_store or cfg["admin"]["store"], cfg["app"]["app_id"])
        watcher = ConfigWatcher(store, eng, cfg["admin"]["poll_seconds"])
        watcher.bootstrap(cfg)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        eng.bus.close()

    app = FastAPI(title="recsys", version="0.2.0", lifespan=lifespan)
    nonces: dict[str, float] = {}
    app.state.engine = eng
    app.state.watcher = watcher

    def resp(code: int, message: str, rid: str = "", items: list | None = None) -> dict:
        return {
            "code": code,
            "message": message,
            "success": code == 200,
            "request_id": rid,
            "value": {"items": items or []},
        }

    async def check(http: HTTPRequest, app_id: str) -> tuple[int, str] | None:
        """Tenant signature + app id; returns (code, message) on failure."""
        h = http.headers
        try:
            verify_signature(
                cfg,
                await http.body(),
                h.get("Tenant-Id"),
                h.get("Tenant-Ts"),
                h.get("Tenant-Nonce"),
                h.get("Tenant-Signature"),
                nonces,
            )
        except HTTPException as e:
            return 401, str(e.detail)
        if app_id != cfg["app"]["app_id"]:
            return 404, f"unknown app_id {app_id}"
        return None

    @app.get("/metrics")
    def metrics() -> Response:
        if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):  # several workers: aggregate all of them
            from prometheus_client import CollectorRegistry, multiprocess

            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry)
            return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/healthz")
    def healthz() -> dict:
        live_ok = eng.users.live.ping()
        return {
            "ok": live_ok,
            "items": eng.catalog.n,
            "snapshot_users": len(eng.users.profiles),
            "live_users": eng.users.live.n_users() if live_ok else None,
            "state": type(eng.users.live).__name__,
            "bus": type(eng.bus).__name__,
            "snapshot_cutoff_ts": eng.info.get("cutoff_ts"),
        }

    @app.post("/predict/api/{app_id}/default")
    async def predict(app_id: str, body: PredictRequest, http: HTTPRequest) -> dict:
        if err := await check(http, app_id):
            REQUESTS.labels("-", str(err[0])).inc()
            return resp(*err)
        if watcher is not None:
            watcher.maybe_reload()
        try:
            rec = eng.recommend_safe(_to_request(body))
        except SlotNotFound:
            REQUESTS.labels("-", "404").inc()
            return resp(404, f"spm {body.context.spm} is not bound to a slot")
        REQUESTS.labels(rec.slot, "200").inc()
        observe(rec)
        items = [
            {"id": it["id"], "rank": n + 1, "transData": trans_data(rec, it), "extra": {"score": f"{it['score']:.6f}"}}
            for n, it in enumerate(rec.items)
        ]
        out = resp(200, "ok", rec.request_id, items)
        if rec.trace.get("degraded"):
            out["degraded"] = rec.trace["degraded"]
        return out

    @app.post("/predict/api/{app_id}/callback")
    async def callback(app_id: str, body: CallbackRequest, http: HTTPRequest) -> dict:
        if err := await check(http, app_id):
            return resp(*err)
        idx = [i for i in (eng.catalog.idx(x.id) for x in body.items) if i is not None]
        eng.record_issued(body.uid, idx, eng.clock.now())
        return {"code": 200, "message": "ok", "success": True}

    @app.post("/data/api/{app_id}/{table}/write")
    async def write(app_id: str, table: str, rows: list[dict[str, Any]], http: HTTPRequest) -> dict:
        if err := await check(http, app_id):
            return {"code": err[0], "message": err[1], "success": False, "errors": []}
        res = ingestor.write(table, rows)
        EVENTS.labels(table, "ok").inc(res.accepted)
        EVENTS.labels(table, "error").inc(len(res.errors))
        code = 200 if not res.errors else (206 if res.accepted else 400)
        return {
            "code": code,
            "message": "ok" if code == 200 else f"{len(res.errors)} of {len(rows)} rows rejected",
            "success": not res.errors,
            "accepted": res.accepted,
            "errors": res.errors[:100],
        }

    @app.post("/data/api/{app_id}/done")
    async def done(app_id: str, body: DoneRequest, http: HTTPRequest) -> dict:
        if err := await check(http, app_id):
            return {"code": err[0], "message": err[1], "success": False}
        ingestor.done(body.table, body.date)
        return {"code": 200, "message": "ok", "success": True}

    if store is not None and watcher is not None:
        mount_admin(app, eng, cfg, store, watcher)

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
        rec: Recommendation = eng.recommend_safe(
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


def app_from_env() -> FastAPI:
    """uvicorn factory for multi-worker serving: RECSYS_ARTIFACT, RECSYS_CONFIG."""
    return create_app(os.environ.get("RECSYS_ARTIFACT", "artifacts/engine.pkl"), os.environ.get("RECSYS_CONFIG"))


# ------------------------------------------------------------------ config console
def mount_admin(app: FastAPI, eng: Engine, cfg: dict, store: ConfigStore, watcher: ConfigWatcher) -> None:
    import copy

    from fastapi import Depends
    from fastapi.responses import JSONResponse

    token = cfg["admin"].get("token") or ""

    def guard(http: HTTPRequest) -> None:
        if token and http.headers.get("X-Admin-Token") != token:
            raise HTTPException(401, "admin token required (X-Admin-Token)")

    def draft_of(body: DraftRequest) -> dict | str:
        if body.toml is not None:
            return body.toml
        if body.config is not None:
            return body.config
        raise HTTPException(422, "give `toml` or `config`")

    def bad(e: ConfigError) -> JSONResponse:
        return JSONResponse({"ok": False, "errors": e.errors}, status_code=422)

    def items_of(rec: Recommendation, lang: str) -> list[dict]:
        out = []
        for n, it in enumerate(rec.items):
            d = eng.catalog.describe(it["idx"], rec.market, lang)
            d.update(rank=n + 1, score=it["score"], sources=it["sources"])
            out.append(d)
        return out

    @app.get("/admin")
    def admin_page() -> FileResponse:
        return FileResponse(WEB / "admin.html")

    @app.get("/admin/api/config", dependencies=[Depends(guard)])
    def get_config() -> dict:
        watcher.maybe_reload(force=True)
        v = store.get()
        return {
            "version": v.summary() if v else None,
            "toml": v.toml if v else "",
            "config": hot_sections(eng.config),
            "running_version": watcher.version,
            "reload_error": watcher.last_error,
            "catalog": {
                "categories": eng.catalog.cat_names,
                "brands": eng.catalog.brand_names,
                "markets": list(eng.config["markets"]),
            },
            "registry": {
                "recalls": {k: list(inspect.signature(fn).parameters)[4:] for k, fn in RECALLS.items()},
                "filters": list(FILTERS),
                "rankers": ["gbdt", "recall_score"],
            },
            "restart_sections": {k: _redact(eng.config.get(k)) for k in RESTART_SECTIONS},
        }

    @app.get("/admin/api/versions", dependencies=[Depends(guard)])
    def versions() -> dict:
        return {"versions": [v.summary() for v in store.versions()], "audit": store.audit(20)}

    @app.get("/admin/api/versions/{vid}", dependencies=[Depends(guard)])
    def version(vid: int) -> dict:
        v = store.get(vid)
        if v is None:
            raise HTTPException(404, f"no version {vid}")
        return {"version": v.summary(), "toml": v.toml, "config": v.config}

    @app.post("/admin/api/validate", dependencies=[Depends(guard)], response_model=None)
    def validate(body: DraftRequest) -> dict | JSONResponse:
        try:
            new, _, warnings = check(draft_of(body), eng.config, eng)
        except ConfigError as e:
            return bad(e)
        return {"ok": True, "warnings": warnings, "diff": diff(hot_sections(eng.config), hot_sections(new))}

    @app.post("/admin/api/preview", dependencies=[Depends(guard)], response_model=None)
    def preview(body: PreviewRequest) -> dict | JSONResponse:
        """Same request under the running config and the draft (nothing recorded)."""
        try:
            new, _, warnings = check(draft_of(body), eng.config, eng)
        except ConfigError as e:
            return bad(e)
        if body.scene not in new["slots"]:
            raise HTTPException(422, f"unknown scene {body.scene}")
        draft = copy.copy(eng)
        draft.config = {**new, **{k: eng.config[k] for k in RESTART_SECTIONS if k in eng.config}}
        out: dict[str, Any] = {}
        for name, en in (("current", eng), ("draft", draft)):
            if body.scene not in en.config["slots"]:
                out[name] = {"items": [], "trace": {"error": "scene not in this config"}}
                continue
            req = Request(
                user_id=body.user_id,
                spm=en.config["slots"][body.scene]["spm"] + "$##$0",
                market=body.market,
                size=body.size,
                parent_id=body.parent_item_id,
            )
            rec = en.recommend(req, record=False)
            out[name] = {"items": items_of(rec, body.lang), "trace": rec.trace}
        cur, dr = {i["id"] for i in out["current"]["items"]}, {i["id"] for i in out["draft"]["items"]}
        return {"ok": True, "warnings": warnings, **out, "overlap": len(cur & dr), "size": max(len(cur), len(dr))}

    @app.post("/admin/api/publish", dependencies=[Depends(guard)], response_model=None)
    def publish(body: DraftRequest) -> dict | JSONResponse:
        try:
            new, text, warnings = check(draft_of(body), eng.config, eng)
        except ConfigError as e:
            return bad(e)
        changes = diff(hot_sections(eng.config), hot_sections(new))
        if not changes:
            return JSONResponse({"ok": False, "errors": ["no changes"]}, status_code=422)
        vid = store.publish(text, hot_sections(new), body.comment or f"{len(changes)} change(s)", body.author)
        watcher.maybe_reload(force=True)
        return {"ok": True, "version": vid, "changes": changes, "warnings": warnings}

    @app.post("/admin/api/rollback", dependencies=[Depends(guard)], response_model=None)
    def rollback(body: RollbackRequest) -> dict | JSONResponse:
        try:
            store.activate(body.version, body.author)
        except KeyError:
            raise HTTPException(404, f"no version {body.version}") from None
        watcher.maybe_reload(force=True)
        if watcher.version != body.version:
            return JSONResponse({"ok": False, "errors": [watcher.last_error or "reload failed"]}, status_code=500)
        return {"ok": True, "version": body.version}


def _redact(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: ("***" if k in ("token", "url", "store") and x else _redact(x)) for k, x in v.items()}
    return v
