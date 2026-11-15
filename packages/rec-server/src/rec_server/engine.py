"""Online orchestrator: SPM -> slot -> (A/B) -> recall -> filter -> rank -> re-rank."""

from __future__ import annotations

import base64
import copy
import inspect
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import abtest
from .bus import Bus, NullBus
from .data import Catalog, Clock, ItemStats, UserStore
from .features import build_features, build_user_ctx
from .models import Models
from .rank import Ranker
from .recall import RECALLS, RecallInput, _from_list, merge
from .rules import FILTERS, RuleCtx, boost_new, disperse, pool_mask

log = logging.getLogger("recsys.engine")


class SlotNotFound(Exception):
    pass


@dataclass
class Request:
    user_id: str
    spm: str
    market: str | None = None
    size: int | None = None
    parent_id: str | None = None
    candidate_ids: list[str] = field(default_factory=list)
    filter_ids: list[str] = field(default_factory=list)
    cart_ids: list[str] = field(default_factory=list)
    filter_spec: dict = field(default_factory=dict)
    allowance_remaining: float | None = None
    force_variants: dict[str, str] | None = None


@dataclass
class Recommendation:
    request_id: str
    slot: str
    market: str
    items: list[dict]  # {idx, id, score, sources}
    trace: dict


class Engine:
    def __init__(
        self,
        config: dict,
        catalog: Catalog,
        users: UserStore,
        stats: ItemStats,
        models: Models,
        ranker: Ranker | None,
        clock: Clock,
        info: dict | None = None,
    ):
        self.config, self.catalog, self.users, self.stats = config, catalog, users, stats
        self.models, self.ranker, self.clock = models, ranker, clock
        self.info = info or {}
        self.market_codes = {m: i for i, m in enumerate(sorted(config["markets"]))}
        self.bus: Bus = NullBus()
        self._item_ver, self._item_sync_at = 0, 0.0
        self.validate_config(config)

    def __getstate__(self) -> dict:
        d = self.__dict__.copy()
        d["bus"] = None
        return d

    def __setstate__(self, d: dict) -> None:
        self.__dict__.update(d)
        self.bus = NullBus()
        self._item_ver, self._item_sync_at = 0, 0.0

    # ------------------------------------------------------------- config hot reload
    def reload_config(self, config: dict) -> None:
        """Swap in a new (already schema-validated) config; restart-only sections are kept."""
        from recsys_core.config import RESTART_SECTIONS

        self.validate_config(config)
        if set(config["markets"]) != set(self.config["markets"]):
            raise ValueError("adding/removing markets needs a retrain + restart (ranker market codes)")
        self.config = {**config, **{k: self.config[k] for k in RESTART_SECTIONS if k in self.config}}

    def catalog_warnings(self, config: dict) -> list[str]:
        """Names in item pools / compliance blocks that do not exist in the current catalog."""
        cat, out = self.catalog, []
        known = {"categories": set(cat.cat_names), "brands": set(cat.brand_names), "items": set(cat.id2idx)}
        refs = []
        for m, mc in config["markets"].items():
            refs += [(f"markets.{m}.blocked_categories", "categories", mc.get("blocked_categories", []))]
            refs += [(f"markets.{m}.blocked_items", "items", mc.get("blocked_items", []))]
        for s, sc in config["slots"].items():
            p = sc.get("pool") or {}
            for key, kind in (
                ("include_categories", "categories"),
                ("exclude_categories", "categories"),
                ("include_brands", "brands"),
                ("exclude_brands", "brands"),
                ("exclude_items", "items"),
            ):
                refs.append((f"slots.{s}.pool.{key}", kind, p.get(key, [])))
        for where, kind, names in refs:
            unknown = [n for n in names if n not in known[kind]]
            if unknown:
                out.append(f"{where}: unknown {kind} {unknown[:5]}")
        return out

    # ------------------------------------------------------------- item price/stock changes
    def publish_item_changes(self, changes: list[tuple[str, str, float | None, int | None]]) -> None:
        recv = time.time()
        self.users.live.push_item_changes([(*c, recv) for c in changes])
        self.sync_items(force=True)

    def sync_items(self, force: bool = False, every: float = 0.5) -> int:
        """Apply price/stock changes other workers received (shared log in live state).
        Changes received before this snapshot was built are already in it."""
        t = time.monotonic()
        if not force and t - self._item_sync_at < every:
            return 0
        self._item_sync_at = t
        self._item_ver, changes = self.users.live.item_changes_since(self._item_ver)
        n = 0
        for market, iid, price, stock, recv in changes:
            if recv >= self.users.built_at:
                n += self.catalog.update_market(iid, market, price, stock)
        return n

    @staticmethod
    def validate_config(config: dict) -> None:
        """Names in config must exist in the recall / filter registries."""
        errs = []
        for name, slot in config["slots"].items():
            for r in slot["recalls"]:
                if r["name"] not in RECALLS:
                    errs.append(f"slot {name}: unknown recall {r['name']!r}")
                    continue
                allowed = list(inspect.signature(RECALLS[r["name"]]).parameters)[4:]
                bad = set(r.get("params") or {}) - set(allowed)
                if bad:
                    errs.append(f"slot {name}: recall {r['name']} has no params {sorted(bad)} (allowed: {allowed})")
            errs += [f"slot {name}: unknown filter {f!r}" for f in slot.get("filters", []) if f not in FILTERS]
        if errs:
            raise ValueError("; ".join(errs))

    # ------------------------------------------------------------- routing
    def route(self, spm: str) -> str:
        prefix = "$##$".join(spm.split("$##$")[:3])
        for name, slot in self.config["slots"].items():
            if slot["spm"] == prefix:
                return name
        raise SlotNotFound(spm)

    def record_issued(self, user_id: str, idxs: list[int], now: float) -> None:
        self.users.live.record_issued(user_id, [str(self.catalog.item_ids[i]) for i in idxs], now)

    def _issued_set(self, user_id: str, now: float) -> set[int]:
        ids = self.users.live.issued(user_id, now)
        return {i for i in (self.catalog.idx(x) for x in ids) if i is not None}

    # ------------------------------------------------------------- main
    def recommend(self, req: Request, overrides: dict | None = None, record: bool = True) -> Recommendation:
        t_start = t0 = time.perf_counter()
        timings: dict[str, float] = {}
        degraded: list[str] = []
        srv = self.serving

        def elapsed() -> float:
            return (time.perf_counter() - t_start) * 1000

        def lap(name: str) -> None:
            nonlocal t0
            t1 = time.perf_counter()
            timings[name] = round((t1 - t0) * 1000, 2)
            t0 = t1

        slot_name = self.route(req.spm)
        try:
            self.sync_items()
        except Exception:
            log.exception("item change sync failed")
            degraded.append("item_sync")
        slot = copy.deepcopy(self.config["slots"][slot_name])
        assignments = abtest.assign(self.config.get("experiments", {}), req.user_id, slot_name, req.force_variants)
        for a in assignments:
            slot.update(a.params)
        if overrides:
            slot.update(overrides)
        now = self.clock.now()
        cat = self.catalog
        ctx = build_user_ctx(
            self.users, cat, self.models, req.user_id, req.market, now, self.config["app"]["default_market"]
        )
        mcfg = self.config["markets"].get(ctx.market, {})
        avail = cat.available(ctx.market, now)
        parent = cat.idx(req.parent_id) if req.parent_id else None
        cart = [i for i in (cat.idx(x) for x in req.cart_ids) if i is not None]
        lap("context")

        # ---- recall
        trace: dict = {
            "slot": slot_name,
            "market": ctx.market,
            "experiments": {a.layer: a.variant for a in assignments},
            "ranker": slot.get("ranker"),
            "recall": {},
            "filters": [],
        }
        if req.candidate_ids:
            cand = np.array([i for i in (cat.idx(x) for x in req.candidate_ids) if i is not None], dtype=int)
            base = np.linspace(1, 0.5, len(cand)) if len(cand) else np.zeros(0)
            sources = {int(i): ["candidate"] for i in cand}
        else:
            rin = RecallInput(ctx=ctx, avail=avail, parent=parent, cart=cart)
            results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            weights: dict[str, float] = {}
            for rc in slot["recalls"]:
                name = rc["name"]
                if results and elapsed() > srv["recall_budget_ms"]:
                    degraded.append(f"recall_budget:{name}")
                    continue
                try:
                    results[name] = RECALLS[name](self.models, cat, rin, rc["quota"], **(rc.get("params") or {}))
                except Exception:
                    log.exception("recall %s failed", name)
                    degraded.append(f"recall_error:{name}")
                    continue
                weights[name] = rc.get("weight", 1.0)
                trace["recall"][name] = len(results[name][0])
            cand, base, sources = merge(results, weights)
            if len(cand) == 0:  # cold fallback
                cand, base = RECALLS["hot"](self.models, cat, rin, 100)
                sources = {int(i): ["fallback_hot"] for i in cand}
        trace["merged"] = len(cand)
        lap("recall")

        # ---- filter
        rctx = RuleCtx(
            ctx=ctx,
            catalog=cat,
            market_cfg=mcfg,
            avail=avail,
            parent=parent,
            cart=cart,
            filter_items={i for i in (cat.idx(x) for x in req.filter_ids) if i is not None},
            filter_spec=req.filter_spec,
            issued=self._issued_set(req.user_id, now) if record else set(),
            allowance_remaining=req.allowance_remaining,
            repurchase_days=getattr(self.models, "repurchase_days", np.zeros(0)),
            pool={k: v for k, v in (slot.get("pool") or {}).items() if v not in (None, [])},
        )
        for fname in slot.get("filters", []):
            if len(cand) == 0:
                break
            keep = FILTERS[fname](cand, rctx)
            trace["filters"].append({"name": fname, "removed": int((~keep).sum()), "left": int(keep.sum())})
            cand, base = cand[keep], base[keep]
        lap("filter")

        # ---- rank
        if len(cand) > srv["rank_max_candidates"]:  # merged recall is sorted by fused score
            cand, base = cand[: srv["rank_max_candidates"]], base[: srv["rank_max_candidates"]]
        score = base.astype(float)
        if slot.get("ranker") == "gbdt" and self.ranker is not None and len(cand):
            if elapsed() > srv["timeout_ms"] * 0.7:
                degraded.append("rank_budget")
            else:
                try:
                    X = build_features(ctx, cand, cat, self.stats, self.models, self.market_codes)
                    score = self.ranker.predict(X)
                except Exception:
                    log.exception("ranker failed")
                    degraded.append("rank_error")
        lap("rank")

        # ---- re-rank
        bn = slot.get("boost_new")
        if bn and len(cand):
            score = boost_new(cand, score, cat, now, bn["days"], bn["weight"])
        order = cand[np.argsort(-score, kind="stable")]
        score_of = dict(zip(cand.tolist(), score.tolist()))
        size = req.size or slot.get("size", 20)
        final = disperse(order, cat, slot.get("dispersion", []), size)
        lap("rerank")

        rid = uuid.uuid4().hex[:16]
        items = [
            dict(idx=int(i), id=str(cat.item_ids[i]), score=float(score_of[int(i)]), sources=sources.get(int(i), []))
            for i in final
        ]
        trace["returned"] = len(items)
        trace["timings_ms"] = timings
        trace["total_ms"] = round(sum(timings.values()), 2)
        trace["degraded"] = degraded
        rec = Recommendation(rid, slot_name, ctx.market, items, trace)
        if record:
            self.record_issued(req.user_id, [int(i) for i in final], now)
            self.log_request(req, rec, now)
        return rec

    # ------------------------------------------------------------- safety net
    @property
    def serving(self) -> dict:
        return {
            "timeout_ms": 150.0,
            "recall_budget_ms": 60.0,
            "rank_max_candidates": 500,
            "fallback_size": 20,
            **(self.config.get("serving") or {}),
        }

    def recommend_safe(self, req: Request, overrides: dict | None = None, record: bool = True) -> Recommendation:
        """recommend(); on any unexpected error serve the market's hot list instead of failing."""
        try:
            return self.recommend(req, overrides, record)
        except SlotNotFound:
            raise
        except Exception as e:
            log.exception("recommend failed, serving fallback")
            return self.fallback(req, f"error:{type(e).__name__}")

    def fallback(self, req: Request, reason: str) -> Recommendation:
        slot_name = self.route(req.spm)
        prof = self.users.profiles.get(req.user_id)
        market = req.market or (prof.market if prof else None) or self.config["app"]["default_market"]
        now = self.clock.now()
        mcfg = self.config["markets"].get(market, {})
        avail = self.catalog.available(market, now) & pool_mask(self.catalog, market, mcfg, None)  # compliance
        size = req.size or self.config["slots"][slot_name].get("size") or self.serving["fallback_size"]
        ix, sc = _from_list(self.models.hot.get(market, np.zeros(0, int)), avail, size)
        items = [
            dict(idx=int(i), id=str(self.catalog.item_ids[i]), score=float(s), sources=["fallback_hot"])
            for i, s in zip(ix, sc)
        ]
        trace: dict[str, Any] = {
            "slot": slot_name,
            "market": market,
            "experiments": {},
            "degraded": [reason],
            "total_ms": 0.0,
        }
        trace["timings_ms"] = {}
        rec = Recommendation(uuid.uuid4().hex[:16], slot_name, market, items, trace)
        self.log_request(req, rec, now)
        return rec

    def log_request(self, req: Request, rec: Recommendation, now: float) -> None:
        try:
            self.bus.publish(
                "rec_log",
                [
                    {
                        "request_id": rec.request_id,
                        "user_id": req.user_id,
                        "ts": int(now),
                        "slot": rec.slot,
                        "market": rec.market,
                        "variants": ",".join(f"{k}={v}" for k, v in rec.trace.get("experiments", {}).items()),
                        "items": "|".join(it["id"] for it in rec.items),
                        "degraded": ",".join(rec.trace.get("degraded", [])),
                    }
                ],
            )
        except Exception:
            log.exception("request log publish failed")


def trans_data(rec: Recommendation, item: dict) -> str:
    payload = {"rid": rec.request_id, "slot": rec.slot, "src": item["sources"], "exp": rec.trace["experiments"]}
    return base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()
