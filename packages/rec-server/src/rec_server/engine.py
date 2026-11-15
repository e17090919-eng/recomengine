"""Online orchestrator: SPM -> slot -> (A/B) -> recall -> filter -> rank -> re-rank."""

from __future__ import annotations

import base64
import copy
import json
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

from . import abtest
from .data import Catalog, Clock, ItemStats, UserStore
from .features import build_features, build_user_ctx
from .models import Models
from .rank import Ranker
from .recall import RECALLS, RecallInput, merge
from .rules import FILTERS, RuleCtx, boost_new, disperse

ISSUED_WINDOW = 30 * 60  # cross-request dedup window (s)
ISSUED_CAP = 200


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
        self.issued: dict[str, deque] = defaultdict(lambda: deque(maxlen=ISSUED_CAP))
        self.validate_config(config)

    @staticmethod
    def validate_config(config: dict) -> None:
        """Names in config must exist in the recall / filter registries."""
        errs = []
        for name, slot in config["slots"].items():
            errs += [f"slot {name}: unknown recall {r['name']!r}" for r in slot["recalls"] if r["name"] not in RECALLS]
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
        dq = self.issued[user_id]
        for i in idxs:
            dq.append((now, i))

    def _issued_set(self, user_id: str, now: float) -> set[int]:
        return {i for ts, i in self.issued.get(user_id, ()) if now - ts <= ISSUED_WINDOW}

    # ------------------------------------------------------------- main
    def recommend(self, req: Request, overrides: dict | None = None, record: bool = True) -> Recommendation:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}

        def lap(name: str) -> None:
            nonlocal t0
            t1 = time.perf_counter()
            timings[name] = round((t1 - t0) * 1000, 2)
            t0 = t1

        slot_name = self.route(req.spm)
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
            results, weights = {}, {}
            for rc in slot["recalls"]:
                fn = RECALLS[rc["name"]]
                results[rc["name"]] = fn(self.models, cat, rin, rc["quota"])
                weights[rc["name"]] = rc.get("weight", 1.0)
                trace["recall"][rc["name"]] = len(results[rc["name"]][0])
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
        )
        for fname in slot.get("filters", []):
            if len(cand) == 0:
                break
            keep = FILTERS[fname](cand, rctx)
            trace["filters"].append({"name": fname, "removed": int((~keep).sum()), "left": int(keep.sum())})
            cand, base = cand[keep], base[keep]
        lap("filter")

        # ---- rank
        if slot.get("ranker") == "gbdt" and self.ranker is not None and len(cand):
            X = build_features(ctx, cand, cat, self.stats, self.models, self.market_codes)
            score = self.ranker.predict(X)
        else:
            score = base.astype(float)
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
        if record:
            self.record_issued(req.user_id, [int(i) for i in final], now)
        trace["returned"] = len(items)
        trace["timings_ms"] = timings
        trace["total_ms"] = round(sum(timings.values()), 2)
        return Recommendation(rid, slot_name, ctx.market, items, trace)


def trans_data(rec: Recommendation, item: dict) -> str:
    payload = {"rid": rec.request_id, "slot": rec.slot, "src": item["sources"], "exp": rec.trace["experiments"]}
    return base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()
