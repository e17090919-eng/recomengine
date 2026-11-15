"""Online A/B simulation: the trained engine serves simulated shoppers for one business
day; their reactions come from the simulator's hidden preference model. Every event goes
through the production path (Ingestor -> live state -> bus -> lake, request log from the
engine), and the T+1 report is computed from the resulting lake — so this exercises
serving, ingestion, attribution and the A/B readout end to end.

Default arms on home_feed (the incumbent is approximated by "popular items only"):
  incumbent     hot recall only, no model
  multi_recall  all recall channels, fused recall score
  gbdt          all recall channels + GBDT ranker
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from rec_server.bus import LakeBus
from rec_server.data import Clock
from rec_server.engine import Request
from rec_server.ingest import Ingestor
from rec_server.pipeline import load_engine
from recsys_core.lake import SNAPSHOT_TABLES, Lake

DEFAULT_ARMS = [
    {
        "name": "incumbent",
        "traffic": 34,
        "params": {"ranker": "recall_score", "recalls": [{"name": "hot", "quota": 200, "weight": 1.0}]},
    },
    {"name": "multi_recall", "traffic": 33, "params": {"ranker": "recall_score"}},
    {"name": "gbdt", "traffic": 33, "params": {"ranker": "gbdt"}},
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def run_ab_sim(
    cfg: dict,
    artifact: Path,
    base_lake: Path,
    out_lake: Path,
    oracle: Callable[[str, np.ndarray], np.ndarray],
    n_sessions: int = 6000,
    arms: list[dict] | None = None,
    slot: str = "home_feed",
    seed: int = 0,
    log: Callable[..., None] = print,
) -> tuple[str, dict]:
    """Returns (business date YYYYMMDD, config used)."""
    rng = np.random.default_rng(seed)
    cfg = {
        **cfg,
        "experiments": {"ab_sim": {"slots": [slot], "salt": f"ab-sim-{seed}", "variants": arms or DEFAULT_ARMS}},
    }

    # fresh output lake with the same snapshot tables
    if out_lake.exists():
        shutil.rmtree(out_lake)
    out_lake.mkdir(parents=True)
    for table in SNAPSHOT_TABLES:
        src = base_lake / f"{table}.parquet"
        if src.exists():
            shutil.copy(src, out_lake / src.name)
    lake = Lake(out_lake)
    bus = LakeBus(lake, flush_rows=200_000, flush_seconds=3600)
    eng = load_engine(artifact, cfg, live=False, bus=bus)
    ing = Ingestor(eng)

    # the business day after the snapshot, in the default market's time zone
    tz = cfg["markets"][cfg["app"]["default_market"]]["timezone"]
    day = (datetime.fromtimestamp(eng.clock.base_ts, ZoneInfo(tz)) + timedelta(days=1)).date()
    date = day.strftime("%Y%m%d")
    windows = {}
    for m, mc in cfg["markets"].items():
        start = datetime(day.year, day.month, day.day, tzinfo=ZoneInfo(mc["timezone"])).timestamp()
        windows[m] = (start, start + 86400)

    users = [(u, p.market) for u, p in eng.users.profiles.items() if p.market in windows]
    pick = rng.integers(0, len(users), n_sessions)
    times = np.array([rng.uniform(*windows[users[i][1]]) for i in pick])
    order = np.argsort(times)
    spm0 = cfg["slots"][slot]["spm"]
    n_ev = n_rej = 0
    for n, k in enumerate(order):
        uid, market = users[pick[k]]
        t = float(times[k])
        eng.clock = Clock(t, live=False)
        rec = eng.recommend(Request(user_id=uid, spm=spm0 + "$##$0", market=market))
        if not rec.items:
            continue
        ids = np.array([it["id"] for it in rec.items])
        p = oracle(uid, ids)
        p = np.nan_to_num(p) / (1 + 0.08 * np.arange(len(ids)))  # position bias as in the simulator
        clicked = rng.random(len(ids)) < p
        rows = [
            {
                "user_id": uid,
                "item_id": i,
                "event_type": "impression",
                "ts": t + pos,
                "request_id": rec.request_id,
                "spm": f"{spm0}$##${pos}",
                "market": market,
            }
            for pos, i in enumerate(ids)
        ]
        tt = t + 30
        for pos in np.flatnonzero(clicked):
            i = ids[pos]
            tt += float(rng.integers(5, 120))
            spm = f"{spm0}$##${pos}"
            rows.append(dict(user_id=uid, item_id=i, event_type="click", ts=tt, request_id=rec.request_id, spm=spm))
            logit = np.log(p[pos] / max(1 - p[pos], 1e-9))
            if rng.random() < _sigmoid(logit + 1.2):
                tt += float(rng.integers(5, 60))
                rows.append(dict(user_id=uid, item_id=i, event_type="add_cart", ts=tt, request_id=rec.request_id))
                if rng.random() < 0.45:
                    tt += float(rng.integers(60, 1800))
                    price = eng.catalog.price[market][eng.catalog.idx(i)]
                    # checkout on the cart page: no request id -> last-click attribution
                    rows.append(
                        dict(
                            user_id=uid,
                            item_id=i,
                            event_type="purchase",
                            ts=tt,
                            paid_amount=float(price),
                            spm="demo$##$cart$##$checkout$##$0",
                        )
                    )
        for r in rows:
            r.setdefault("market", market)
        eng.clock = Clock(tt + 1, live=False)
        res = ing.write("behavior", rows)
        n_ev += res.accepted
        n_rej += len(res.errors)
        if (n + 1) % 2000 == 0:
            log(f"[ab-sim] {n + 1:,}/{n_sessions:,} sessions, {n_ev:,} events")
    bus.close()
    log(f"[ab-sim] done: {n_sessions:,} sessions, {n_ev:,} events accepted, {n_rej} rejected -> {out_lake}")
    return date, cfg
