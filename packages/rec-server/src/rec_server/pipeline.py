"""Batch pipeline: build state + models at a cutoff and train the ranker.

Ranker samples come from impressions in [cutoff - window, cutoff) with features
computed from state *before* that window, so there is no label leakage."""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

from .bus import Bus, NullBus
from .data import DAY, Catalog, Clock, ItemStats, Tables, UserStore
from .engine import Engine
from .features import build_features, build_user_ctx
from .models import Models
from .rank import train_ranker
from .state import LiveState, make_live


def build_samples(t: Tables, catalog: Catalog, config: dict, start_ts: float, end_ts: float):
    """Point-in-time samples: item-side state/models from the daily batch at `start_ts`;
    user-side features from the user's history strictly before each request (what the
    online feature log would have recorded)."""
    users = UserStore.build(t, catalog, end_ts)
    stats = ItemStats.build(t, catalog, start_ts)
    models = Models.train(t, catalog, stats, start_ts)
    b = t.behaviors
    win = b[(b["ts"] >= start_ts) & (b["ts"] < end_ts)]
    imp = win[win["event_type"] == "impression"]
    clk = win[win["event_type"] == "click"]
    clicks = set(zip(clk["request_id"], clk["item_id"]))
    codes = {m: i for i, m in enumerate(sorted(config["markets"]))}
    uidx = {u: i for i, u in enumerate(imp["user_id"].unique())}
    Xs, ys, gs, ts = [], [], [], []
    for (rid, uid), grp in imp.groupby(["request_id", "user_id"], sort=False):
        now = float(grp["ts"].min()) - 1
        cand = grp["item_id"].map(catalog.id2idx).to_numpy()
        ctx = build_user_ctx(
            users, catalog, models, str(uid), grp["market"].iloc[0], now, config["app"]["default_market"]
        )
        Xs.append(build_features(ctx, cand, catalog, stats, models, codes))
        ys.append([(rid, i) in clicks for i in grp["item_id"]])
        gs.append(np.full(len(grp), uidx[uid]))
        ts.append(grp["ts"].to_numpy())
    return (np.vstack(Xs), np.concatenate(ys).astype(int), np.concatenate(gs), np.concatenate(ts))


def build_engine(
    config: dict, t: Tables, cutoff_ts: float, sample_days: float = 5, live: bool = False, log=print
) -> Engine:
    t_start = time.time()
    catalog = Catalog.build(t)
    log(f"[pipeline] catalog: {catalog.n} items, markets={catalog.markets}")
    s0 = cutoff_ts - sample_days * DAY
    X, y, g, ts = build_samples(t, catalog, config, s0, cutoff_ts)
    valid = ts >= cutoff_ts - DAY
    ranker, metrics = train_ranker(X, y, g, valid)
    log(
        f"[pipeline] ranker ({metrics['backend']}): AUC={metrics['auc']:.4f} GAUC={metrics['gauc']:.4f} "
        f"train={metrics['train_rows']} valid={metrics['valid_rows']} onnx={metrics.get('onnx')} "
        f"serving={metrics.get('serving_backend')}"
    )
    users = UserStore.build(t, catalog, cutoff_ts)
    stats = ItemStats.build(t, catalog, cutoff_ts)
    models = Models.train(t, catalog, stats, cutoff_ts)
    log(
        f"[pipeline] swing nnz={models.swing.nnz} buy_swing nnz={models.buy_swing.nnz} "
        f"emb={models.item_vec.shape} ({time.time() - t_start:.1f}s)"
    )
    return Engine(
        config,
        catalog,
        users,
        stats,
        models,
        ranker,
        Clock(cutoff_ts, live=live),
        info=dict(cutoff_ts=cutoff_ts, ranker=metrics, built_at=time.time()),
    )


def save_engine(engine: Engine, path: Path) -> None:
    """Pickle the snapshot. Live state and the bus are connections, never pickled
    (see Engine/UserStore __getstate__)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(engine, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_engine(
    path: Path,
    config: dict | None = None,
    live: bool = True,
    state: LiveState | None = None,
    bus: Bus | None = None,
) -> Engine:
    """Load a snapshot and attach real-time state + bus (from config unless given)."""
    with open(path, "rb") as f:
        engine: Engine = pickle.load(f)
    if config is not None:
        Engine.validate_config(config)
        engine.config = config
    engine.clock = Clock(engine.clock.base_ts, live=live)
    engine.users.live = state if state is not None else make_live(engine.config)
    engine.bus = bus if bus is not None else NullBus()
    return engine
