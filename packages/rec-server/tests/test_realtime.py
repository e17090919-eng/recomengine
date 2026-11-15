"""P1 real-time path: shared live state (memory / Redis), ingest validation, bus -> lake,
multi-worker consistency, degradation."""

from __future__ import annotations

import time

import fakeredis
import numpy as np
import pytest
from fastapi.testclient import TestClient

from rec_server.api import create_app
from rec_server.bus import LakeBus, MemoryBus
from rec_server.data import Tables, load_config
from rec_server.engine import Request
from rec_server.pipeline import build_engine, load_engine, save_engine
from rec_server.recall import RECALLS
from rec_server.state import MemoryLive, RedisLive
from recsys_core.lake import Lake
from recsys_tools.simulator import generate

HOME = "demo$##$home$##$feed$##$0"


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    d = tmp_path_factory.mktemp("rt")
    generate(d / "lake", n_users=400, n_items=400, n_brands=30, n_days=10, seed=3)
    cfg = load_config()
    t = Tables.load(d / "lake")
    eng = build_engine(cfg, t, t.end_ts, sample_days=3, log=lambda *a: None)
    path = d / "engine.pkl"
    save_engine(eng, path)
    return cfg, path, d


def redis_live(server=None) -> RedisLive:
    return RedisLive(fakeredis.FakeRedis(server=server or fakeredis.FakeServer(), decode_responses=True), "t")


def workers(snapshot, n=2):
    """n engines loaded from the same snapshot, sharing one Redis (= n serving pods)."""
    cfg, path, _ = snapshot
    server = fakeredis.FakeServer()
    return [load_engine(path, cfg, live=False, state=redis_live(server), bus=MemoryBus()) for _ in range(n)]


# ------------------------------------------------------------------ live state backends
@pytest.mark.parametrize("make", [MemoryLive, redis_live], ids=["memory", "redis"])
def test_live_state_backends_behave_the_same(make):
    s = make()
    s.add_events("u", [(10.0, "a", "click", 0.0, 1.0), (11.0, "b", "impression", 0.0, 1.0)])
    s.add_events("u", [(12.0, "c", "purchase", 9.5, 2.0)])
    assert [e[1] for e in s.events("u")] == ["a", "c"]
    assert [e[1] for e in s.impressions("u")] == ["b"]
    s.patch_profile("u", {"member_level": 3})
    s.patch_profile("u", {"language": "zh"})
    assert s.profile_patch("u") == {"member_level": 3, "language": "zh"}
    s.record_issued("u", ["a", "b"], 1000.0)
    s.record_issued("u", ["c"], 1000.0 + 3600)
    assert s.issued("u", 1000.0 + 3600) == {"c"}  # a, b fell out of the 30 min window
    s.push_item_changes([("M", "a", 1.0, None, 5.0)])
    v, ch = s.item_changes_since(0)
    s.push_item_changes([("M", "b", None, 0, 6.0)])
    v2, ch2 = s.item_changes_since(v)
    assert (v, v2) == (1, 2) and ch[0][1] == "a" and ch2[0][1] == "b"
    assert s.n_users() == 1 and s.ping()


def test_snapshot_pickle_has_no_live_data(snapshot):
    cfg, path, _ = snapshot
    eng = load_engine(path, cfg, live=False)
    uid = next(iter(eng.users.profiles))
    eng.users.add_event(uid, eng.clock.now(), 0, "click")
    assert len(eng.users.history(uid)) == len(eng.users.events[uid]) + 1
    eng2 = load_engine(path, cfg, live=False)
    assert len(eng2.users.history(uid)) == len(eng2.users.events[uid])
    assert isinstance(eng2.users.live, MemoryLive)


# ------------------------------------------------------------------ multi-worker consistency
def test_click_on_worker_a_is_seen_by_worker_b(snapshot):
    a, b = workers(snapshot)
    ca, cb = TestClient(create_app("x", engine=a)), TestClient(create_app("x", engine=b))
    uid = "mw-user"
    target = str(a.catalog.item_ids[np.flatnonzero(a.catalog.available("NZ-AKL", a.clock.now()))[0]])
    w = ca.post(
        "/data/api/demo/behavior/write",
        json=[{"user_id": uid, "item_id": target, "event_type": "add_cart", "market": "NZ-AKL"}],
    ).json()
    assert w["code"] == 200 and w["accepted"] == 1
    rec = b.recommend(
        Request(user_id=uid, spm=HOME, market="NZ-AKL"),
        overrides=dict(recalls=[{"name": "retarget", "quota": 5}]),
        record=False,
    )
    assert rec.items and rec.items[0]["id"] == target
    assert cb.get("/healthz").json()["state"] == "RedisLive"


def test_issued_dedup_across_workers(snapshot):
    a, b = workers(snapshot)
    uid = next(u for u in a.users.profiles if len(a.users.history(u)) >= 5)
    first = {it["id"] for it in a.recommend(Request(user_id=uid, spm=HOME)).items}
    second = {it["id"] for it in b.recommend(Request(user_id=uid, spm=HOME)).items}
    assert first and second and not first & second


def test_price_and_stock_change_reaches_other_worker(snapshot):
    a, b = workers(snapshot)
    m = "CN-HAINAN"
    i = int(np.flatnonzero(a.catalog.available(m, a.clock.now()))[0])
    iid = str(a.catalog.item_ids[i])
    r = TestClient(create_app("x", engine=a)).post(
        "/data/api/demo/item/write", json=[{"item_id": iid, "market": m, "stock": 0, "price": 1.5}]
    )
    assert r.json()["accepted"] == 1
    assert a.catalog.stock[m][i] == 0
    b.sync_items(force=True)
    assert b.catalog.stock[m][i] == 0 and b.catalog.price[m][i] == 1.5
    assert not b.catalog.available(m, b.clock.now())[i]


# ------------------------------------------------------------------ ingest validation
def test_behavior_validation_and_bus(snapshot):
    (eng,) = workers(snapshot, 1)
    c = TestClient(create_app("x", engine=eng))
    item = str(eng.catalog.item_ids[0])
    now_ms = int(eng.clock.now() * 1000)
    rows = [
        {"user_id": "v1", "item_id": item, "event_type": "click", "ts": now_ms, "spm": "demo$##$home$##$feed$##$3"},
        {"user_id": "v1", "item_id": item, "event_type": "click", "spm": "home-feed"},
        {"user_id": "v1", "item_id": item, "event_type": "click", "market": "MARS"},
        {"user_id": "v1", "item_id": item, "event_type": "purchase", "paid_amount": -3},
        {"user_id": "", "item_id": item, "event_type": "click"},
        {"user_id": "v1", "item_id": item, "event_type": "click", "ts": 10},
        {"user_id": "v1", "item_id": item, "event_type": "purchase", "paid_amount": "88.5"},
    ]
    r = c.post("/data/api/demo/behavior/write", json=rows).json()
    assert r["code"] == 206 and r["accepted"] == 2
    msgs = {e["index"]: e["message"] for e in r["errors"]}
    assert set(msgs) == {1, 2, 3, 4, 5}
    assert "spm" in msgs[1] and "market" in msgs[2] and "paid_amount" in msgs[3] and "user_id" in msgs[4]
    out = eng.bus.records["behavior"]  # type: ignore[attr-defined]
    assert out[0]["ts"] == now_ms // 1000  # epoch millis accepted
    assert out[1]["paid_amount"] == 88.5 and out[0]["paid_amount"] is None
    assert c.post("/data/api/demo/nope/write", json=[{}]).json()["code"] == 400


def test_user_profile_patch_and_allowance(snapshot):
    (eng,) = workers(snapshot, 1)
    c = TestClient(create_app("x", engine=eng))
    uid = next(u for u, p in eng.users.profiles.items() if p.market == "CN-HAINAN")
    r = c.post("/data/api/demo/user/write", json=[{"user_id": uid, "allowance_used": 99000, "member_level": 4}])
    assert r.json()["accepted"] == 1
    prof = eng.users.profile(uid)
    assert prof and prof.member_level == 4 and prof.allowance_used == 99000
    item = str(eng.catalog.item_ids[0])
    c.post(
        "/data/api/demo/behavior/write",
        json=[{"user_id": uid, "item_id": item, "event_type": "purchase", "paid_amount": 500}],
    )
    assert eng.users.profile(uid).allowance_used == 99500  # type: ignore[union-attr]
    rec = eng.recommend(Request(user_id=uid, spm=HOME), record=False)  # only items <= 500 CNY remain
    assert np.all(eng.catalog.price["CN-HAINAN"][[it["idx"] for it in rec.items]] <= 500)
    bad = c.post("/data/api/demo/user/write", json=[{"user_id": uid, "vip": True}, {"user_id": uid, "market": "X"}])
    assert bad.json()["code"] == 400 and len(bad.json()["errors"]) == 2


def test_lake_bus_writes_behavior_rec_log_and_done(snapshot, tmp_path):
    cfg, path, _ = snapshot
    lake = Lake(tmp_path / "lake")
    bus = LakeBus(lake, flush_rows=10_000, flush_seconds=60)
    eng = load_engine(path, cfg, live=False, bus=bus)
    c = TestClient(create_app("x", engine=eng))
    uid = next(iter(eng.users.profiles))
    rec = c.post("/predict/api/demo/default", json={"user": {"uid": uid}, "context": {"spm": HOME}, "size": 4}).json()
    item = rec["value"]["items"][0]["id"]
    c.post(
        "/data/api/demo/behavior/write",
        json=[{"user_id": uid, "item_id": item, "event_type": "click", "request_id": rec["request_id"]}],
    )
    c.post("/data/api/demo/item/write", json=[{"item_id": item, "market": "CN-HAINAN", "stock": 7}])
    assert c.post("/data/api/demo/done", json={"table": "behavior", "date": "20261114"}).json()["success"]
    beh, log = lake.read("behavior"), lake.read("rec_log")
    assert list(beh["request_id"]) == [rec["request_id"]]
    assert log["request_id"].tolist() == [rec["request_id"]] and log["items"][0].split("|")[0] == item
    assert "rank_layer=" in log["variants"][0]
    assert lake.done_dates("behavior") == ["20261114"]
    assert lake.changes("item")["stock"].tolist() == [7]
    bus.close()


def test_batch_applies_realtime_item_changes(snapshot, tmp_path):
    _, _, d = snapshot
    import shutil

    shutil.copytree(d / "lake", tmp_path / "lake")
    lake = Lake(tmp_path / "lake")
    im = lake.read("item_market")
    iid, m = im["item_id"][0], im["market"][0]
    w = LakeBus(lake, flush_rows=1, flush_seconds=60).writer
    w.add("item", [{"item_id": iid, "market": m, "price": None, "stock": 3, "recv": time.time()}])
    w.add("item", [{"item_id": iid, "market": m, "price": 42.0, "stock": None, "recv": time.time() + 1}])
    w.flush()
    t = Tables.load(tmp_path / "lake")
    row = t.item_market[(t.item_market["item_id"] == iid) & (t.item_market["market"] == m)].iloc[0]
    assert (row["stock"], row["price"]) == (3, 42.0) and t.meta["applied_item_changes"] == 2


# ------------------------------------------------------------------ degradation
def test_failing_recall_channel_degrades_not_fails(snapshot, monkeypatch):
    (eng,) = workers(snapshot, 1)
    uid = next(u for u in eng.users.profiles if len(eng.users.history(u)) >= 5)

    def boom(*a, **k):
        raise RuntimeError("index down")

    monkeypatch.setitem(RECALLS, "swing", boom)
    rec = eng.recommend_safe(Request(user_id=uid, spm=HOME), record=False)
    assert rec.items and "recall_error:swing" in rec.trace["degraded"]


def test_failing_ranker_falls_back_to_recall_score(snapshot, monkeypatch):
    (eng,) = workers(snapshot, 1)
    uid = next(iter(eng.users.profiles))
    monkeypatch.setattr(eng.ranker, "predict", lambda X: 1 / 0)
    rec = eng.recommend_safe(Request(user_id=uid, spm=HOME, force_variants={"rank_layer": "treatment"}), record=False)
    assert rec.items and rec.trace["degraded"] == ["rank_error"]


def test_total_failure_serves_hot_fallback(snapshot, monkeypatch):
    (eng,) = workers(snapshot, 1)
    monkeypatch.setattr(eng, "recommend", lambda *a, **k: 1 / 0)
    c = TestClient(create_app("x", engine=eng))
    r = c.post("/predict/api/demo/default", json={"user": {"uid": "u"}, "context": {"spm": HOME}, "size": 6}).json()
    assert r["code"] == 200 and len(r["value"]["items"]) == 6 and r["degraded"] == ["error:ZeroDivisionError"]
    assert 'rec_degraded_total{reason="error",slot="home_feed"}' in c.get("/metrics").text
    assert eng.bus.records["rec_log"][-1]["degraded"] == "error:ZeroDivisionError"  # type: ignore[attr-defined]


def test_recall_budget_skips_remaining_channels(snapshot):
    (eng,) = workers(snapshot, 1)
    eng.config = {**eng.config, "serving": {**eng.config["serving"], "recall_budget_ms": 1e-9}}
    uid = next(u for u in eng.users.profiles if len(eng.users.history(u)) >= 5)
    rec = eng.recommend(Request(user_id=uid, spm=HOME), record=False)
    assert len(rec.trace["recall"]) == 1 and rec.items
    assert all(d.startswith("recall_budget:") for d in rec.trace["degraded"])


# ------------------------------------------------------------------ repurchase
def test_repurchase_recall_and_purchased_filter(snapshot):
    (eng,) = workers(snapshot, 1)
    cat = eng.catalog
    now = eng.clock.now()
    m = "CN-HAINAN"
    avail = np.flatnonzero(cat.available(m, now))
    i = int(avail[0])
    cyc = np.full(len(cat.cat_names), np.inf)
    cyc[cat.cat[i]] = 20.0  # this item's category is a consumable with a 20-day cycle
    eng.models.repurchase_days = cyc

    def run(uid: str):
        return eng.recommend(
            Request(user_id=uid, spm=HOME, market=m),
            overrides=dict(
                recalls=[{"name": "repurchase", "quota": 10}], dispersion=[], boost_new=None, ranker="recall_score"
            ),
            record=False,
        )

    eng.users.add_events("due", [(now - 21 * 86400, str(cat.item_ids[i]), "purchase", 10.0)])
    rec = run("due")
    assert rec.items[0]["idx"] == i and rec.items[0]["sources"] == ["repurchase"]
    assert all(cat.cat[it["idx"]] == cat.cat[i] for it in rec.items)
    eng.users.add_events("recent", [(now - 2 * 86400, str(cat.item_ids[i]), "purchase", 10.0)])
    rec = run("recent")  # bought 2 days ago: not due -> nothing recalled (cold fallback), and the item is hidden
    assert {s for it in rec.items for s in it["sources"]} == {"fallback_hot"}
    assert i not in {it["idx"] for it in rec.items}


def test_recall_params_are_validated(snapshot):
    from rec_server.engine import Engine

    cfg = snapshot[0]
    bad = {
        **cfg,
        "slots": {
            "x": {**cfg["slots"]["home_feed"], "recalls": [{"name": "repurchase", "quota": 5, "params": {"erly": 1}}]}
        },
    }
    with pytest.raises(ValueError, match="no params"):
        Engine.validate_config(bad)


# ------------------------------------------------------------------ Kafka path (fake client)
class FakeProducer:
    def __init__(self):
        self.sent = []

    def produce(self, topic, value, key=b""):
        self.sent.append((topic, key, value))

    def poll(self, t):
        return 0

    def flush(self, t=None):
        return 0


class FakeMsg:
    def __init__(self, topic, value):
        self._t, self._v = topic, value

    def topic(self):
        return self._t

    def value(self):
        return self._v

    def error(self):
        return None


class FakeConsumer:
    def __init__(self, msgs):
        self.msgs, self.commits, self.closed = list(msgs), 0, False

    def subscribe(self, topics):
        self.topics = topics

    def poll(self, t):
        return self.msgs.pop(0) if self.msgs else None

    def commit(self, asynchronous=True):
        self.commits += 1

    def close(self):
        self.closed = True


def test_kafka_bus_to_consumer_to_lake(snapshot, tmp_path):
    from rec_server.bus import KafkaBus
    from recsys_tools.consume import consume

    cfg, path, _ = snapshot
    prod = FakeProducer()
    eng = load_engine(path, cfg, live=False, bus=KafkaBus("unused", cfg["ingest"]["topics"], producer=prod))
    c = TestClient(create_app("x", engine=eng))
    item = str(eng.catalog.item_ids[0])
    c.post("/predict/api/demo/default", json={"user": {"uid": "k1"}, "context": {"spm": HOME}, "size": 3})
    c.post("/data/api/demo/behavior/write", json=[{"user_id": "k1", "item_id": item, "event_type": "click"}])
    c.post("/data/api/demo/done", json={"table": "behavior", "date": "20261101"})
    topics = [t for t, _, _ in prod.sent]
    assert topics == ["rec-log", "behavior-events", "recsys-done"]
    assert prod.sent[1][1] == b"k1"  # keyed by user: per-user ordering within a partition
    fc = FakeConsumer([FakeMsg(t, v) for t, _, v in prod.sent] + [FakeMsg("behavior-events", b"not json")])
    kcfg = {**cfg, "ingest": {**cfg["ingest"], "flush_rows": 1000, "flush_seconds": 0.01}}
    assert consume(kcfg, tmp_path / "lake", "g", max_seconds=0.3, consumer=fc) == 0
    lake = Lake(tmp_path / "lake")
    assert lake.read("behavior")["user_id"].tolist() == ["k1"]
    assert len(lake.read("rec_log")) == 1 and lake.done_dates("behavior") == ["20261101"]
    assert fc.commits >= 1 and fc.closed
