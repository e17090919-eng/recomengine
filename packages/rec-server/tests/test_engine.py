from __future__ import annotations

import hashlib
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rec_server.abtest import assign, bucket
from rec_server.api import create_app
from rec_server.data import Tables, load_config
from rec_server.engine import Engine, Request, SlotNotFound
from rec_server.pipeline import build_engine
from rec_server.rules import disperse
from recsys_tools.simulator import generate

HOME = "demo$##$home$##$feed$##$0"


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    generate(d, n_users=600, n_items=500, n_brands=40, n_days=12, seed=1)
    cfg = load_config()
    t = Tables.load(d)
    eng = build_engine(cfg, t, t.end_ts, sample_days=4, log=lambda *a: None)
    return cfg, t, eng


def active_user(eng, market=None):
    for uid, p in eng.users.profiles.items():
        if len(eng.users.history(uid)) >= 5 and (market is None or p.market == market):
            return uid
    raise AssertionError("no active user")


def test_home_feed_basic(setup):
    cfg, _, eng = setup
    uid = active_user(eng)
    rec = eng.recommend(Request(user_id=uid, spm=HOME), record=False)
    ids = [it["id"] for it in rec.items]
    assert len(ids) == cfg["slots"]["home_feed"]["size"]
    assert len(set(ids)) == len(ids)
    avail = eng.catalog.available(rec.market, eng.clock.now())
    assert all(avail[it["idx"]] for it in rec.items)
    assert rec.trace["merged"] >= len(ids)


def test_cold_start_user_gets_results(setup):
    _, _, eng = setup
    rec = eng.recommend(Request(user_id="brand-new-user", spm=HOME, market="NZ-AKL"), record=False)
    assert len(rec.items) > 0 and rec.market == "NZ-AKL"


def test_unknown_spm(setup):
    with pytest.raises(SlotNotFound):
        setup[2].recommend(Request(user_id="x", spm="demo$##$nope$##$x$##$0"))


def test_purchased_and_filter_items_excluded(setup):
    _, _, eng = setup
    uid = active_user(eng)
    first = eng.recommend(Request(user_id=uid, spm=HOME), record=False).items
    bought, banned = first[0], first[1]
    eng.users.add_event(uid, eng.clock.now() - 1, bought["idx"], "purchase", 0.0)
    rec = eng.recommend(Request(user_id=uid, spm=HOME, filter_ids=[banned["id"]]), record=False)
    ids = {it["id"] for it in rec.items}
    assert bought["id"] not in ids and banned["id"] not in ids


def test_allowance_filter(setup):
    _, _, eng = setup
    uid = active_user(eng, "CN-HAINAN")
    rec = eng.recommend(Request(user_id=uid, spm=HOME, allowance_remaining=300.0), record=False)
    prices = eng.catalog.price["CN-HAINAN"][[it["idx"] for it in rec.items]]
    assert len(prices) and np.all(prices <= 300.0)


def test_issued_dedup_across_refreshes(setup):
    _, _, eng = setup
    uid = active_user(eng)
    a = {it["id"] for it in eng.recommend(Request(user_id=uid, spm=HOME)).items}
    b = {it["id"] for it in eng.recommend(Request(user_id=uid, spm=HOME)).items}
    assert a and b and not (a & b)


def test_realtime_click_drives_retarget(setup):
    _, _, eng = setup
    uid = "rt-user"
    avail = np.flatnonzero(eng.catalog.available("NZ-AKL", eng.clock.now()))
    target = int(avail[0])
    eng.users.add_event(uid, eng.clock.now() - 5, target, "add_cart")
    rec = eng.recommend(
        Request(user_id=uid, spm=HOME, market="NZ-AKL"),
        overrides=dict(recalls=[{"name": "retarget", "quota": 10}]),
        record=False,
    )
    assert rec.items[0]["idx"] == target and "retarget" in rec.items[0]["sources"]


def test_related_excludes_parent(setup):
    _, _, eng = setup
    uid = active_user(eng)
    parent = eng.recommend(Request(user_id=uid, spm=HOME), record=False).items[0]["id"]
    rec = eng.recommend(Request(user_id=uid, spm="demo$##$item$##$related$##$0", parent_id=parent), record=False)
    assert rec.items and parent not in {it["id"] for it in rec.items}


def test_dispersion():
    class C:
        brand = np.array([0, 0, 0, 1, 1, 2, 2, 3])
        cat = np.zeros(8, int)

    out = disperse(np.arange(8), C, [{"field": "brand", "window": 3, "max": 1}], 6)
    b = C.brand[out]
    assert all(len(set(b[i : i + 3])) == len(b[i : i + 3]) for i in range(len(b) - 2))


def test_ab_bucketing_stable_and_split():
    exps = {
        "l": {
            "slots": ["home_feed"],
            "salt": "s",
            "variants": [{"name": "a", "traffic": 50, "params": {}}, {"name": "b", "traffic": 50, "params": {}}],
        }
    }
    names = [assign(exps, f"u{i}", "home_feed")[0].variant for i in range(4000)]
    assert 0.45 < names.count("a") / len(names) < 0.55
    assert assign(exps, "u1", "home_feed")[0].variant == assign(exps, "u1", "home_feed")[0].variant
    assert assign(exps, "u1", "home_feed", force={"l": "b"})[0].variant == "b"
    assert bucket("x", "s") == bucket("x", "s")


def test_api_predict_write_and_signature(setup):
    cfg, _, eng = setup
    app = create_app("unused", engine=eng)
    c = TestClient(app)
    uid = active_user(eng)
    body = {"user": {"uid": uid}, "context": {"spm": HOME}, "size": 5}
    r = c.post("/predict/api/demo/default", json=body).json()
    assert r["code"] == 200 and r["success"] and len(r["value"]["items"]) == 5
    assert all(set(it) >= {"id", "rank", "transData", "extra"} for it in r["value"]["items"])
    assert (
        c.post("/predict/api/demo/default", json={**body, "context": {"spm": "x$##$y$##$z$##$0"}}).json()["code"] == 404
    )
    w = c.post(
        "/data/api/demo/behavior/write",
        json=[
            {"user_id": uid, "item_id": r["value"]["items"][0]["id"], "event_type": "click"},
            {"user_id": uid, "item_id": "nope", "event_type": "click"},
            {"user_id": uid, "item_id": r["value"]["items"][0]["id"], "event_type": "teleport"},
        ],
    ).json()
    assert [e["index"] for e in w["errors"]] == [1, 2]

    # signature check (enabled via config)
    eng_cfg = {**cfg, "auth": {"enabled": True, "tenant_id": "demo", "token": "secret"}}
    assert c.get("/metrics").text.count("rec_requests_total") > 0
    import rec_server.api as api_mod

    orig = api_mod.load_config
    api_mod.load_config = lambda _p=None: eng_cfg
    try:
        c2 = TestClient(create_app("unused", engine=eng))
    finally:
        api_mod.load_config = orig
    import json

    raw = json.dumps(body).encode()
    assert (
        c2.post("/predict/api/demo/default", content=raw, headers={"Content-Type": "application/json"}).json()["code"]
        == 401
    )
    ts, nonce = str(int(time.time())), "n1"
    sig = hashlib.sha256(b"secret" + raw + b"demo" + ts.encode() + nonce.encode()).hexdigest()
    h = {
        "Content-Type": "application/json",
        "Tenant-Id": "demo",
        "Tenant-Ts": ts,
        "Tenant-Nonce": nonce,
        "Tenant-Signature": sig,
    }
    assert c2.post("/predict/api/demo/default", content=raw, headers=h).json()["code"] == 200
    assert c2.post("/predict/api/demo/default", content=raw, headers=h).json()["code"] == 401  # replay


def test_config_names_validated(setup):
    cfg = setup[0]
    bad = {
        **cfg,
        "slots": {**cfg["slots"], "x": {**cfg["slots"]["home_feed"], "recalls": [{"name": "nope", "quota": 5}]}},
    }
    with pytest.raises(ValueError, match="unknown recall"):
        Engine.validate_config(bad)
