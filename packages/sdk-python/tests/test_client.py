from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from rec_server.api import create_app
from rec_server.data import Tables, load_config
from rec_server.pipeline import build_engine
from recsys_client import AsyncRecsysClient, RecsysClient, RecsysError, sign
from recsys_core import signing
from recsys_tools.simulator import generate

HOME = "demo$##$home$##$feed$##$0"


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    d = tmp_path_factory.mktemp("sdk")
    generate(d, n_users=300, n_items=300, n_brands=25, n_days=8, seed=5)
    t = Tables.load(d)
    return build_engine(load_config(), t, t.end_ts, sample_days=3, log=lambda *a: None)


def app_with(engine, auth: dict | None = None):
    import rec_server.api as api_mod

    cfg = load_config()
    if auth:
        cfg = {**cfg, "auth": auth}
    orig = api_mod.load_config
    api_mod.load_config = lambda _p=None: cfg
    try:
        return create_app("unused", engine=engine)
    finally:
        api_mod.load_config = orig


def client(engine, **kw) -> RecsysClient:
    return RecsysClient("http://testserver", "demo", http=TestClient(app_with(engine, kw.pop("auth", None))), **kw)


def test_sign_matches_server():
    args = ("tok", b'{"a":1}', "t", "123", "n")
    assert sign(*args) == signing.sign(*args)


def test_predict_write_callback_done(engine):
    rc = client(engine)
    uid = next(iter(engine.users.profiles))
    res = rc.predict(uid, HOME, 5, market="NZ-AKL", categories=["skincare"])
    assert not res.fallback and res.request_id and len(res.items) == 5
    assert all(engine.catalog.cat_names[engine.catalog.cat[engine.catalog.idx(i)]] == "skincare" for i in res.ids)
    assert all(it.trans_data and it.score is not None for it in res.items)
    rows = [{"user_id": uid, "item_id": i, "event_type": "impression", "request_id": res.request_id} for i in res.ids]
    rows.insert(2, {"user_id": uid, "item_id": "nope", "event_type": "click"})
    rc.chunk_size = 2
    w = rc.write_behaviors(rows)
    assert w.accepted == 5 and [e["index"] for e in w.errors] == [2]  # index in the caller's list, across chunks
    rc.callback(uid, res.request_id, res.ids[:2])
    rc.done("behavior", "20261114")
    with pytest.raises(ValueError, match="spm"):
        rc.predict(uid, "home_feed")


def test_signed_requests(engine):
    auth = {"enabled": True, "tenant_id": "demo", "token": "s3cret"}
    ok = client(engine, auth=auth, token="s3cret")
    assert not ok.predict("u1", HOME, 3).fallback
    bad = client(engine, auth=auth, token="wrong")
    with pytest.raises(RecsysError) as e:
        bad.predict("u1", HOME, 3)
    assert e.value.code == 401
    fb = bad.predict("u1", HOME, 3, fallback_ids=["a", "b", "c", "d"])
    assert fb.fallback and fb.ids == ["a", "b", "c"]


def test_transport_failure_serves_fallback():
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("down", request=request)

    rc = RecsysClient("http://x", "demo", http=httpx.Client(transport=httpx.MockTransport(boom)))
    res = rc.predict("u", HOME, 2, fallback_ids=["h1", "h2", "h3"])
    assert res.fallback and res.ids == ["h1", "h2"] and "ConnectTimeout" in (res.error or "")


def test_write_retries_on_5xx():
    calls = []

    def flaky(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"code": 200, "accepted": 1, "errors": []})

    rc = RecsysClient("http://x", "demo", http=httpx.Client(transport=httpx.MockTransport(flaky)), retries=2)
    assert rc.write_behaviors([{"user_id": "u", "item_id": "i", "event_type": "click"}]).accepted == 1
    assert len(calls) == 2


def test_async_client(engine):
    app = app_with(engine)

    async def run():
        async with AsyncRecsysClient(
            "http://testserver", "demo", http=httpx.AsyncClient(transport=httpx.ASGITransport(app=app))
        ) as rc:
            res = await rc.predict("u-async", HOME, 4, market="CN-HAINAN")
            w = await rc.write("behavior", [{"user_id": "u-async", "item_id": res.ids[0], "event_type": "click"}])
            return res, w

    res, w = asyncio.run(run())
    assert len(res.items) == 4 and w.accepted == 1
