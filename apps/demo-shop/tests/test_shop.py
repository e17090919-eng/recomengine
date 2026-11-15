"""Demo shop against a real (small) engine: the shop only talks to it over HTTP via the SDK."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from demo_shop.app import create_app as create_shop
from rec_server.api import create_app as create_engine
from rec_server.bus import MemoryBus
from rec_server.data import Tables, load_config
from rec_server.pipeline import build_engine, load_engine, save_engine
from recsys_tools.simulator import generate


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("shop")
    generate(d / "lake", n_users=400, n_items=400, n_brands=30, n_days=10, seed=5)
    cfg = load_config()
    t = Tables.load(d / "lake")
    save_engine(build_engine(cfg, t, t.end_ts, sample_days=3, log=lambda *a: None), d / "engine.pkl")
    bus = MemoryBus()
    eng = load_engine(d / "engine.pkl", cfg, live=True, bus=bus)
    engine = TestClient(create_engine(d / "engine.pkl", engine=eng), base_url="http://engine")
    shop = TestClient(create_shop("http://engine", d / "lake", http=engine, probe=False))
    return shop, bus


def test_home_is_personalised_and_reacts_to_clicks(setup):
    shop, bus = setup
    boot = shop.get("/api/bootstrap").json()
    assert {"CN-HAINAN", "NZ-AKL"} <= set(boot["markets"]) and boot["personas"][0]["id"] == "visitor-001"
    base = {"uid": "visitor-new", "market": "CN-HAINAN", "lang": "zh"}
    first = shop.get("/api/home", params=base).json()
    assert first["items"] and not first["fallback"]
    assert {s for i in first["items"] for s in i["sources"]} <= {"hot", "new_items"}  # cold start
    assert all(i["spm"].startswith("demo$##$home$##$feed$##$") for i in first["items"])

    cat = next(c["id"] for c in boot["categories"] if c["id"] == "fragrance")
    listing = shop.get(f"/api/category/{cat}", params=base).json()["items"]
    for it in listing[:3]:
        r = shop.get(f"/api/item/{it['id']}", params={**base, "spm": it["spm"], "rid": "r1"})
        assert r.status_code == 200 and r.json()["related"]["items"]
    clicks = [e for e in bus.records["behavior"] if e["user_id"] == "visitor-new" and e["event_type"] == "click"]
    assert len(clicks) == 3 and clicks[0]["request_id"] == "r1"

    after = shop.get("/api/home", params=base).json()
    frag = sum(i["category"] == "fragrance" for i in after["items"])
    assert frag > sum(i["category"] == "fragrance" for i in first["items"])
    prof = shop.get("/api/profile", params=base).json()
    assert prof["categories"][0]["name"] == "fragrance"


def test_cart_allowance_checkout_and_orders(setup):
    shop, bus = setup
    base = {"uid": "visitor-cart", "market": "CN-HAINAN", "lang": "en"}
    item = shop.get("/api/category/watches", params=base).json()["items"][0]
    assert shop.post("/api/cart", json={**base, "item_id": item["id"], "qty": 2}).json()["count"] == 2
    cart = shop.get("/api/cart", params=base).json()
    a = cart["allowance"]
    assert a["in_cart"] == pytest.approx(2 * item["price"])
    assert a["left"] == pytest.approx(max(a["cap"] - a["used"] - a["in_cart"], 0))
    assert all(i["price"] <= a["left"] + 1e-6 for i in cart["bundle"]["items"])  # add-ons fit the allowance
    assert all(i["id"] != item["id"] for i in cart["bundle"]["items"])

    o = shop.post("/api/checkout", json={"uid": base["uid"], "market": base["market"]}).json()
    assert o["total"] == pytest.approx(2 * item["price"])
    buys = [e for e in bus.records["behavior"] if e["user_id"] == "visitor-cart" and e["event_type"] == "purchase"]
    assert len(buys) == 1 and buys[0]["paid_amount"] == pytest.approx(2 * item["price"])
    orders = shop.get("/api/orders", params=base).json()
    assert orders["orders"][0]["live"] and "reorder" in orders
    assert shop.get("/api/cart", params=base).json()["lines"] == []


def test_compare_and_reset(setup, monkeypatch):
    shop, _ = setup
    r = shop.get("/api/compare", params={"uid": "visitor-new", "market": "NZ-AKL", "lang": "en"}).json()
    assert r["incumbent"] and r["engine"]["items"] and r["engine"]["items"][0]["currency"] == "NZD"
    assert shop.post("/api/reset").status_code == 502  # engine refuses unless RECSYS_DEMO=1
    monkeypatch.setenv("RECSYS_DEMO", "1")
    assert shop.post("/api/reset").json()["ok"]
    prof = shop.get("/api/profile", params={"uid": "visitor-new", "market": "CN-HAINAN"}).json()
    assert prof["categories"] == [] and prof["recent"] == []
