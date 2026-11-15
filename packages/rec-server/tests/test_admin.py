"""Config console: store, validate, preview, publish, rollback, hot reload across workers, item pools."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from fastapi.testclient import TestClient

from rec_server.admin import ConfigStore, ConfigWatcher, check
from rec_server.api import create_app
from rec_server.data import Tables, load_config
from rec_server.engine import Request
from rec_server.pipeline import build_engine, load_engine, save_engine
from recsys_core.config import hot_sections
from recsys_tools.simulator import generate

HOME = "demo$##$home$##$feed$##$0"


@pytest.fixture(scope="module")
def snap(tmp_path_factory):
    d = tmp_path_factory.mktemp("adm")
    generate(d / "lake", n_users=300, n_items=300, n_brands=25, n_days=8, seed=9)
    cfg = load_config()
    t = Tables.load(d / "lake")
    save_engine(build_engine(cfg, t, t.end_ts, sample_days=3, log=lambda *a: None), d / "e.pkl")
    return cfg, d


def console(snap, tmp_path, n=1):
    """n workers sharing one config store (sqlite file)."""
    cfg, d = snap
    url = f"sqlite:///{tmp_path / 'cfg.db'}"
    out = []
    for _ in range(n):
        eng = load_engine(d / "e.pkl", cfg, live=False)
        out.append((eng, TestClient(create_app("x", engine=eng, admin_store=url))))
    return out


def test_bootstrap_has_no_secrets_and_lists_registry(snap, tmp_path):
    ((_, c),) = console(snap, tmp_path)
    j = c.get("/admin/api/config").json()
    assert j["version"]["created_by"] == "bootstrap" and j["running_version"] == j["version"]["id"]
    assert "auth" not in j["config"] and "auth" not in j["toml"] and "change-me" not in j["toml"]
    assert j["restart_sections"]["auth"]["token"] == "***"
    assert "repurchase" in j["registry"]["recalls"] and "early" in j["registry"]["recalls"]["repurchase"]
    assert c.get("/admin").status_code == 200


def test_validate_rejects_bad_drafts(snap, tmp_path):
    ((_, c),) = console(snap, tmp_path)
    base = c.get("/admin/api/config").json()["config"]
    bad = copy.deepcopy(base)
    bad["slots"]["home_feed"]["recalls"].append({"name": "nope", "quota": 5})
    bad["experiments"]["rank_layer"]["variants"][0]["traffic"] = 90
    r = c.post("/admin/api/validate", json={"config": bad})
    assert r.status_code == 422
    errs = " ".join(r.json()["errors"])
    assert "more than 100" in errs
    r = c.post("/admin/api/validate", json={"config": {**base, "auth": {"enabled": False}}})
    assert r.status_code == 422 and "restart" in r.json()["errors"][0]
    r = c.post("/admin/api/validate", json={"toml": "[slots.home_feed\nsize = 1"})
    assert r.status_code == 422 and r.json()["errors"][0].startswith("TOML")
    r = c.post("/admin/api/validate", json={"config": {**base, "markets": {"CN-HAINAN": base["markets"]["CN-HAINAN"]}}})
    assert r.status_code == 422 and "markets" in " ".join(r.json()["errors"])


def test_publish_hot_reloads_every_worker_and_rollback(snap, tmp_path):
    (a, ca), (b, cb) = console(snap, tmp_path, n=2)
    base = ca.get("/admin/api/config").json()
    draft = copy.deepcopy(base["config"])
    draft["slots"]["home_feed"]["size"] = 7
    v = ca.post("/admin/api/validate", json={"config": draft}).json()
    assert v["ok"] and v["diff"] == [{"path": "slots.home_feed.size", "old": 20, "new": 7}]
    p = ca.post("/admin/api/publish", json={"config": draft, "comment": "smaller feed", "author": "ops"}).json()
    assert p["ok"] and a.config["slots"]["home_feed"]["size"] == 7
    assert a.config["auth"] == snap[0]["auth"]  # restart sections untouched
    # worker b picks it up on its next request (poll interval elapsed)
    cb.app.state.watcher._next = 0
    r = cb.post("/predict/api/demo/default", json={"user": {"uid": "u1"}, "context": {"spm": HOME}}).json()
    assert len(r["value"]["items"]) == 7 and b.config["slots"]["home_feed"]["size"] == 7
    assert ca.post("/admin/api/publish", json={"config": draft, "comment": "again"}).status_code == 422  # no changes
    hist = ca.get("/admin/api/versions").json()
    assert [v["is_active"] for v in hist["versions"]] == [True, False]
    assert hist["audit"][0]["actor"] == "ops"
    rb = ca.post("/admin/api/rollback", json={"version": base["version"]["id"]}).json()
    assert rb["ok"] and a.config["slots"]["home_feed"]["size"] == 20
    assert ca.post("/admin/api/rollback", json={"version": 999}).status_code == 404


def test_toml_publish_and_preview(snap, tmp_path):
    ((eng, c),) = console(snap, tmp_path)
    j = c.get("/admin/api/config").json()
    toml = j["toml"].replace('ranker = "gbdt"', 'ranker = "recall_score"', 1)
    uid = next(u for u in eng.users.profiles if len(eng.users.history(u)) >= 5)
    pv = c.post("/admin/api/preview", json={"toml": toml, "user_id": uid, "scene": "home_feed", "size": 10}).json()
    assert pv["ok"] and len(pv["current"]["items"]) == 10 and len(pv["draft"]["items"]) == 10
    assert pv["draft"]["trace"]["ranker"] != pv["current"]["trace"]["ranker"] or pv["draft"]["trace"]["experiments"]
    assert eng.config["slots"]["home_feed"]["ranker"] == "gbdt"  # preview publishes nothing
    p = c.post("/admin/api/publish", json={"toml": toml, "comment": "toml edit"}).json()
    assert p["ok"] and eng.config["slots"]["home_feed"]["ranker"] == "recall_score"
    assert c.get(f"/admin/api/versions/{p['version']}").json()["toml"] == toml


def test_admin_token(snap, tmp_path):
    cfg, d = snap
    import rec_server.api as api_mod

    locked = {**cfg, "admin": {**cfg["admin"], "token": "t0k"}}
    orig = api_mod.load_config
    api_mod.load_config = lambda _p=None: locked
    try:
        eng = load_engine(d / "e.pkl", cfg, live=False)
        c = TestClient(create_app("x", engine=eng, admin_store=f"sqlite:///{tmp_path / 'c.db'}"))
    finally:
        api_mod.load_config = orig
    assert c.get("/admin/api/config").status_code == 401
    assert c.get("/admin/api/config", headers={"X-Admin-Token": "t0k"}).status_code == 200


def test_bad_stored_version_keeps_serving(snap, tmp_path):
    cfg, d = snap
    eng = load_engine(d / "e.pkl", cfg, live=False)
    store = ConfigStore(f"sqlite:///{tmp_path / 'x.db'}", "demo")
    w = ConfigWatcher(store, eng)
    w.bootstrap(cfg)
    broken = hot_sections(cfg) | {
        "slots": {"home_feed": {**cfg["slots"]["home_feed"], "recalls": [{"name": "gone", "quota": 1}]}}
    }
    store.publish("", broken, "written by an older build", "x")
    assert not w.maybe_reload(force=True) and "unknown recall" in (w.last_error or "")
    assert eng.config["slots"]["home_feed"]["recalls"] == cfg["slots"]["home_feed"]["recalls"]


# ------------------------------------------------------------------ item pool / compliance
def test_slot_pool_and_market_blocks(snap):
    cfg, d = snap
    eng = load_engine(d / "e.pkl", cfg, live=False)
    cat = eng.catalog
    uid = next(u for u, p in eng.users.profiles.items() if p.market == "CN-HAINAN" and len(eng.users.history(u)) >= 3)
    blocked = "liquor"
    new = copy.deepcopy(cfg)
    new["markets"]["CN-HAINAN"]["blocked_categories"] = [blocked]
    new["slots"]["home_feed"]["pool"] = {"include_categories": ["skincare", "makeup", blocked], "max_price": 300.0}
    full, _, warn = check(hot_sections(new), cfg, eng)
    assert warn == []
    eng.reload_config(full)
    rec = eng.recommend(Request(user_id=uid, spm=HOME, market="CN-HAINAN"), record=False)
    cats = {cat.cat_names[cat.cat[it["idx"]]] for it in rec.items}
    assert rec.items and cats <= {"skincare", "makeup"}
    assert np.all(cat.price["CN-HAINAN"][[it["idx"] for it in rec.items]] <= 300)
    fb = eng.fallback(Request(user_id=uid, spm=HOME, market="CN-HAINAN", size=50), "test")
    assert blocked not in {cat.cat_names[cat.cat[it["idx"]]] for it in fb.items}  # compliance holds in fallback
    new["slots"]["home_feed"]["pool"]["exclude_brands"] = ["No Such Brand"]
    assert "unknown brands" in check(hot_sections(new), cfg, eng)[2][0]
