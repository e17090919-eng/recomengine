"""End-to-end P0 path on a small dataset: simulate -> raw export -> backfill -> lake -> baseline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from recsys_core.config import find_root, load_config
from recsys_core.lake import Lake
from recsys_tools.backfill import TableStats, apply_column, load_mapping, run_backfill
from recsys_tools.raw_export import export_raw
from recsys_tools.simulator import simulate

ROOT = find_root(Path(__file__).parent)
MAPPING = ROOT / "configs" / "mappings" / "sample_vendor.toml"


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p0")
    sim = simulate(n_users=400, n_items=300, n_brands=30, n_days=10, seed=3)
    dirty = export_raw(sim, tmp / "raw")
    res = run_backfill(MAPPING, tmp / "raw", tmp / "lake", log=lambda *a: None)
    return sim, dirty, res, tmp


def test_backfill_reconciles_with_source(pipeline):
    sim, _, res, _tmp = pipeline
    st = res.stats["behavior"]
    assert st.rows_out == len(sim.tables["behavior"])
    assert st.dropped["unmapped event_type"] > 0 and st.dropped["duplicate"] > 0
    assert "$AppStart" in st.unmapped["event_type"]
    assert res.quality.passed


def test_backfill_preserves_values(pipeline):
    sim, _, _, tmp = pipeline
    lake = Lake(tmp / "lake")
    got = lake.read("behavior").sort_values(["ts", "user_id", "item_id", "event_type"]).reset_index(drop=True)
    exp = sim.tables["behavior"]
    # rows whose item id was deliberately corrupted in the raw export are excluded from the comparison
    got = got[~got["item_id"].str.startswith("SKU-UNKNOWN")]
    exp = exp[
        exp.set_index(["ts", "user_id", "event_type"]).index.isin(got.set_index(["ts", "user_id", "event_type"]).index)
    ]
    exp = exp.sort_values(["ts", "user_id", "item_id", "event_type"]).reset_index(drop=True)
    merged = got.merge(exp, on=["ts", "user_id", "item_id", "event_type"], suffixes=("", "_src"))
    assert len(merged) / len(got) > 0.99  # timestamps survive local-time -> UTC round trip
    assert (merged["market"] == merged["market_src"]).all()
    assert (merged["request_id"] == merged["request_id_src"]).all()
    users = lake.read("user").set_index("user_id")
    src = sim.tables["user"].set_index("user_id")
    assert (users.loc[src.index, "member_level"] == src["member_level"]).all()
    im = lake.read("item_market")
    assert im["channels"].str.contains(r"\|").all()


def test_backfill_is_idempotent(pipeline):
    _, _, _res, tmp = pipeline
    before = len(Lake(tmp / "lake").read("behavior"))
    run_backfill(MAPPING, tmp / "raw", tmp / "lake", log=lambda *a: None)
    assert len(Lake(tmp / "lake").read("behavior")) == before


def test_column_transforms():
    src = pd.DataFrame(
        {"t": ["2026-10-01 08:00:00", None], "e": ["A", "Z"], "a": ["x", None], "b": ["1", "2"], "d": ["p,q", "r"]}
    )
    st = TableStats()
    ts, _ = apply_column(src, {"field": "t", "time_format": "%Y-%m-%d %H:%M:%S", "tz": "Asia/Shanghai"}, st, "ts")
    assert ts.iloc[0] == 1790812800 and pd.isna(ts.iloc[1])  # 08:00 Beijing = 00:00 UTC
    ev, drop = apply_column(src, {"field": "e", "map": {"A": "click"}, "drop_unmapped": True}, st, "ev")
    assert ev.iloc[0] == "click" and drop.tolist() == [False, True] and st.unmapped["ev"] == {"Z": 1}
    tp, _ = apply_column(src, {"template": "app$##${a}$##${b}"}, st, "spm")
    assert tp.iloc[0] == "app$##$x$##$1" and pd.isna(tp.iloc[1])
    ch, _ = apply_column(src, {"field": "d", "replace": {",": "|"}}, st, "c")
    assert ch.tolist() == ["p|q", "r"]
    dv, _ = apply_column(src, {"field": "a", "default": "0"}, st, "r")
    assert dv.tolist() == ["x", "0"]


def test_mapping_validation(tmp_path):
    p = tmp_path / "m.toml"
    p.write_text('[tables.orders]\npath = "x"\ncolumns = {}\n')
    with pytest.raises(ValueError, match="unknown tables"):
        load_mapping(p)


def test_baseline_report_on_backfilled_lake(pipeline):
    from recsys_batch.baseline import run

    sim, _, _, tmp = pipeline
    report = run(
        load_config(ROOT / "configs" / "engine.toml"),
        str(tmp / "lake"),
        test_days=2,
        n_users=60,
        oracle=sim.truth.oracle(),
        log=lambda *a: None,
    )
    for s in ("Ranking quality", "GBDT ranker", "Hot score", "Swing", "HitRate@20", "NDCG@20", "sim. CTR"):
        assert s in report


def test_ab_sim_end_to_end(tmp_path):
    from rec_server.data import Tables, load_config
    from rec_server.pipeline import build_engine, save_engine
    from recsys_batch.report import daily_report
    from recsys_tools.abtest_sim import run_ab_sim
    from recsys_tools.simulator import simulate, write_lake

    sim = simulate(n_users=400, n_items=400, n_brands=30, n_days=10, seed=11)
    write_lake(sim, tmp_path / "lake")
    cfg = load_config()
    t = Tables.load(tmp_path / "lake")
    save_engine(build_engine(cfg, t, t.end_ts, sample_days=3, log=lambda *a: None), tmp_path / "e.pkl")
    date, cfg_ab = run_ab_sim(
        cfg, tmp_path / "e.pkl", tmp_path / "lake", tmp_path / "ab", sim.truth.oracle(), 300, log=lambda *a: None
    )
    rep = daily_report(str(tmp_path / "ab"), cfg_ab, date)
    assert set(rep.rows["variant"]) == {"incumbent", "multi_recall", "gbdt"}
    log = Lake(tmp_path / "ab").read("rec_log")
    assert rep.rows["requests"].sum() == len(log) == 300
    assert rep.rows["impressions"].sum() == log["items"].str.count(r"\|").add(1).sum()
    assert rep.rows["purchase"].sum() > 0 and rep.meta["variant_source"] == "request log"
