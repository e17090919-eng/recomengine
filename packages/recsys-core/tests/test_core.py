from __future__ import annotations

import copy
import tomllib
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from recsys_core.config import EngineConfig, find_root, load_config
from recsys_core.lake import Lake
from recsys_core.quality import run_checks
from recsys_core.schema import conform, to_markdown, valid_spm
from recsys_core.signing import headers, sign, verify

ROOT = find_root(Path(__file__).parent)


def raw_cfg() -> dict:
    with open(ROOT / "configs" / "engine.toml", "rb") as f:
        return tomllib.load(f)


def test_repo_config_is_valid():
    cfg = load_config(ROOT / "configs" / "engine.toml")
    assert cfg["app"]["default_market"] in cfg["markets"]
    assert cfg["quality"]["max_unknown_item_rate"] > 0  # defaults filled in


@pytest.mark.parametrize(
    "mutate, msg",
    [
        (lambda c: c["app"].update(default_market="XX"), "default_market"),
        (lambda c: c["markets"]["NZ-AKL"].update(timezone="Mars/Base"), "timezone"),
        (lambda c: c["markets"]["NZ-AKL"].update(currency="nzd"), "currency"),
        (lambda c: c["slots"]["cart"].update(spm=c["slots"]["home_feed"]["spm"]), "same spm"),
        (lambda c: c["slots"]["cart"].update(spm="demo$##$cart"), "spm"),
        (lambda c: c["experiments"]["rank_layer"]["variants"][0].update(traffic=80), "traffic"),
        (lambda c: c["experiments"]["rank_layer"].update(slots=["nope"]), "unknown slots"),
        (lambda c: c["slots"]["home_feed"].update(typo_field=1), "typo_field"),
    ],
)
def test_config_rejects(mutate, msg):
    c = copy.deepcopy(raw_cfg())
    mutate(c)
    with pytest.raises(ValidationError, match=msg):
        EngineConfig.model_validate(c)


def test_conform_coerces_and_counts_errors():
    df = pd.DataFrame(
        {
            "user_id": ["a", "b", "c"],
            "item_id": ["i", "j", "k"],
            "event_type": ["click"] * 3,
            "ts": ["100", "oops", "102.5"],
            "market": ["M"] * 3,
            "junk": [1, 2, 3],
        }
    )
    r = conform(df, "behavior")
    assert r.type_errors == {"ts": 2}
    assert r.df["ts"].tolist()[0] == 100 and r.df["ts"].isna().sum() == 2
    assert "junk" in r.dropped_columns and "spm" in r.df.columns and not r.missing_required


def test_conform_item_needs_a_title():
    df = pd.DataFrame({"item_id": ["i"], "category": ["c"], "brand": ["b"], "price_usd": [1.0], "publish_ts": [1]})
    assert conform(df, "item").missing_required
    assert not conform(df.assign(title_fr=["x"]), "item").missing_required


def test_spm_validation():
    s = pd.Series(["a$##$b$##$c$##$0", "a$##$b$##$c", None, "a$##$$##$c$##$1"])
    assert valid_spm(s).tolist() == [True, False, False, False]


def test_lake_roundtrip_and_partition_overwrite(tmp_path):
    lake = Lake(tmp_path)
    day1, day2 = 1790812800, 1790812800 + 86400
    b = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u1"],
            "item_id": ["i1", "i2", "i3"],
            "event_type": ["click", "click", "purchase"],
            "ts": [day1 + 5, day1 + 9, day2 + 1],
            "market": ["M"] * 3,
            "paid_amount": [0, 0, 9.5],
        }
    )
    assert lake.write("behavior", b) == 3
    assert lake.partitions() == ["20261001", "20261002"]
    # re-running day 1 replaces it (idempotent), day 2 untouched
    lake.write("behavior", b.iloc[:1])
    out = lake.read("behavior")
    assert len(out) == 2 and out["ts"].dtype == "int64"
    assert len(lake.read("behavior", start_ts=day2)) == 1
    lake.write("behavior", b.iloc[1:2], mode="append")
    assert len(lake.read("behavior")) == 3


def test_quality_gate_flags_bad_data():
    items = pd.DataFrame({"item_id": ["i1", "i2"]})
    users = pd.DataFrame({"user_id": ["u1"], "market": ["M"]})
    b = pd.DataFrame(
        {
            "user_id": ["u1"] * 10,
            "item_id": ["i1"] * 8 + ["zz", "zz"],
            "event_type": ["impression"] * 8 + ["click", "teleport"],
            "ts": [1790812800 + i for i in range(10)],
            "market": ["M"] * 10,
            "spm": ["a$##$b$##$c$##$0"] * 10,
        }
    )
    rep = run_checks({"item": items, "user": users, "behavior": b})
    by = {c.name: c.status for c in rep.checks}
    assert by["unknown item_id"] == "fail" and by["unregistered event_type"] == "fail"
    assert not rep.passed
    assert "Gate: FAILED" in rep.to_markdown()


def test_signing_roundtrip():
    body = b'{"a":1}'
    h = headers("tok", "t1", body)
    assert verify("tok", body, "t1", h["Tenant-Ts"], h["Tenant-Nonce"], h["Tenant-Signature"])
    assert not verify("tok", body + b" ", "t1", h["Tenant-Ts"], h["Tenant-Nonce"], h["Tenant-Signature"])
    assert sign("a", b"b", "c", "d", "e") == sign("a", b"b", "c", "d", "e")


def test_schema_doc_lists_all_tables():
    md = to_markdown()
    for t in ("`user`", "`item`", "`item_market`", "`behavior`", "Event types"):
        assert t in md


def test_env_expansion_in_config(monkeypatch):
    from recsys_core.config import expand_env

    monkeypatch.setenv("RS_TOKEN", "s3cret")
    monkeypatch.delenv("RS_MISSING", raising=False)
    out = expand_env({"a": "${RS_TOKEN}", "b": ["x-${RS_MISSING:-dflt}"], "c": 3, "d": "${RS_MISSING:-}"})
    assert out == {"a": "s3cret", "b": ["x-dflt"], "c": 3, "d": ""}
    import pytest

    with pytest.raises(ValueError, match="RS_MISSING"):
        expand_env({"a": "${RS_MISSING}"})
