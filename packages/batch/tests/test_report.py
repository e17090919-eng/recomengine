from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from rec_server.abtest import assign
from rec_server.data import load_config
from recsys_batch.report import ab_readout, attribute, daily_report
from recsys_core.lake import Lake

SPM = "demo$##$home$##$feed$##$"
T0 = int(datetime(2026, 11, 1, 10, tzinfo=ZoneInfo("Asia/Shanghai")).timestamp())


def ev(rid, uid, iid, et, ts, spm=SPM + "0", market="CN-HAINAN", paid=None):
    return dict(
        request_id=rid, user_id=uid, item_id=iid, event_type=et, ts=ts, spm=spm, market=market, paid_amount=paid
    )


def test_last_click_attribution_and_window():
    cfg = load_config()
    b = pd.DataFrame(
        [
            ev("r1", "u", "a", "impression", T0),
            ev("r1", "u", "a", "click", T0 + 10),
            ev("0", "u", "a", "purchase", T0 + 3600, spm="demo$##$cart$##$x$##$0", paid=100.0),  # -> r1
            ev("0", "u", "a", "purchase", T0 + 80 * 3600, paid=100.0),  # > 72h after the click -> organic
            ev("0", "u", "b", "add_cart", T0 + 60),  # never clicked from a rec -> organic
        ]
    )
    log = pd.DataFrame(
        [
            dict(
                request_id="r1",
                user_id="u",
                ts=T0,
                slot="home_feed",
                market="CN-HAINAN",
                variants="x=t",
                items="a",
                degraded="",
            )
        ]
    )
    out, source = attribute(b, log, cfg)
    assert source == "request log"
    assert out["attr_rid"].tolist()[:3] == ["r1", "r1", "r1"]
    assert pd.isna(out["attr_rid"][3]) and pd.isna(out["attr_rid"][4])
    assert out["slot"].tolist()[:3] == ["home_feed"] * 3 and out["variants"][2] == "x=t"


def test_variant_inferred_without_request_log():
    cfg = load_config()
    b = pd.DataFrame([ev("r9", "u42", "a", "impression", T0), ev("r9", "u42", "a", "click", T0 + 5)])
    out, source = attribute(b, pd.DataFrame(), cfg)
    expected = ",".join(f"{a.layer}={a.variant}" for a in assign(cfg["experiments"], "u42", "home_feed"))
    assert source == "inferred (hash)" and set(out["slot"]) == {"home_feed"} and set(out["variants"]) == {expected}


def _synthetic(ctr_c, ctr_t, n_users=3000, imps=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        v = "control" if u % 2 == 0 else "treat"
        p = ctr_c if v == "control" else ctr_t
        for _ in range(imps):
            rows.append(
                dict(
                    user_id=f"u{u}", event_type="impression", variants=f"l={v}", slot="s", market="M", paid_amount=None
                )
            )
            if rng.random() < p:
                rows.append(
                    dict(user_id=f"u{u}", event_type="click", variants=f"l={v}", slot="s", market="M", paid_amount=None)
                )
    return pd.DataFrame(rows)


def test_ab_readout_detects_true_lift_and_not_aa():
    cfg = {
        "experiments": {
            "l": {"slots": ["s"], "variants": [{"name": "control", "traffic": 50}, {"name": "treat", "traffic": 50}]}
        }
    }
    ab, srm = ab_readout(_synthetic(0.10, 0.12), cfg)
    ctr = next(a for a in ab if a["metric"] == "CTR")
    assert ctr["p"] < 0.001 and ctr["ci_low"] < 0.2 < ctr["ci_high"]
    assert srm[0]["p"] > 0.5  # 1500 / 1500
    aa = [
        next(a for a in ab_readout(_synthetic(0.10, 0.10, seed=s), cfg)[0] if a["metric"] == "CTR") for s in range(20)
    ]
    assert sum(a["p"] < 0.05 for a in aa) <= 3  # ~5% false positives


def test_daily_report_from_lake(tmp_path):
    cfg = load_config()
    lake = Lake(tmp_path)
    beh = pd.DataFrame(
        [
            ev("r1", "u1", "a", "impression", T0),
            ev("r1", "u1", "a", "click", T0 + 5),
            ev("0", "u1", "a", "purchase", T0 + 99, paid=250.0),
            ev("r1", "u1", "b", "impression", T0 + 1),
            ev("r0", "u1", "c", "impression", T0 - 86400),  # previous local day: excluded
        ]
    )
    lake.write("behavior", beh)
    lake.write(
        "rec_log",
        pd.DataFrame(
            [
                dict(
                    request_id="r1",
                    user_id="u1",
                    ts=T0,
                    slot="home_feed",
                    market="CN-HAINAN",
                    variants="rank_layer=treatment",
                    items="a|b",
                    degraded="",
                )
            ]
        ),
    )
    rep = daily_report(str(tmp_path), cfg, "20261101")
    r = rep.rows.iloc[0]
    assert (r["impressions"], r["clicks"], r["purchase"], r["gmv"], r["requests"]) == (2, 1, 1, 250.0, 1)
    assert r["variant"] == "treatment" and r["currency"] == "CNY"
    md = rep.to_markdown()
    assert "home_feed" in md and "50.00%" in md
