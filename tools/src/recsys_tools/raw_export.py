"""Turn simulated canonical data into a *vendor-style raw export* (the kind of files a
customer actually hands over: analytics-SDK event JSONL in local time, ERP/CRM CSVs
with their own column names and code values, plus realistic dirt). Used to exercise
and test the backfill tool before real customer data arrives."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .simulator import SimResult

EVENT_NAMES = {
    "impression": "ProductExposure",
    "click": "ProductClick",
    "favorite": "AddWishlist",
    "add_cart": "AddToCart",
    "purchase": "PayOrderSuccess",
}
SITE = {"CN-HAINAN": "HN", "NZ-AKL": "AKL"}
GRADES = ["普卡", "金卡", "铂金卡", "黑卡"]
TZ = "Asia/Shanghai"  # the analytics project records local Beijing time


def export_raw(sim: SimResult, raw_dir: Path, seed: int = 0, dirt: bool = True) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    raw = Path(raw_dir)
    (raw / "events").mkdir(parents=True, exist_ok=True)
    t = sim.tables

    u = t["user"]
    pd.DataFrame(
        {
            "member_no": u["user_id"],
            "site": u["market"].map(SITE),
            "nationality": u["country"],
            "lang": u["language"],
            "grade": u["member_level"].map(lambda x: GRADES[int(x)]),
            "reg_date": pd.to_datetime(u["register_ts"], unit="s", utc=True).dt.tz_convert(TZ).dt.strftime("%Y-%m-%d"),
            "used_quota": u["allowance_used"],
        }
    ).to_csv(raw / "members.csv", index=False)

    it = t["item"]
    pd.DataFrame(
        {
            "sku_code": it["item_id"],
            "name_cn": it["title_zh"],
            "name_en": it["title_en"],
            "cat_code": it["category"],
            "brand_name": it["brand"],
            "ref_price_usd": it["price_usd"],
            "online_time": pd.to_datetime(it["publish_ts"], unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pic": it["image"],
        }
    ).to_csv(raw / "products.csv", index=False)

    im = t["item_market"]
    pd.DataFrame(
        {
            "sku_code": im["item_id"],
            "site": im["market"].map(SITE),
            "sale_price": im["price"],
            "currency": im["currency"],
            "stock_qty": im["stock"],
            "delivery": im["channels"].str.replace("|", ","),
        }
    ).to_csv(raw / "store_prices.csv", index=False)

    c = t["category"]
    pd.DataFrame(
        {"cat_code": c["category"], "cat_name_cn": c["label_zh"], "cat_name_en": c["label_en"], "icon": c["image"]}
    ).to_csv(raw / "categories.csv", index=False)

    b = t["behavior"].copy()
    spm = b["spm"].str.split(r"\$##\$", regex=True, expand=True)
    local = pd.to_datetime(b["ts"], unit="s", utc=True).dt.tz_convert(TZ)
    recs = pd.DataFrame(
        {
            "distinct_id": b["user_id"],
            "event": b["event_type"].map(EVENT_NAMES),
            "time": local.dt.strftime("%Y-%m-%d %H:%M:%S"),
            "day": local.dt.strftime("%Y-%m-%d"),
            "product_id": b["item_id"],
            "page_type": spm[1],
            "module": spm[2],
            "position": spm[3],
            "rec_request_id": b["request_id"],
            "parent_product_id": b["parent_item_id"].replace("", None),
            "order_amount": b["paid_amount"].where(b["event_type"] == "purchase"),
            "site": b["market"].map(SITE),
        }
    )
    n_dirty = 0
    if dirt:
        n = len(recs)
        extra = recs.sample(n=n // 100, random_state=seed).copy()  # 1% app-start noise events
        extra["event"], extra["product_id"] = "$AppStart", None
        unknown = recs.sample(n=n // 200, random_state=seed + 1).index  # 0.5% unknown SKUs
        recs.loc[unknown, "product_id"] = "SKU-UNKNOWN-" + pd.Series(
            rng.integers(0, 50, len(unknown)), index=unknown
        ).astype(str)
        nopos = recs.sample(n=n // 500, random_state=seed + 2).index  # 0.2% missing position
        recs.loc[nopos, "position"] = None
        dup = recs.sample(n=n // 300, random_state=seed + 3)  # 0.33% duplicated lines
        recs = pd.concat([recs, extra, dup]).sort_values("time", kind="stable")
        n_dirty = len(extra) + len(unknown) + len(nopos) + len(dup)

    for day, g in recs.groupby("day"):
        with open(raw / "events" / f"{day}.jsonl", "w", encoding="utf-8") as f:
            for r in g.itertuples(index=False):
                props = {
                    k: v
                    for k, v in (
                        ("product_id", r.product_id),
                        ("page_type", r.page_type),
                        ("module", r.module),
                        ("position", r.position),
                        ("rec_request_id", r.rec_request_id),
                        ("parent_product_id", r.parent_product_id),
                        ("order_amount", r.order_amount),
                        ("site", r.site),
                    )
                    if v is not None and not (isinstance(v, float) and np.isnan(v))
                }
                f.write(
                    json.dumps(
                        {"distinct_id": r.distinct_id, "event": r.event, "time": r.time, "properties": props},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    sim.truth.save(raw / "_truth.npz")
    return {"event_lines": len(recs), "dirty_rows": n_dirty, "event_files": recs["day"].nunique()}
