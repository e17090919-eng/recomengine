"""Canonical data contract. Every customer's data is mapped into these tables
(by the backfill tool or the real-time ingest API) before anything else reads it."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import pandas as pd

DType = Literal["string", "int64", "float64"]

SPM_RE = re.compile(r"^[^$]+\$##\$[^$]+\$##\$[^$]+\$##\$[^$]+$")


@dataclass(frozen=True)
class Column:
    name: str
    dtype: DType
    required: bool
    doc: str
    doc_zh: str


@dataclass(frozen=True)
class Table:
    name: str
    key: tuple[str, ...]
    columns: tuple[Column, ...]
    doc: str
    pattern_columns: tuple[tuple[str, DType, str], ...] = ()  # (regex, dtype, doc) e.g. title_<lang>
    min_pattern: int = 0  # how many pattern columns must exist

    def column(self, name: str) -> Column | None:
        return next((c for c in self.columns if c.name == name), None)

    @property
    def required(self) -> list[str]:
        return [c.name for c in self.columns if c.required]


C = Column
TABLES: dict[str, Table] = {
    "user": Table(
        "user",
        ("user_id",),
        (
            C("user_id", "string", True, "Stable member id (or device id for guests)", "会员 ID（访客用设备 ID）"),
            C("market", "string", True, "Home market code, e.g. CN-HAINAN", "所属市场代码"),
            C("country", "string", False, "ISO-3166 alpha-2 nationality", "国籍（ISO 两位码）"),
            C("language", "string", False, "Preferred language (BCP-47 primary tag)", "偏好语言"),
            C("member_level", "int64", False, "Loyalty tier, 0 = base", "会员等级，0 为普卡"),
            C("register_ts", "int64", False, "Registration time, unix seconds UTC", "注册时间（UTC 秒）"),
            C(
                "allowance_used",
                "float64",
                False,
                "Duty-free allowance used this period (local currency)",
                "本期已用免税额度（本币）",
            ),
        ),
        "One row per shopper.",
    ),
    "item": Table(
        "item",
        ("item_id",),
        (
            C("item_id", "string", True, "SPU id", "商品 SPU ID"),
            C("category", "string", True, "Leaf category code", "类目代码"),
            C("brand", "string", True, "Brand name or id", "品牌"),
            C("price_usd", "float64", True, "Reference price in USD (for cross-market features)", "参考价（美元）"),
            C("publish_ts", "int64", True, "First on-shelf time, unix seconds UTC", "首次上架时间（UTC 秒）"),
            C("image", "string", False, "Image URL (or placeholder)", "图片 URL"),
        ),
        "One row per sellable product (SPU), market-independent.",
        pattern_columns=((r"^title_[a-z]{2}$", "string", "Title in a language, e.g. title_en, title_zh"),),
        min_pattern=1,
    ),
    "item_market": Table(
        "item_market",
        ("item_id", "market"),
        (
            C("item_id", "string", True, "SPU id", "商品 SPU ID"),
            C("market", "string", True, "Market code", "市场代码"),
            C("price", "float64", True, "Selling price in local currency", "本币售价"),
            C("currency", "string", True, "ISO-4217 code", "币种"),
            C("stock", "int64", True, "Sellable stock", "可售库存"),
            C("channels", "string", False, "Pickup/delivery channels, '|' separated", "提货/配送渠道，竖线分隔"),
        ),
        "Availability, price and stock of an item in a market.",
    ),
    "category": Table(
        "category",
        ("category",),
        (
            C("category", "string", True, "Category code", "类目代码"),
            C("image", "string", False, "Placeholder image / icon", "类目图标"),
        ),
        "Optional category labels.",
        pattern_columns=((r"^label_[a-z]{2}$", "string", "Display label in a language, e.g. label_zh"),),
    ),
    "behavior": Table(
        "behavior",
        (),
        (
            C(
                "request_id",
                "string",
                False,
                "Recommendation request id; '0' for organic traffic",
                "推荐请求 ID，自然流量为 0",
            ),
            C("user_id", "string", True, "Shopper id", "用户 ID"),
            C("item_id", "string", True, "SPU id", "商品 ID"),
            C("event_type", "string", True, "One of the registered event types", "已注册的行为类型"),
            C("ts", "int64", True, "Event time, unix seconds UTC", "行为时间（UTC 秒）"),
            C("spm", "string", False, "app$##$page$##$block$##$position", "坑位 SPM"),
            C("market", "string", True, "Market where the event happened", "发生的市场"),
            C("parent_item_id", "string", False, "Seed item for related-item scenes", "相关推荐的种子商品"),
            C("paid_amount", "float64", False, "Amount paid (purchase events, local currency)", "支付金额（本币）"),
        ),
        "Append-only event log, partitioned by UTC day (dt=YYYYMMDD).",
    ),
    "rec_log": Table(
        "rec_log",
        (),
        (
            C("request_id", "string", True, "Recommendation request id", "推荐请求 ID"),
            C("user_id", "string", True, "Shopper id", "用户 ID"),
            C("ts", "int64", True, "Serving time, unix seconds UTC", "下发时间（UTC 秒）"),
            C("slot", "string", True, "Slot that served the request", "栏位"),
            C("market", "string", True, "Market", "市场"),
            C("variants", "string", False, "A/B assignment, 'layer=variant' joined by ','", "实验分组"),
            C("items", "string", True, "Returned item ids in order, '|' separated", "下发的商品（有序，竖线分隔）"),
            C("degraded", "string", False, "Degradation reasons, ',' separated; empty = none", "降级原因"),
        ),
        "Written by the engine for every request: joins behaviors to slots/variants for attribution.",
    ),
}
PARTITIONED = ("behavior", "rec_log")

EVENT_TYPES: dict[str, str] = {
    "impression": "Item visible ≥50% for ≥1s (client-side, deduplicated per request)",
    "click": "Opened item detail",
    "favorite": "Added to wishlist",
    "add_cart": "Added to cart",
    "order_submit": "Order placed (not yet paid)",
    "purchase": "Payment succeeded",
    "dislike": "Explicit negative feedback ('not interested')",
    "share": "Shared item",
    "search": "Search result click (item_id = clicked item)",
}


@dataclass
class ConformResult:
    df: pd.DataFrame
    missing_required: list[str]
    type_errors: dict[str, int]  # column -> rows that failed coercion
    dropped_columns: list[str]


def conform(df: pd.DataFrame, table: str) -> ConformResult:
    """Project a frame onto the canonical schema: keep known columns, add missing optional
    ones as null, coerce dtypes (counting failures)."""
    t = TABLES[table]
    out = pd.DataFrame(index=df.index)
    type_errors: dict[str, int] = {}
    cols: list[tuple[str, DType]] = [(c.name, c.dtype) for c in t.columns]
    for regex, dt, _ in t.pattern_columns:
        cols += [(c, dt) for c in df.columns if re.match(regex, str(c))]
    for name, dtype in cols:
        if name not in df.columns:
            out[name] = pd.Series(pd.NA, index=df.index, dtype=_pd_dtype(dtype))
            continue
        s = df[name]
        if dtype == "string":
            out[name] = s.astype("string")
        else:
            num = pd.to_numeric(s, errors="coerce")
            bad = int((num.isna() & s.notna()).sum())
            if bad:
                type_errors[name] = bad
            if dtype == "int64":
                frac = num.notna() & (num != num.round())
                if frac.any():
                    type_errors[name] = type_errors.get(name, 0) + int(frac.sum())
                    num = num.where(~frac)
                out[name] = num.round().astype("Int64")
            else:
                out[name] = num.astype("Float64")
    missing = [c for c in t.required if c not in df.columns]
    if (
        t.min_pattern
        and sum(any(re.match(rx, str(c)) for rx, _, _ in t.pattern_columns) for c in df.columns) < t.min_pattern
    ):
        missing.append(" | ".join(rx for rx, _, _ in t.pattern_columns))
    known = {n for n, _ in cols}
    dropped = [str(c) for c in df.columns if c not in known and c != "dt"]
    return ConformResult(out, missing, type_errors, dropped)


def _pd_dtype(d: DType) -> str:
    return {"string": "string", "int64": "Int64", "float64": "Float64"}[d]


def valid_spm(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.match(SPM_RE.pattern)


def to_markdown() -> str:
    """Render the data contract (docs/data-contract.md is generated from this)."""
    lines = [
        "# Data contract — canonical tables",
        "",
        "_Generated by `recsys schema-doc` from `recsys_core.schema`; do not edit by hand._",
        "",
        "All timestamps are **unix seconds, UTC**. Prices are in the market's local currency except "
        "`item.price_usd`. Customer-specific fields are mapped into these tables by the backfill tool "
        "(`configs/mappings/*.toml`) or the real-time `write` API.",
        "",
    ]
    for t in TABLES.values():
        key = ", ".join(t.key) if t.key else "—"
        lines += [
            f"## `{t.name}`",
            "",
            t.doc,
            "",
            f"Primary key: `{key}`",
            "",
            "| column | type | required | description | 说明 |",
            "|---|---|---|---|---|",
        ]
        for c in t.columns:
            lines.append(f"| `{c.name}` | {c.dtype} | {'✔' if c.required else ''} | {c.doc} | {c.doc_zh} |")
        for rx, dt, doc in t.pattern_columns:
            req = f"≥{t.min_pattern}" if t.min_pattern else ""
            lines.append(f"| `{rx}` | {dt} | {req} | {doc} | 多语言列 |")
        lines.append("")
    lines += ["## Event types", "", "| event_type | meaning |", "|---|---|"]
    lines += [f"| `{k}` | {v} |" for k, v in EVENT_TYPES.items()]
    lines += ["", "Unregistered event types are rejected by ingest and counted by the quality gate.", ""]
    return "\n".join(lines)
