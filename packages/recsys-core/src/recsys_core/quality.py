"""Data-quality gate. Runs on canonical tables (after mapping, before training).
A `fail` blocks training/publishing; `warn` is reported for humans to look at."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from .config import QualityCfg
from .schema import EVENT_TYPES, TABLES, valid_spm

Status = Literal["pass", "warn", "fail"]
EARLIEST_TS = 1420070400  # 2015-01-01


@dataclass
class Check:
    table: str
    name: str
    status: Status
    value: float
    threshold: str
    detail: str = ""


@dataclass
class QualityReport:
    checks: list[Check] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not any(c.status == "fail" for c in self.checks)

    def add(
        self,
        table: str,
        name: str,
        value: float,
        *,
        fail_above: float | None = None,
        warn_above: float | None = None,
        detail: str = "",
    ) -> None:
        status: Status = "pass"
        thr = []
        if warn_above is not None:
            thr.append(f"warn>{warn_above:g}")
            if value > warn_above:
                status = "warn"
        if fail_above is not None:
            thr.append(f"fail>{fail_above:g}")
            if value > fail_above:
                status = "fail"
        self.checks.append(Check(table, name, status, float(value), ", ".join(thr), detail))

    def to_markdown(self) -> str:
        icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}
        n = {s: sum(c.status == s for c in self.checks) for s in ("pass", "warn", "fail")}
        lines = [
            "# Data quality report",
            "",
            f"**Gate: {'PASSED' if self.passed else 'FAILED'}** — "
            f"{n['pass']} pass · {n['warn']} warn · {n['fail']} fail",
            "",
            "## Summary",
            "",
        ]
        lines += [f"- **{k}**: {v}" for k, v in self.stats.items()]
        lines += ["", "## Checks", "", "| | table | check | value | threshold | detail |", "|---|---|---|---|---|---|"]
        for c in self.checks:
            v = f"{c.value:.4f}" if c.value < 1 else f"{c.value:,.0f}"
            lines.append(f"| {icon[c.status]} | {c.table} | {c.name} | {v} | {c.threshold} | {c.detail} |")
        return "\n".join(lines) + "\n"


def _rate(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) else 0.0


def run_checks(
    tables: dict[str, pd.DataFrame],
    cfg: QualityCfg | None = None,
    type_errors: dict[str, dict[str, int]] | None = None,
    now: float | None = None,
) -> QualityReport:
    cfg = cfg or QualityCfg()
    now = now or time.time()
    r = QualityReport()

    # --- generic per-table checks
    for name, df in tables.items():
        t = TABLES[name]
        r.stats[f"{name} rows"] = f"{len(df):,}"
        missing = [c for c in t.required if c not in df.columns]
        r.add(name, "required columns present", len(missing), fail_above=0, detail=", ".join(missing))
        for c in t.required:
            if c in df.columns and len(df):
                r.add(
                    name,
                    f"null rate: {c}",
                    _rate(df[c].isna() | (df[c].astype(str) == "")),
                    fail_above=cfg.max_required_null_rate,
                )
        for col, bad in (type_errors or {}).get(name, {}).items():
            r.add(
                name,
                f"type errors: {col}",
                bad / max(len(df), 1),
                fail_above=cfg.max_type_error_rate,
                detail=f"{bad:,} rows",
            )
        if t.key and all(k in df.columns for k in t.key) and len(df):
            r.add(name, "duplicate primary keys", _rate(df.duplicated(list(t.key))), warn_above=0, fail_above=0.01)

    users, items, im, b = (tables.get(k) for k in ("user", "item", "item_market", "behavior"))

    if im is not None and len(im):
        r.add(
            "item_market",
            "price <= 0 or missing",
            _rate(~(pd.to_numeric(im["price"], errors="coerce") > 0)),
            fail_above=0.01,
        )
        if items is not None:
            r.add("item_market", "unknown item_id", _rate(~im["item_id"].isin(items["item_id"])), fail_above=0.01)

    if b is not None and len(b):
        r.stats["behavior period (UTC)"] = (
            f"{pd.to_datetime(b['ts'].min(), unit='s'):%Y-%m-%d} → {pd.to_datetime(b['ts'].max(), unit='s'):%Y-%m-%d}"
        )
        counts = b["event_type"].value_counts()
        r.stats["events by type"] = ", ".join(f"{k}={v:,}" for k, v in counts.items())
        r.add(
            "behavior",
            "unregistered event_type",
            _rate(~b["event_type"].isin(list(EVENT_TYPES))),
            warn_above=0,
            fail_above=cfg.max_unregistered_event_rate,
            detail=", ".join(sorted(set(b["event_type"].dropna()) - set(EVENT_TYPES)))[:120],
        )
        if items is not None:
            r.add(
                "behavior",
                "unknown item_id",
                _rate(~b["item_id"].isin(items["item_id"])),
                warn_above=0.001,
                fail_above=cfg.max_unknown_item_rate,
            )
        if users is not None:
            r.add(
                "behavior",
                "user_id not in user table",
                _rate(~b["user_id"].isin(users["user_id"])),
                warn_above=cfg.max_unknown_user_rate,
            )
        if "spm" in b.columns:
            r.add("behavior", "invalid spm", _rate(~valid_spm(b["spm"])), warn_above=cfg.max_invalid_spm_rate)
        ts = pd.to_numeric(b["ts"], errors="coerce")
        r.add(
            "behavior",
            "timestamp out of range",
            _rate((ts < EARLIEST_TS) | (ts > now + 86400) | ts.isna()),
            fail_above=cfg.max_bad_ts_rate,
        )
        r.add(
            "behavior",
            "duplicate events",
            _rate(b.duplicated(["user_id", "item_id", "event_type", "ts"])),
            warn_above=cfg.max_duplicate_event_rate,
        )
        imp, clk = int(counts.get("impression", 0)), int(counts.get("click", 0))
        if imp:
            ctr = clk / imp
            lo, hi = cfg.ctr_range
            r.checks.append(
                Check(
                    "behavior",
                    "overall CTR (click/impression)",
                    "pass" if lo <= ctr <= hi else "warn",
                    ctr,
                    f"expected {lo:g}–{hi:g}",
                )
            )
        else:
            r.add(
                "behavior",
                "impressions present",
                1,
                warn_above=0,
                detail="no impression events: CTR models cannot be trained; ask for exposure logging",
            )
        if "purchase" in counts and "paid_amount" in b.columns:
            p = b[b["event_type"] == "purchase"]
            r.add(
                "behavior",
                "purchase without paid_amount",
                _rate(~(pd.to_numeric(p["paid_amount"], errors="coerce") > 0)),
                warn_above=0.05,
            )
        if "request_id" in b.columns:
            rec = b[b["event_type"] == "click"]
            r.stats["clicks attributable to a request"] = (
                f"{_rate(rec['request_id'].notna() & (rec['request_id'].astype(str) != '0')):.1%}"
            )
        day = pd.to_datetime(ts, unit="s").dt.floor("D")
        daily = b.groupby(day).size().sort_index()
        if len(daily) >= 3:
            chg = (daily.pct_change().abs()).iloc[1:-1]  # first/last day are often partial
            worst = chg.max() if len(chg) else 0.0
            bad_days = [f"{d:%m-%d}" for d, v in chg.items() if v > cfg.max_daily_volume_change]
            r.add(
                "behavior",
                "max day-over-day volume change",
                float(worst),
                warn_above=cfg.max_daily_volume_change,
                detail=" ".join(bad_days[:10]),
            )
        r.stats["distinct users / items in behavior"] = f"{b['user_id'].nunique():,} / {b['item_id'].nunique():,}"
    return r
