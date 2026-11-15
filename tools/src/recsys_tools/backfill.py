"""Historical backfill: customer raw exports -> canonical lake, driven by a mapping file.

Mapping (TOML), one section per canonical table:

    [tables.behavior]
    path = "events/*.jsonl"          # glob relative to --raw
    format = "jsonl"                 # jsonl | csv | parquet
    [tables.behavior.columns]
    user_id = "distinct_id"                                  # plain source field (dotted = nested JSON)
    event_type = { field = "event", map = { ProductClick = "click" }, drop_unmapped = true }
    ts = { field = "time", time_format = "%Y-%m-%d %H:%M:%S", tz = "Asia/Shanghai" }
    spm = { template = "app$##${properties.page}$##${properties.module}$##${properties.pos}" }
    request_id = { field = "properties.rid", default = "0" }
    channels = { field = "delivery", replace = { "," = "|" } }
    market = { const = "CN-HAINAN" }
    dedupe = true                    # (table level) drop exact duplicate rows after mapping

time_format: unix_s | unix_ms | iso | any strftime pattern (+ tz for naive local times).
"""

from __future__ import annotations

import glob
import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from recsys_core.config import QualityCfg
from recsys_core.lake import Lake
from recsys_core.quality import QualityReport, run_checks
from recsys_core.schema import TABLES, conform

ORDER = ("category", "item", "item_market", "user", "behavior")
_PLACEHOLDER = re.compile(r"\{([^{}]+)\}")


@dataclass
class TableStats:
    files: int = 0
    rows_in: int = 0
    rows_out: int = 0
    dropped: dict[str, int] = field(default_factory=dict)  # reason -> rows
    unmapped: dict[str, dict[str, int]] = field(default_factory=dict)  # column -> value -> rows
    type_errors: dict[str, int] = field(default_factory=dict)


@dataclass
class BackfillResult:
    stats: dict[str, TableStats]
    quality: QualityReport
    partitions: list[str]

    def to_markdown(self) -> str:
        lines = [
            "# Backfill report",
            "",
            "| table | files | rows in | rows written | dropped | type errors |",
            "|---|---|---|---|---|---|",
        ]
        for name, s in self.stats.items():
            dropped = ", ".join(f"{k}: {v:,}" for k, v in s.dropped.items()) or "—"
            te = ", ".join(f"{k}: {v:,}" for k, v in s.type_errors.items()) or "—"
            lines.append(f"| {name} | {s.files} | {s.rows_in:,} | {s.rows_out:,} | {dropped} | {te} |")
        um = [(t, c, v, n) for t, s in self.stats.items() for c, vals in s.unmapped.items() for v, n in vals.items()]
        if um:
            lines += [
                "",
                "**Unmapped source values** (extend the mapping if these matter):",
                "",
                "| table | column | source value | rows |",
                "|---|---|---|---|",
            ]
            lines += [f"| {t} | {c} | `{v}` | {n:,} |" for t, c, v, n in sorted(um, key=lambda x: -x[3])[:30]]
        lines += [
            "",
            f"Behavior partitions written: {len(self.partitions)} "
            f"({self.partitions[0] if self.partitions else '—'} → {self.partitions[-1] if self.partitions else '—'})",
            "",
            self.quality.to_markdown().replace("# Data quality report", "## Data quality"),
        ]
        return "\n".join(lines)


def load_mapping(path: Path | str) -> dict[str, Any]:
    with open(path, "rb") as f:
        m = tomllib.load(f)
    unknown = set(m.get("tables", {})) - set(TABLES)
    if unknown:
        raise ValueError(f"mapping has unknown tables: {sorted(unknown)}")
    for t, spec in m.get("tables", {}).items():
        if "path" not in spec or "columns" not in spec:
            raise ValueError(f"tables.{t} needs 'path' and 'columns'")
    return m


def read_source(path: str, fmt: str) -> pd.DataFrame:
    if fmt == "csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    if fmt == "parquet":
        return pd.read_parquet(path)
    if fmt == "jsonl":
        with open(path, encoding="utf-8") as f:
            recs = [json.loads(line) for line in f if line.strip()]
        return pd.json_normalize(recs)
    raise ValueError(f"unknown format {fmt}")


def _parse_time(s: pd.Series, fmt: str, tz: str | None) -> pd.Series:
    if fmt in ("unix_s", "unix_ms"):
        n = pd.to_numeric(s, errors="coerce")
        return (n / 1000 if fmt == "unix_ms" else n).round()
    if fmt == "iso":
        t = pd.to_datetime(s, errors="coerce", utc=tz is None, format="ISO8601")
    else:
        t = pd.to_datetime(s, errors="coerce", format=fmt)
    if tz is not None:
        if t.dt.tz is None:
            t = t.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
        t = t.dt.tz_convert("UTC")
    elif t.dt.tz is None:
        t = t.dt.tz_localize("UTC")
    secs = (t - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    return secs.astype("Float64")


def apply_column(src: pd.DataFrame, spec: Any, stats: TableStats, col: str) -> tuple[pd.Series, pd.Series | None]:
    """Returns (values, drop_mask or None)."""
    if isinstance(spec, str):
        spec = {"field": spec}
    if "const" in spec:
        s = pd.Series(spec["const"], index=src.index)
    elif "template" in spec:
        parts = _PLACEHOLDER.split(spec["template"])  # literal, field, literal, field, ...
        s = pd.Series("", index=src.index, dtype=object)
        ok = pd.Series(True, index=src.index)
        for n, part in enumerate(parts):
            if n % 2 == 0:
                s = s + part
                continue
            v = src[part] if part in src.columns else pd.Series(pd.NA, index=src.index)
            ok &= v.notna() & (v.astype(str) != "")
            s = s + v.astype(str).str.replace(r"\.0$", "", regex=True)
        s = s.where(ok)
    else:
        f = spec["field"]
        s = src[f] if f in src.columns else pd.Series(pd.NA, index=src.index, dtype=object)
    drop = None
    if "map" in spec:
        mapping = {str(k): v for k, v in spec["map"].items()}
        key = s.astype(str)
        hit = key.isin(mapping)
        miss = s.notna() & ~hit
        if miss.any():
            stats.unmapped.setdefault(col, {})
            for val, cnt in key[miss].value_counts().items():
                stats.unmapped[col][str(val)] = stats.unmapped[col].get(str(val), 0) + int(cnt)
        mapped = key.map(mapping)
        if spec.get("drop_unmapped"):
            drop = ~hit
            s = mapped
        else:
            s = mapped.where(hit, s)
    if "replace" in spec:
        s = s.astype("string")
        for a, b in spec["replace"].items():
            s = s.str.replace(a, b, regex=False)
    if spec.get("lower"):
        s = s.astype("string").str.lower()
    if spec.get("upper"):
        s = s.astype("string").str.upper()
    if "time_format" in spec:
        s = _parse_time(s, spec["time_format"], spec.get("tz"))
    if "scale" in spec:
        s = pd.to_numeric(s, errors="coerce") * float(spec["scale"])
    if "default" in spec:
        s = s.where(s.notna() & (s.astype(str) != ""), spec["default"])
    return s, drop


def map_frame(src: pd.DataFrame, spec: dict[str, Any], table: str, stats: TableStats) -> pd.DataFrame:
    out = pd.DataFrame(index=src.index)
    drop = pd.Series(False, index=src.index)
    for col, cspec in spec["columns"].items():
        out[col], d = apply_column(src, cspec, stats, col)
        if d is not None:
            n = int((d & ~drop).sum())
            if n:
                stats.dropped[f"unmapped {col}"] = stats.dropped.get(f"unmapped {col}", 0) + n
            drop |= d
    out = out[~drop]
    if spec.get("dedupe"):
        dup = out.duplicated(keep="first")
        if dup.any():
            stats.dropped["duplicate"] = stats.dropped.get("duplicate", 0) + int(dup.sum())
            out = out[~dup]
    req = [c for c in TABLES[table].required if c in out.columns]
    missing_req = out[req].isna().any(axis=1) if req else pd.Series(False, index=out.index)
    if missing_req.any():
        stats.dropped["missing required value"] = stats.dropped.get("missing required value", 0) + int(
            missing_req.sum()
        )
        out = out[~missing_req]
    return out


def run_backfill(
    mapping_path: Path | str,
    raw_dir: Path | str,
    lake_dir: Path | str,
    quality_cfg: QualityCfg | None = None,
    log: Any = print,
) -> BackfillResult:
    mapping = load_mapping(mapping_path)
    lake = Lake(lake_dir)
    stats: dict[str, TableStats] = {}
    written: dict[str, pd.DataFrame] = {}
    seen_dt: set[str] = set()
    for table in ORDER:
        spec = mapping.get("tables", {}).get(table)
        if not spec:
            continue
        st = stats[table] = TableStats()
        files = sorted(glob.glob(str(Path(raw_dir) / spec["path"])))
        if not files:
            raise FileNotFoundError(f"tables.{table}: no files match {spec['path']} under {raw_dir}")
        frames = []
        for fp in files:
            src = read_source(fp, spec.get("format", Path(fp).suffix.lstrip(".")))
            st.files += 1
            st.rows_in += len(src)
            mapped = map_frame(src, spec, table, st)
            res = conform(mapped, table)
            for k, v in res.type_errors.items():
                st.type_errors[k] = st.type_errors.get(k, 0) + v
            if table == "behavior":
                # idempotent: first time we touch a day partition in this run we replace it, then append
                dts = pd.to_datetime(res.df["ts"].astype("float64"), unit="s", utc=True).dt.strftime("%Y%m%d")
                new = ~dts.isin(seen_dt)
                if new.any():
                    st.rows_out += lake.write(table, mapped[new.to_numpy()], mode="overwrite")
                if (~new).any():
                    st.rows_out += lake.write(table, mapped[(~new).to_numpy()], mode="append")
                seen_dt |= set(dts.dropna())
                log(f"[backfill] {table}: {Path(fp).name} {len(src):,} -> {len(mapped):,}")
            else:
                frames.append(mapped)
        if table != "behavior":
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            st.rows_out = lake.write(table, df)
            log(f"[backfill] {table}: {st.rows_in:,} -> {st.rows_out:,}")
        written[table] = lake.read(table, numpy=False)
    meta = {"source": f"backfill:{Path(mapping_path).name}", **mapping.get("meta", {})}
    lake.write_meta(meta)
    q = run_checks(
        {k: v for k, v in written.items() if k != "category"},
        quality_cfg,
        type_errors={k: s.type_errors for k, s in stats.items()},
    )
    return BackfillResult(stats, q, lake.partitions())
