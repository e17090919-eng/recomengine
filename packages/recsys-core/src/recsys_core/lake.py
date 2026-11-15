"""Canonical data lake on local disk or any fsspec-style path (S3/OSS later).

Layout:
    <root>/user.parquet, item.parquet, item_market.parquet, category.parquet   (snapshots)
    <root>/behavior/dt=YYYYMMDD/part-*.parquet                               (append log)
    <root>/_meta.json
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from .schema import TABLES, conform

SNAPSHOT_TABLES = ("user", "item", "item_market", "category")


def numpy_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Nullable pandas dtypes -> plain numpy dtypes (what numeric code expects)."""
    out = df.copy()
    for c in out.columns:
        dt = str(out[c].dtype)
        if dt in ("string", "large_string") or dt.startswith("string"):
            out[c] = out[c].astype(object).where(out[c].notna(), None)
        elif dt in ("Int64", "Int32"):
            out[c] = out[c].astype("int64") if not out[c].isna().any() else out[c].astype("float64")
        elif dt in ("Float64", "Float32"):
            out[c] = out[c].astype("float64")
    return out


class Lake:
    def __init__(self, root: Path | str):
        self.root = Path(root)

    # ------------------------------------------------------------ write
    def write(self, table: str, df: pd.DataFrame, mode: str = "overwrite") -> int:
        """Conform to the canonical schema and write. Behavior partitions present in `df`
        are replaced when mode='overwrite' (idempotent re-runs of a backfill day)."""
        if table not in TABLES:
            raise KeyError(f"unknown table {table}")
        res = conform(df, table)
        if res.missing_required:
            raise ValueError(f"{table}: missing required columns {res.missing_required}")
        self.root.mkdir(parents=True, exist_ok=True)
        if table in SNAPSHOT_TABLES:
            res.df.to_parquet(self.root / f"{table}.parquet", index=False)
            return len(res.df)
        out = res.df.copy()
        ts = pd.to_datetime(out["ts"].astype("float64"), unit="s", utc=True)
        out["dt"] = ts.dt.strftime("%Y%m%d")
        out = out[out["dt"].notna()]
        tbl = pa.Table.from_pandas(out, preserve_index=False)
        ds.write_dataset(
            tbl,
            self.root / "behavior",
            format="parquet",
            partitioning=["dt"],
            partitioning_flavor="hive",
            existing_data_behavior="delete_matching" if mode == "overwrite" else "overwrite_or_ignore",
            basename_template=f"part-{uuid.uuid4().hex[:8]}-{{i}}.parquet",
        )
        return len(out)

    # ------------------------------------------------------------ read
    def exists(self, table: str) -> bool:
        if table in SNAPSHOT_TABLES:
            return (self.root / f"{table}.parquet").exists()
        return (self.root / "behavior").exists()

    def read(
        self, table: str, start_ts: float | None = None, end_ts: float | None = None, numpy: bool = True
    ) -> pd.DataFrame:
        if table in SNAPSHOT_TABLES:
            p = self.root / f"{table}.parquet"
            if not p.exists():
                cols = [c.name for c in TABLES[table].columns]
                return pd.DataFrame(columns=cols)
            df = pd.read_parquet(p)
        else:
            dset = ds.dataset(self.root / "behavior", format="parquet", partitioning="hive")
            flt = None
            if start_ts is not None:
                flt = ds.field("ts") >= int(start_ts)
            if end_ts is not None:
                e = ds.field("ts") < int(end_ts)
                flt = e if flt is None else flt & e
            df = dset.to_table(filter=flt).to_pandas()
            df = (
                df.drop(columns=["dt"], errors="ignore")
                .sort_values(["ts", "user_id", "item_id", "event_type"], kind="stable")
                .reset_index(drop=True)
            )
        return numpy_dtypes(df) if numpy else df

    def partitions(self) -> list[str]:
        d = self.root / "behavior"
        return sorted(p.name.removeprefix("dt=") for p in d.glob("dt=*")) if d.exists() else []

    # ------------------------------------------------------------ meta
    def write_meta(self, meta: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    def meta(self) -> dict[str, Any]:
        p = self.root / "_meta.json"
        return json.loads(p.read_text()) if p.exists() else {}
