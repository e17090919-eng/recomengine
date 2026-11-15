"""Event bus: where accepted rows (behavior / item / user) and engine request logs go
after the live state has been updated.

- `LakeBus`   micro-batches straight into the canonical lake (single node, dev, demo)
- `KafkaBus`  one topic per table; `recsys consume` (and ClickHouse) read them
- `MemoryBus` keeps records in memory (tests)
- `NullBus`   drop

Records are canonical rows (see recsys_core.schema). Publishing never blocks serving on
I/O: the lake writer flushes from a background thread, Kafka produces asynchronously.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from recsys_core.config import find_root
from recsys_core.lake import Lake

log = logging.getLogger("recsys.bus")

Record = dict[str, Any]


class Bus(Protocol):
    def publish(self, table: str, records: list[Record]) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


class NullBus:
    def publish(self, table: str, records: list[Record]) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class MemoryBus:
    def __init__(self) -> None:
        self.records: dict[str, list[Record]] = defaultdict(list)

    def publish(self, table: str, records: list[Record]) -> None:
        self.records[table].extend(records)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class LakeWriter:
    """Buffers rows per table and appends them to the lake as parquet micro-batches.
    Snapshot tables (item/user changes) are not rewritten here — they are applied to live
    state and kept in a change log `<lake>/_changes/<table>/` for the next batch."""

    def __init__(self, lake: Lake, flush_rows: int = 5000, flush_seconds: float = 10.0):
        self.lake, self.flush_rows, self.flush_seconds = lake, flush_rows, flush_seconds
        self._buf: dict[str, list[Record]] = defaultdict(list)
        self._lock = threading.Lock()
        self._last = time.monotonic()
        self.written: dict[str, int] = defaultdict(int)

    def add(self, table: str, records: list[Record]) -> bool:
        """Buffer; returns True when a flush is due."""
        with self._lock:
            self._buf[table].extend(records)
            n = sum(len(v) for v in self._buf.values())
        return n >= self.flush_rows or time.monotonic() - self._last >= self.flush_seconds

    def flush(self) -> dict[str, int]:
        with self._lock:
            buf, self._buf = self._buf, defaultdict(list)
            self._last = time.monotonic()
        out: dict[str, int] = {}
        for table, rows in buf.items():
            if not rows:
                continue
            df = pd.DataFrame.from_records(rows)
            if table in ("behavior", "rec_log"):
                n = self.lake.write(table, df, mode="append")
            elif table == "done":
                for r in rows:
                    self.lake.mark_done(str(r["table"]), str(r["date"]))
                n = len(rows)
            else:
                d = self.lake.root / "_changes" / table
                d.mkdir(parents=True, exist_ok=True)
                df.to_parquet(d / f"part-{time.time_ns()}.parquet", index=False)
                n = len(df)
            out[table] = n
            self.written[table] += n
        return out


class LakeBus:
    def __init__(self, lake: Lake, flush_rows: int = 5000, flush_seconds: float = 10.0):
        self.writer = LakeWriter(lake, flush_rows, flush_seconds)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="lake-bus", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.wait(min(self.writer.flush_seconds, 1.0)):
            if time.monotonic() - self.writer._last >= self.writer.flush_seconds:
                self._safe_flush()

    def _safe_flush(self) -> None:
        try:
            self.writer.flush()
        except Exception:  # never take serving down because the lake is unavailable
            log.exception("lake flush failed")

    def publish(self, table: str, records: list[Record]) -> None:
        if records and self.writer.add(table, records):
            threading.Thread(target=self._safe_flush, daemon=True).start()

    def flush(self) -> None:
        self._safe_flush()

    def close(self) -> None:
        self._stop.set()
        self._safe_flush()


class KafkaBus:
    def __init__(self, bootstrap: str, topics: dict[str, str], producer: Any = None):
        if producer is None:
            from confluent_kafka import Producer

            producer = Producer({"bootstrap.servers": bootstrap, "linger.ms": 20, "compression.type": "lz4"})
        self.p, self.topics = producer, topics

    def publish(self, table: str, records: list[Record]) -> None:
        topic = self.topics.get(table) or f"recsys-{table}"
        for r in records:
            key = str(r.get("user_id") or r.get("item_id") or "")
            self.p.produce(topic, json.dumps(r, separators=(",", ":"), default=str).encode(), key=key.encode())
        self.p.poll(0)

    def flush(self) -> None:
        self.p.flush(5)

    def close(self) -> None:
        self.flush()


def make_bus(cfg: dict, root: Path | None = None) -> Bus:
    ic = cfg.get("ingest", {}) or {}
    kind = ic.get("bus", "none")
    if kind == "lake":
        p = Path(ic.get("lake_dir", "lake"))
        lake = Lake(p if p.is_absolute() else (root or find_root()) / p)
        return LakeBus(lake, ic.get("flush_rows", 5000), ic.get("flush_seconds", 10.0))
    if kind == "kafka":
        return KafkaBus(ic["kafka_bootstrap"], ic["topics"])
    return NullBus()
