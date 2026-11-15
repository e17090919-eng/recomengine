"""Kafka -> lake consumer (production path when [ingest] bus = "kafka").

Reads the behavior / item / user / rec_log / done topics, micro-batches them into the
lake with the same writer the single-node LakeBus uses, and commits offsets only after a
successful flush (at-least-once; the daily batch de-duplicates behaviors)."""

from __future__ import annotations

import json
import signal
import time
from pathlib import Path
from typing import Any

from rec_server.bus import LakeWriter
from recsys_core.lake import Lake


def consume(cfg: dict, lake_dir: Path, group: str, max_seconds: float | None = None, consumer: Any = None) -> int:
    ic = cfg["ingest"]
    topics: dict[str, str] = {**ic["topics"], "done": ic["topics"].get("done", "recsys-done")}
    table_of = {v: k for k, v in topics.items()}
    if consumer is None:
        from confluent_kafka import Consumer

        consumer = Consumer(
            {
                "bootstrap.servers": ic["kafka_bootstrap"],
                "group.id": group,
                "enable.auto.commit": False,
                "auto.offset.reset": "earliest",
            }
        )
    consumer.subscribe(list(topics.values()))
    writer = LakeWriter(Lake(lake_dir), ic["flush_rows"], ic["flush_seconds"])
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    t0, pending, bad = time.monotonic(), 0, 0
    try:
        while not stop["flag"] and (max_seconds is None or time.monotonic() - t0 < max_seconds):
            msg = consumer.poll(0.5)
            due = False
            if msg is not None:
                if msg.error():
                    print(f"[consume] kafka error: {msg.error()}")
                    continue
                try:
                    rec = json.loads(msg.value())
                except ValueError:
                    bad += 1
                    continue
                due = writer.add(table_of[msg.topic()], [rec])
                pending += 1
            elif pending and time.monotonic() - writer._last >= writer.flush_seconds:
                due = True
            if due and pending:
                out = writer.flush()
                consumer.commit(asynchronous=False)
                print(f"[consume] flushed {out} (skipped {bad} malformed)")
                pending = 0
    finally:
        if pending:
            writer.flush()
            consumer.commit(asynchronous=False)
        consumer.close()
    return 0
