"""Real-time ingestion: validate rows from the data API, apply them to live state, and
forward canonical rows to the bus (lake / Kafka) for the next batch.

Per-row validation errors are returned to the caller (index + reason); valid rows of
the same request are still accepted, as in the Volcengine write API.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from recsys_core.quality import EARLIEST_TS
from recsys_core.schema import EVENT_TYPES, SPM_RE

from .data import PROFILE_FIELDS

if TYPE_CHECKING:
    from .engine import Engine

TABLES = ("behavior", "item", "user")


@dataclass
class IngestResult:
    accepted: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)


class RowError(ValueError):
    pass


def _str(r: dict, k: str, required: bool = True) -> str | None:
    v = r.get(k)
    if v is None or (isinstance(v, str) and not v.strip()):
        if required:
            raise RowError(f"missing {k}")
        return None
    return str(v)


def _num(r: dict, k: str, *, ge: float | None = None, integer: bool = False) -> float | None:
    v = r.get(k)
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise RowError(f"{k} is not a number: {v!r}") from None
    if x != x:
        raise RowError(f"{k} is NaN")
    if integer and x != int(x):
        raise RowError(f"{k} must be an integer")
    if ge is not None and x < ge:
        raise RowError(f"{k} must be >= {ge:g}")
    return x


class Ingestor:
    def __init__(self, engine: Engine):
        self.eng = engine

    def write(self, table: str, rows: list[dict[str, Any]]) -> IngestResult:
        if table not in TABLES:
            return IngestResult(0, [{"index": -1, "message": f"unknown table {table}; expected one of {TABLES}"}])
        fn = {"behavior": self._behavior, "item": self._item, "user": self._user}[table]
        return fn(rows)

    # ------------------------------------------------------------- behavior
    def _behavior(self, rows: list[dict[str, Any]]) -> IngestResult:
        eng, res = self.eng, IngestResult()
        now = eng.clock.now()
        markets = eng.config["markets"]
        by_user: dict[str, list[tuple[float, str, str, float]]] = defaultdict(list)
        out: list[dict[str, Any]] = []
        for n, r in enumerate(rows):
            try:
                uid = _str(r, "user_id")
                iid = _str(r, "item_id")
                et = _str(r, "event_type")
                assert uid and iid and et
                if et not in EVENT_TYPES:
                    raise RowError(f"unregistered event_type {et!r}")
                if eng.catalog.idx(iid) is None:
                    raise RowError(f"unknown item_id {iid!r}")
                ts = _num(r, "ts")
                ts = now if ts is None else (ts / 1000 if ts > 1e11 else ts)  # accept epoch millis
                if ts < EARLIEST_TS or ts > now + 86400:
                    raise RowError(f"ts out of range: {r.get('ts')!r}")
                market = _str(r, "market", required=False)
                if market is None:
                    prof = eng.users.profiles.get(uid)
                    market = prof.market if prof else eng.config["app"]["default_market"]
                if market not in markets:
                    raise RowError(f"unknown market {market!r}")
                spm = _str(r, "spm", required=False)
                if spm is not None and not SPM_RE.match(spm):
                    raise RowError(f"invalid spm {spm!r}")
                paid = _num(r, "paid_amount", ge=0) or 0.0
            except RowError as e:
                res.errors.append({"index": n, "message": str(e)})
                continue
            by_user[uid].append((float(ts), iid, et, paid))
            out.append(
                {
                    "request_id": _str(r, "request_id", required=False) or "0",
                    "user_id": uid,
                    "item_id": iid,
                    "event_type": et,
                    "ts": int(ts),
                    "spm": spm,
                    "market": market,
                    "parent_item_id": _str(r, "parent_item_id", required=False),
                    "paid_amount": paid if et == "purchase" else None,
                }
            )
        for uid, evs in by_user.items():
            eng.users.add_events(uid, evs)
        eng.bus.publish("behavior", out)
        res.accepted = len(out)
        return res

    # ------------------------------------------------------------- item price / stock
    def _item(self, rows: list[dict[str, Any]]) -> IngestResult:
        eng, res = self.eng, IngestResult()
        changes, out = [], []
        for n, r in enumerate(rows):
            try:
                iid, market = _str(r, "item_id"), _str(r, "market")
                assert iid and market
                if eng.catalog.idx(iid) is None:
                    raise RowError(f"unknown item_id {iid!r} (new items are picked up by the next batch build)")
                if market not in eng.catalog.price:
                    raise RowError(f"unknown market {market!r}")
                price = _num(r, "price", ge=0)
                stock = _num(r, "stock", ge=0, integer=True)
                if price is None and stock is None:
                    raise RowError("nothing to update: give price and/or stock")
            except RowError as e:
                res.errors.append({"index": n, "message": str(e)})
                continue
            st = None if stock is None else int(stock)
            changes.append((market, iid, price, st))
            out.append({"item_id": iid, "market": market, "price": price, "stock": st, "recv": time.time()})
        if changes:
            eng.publish_item_changes(changes)
        eng.bus.publish("item", out)
        res.accepted = len(out)
        return res

    # ------------------------------------------------------------- user profile
    def _user(self, rows: list[dict[str, Any]]) -> IngestResult:
        eng, res = self.eng, IngestResult()
        out = []
        for n, r in enumerate(rows):
            try:
                uid = _str(r, "user_id")
                assert uid
                fields = {k: r[k] for k in PROFILE_FIELDS if k in r and r[k] is not None}
                unknown = set(r) - PROFILE_FIELDS - {"user_id", "register_ts"}
                if unknown:
                    raise RowError(f"unknown fields {sorted(unknown)}")
                if "market" in fields and fields["market"] not in eng.config["markets"]:
                    raise RowError(f"unknown market {fields['market']!r}")
                if "member_level" in fields:
                    fields["member_level"] = int(_num(fields, "member_level", ge=0, integer=True) or 0)
                if "allowance_used" in fields:
                    fields["allowance_used"] = float(_num(fields, "allowance_used", ge=0) or 0.0)
                for k in ("country", "language"):
                    if k in fields:
                        fields[k] = str(fields[k])
                if not fields:
                    raise RowError("nothing to update")
            except RowError as e:
                res.errors.append({"index": n, "message": str(e)})
                continue
            eng.users.patch_profile(uid, fields)
            out.append({"user_id": uid, **fields, "recv": time.time()})
        eng.bus.publish("user", out)
        res.accepted = len(out)
        return res

    # ------------------------------------------------------------- completeness
    def done(self, table: str, date: str) -> None:
        """Completeness marker; flushed so it lands after the rows it covers."""
        self.eng.bus.publish("done", [{"table": table, "date": date, "recv": time.time()}])
        self.eng.bus.flush()
