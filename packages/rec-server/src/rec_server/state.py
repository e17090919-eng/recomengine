"""Real-time state shared by all serving workers.

The daily batch produces an immutable snapshot (profiles + history up to `cutoff_ts`,
pickled with the engine). Everything that happens after the cutoff — behaviors written
through the data API, profile updates, what was issued to whom — goes to a `LiveState`
and is overlaid on the snapshot at request time.

- `MemoryLive`: single process (dev, tests, demo).
- `RedisLive`: shared by every worker / pod. Items are stored by *id*, not catalog index,
  so live data stays valid when a new snapshot with a different catalog is loaded.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from functools import partial
from typing import Any, Protocol

MAX_EVENTS = 500
MAX_IMPRESSIONS = 300
ISSUED_WINDOW = 30 * 60  # cross-request dedup window (s)
ISSUED_CAP = 200

Event = tuple[float, str, str, float, float]  # ts, item_id, event_type, paid_amount, received (wall clock)
ItemChange = tuple[str, str, float | None, int | None, float]  # market, item_id, price, stock, received


class LiveState(Protocol):
    def add_events(self, user_id: str, events: list[Event]) -> None: ...
    def events(self, user_id: str) -> list[Event]: ...
    def impressions(self, user_id: str) -> list[tuple[float, str]]: ...
    def patch_profile(self, user_id: str, fields: dict[str, Any]) -> None: ...
    def profile_patch(self, user_id: str) -> dict[str, Any]: ...
    def record_issued(self, user_id: str, item_ids: list[str], now: float) -> None: ...
    def issued(self, user_id: str, now: float) -> set[str]: ...
    def push_item_changes(self, changes: list[ItemChange]) -> None: ...
    def item_changes_since(self, version: int) -> tuple[int, list[ItemChange]]: ...
    def n_users(self) -> int: ...
    def ping(self) -> bool: ...


class MemoryLive:
    def __init__(self) -> None:
        self._ev: dict[str, deque] = defaultdict(partial(deque, maxlen=MAX_EVENTS))
        self._imp: dict[str, deque] = defaultdict(partial(deque, maxlen=MAX_IMPRESSIONS))
        self._iss: dict[str, deque] = defaultdict(partial(deque, maxlen=ISSUED_CAP))
        self._prof: dict[str, dict[str, Any]] = {}
        self._items: list[ItemChange] = []

    def add_events(self, user_id: str, events: list[Event]) -> None:
        for e in events:
            if e[2] == "impression":
                self._imp[user_id].append((e[0], e[1]))
            else:
                self._ev[user_id].append(e)

    def events(self, user_id: str) -> list[Event]:
        return list(self._ev.get(user_id, ()))

    def impressions(self, user_id: str) -> list[tuple[float, str]]:
        return list(self._imp.get(user_id, ()))

    def patch_profile(self, user_id: str, fields: dict[str, Any]) -> None:
        self._prof.setdefault(user_id, {}).update(fields)

    def profile_patch(self, user_id: str) -> dict[str, Any]:
        return dict(self._prof.get(user_id, {}))

    def record_issued(self, user_id: str, item_ids: list[str], now: float) -> None:
        dq = self._iss[user_id]
        for i in item_ids:
            dq.append((now, i))

    def issued(self, user_id: str, now: float) -> set[str]:
        return {i for ts, i in self._iss.get(user_id, ()) if now - ts <= ISSUED_WINDOW}

    def push_item_changes(self, changes: list[ItemChange]) -> None:
        self._items.extend(changes)

    def item_changes_since(self, version: int) -> tuple[int, list[ItemChange]]:
        return len(self._items), self._items[version:]

    def n_users(self) -> int:
        return len(set(self._ev) | set(self._prof))

    def ping(self) -> bool:
        return True


class RedisLive:
    """Keys (all under `prefix`): ev:{uid} list, imp:{uid} list, iss:{uid} zset, prof:{uid} hash,
    itemlog list (price/stock changes; workers replay it from their last offset)."""

    def __init__(self, client: Any, prefix: str = "recsys", ttl_days: float = 30):
        self.r, self.p, self.ttl = client, prefix.rstrip(":") + ":", int(ttl_days * 86400)

    @classmethod
    def from_url(cls, url: str, prefix: str = "recsys") -> RedisLive:
        import redis

        return cls(redis.Redis.from_url(url, decode_responses=True, socket_timeout=0.05), prefix)

    def _k(self, kind: str, uid: str) -> str:
        return f"{self.p}{kind}:{uid}"

    def add_events(self, user_id: str, events: list[Event]) -> None:
        ev = [json.dumps(list(e), separators=(",", ":")) for e in events if e[2] != "impression"]
        imp = [json.dumps([e[0], e[1]], separators=(",", ":")) for e in events if e[2] == "impression"]
        pipe = self.r.pipeline(transaction=False)
        for kind, vals, cap in (("ev", ev, MAX_EVENTS), ("imp", imp, MAX_IMPRESSIONS)):
            if vals:
                k = self._k(kind, user_id)
                pipe.rpush(k, *vals)
                pipe.ltrim(k, -cap, -1)
                pipe.expire(k, self.ttl)
        pipe.sadd(self.p + "users", user_id)
        pipe.execute()

    def events(self, user_id: str) -> list[Event]:
        return [tuple(json.loads(x)) for x in self.r.lrange(self._k("ev", user_id), 0, -1)]

    def impressions(self, user_id: str) -> list[tuple[float, str]]:
        return [tuple(json.loads(x)) for x in self.r.lrange(self._k("imp", user_id), 0, -1)]

    def patch_profile(self, user_id: str, fields: dict[str, Any]) -> None:
        k = self._k("prof", user_id)
        self.r.hset(k, mapping={f: json.dumps(v) for f, v in fields.items()})
        self.r.expire(k, self.ttl)
        self.r.sadd(self.p + "users", user_id)

    def profile_patch(self, user_id: str) -> dict[str, Any]:
        return {f: json.loads(v) for f, v in self.r.hgetall(self._k("prof", user_id)).items()}

    def record_issued(self, user_id: str, item_ids: list[str], now: float) -> None:
        if not item_ids:
            return
        k = self._k("iss", user_id)
        pipe = self.r.pipeline(transaction=False)
        pipe.zadd(k, {i: now for i in item_ids})
        pipe.zremrangebyscore(k, "-inf", now - ISSUED_WINDOW)
        pipe.zremrangebyrank(k, 0, -ISSUED_CAP - 1)
        pipe.expire(k, ISSUED_WINDOW * 2)
        pipe.execute()

    def issued(self, user_id: str, now: float) -> set[str]:
        return set(self.r.zrangebyscore(self._k("iss", user_id), now - ISSUED_WINDOW, "+inf"))

    def push_item_changes(self, changes: list[ItemChange]) -> None:
        if changes:
            self.r.rpush(self.p + "itemlog", *[json.dumps(list(c), separators=(",", ":")) for c in changes])

    def item_changes_since(self, version: int) -> tuple[int, list[ItemChange]]:
        vals = self.r.lrange(self.p + "itemlog", version, -1)
        return version + len(vals), [tuple(json.loads(v)) for v in vals]

    def n_users(self) -> int:
        return int(self.r.scard(self.p + "users"))

    def ping(self) -> bool:
        try:
            return bool(self.r.ping())
        except Exception:
            return False


def make_live(cfg: dict) -> LiveState:
    st = cfg.get("state", {}) or {}
    if st.get("backend", "memory") == "redis":
        return RedisLive.from_url(st["url"], st.get("prefix") or f"recsys:{cfg['app']['app_id']}")
    return MemoryLive()
