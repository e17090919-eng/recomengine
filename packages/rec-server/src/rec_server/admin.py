"""Config console backend: versioned config store + validation / preview / publish /
rollback, and the watcher that hot-reloads every serving worker.

Store schema = deploy/postgres/init.sql (`config_version`, `audit_log`); SQLite is used
for single-node / dev. Content is stored as {"toml": <text>, "config": <validated dict>}.
"""

from __future__ import annotations

import copy
import json
import logging
import sqlite3
import time
import tomllib
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli_w
from pydantic import ValidationError

from recsys_core.config import RESTART_SECTIONS, find_root, hot_sections, validate_config

log = logging.getLogger("recsys.admin")

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS config_version (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    app_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('engine', 'mapping')),
    content TEXT NOT NULL,
    comment TEXT,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    is_active INTEGER NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX IF NOT EXISTS config_version_one_active ON config_version (app_id, kind) WHERE is_active;
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    target TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '{}',
    at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""


class ConfigError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class Version:
    id: int
    comment: str
    created_by: str
    created_at: str
    is_active: bool
    toml: str
    config: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "comment": self.comment,
            "created_by": self.created_by,
            "created_at": str(self.created_at),
            "is_active": self.is_active,
        }


class ConfigStore:
    def __init__(self, url: str, app_id: str):
        self.url, self.app_id = url, app_id
        self.pg = url.startswith(("postgres://", "postgresql://"))
        if not self.pg:
            p = Path(url.removeprefix("sqlite:///"))
            self.path = p if p.is_absolute() else find_root() / p
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(self._conn()) as c:
                c.executescript(SQLITE_SCHEMA)

    # ------------------------------------------------------------- db plumbing
    def _conn(self) -> Any:
        if self.pg:
            import psycopg

            return psycopg.connect(self.url, autocommit=False)
        c = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def _sql(self, q: str) -> str:
        return q.replace("?", "%s") if self.pg else q

    def _rows(self, q: str, args: tuple = ()) -> list[tuple]:
        with closing(self._conn()) as c:
            cur = c.execute(self._sql(q), args)
            return list(cur.fetchall())

    def _version(self, r: tuple) -> Version:
        content = r[1] if isinstance(r[1], dict) else json.loads(r[1])
        return Version(int(r[0]), r[2] or "", r[3], str(r[4]), bool(r[5]), content["toml"], content["config"])

    _COLS = "id, content, comment, created_by, created_at, is_active"

    # ------------------------------------------------------------- reads
    def active_id(self) -> int | None:
        r = self._rows(
            "SELECT id FROM config_version WHERE app_id = ? AND kind = 'engine' AND is_active", (self.app_id,)
        )
        return int(r[0][0]) if r else None

    def get(self, version_id: int | None = None) -> Version | None:
        if version_id is None:
            q = f"SELECT {self._COLS} FROM config_version WHERE app_id = ? AND kind = 'engine' AND is_active"
            r = self._rows(q, (self.app_id,))
        else:
            q = f"SELECT {self._COLS} FROM config_version WHERE app_id = ? AND kind = 'engine' AND id = ?"
            r = self._rows(q, (self.app_id, version_id))
        return self._version(r[0]) if r else None

    def versions(self, limit: int = 50) -> list[Version]:
        q = f"SELECT {self._COLS} FROM config_version WHERE app_id = ? AND kind = 'engine' ORDER BY id DESC LIMIT ?"
        return [self._version(r) for r in self._rows(q, (self.app_id, limit))]

    def audit(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._rows("SELECT actor, action, target, detail, at FROM audit_log ORDER BY id DESC LIMIT ?", (limit,))
        return [
            {"actor": a, "action": b, "target": t, "detail": d if isinstance(d, dict) else json.loads(d), "at": str(at)}
            for a, b, t, d, at in rows
        ]

    # ------------------------------------------------------------- writes
    def publish(
        self, toml_text: str, config: dict[str, Any], comment: str, author: str, if_none_active: bool = False
    ) -> int:
        """New active version. `if_none_active`: only when nothing is active yet (race-free seeding
        when several workers start at once); returns the existing active id otherwise."""
        content = json.dumps({"toml": toml_text, "config": config}, ensure_ascii=False, default=str)
        with closing(self._conn()) as c:
            if not self.pg:
                c.execute("BEGIN IMMEDIATE")
            else:
                c.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"config:{self.app_id}",))
            if if_none_active:
                cur = c.execute(
                    self._sql("SELECT id FROM config_version WHERE app_id = ? AND kind = 'engine' AND is_active"),
                    (self.app_id,),
                ).fetchone()
                if cur:
                    c.execute("ROLLBACK") if not self.pg else c.rollback()
                    return int(cur[0])
            c.execute(
                self._sql("UPDATE config_version SET is_active = ? WHERE app_id = ? AND kind = 'engine' AND is_active"),
                (False, self.app_id),
            )
            cur = c.execute(
                self._sql(
                    "INSERT INTO config_version (app_id, kind, content, comment, created_by, is_active) "
                    "VALUES (?, 'engine', ?, ?, ?, ?) RETURNING id"
                ),
                (self.app_id, content, comment, author, True),
            )
            vid = int(cur.fetchone()[0])
            c.execute(
                self._sql("INSERT INTO audit_log (actor, action, target, detail) VALUES (?, ?, ?, ?)"),
                (author, "publish", f"config:{self.app_id}", json.dumps({"version": vid, "comment": comment})),
            )
            c.execute("COMMIT") if not self.pg else c.commit()
        return vid

    def activate(self, version_id: int, author: str) -> None:
        with closing(self._conn()) as c:
            if not self.pg:
                c.execute("BEGIN IMMEDIATE")
            exists = c.execute(
                self._sql("SELECT 1 FROM config_version WHERE app_id = ? AND kind = 'engine' AND id = ?"),
                (self.app_id, version_id),
            ).fetchone()
            if not exists:
                c.execute("ROLLBACK") if not self.pg else c.rollback()
                raise KeyError(version_id)
            c.execute(
                self._sql("UPDATE config_version SET is_active = ? WHERE app_id = ? AND kind = 'engine' AND is_active"),
                (False, self.app_id),
            )
            c.execute(self._sql("UPDATE config_version SET is_active = ? WHERE id = ?"), (True, version_id))
            c.execute(
                self._sql("INSERT INTO audit_log (actor, action, target, detail) VALUES (?, ?, ?, ?)"),
                (author, "rollback", f"config:{self.app_id}", json.dumps({"version": version_id})),
            )
            c.execute("COMMIT") if not self.pg else c.commit()


# ------------------------------------------------------------------ validation
def check(raw: dict[str, Any] | str, running: dict[str, Any], engine: Any = None) -> tuple[dict, str, list[str]]:
    """Validate a draft of the hot sections (TOML text or dict) against the schema, the engine
    registries and the running config. Returns (full config, toml text of the hot sections,
    warnings); raises ConfigError."""
    from .engine import Engine

    if isinstance(raw, str):
        try:
            parsed = tomllib.loads(raw)
        except tomllib.TOMLDecodeError as e:
            raise ConfigError([f"TOML: {e}"]) from None
    else:
        parsed = copy.deepcopy(raw)
    restart = [k for k in parsed if k in RESTART_SECTIONS]
    if restart:
        raise ConfigError(
            [f"[{k}] cannot be changed from the console (needs a restart): edit configs/engine.toml" for k in restart]
        )
    full = {**{k: running[k] for k in RESTART_SECTIONS if k in running}, **parsed}
    try:
        cfg = validate_config(full)
    except ValidationError as e:
        raise ConfigError(
            [f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors(include_url=False)]
        ) from None
    errors = []
    try:
        Engine.validate_config(cfg)
    except ValueError as e:
        errors += str(e).split("; ")
    if set(cfg["markets"]) != set(running["markets"]):
        errors.append("adding/removing markets needs a retrain + restart")
    if errors:
        raise ConfigError(errors)
    text = raw if isinstance(raw, str) else tomli_w.dumps(_strip_none(hot_sections(cfg)))
    warnings = engine.catalog_warnings(cfg) if engine is not None else []
    return cfg, text, warnings


def _strip_none(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _strip_none(v) for k, v in x.items() if v is not None}
    if isinstance(x, list):
        return [_strip_none(v) for v in x]
    return x


def diff(a: Any, b: Any, path: str = "") -> list[dict[str, Any]]:
    """Leaf-level differences between two configs (for the publish confirmation)."""
    if not path:  # compare as JSON (tuples vs lists, etc.)
        a, b = json.loads(json.dumps(a, default=str)), json.loads(json.dumps(b, default=str))
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            p = f"{path}.{k}" if path else str(k)
            if k not in a:
                out.append({"path": p, "old": None, "new": b[k]})
            elif k not in b:
                out.append({"path": p, "old": a[k], "new": None})
            else:
                out += diff(a[k], b[k], p)
        return out
    return [] if a == b else [{"path": path, "old": a, "new": b}]


# ------------------------------------------------------------------ hot reload
class ConfigWatcher:
    """Polls the store's active version; reloads the engine when it changes. Called from
    the request path (cheap time check), so every worker converges within `poll_seconds`."""

    def __init__(self, store: ConfigStore, engine: Any, poll_seconds: float = 5.0):
        self.store, self.engine, self.poll = store, engine, poll_seconds
        self.version: int | None = None
        self._next = 0.0
        self.last_error: str | None = None

    def bootstrap(self, file_cfg: dict[str, Any]) -> None:
        """Seed the store from the config file once; afterwards the store wins for hot sections."""
        if self.store.active_id() is None:
            hot = hot_sections(file_cfg)
            text = tomli_w.dumps(_strip_none(hot))
            self.store.publish(text, hot, "seeded from configs/engine.toml", "bootstrap", if_none_active=True)
        self.maybe_reload(force=True)

    def maybe_reload(self, force: bool = False) -> bool:
        now = time.monotonic()
        if not force and now < self._next:
            return False
        self._next = now + self.poll
        try:
            vid = self.store.active_id()
            if vid is None or vid == self.version:
                return False
            v = self.store.get(vid)
            if v is None:
                return False
            self.engine.reload_config(v.config)
            self.version = vid
            self.last_error = None
            log.info("config version %s active", vid)
            return True
        except Exception as e:  # keep serving with the previous config
            self.last_error = repr(e)
            log.exception("config reload failed")
            return False
