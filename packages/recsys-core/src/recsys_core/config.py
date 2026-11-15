"""Engine configuration: load TOML and validate it against a typed model.

Validation catches mistakes before they reach production (unknown default market,
duplicate SPMs, experiment traffic > 100%, bad timezone...). The engine keeps
working with the plain dict, so config stays easy to diff and hot-reload."""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SPM_SEP = "$##$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AppCfg(_Strict):
    app_id: str = Field(min_length=1)
    default_market: str


class AuthCfg(_Strict):
    enabled: bool = False
    tenant_id: str = ""
    token: str = ""


class MarketCfg(_Strict):
    name: dict[str, str] = Field(default_factory=dict)
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    timezone: str = "UTC"
    languages: list[str] = Field(min_length=1)
    channels: list[str] = Field(default_factory=list)
    allowance: float = Field(default=0.0, ge=0)
    category_limits: dict[str, int] = Field(default_factory=dict)
    # compliance: never recommended in this market, in any slot (also applies to the fallback)
    blocked_categories: list[str] = Field(default_factory=list)
    blocked_items: list[str] = Field(default_factory=list)

    @field_validator("timezone")
    @classmethod
    def _tz(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError) as e:
            raise ValueError(f"unknown timezone {v!r}") from e
        return v


class RecallCfg(_Strict):
    name: str
    quota: int = Field(gt=0, le=5000)
    weight: float = Field(default=1.0, ge=0)
    params: dict[str, float] = Field(default_factory=dict)  # channel-specific keyword arguments


class DispersionCfg(_Strict):
    field: Literal["brand", "category"]
    window: int = Field(ge=2)
    max: int = Field(ge=1)


class BoostNewCfg(_Strict):
    days: float = Field(gt=0)
    weight: float = Field(ge=0)


class PoolCfg(_Strict):
    """Item pool of a slot (applied by the `item_pool` filter on top of market availability)."""

    include_categories: list[str] = Field(default_factory=list)  # empty = all
    exclude_categories: list[str] = Field(default_factory=list)
    include_brands: list[str] = Field(default_factory=list)
    exclude_brands: list[str] = Field(default_factory=list)
    exclude_items: list[str] = Field(default_factory=list)
    min_price: float | None = Field(default=None, ge=0)  # local currency
    max_price: float | None = Field(default=None, ge=0)


class SlotCfg(_Strict):
    spm: str
    size: int = Field(default=20, ge=1, le=500)
    ranker: Literal["gbdt", "recall_score"] = "gbdt"
    recalls: list[RecallCfg] = Field(min_length=1)
    filters: list[str] = Field(default_factory=list)
    boost_new: BoostNewCfg | None = None
    dispersion: list[DispersionCfg] = Field(default_factory=list)
    pool: PoolCfg = Field(default_factory=PoolCfg)

    @field_validator("spm")
    @classmethod
    def _spm(cls, v: str) -> str:
        parts = v.split(SPM_SEP)
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"slot spm must be 'app{SPM_SEP}page{SPM_SEP}block', got {v!r}")
        return v


class VariantCfg(_Strict):
    name: str
    traffic: int = Field(ge=0, le=100)
    params: dict[str, Any] = Field(default_factory=dict)


class ExperimentCfg(_Strict):
    slots: list[str] = Field(min_length=1)
    salt: str
    variants: list[VariantCfg] = Field(min_length=1)

    @model_validator(mode="after")
    def _traffic(self) -> ExperimentCfg:
        if sum(v.traffic for v in self.variants) > 100:
            raise ValueError("variant traffic sums to more than 100")
        names = [v.name for v in self.variants]
        if len(set(names)) != len(names):
            raise ValueError("duplicate variant names")
        return self


class QualityCfg(_Strict):
    """Thresholds (fractions) for the data-quality gate."""

    max_required_null_rate: float = 0.01
    max_type_error_rate: float = 0.01
    max_unknown_item_rate: float = 0.05
    max_unknown_user_rate: float = 0.30
    max_unregistered_event_rate: float = 0.05
    max_invalid_spm_rate: float = 0.02
    max_bad_ts_rate: float = 0.001
    max_duplicate_event_rate: float = 0.02
    ctr_range: tuple[float, float] = (0.002, 0.5)
    max_daily_volume_change: float = 0.6


class ServingCfg(_Strict):
    """Latency budget and degradation. Stages that would overrun the budget are
    skipped (recall channels, GBDT) and the response is marked degraded."""

    timeout_ms: float = Field(default=150, gt=0)  # whole-request budget
    recall_budget_ms: float = Field(default=60, gt=0)  # stop calling further recall channels after this
    rank_max_candidates: int = Field(default=500, ge=1)  # truncate merged recall before GBDT
    fallback_size: int = Field(default=20, ge=1)


class StateCfg(_Strict):
    """Where real-time user state lives. `memory` = per process (dev); `redis` = shared by all workers."""

    backend: Literal["memory", "redis"] = "memory"
    url: str = "redis://localhost:6379/0"
    prefix: str = ""  # default recsys:<app_id>


class IngestCfg(_Strict):
    """Where ingested rows and request logs go after updating live state.
    `lake` = micro-batches straight into the lake (single node); `kafka` = topics consumed
    by `recsys consume` / ClickHouse; `none` = live state only."""

    bus: Literal["none", "lake", "kafka"] = "lake"
    lake_dir: str = "lake"  # relative to the repo root
    kafka_bootstrap: str = "localhost:9092"
    topics: dict[str, str] = Field(
        default_factory=lambda: {
            "behavior": "behavior-events",
            "item": "item-changes",
            "user": "user-changes",
            "rec_log": "rec-log",
        }
    )
    flush_rows: int = Field(default=5000, ge=1)
    flush_seconds: float = Field(default=10.0, gt=0)


class AdminCfg(_Strict):
    """Config console. The store holds versioned configs; once seeded from this file it is the
    source of truth for hot-reloadable sections (everything except RESTART_SECTIONS)."""

    enabled: bool = True
    token: str = ""  # X-Admin-Token required when set
    store: str = "sqlite:///artifacts/config.db"  # or postgresql://user:pass@host:5432/db
    poll_seconds: float = Field(default=5.0, gt=0)


# Sections that need a restart (identity, secrets, connections). They always come from the
# config file and are never stored in / changed by the console; everything else is "hot".
RESTART_SECTIONS = ("app", "auth", "state", "ingest", "admin")


def hot_sections(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in cfg.items() if k not in RESTART_SECTIONS}


class EngineConfig(_Strict):
    app: AppCfg
    auth: AuthCfg = Field(default_factory=AuthCfg)
    markets: dict[str, MarketCfg] = Field(min_length=1)
    slots: dict[str, SlotCfg] = Field(min_length=1)
    experiments: dict[str, ExperimentCfg] = Field(default_factory=dict)
    quality: QualityCfg = Field(default_factory=QualityCfg)
    serving: ServingCfg = Field(default_factory=ServingCfg)
    state: StateCfg = Field(default_factory=StateCfg)
    ingest: IngestCfg = Field(default_factory=IngestCfg)
    admin: AdminCfg = Field(default_factory=AdminCfg)

    @model_validator(mode="after")
    def _cross(self) -> EngineConfig:
        if self.app.default_market not in self.markets:
            raise ValueError(f"default_market {self.app.default_market!r} is not a configured market")
        spms = [s.spm for s in self.slots.values()]
        if len(set(spms)) != len(spms):
            raise ValueError("two slots share the same spm")
        for name, exp in self.experiments.items():
            unknown = set(exp.slots) - set(self.slots)
            if unknown:
                raise ValueError(f"experiment {name!r} references unknown slots {sorted(unknown)}")
        if self.auth.enabled and not self.auth.token:
            raise ValueError("auth.enabled requires auth.token")
        return self


def find_root(start: Path | None = None) -> Path:
    """Repository root: $RECSYS_HOME, else the nearest parent containing configs/engine.toml."""
    env = os.environ.get("RECSYS_HOME")
    if env:
        return Path(env)
    p = (start or Path.cwd()).resolve()
    for cand in (p, *p.parents):
        if (cand / "configs" / "engine.toml").exists():
            return cand
    return p


_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(x: Any) -> Any:
    """Replace ${NAME} / ${NAME:-default} in string values with environment variables, so
    secrets (tokens, Redis / Postgres URLs) stay out of the config file."""
    if isinstance(x, dict):
        return {k: expand_env(v) for k, v in x.items()}
    if isinstance(x, list):
        return [expand_env(v) for v in x]
    if isinstance(x, str):

        def sub(m: re.Match[str]) -> str:
            val = os.environ.get(m.group(1), m.group(2))
            if val is None:
                raise ValueError(f"environment variable {m.group(1)} is not set (referenced in config)")
            return val

        return _ENV_RE.sub(sub, x)
    return x


def validate_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate a raw (TOML-parsed) config; returns it as a plain dict with defaults filled in."""
    return EngineConfig.model_validate(raw).model_dump(mode="python")


def parse_toml(text: str) -> dict[str, Any]:
    return validate_config(tomllib.loads(text))


def config_path(path: Path | str | None = None) -> Path:
    return Path(path) if path else find_root() / "configs" / "engine.toml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load and validate; returns the config as a plain dict with defaults filled in."""
    with open(config_path(path), "rb") as f:
        raw = tomllib.load(f)
    return validate_config(expand_env(raw))
