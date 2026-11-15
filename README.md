# recsys — market-agnostic e-commerce recommendation engine

Self-built recommendation engine (recall → rank → business rules → A/B) designed for
**multiple markets** — currency, language, duty-free allowance, purchase limits, compliance
blocks, channels and slots are configuration, not code. First target customer: travel retail
(CDF, CN + NZ).

Plan: [`../docs/ecommerce-recsys-plan.md`](../docs/ecommerce-recsys-plan.md) · Data contract:
[`docs/data-contract.md`](docs/data-contract.md) · Tracking spec: [`docs/tracking-spec.md`](docs/tracking-spec.md) ·
Customer alignment: [`docs/p0-alignment-checklist.md`](docs/p0-alignment-checklist.md) · Python SDK:
[`packages/sdk-python`](packages/sdk-python/README.md)

## Quick start

```bash
make setup        # uv sync --all-packages
make check        # ruff + mypy + pytest + config validation (what CI runs)
make p0           # raw export -> backfill (+quality gate) -> offline baseline report
make serve        # train on the lake, then API + demo UI on :8000, config console on :8000/admin
make ab-sim       # online A/B simulation through the production path + daily report
make help         # everything else
```

Optional: `uv sync --all-packages --extra lgb` uses LightGBM for the ranker (macOS: `brew install libomp`);
otherwise scikit-learn's HistGradientBoosting is used and exported to ONNX for serving.

## Workspace layout (uv workspace)

| path | package | what |
|---|---|---|
| `packages/recsys-core` | `recsys_core` | config schema & validation (`${ENV}` expansion), data contract, lake storage, data-quality gate, signing |
| `packages/rec-server` | `rec_server` | online engine: recall, features, GBDT ranker (+ ONNX), rules, A/B, degradation, live state (memory / Redis), ingest + bus (lake / Kafka), HTTP API, `/metrics`, demo UI, config console |
| `packages/batch` | `recsys_batch` | batch jobs: offline baseline, T+1 daily report (attribution, funnel, A/B readout) |
| `packages/sdk-python` | `recsys_client` | Python SDK (only needs httpx; Python ≥ 3.9) |
| `tools` | `recsys_tools` | `recsys` CLI, backfill, simulator, online A/B simulation, Kafka consumer, load test |
| `configs/engine.toml` | | markets, slots (SPM → recall/rank/rules/item pool), experiments, serving budget, state, ingest, console |
| `configs/mappings/*.toml` | | per-customer field mappings for backfill |
| `deploy/` | | docker-compose (Kafka, Redis, Postgres, ClickHouse, MinIO, Prometheus, Grafana), Dockerfile, Helm chart |

## Architecture (P1)

```
             predict / callback                     write / done
  client ─────────────────────────┐        ┌──────────────────────────── client / SDK
                                  ▼        ▼
                        ┌──────────────────────────┐   hot reload   ┌──────────────────┐
                        │ rec-server (N workers)   │◀──────────────│ config store      │◀── /admin console
                        │ recall → filter → rank   │   (poll 5 s)   │ sqlite / Postgres │    validate·preview
                        │ (ONNX) → rules → A/B     │                └──────────────────┘    publish·rollback
                        │ budgets + hot fallback   │
                        └───┬──────────────┬───────┘
          live state        │              │ canonical rows + request log
   (events, profiles,       ▼              ▼
    issued, price/stock) ┌───────┐   ┌──────────────┐   recsys consume   ┌──────────────┐
                         │ Redis │   │ lake bus  or │ ─────────────────▶ │ lake (parquet)│──▶ daily batch: train,
                         └───────┘   │ Kafka topics │                    └──────────────┘    baseline, T+1 report
                                     └──────────────┘
```

- **Snapshot + live overlay.** The daily batch pickles an immutable snapshot (catalog, user history up to the
  cutoff, models). Everything after the cutoff goes to *live state* and is overlaid at request time:
  `memory` for one process, `redis` when several workers/pods must agree (behaviors, profile patches,
  cross-request dedup, price/stock changes).
- **Degradation.** Whole-request and recall budgets; a failing recall channel or ranker degrades that stage;
  any other error serves the market's hot list (compliance blocks still apply). Every degraded response is
  marked (`degraded` in the response, `rec_degraded_total` metric, request log).
- **Ingestion.** `/data/api/{app}/{behavior,item,user}/write` validates each row (unknown items/markets,
  bad SPM, timestamps in s or ms, negative amounts …), applies it to live state, and forwards canonical rows to
  the bus: straight into the lake (single node) or Kafka (`recsys consume` → lake, ClickHouse reads the same
  topic). `/data/api/{app}/done` marks a day complete.
- **Attribution & reporting.** The engine logs every request (`rec_log`: slot, variants, items). The T+1 report
  attributes conversions by request id or last click (≤ 72 h), reports the funnel per slot × market × variant and
  an A/B readout with 95% CIs (delta method for CTR) and a sample-ratio check.
- **Ranker.** GBDT; sklearn models are exported to ONNX by our own converter (exact parity, categorical splits)
  and served with ONNX Runtime (≈5× faster); `artifacts/ranker.onnx` + `ranker.json` for other runtimes.

## CLI

```
recsys simulate [--raw DIR] [--no-lake]   synthetic data -> lake and/or a vendor-style raw export
recsys backfill --mapping M --raw DIR     raw export -> lake, with backfill + quality report (exit 1 if gate fails)
recsys quality                            run the quality gate on the lake
recsys baseline [--test-days 5] [--truth] offline baseline report
recsys train                              models + ranker (+ ONNX) -> artifacts/
recsys serve [--workers N]                API + demo UI + console
recsys report --date YYYYMMDD             T+1 daily report (attribution, funnel, A/B readout)
recsys ab-sim [--sessions N]              online A/B simulation through the production path
recsys consume                            Kafka topics -> lake
recsys loadtest [--rate 2000]             load test a running server
recsys config-check | schema-doc
```

## API

| endpoint | |
|---|---|
| `POST /predict/api/{app}/default` | recommendations (request/response shape compatible with Volcengine/BytePlus predict) |
| `POST /predict/api/{app}/callback` | report what was actually shown |
| `POST /data/api/{app}/{behavior,item,user}/write` | real-time ingestion, per-row errors |
| `POST /data/api/{app}/done` | `{table, date}`: data for a business day is complete |
| `GET /admin`, `/admin/api/*` | config console: config, versions, validate, preview, publish, rollback |
| `GET /metrics`, `/healthz` | Prometheus metrics (aggregated across workers), health |
| `GET /` | demo UI |

Signing (`[auth]`): `Tenant-Signature = hex(sha256(token‖body‖tenant_id‖ts‖nonce))` on predict and data APIs;
the console uses `X-Admin-Token`. Secrets come from the environment (`token = "${RECSYS_TENANT_TOKEN}"`).

## Running more than one worker

```toml
[state]
backend = "redis"
url = "${RECSYS_REDIS_URL}"
[ingest]
bus = "kafka"                      # or "lake" on a single node
[admin]
store = "${RECSYS_CONFIG_STORE}"   # postgresql://... ; sections app/auth/state/ingest/admin stay file-only
```

`recsys serve --workers N` (or `WORKERS` in the image / Helm `workers`). Measured on a 12-core laptop with the
client on the same machine: 2,000 req/s at P99 38 ms, 3,000 req/s at P99 79 ms (`reports/loadtest.md`).
