# recsys — market-agnostic e-commerce recommendation engine

Self-built recommendation engine (recall → rank → business rules → A/B) designed for
**multiple markets** — currency, language, duty-free allowance, purchase limits, channels and
slots are configuration, not code. First target customer: travel retail (CDF, CN + NZ).

Plan: [`../docs/ecommerce-recsys-plan.md`](../docs/ecommerce-recsys-plan.md) · Data contract:
[`docs/data-contract.md`](docs/data-contract.md) · Tracking spec: [`docs/tracking-spec.md`](docs/tracking-spec.md) ·
Customer alignment: [`docs/p0-alignment-checklist.md`](docs/p0-alignment-checklist.md)

## Quick start

```bash
make setup        # uv sync --all-packages
make check        # ruff + mypy + pytest + config validation (what CI runs)
make p0           # raw export -> backfill (+quality gate) -> offline baseline report
make serve        # train on the lake, then API + demo UI on http://127.0.0.1:8000
make help         # everything else
```

Optional: `uv sync --all-packages --extra lgb` uses LightGBM for the ranker (macOS: `brew install libomp`);
otherwise scikit-learn's HistGradientBoosting is used.

## Workspace layout (uv workspace)

| path | package | what |
|---|---|---|
| `packages/recsys-core` | `recsys_core` | config schema & validation, canonical data contract, lake storage, data-quality gate, request signing |
| `packages/rec-server` | `rec_server` | online engine: recall, features, GBDT ranker, rules, A/B, HTTP API, `/metrics`, demo UI |
| `packages/batch` | `recsys_batch` | batch jobs: offline baseline report |
| `tools` | `recsys_tools` | `recsys` CLI, historical backfill, simulator, vendor-style raw export |
| `configs/engine.toml` | | markets, slots (SPM → recall/rank/rules), experiments, quality thresholds |
| `configs/mappings/*.toml` | | per-customer field mappings for backfill |
| `deploy/` | | docker-compose (Kafka, Redis, Postgres, ClickHouse, MinIO, Prometheus, Grafana), Dockerfile, Helm chart |
| `docs/` | | data contract (generated), tracking spec, alignment checklist |

## Data flow

```
customer raw export ──(mapping.toml)──> recsys backfill ──> lake/ (canonical parquet, behavior partitioned by day)
                                             │                    │
                                   reports/backfill.md      recsys quality  ──> reports/quality.md (gate)
                                                                  │
                                                            recsys baseline ──> reports/baseline.md
                                                            recsys train    ──> artifacts/engine.pkl ──> recsys serve
```

## CLI

```
recsys simulate [--raw DIR] [--no-lake]   synthetic data -> lake and/or a vendor-style raw export
recsys backfill --mapping M --raw DIR     raw export -> lake, with backfill + quality report (exit 1 if gate fails)
recsys quality                            run the quality gate on the lake
recsys baseline [--test-days 5] [--truth] offline baseline report
recsys train | serve | config-check | schema-doc
```

## API

| endpoint | |
|---|---|
| `POST /predict/api/{app}/default` | recommendations (request/response shape compatible with Volcengine/BytePlus predict) |
| `POST /predict/api/{app}/callback` | report what was actually shown |
| `POST /data/api/{app}/{behavior,item,user}/write` | real-time ingestion |
| `GET /metrics`, `/healthz` | Prometheus metrics, health |
| `GET /` | demo UI |

Signing (optional, `[auth]` in config): `Tenant-Signature = hex(sha256(token‖body‖tenant_id‖ts‖nonce))`,
see `recsys_core.signing.headers()`.
