# deploy

| file | purpose |
|---|---|
| `docker-compose.yml` | local infra: Redis, PostgreSQL (metadata), Kafka (KRaft) + topics, ClickHouse (behavior log, Kafka ingestion, daily funnel view), MinIO (S3 lake/models) + buckets, Prometheus, Grafana (rec-server dashboard) |
| `check-infra.sh` | smoke test for all of the above (`make infra-check`) |
| `postgres/init.sql` | config versions, model registry, audit log |
| `clickhouse/init.sql` | `recsys.behavior` + Kafka engine + materialized view + `daily_funnel` |
| `Dockerfile` | rec-server image (`docker build -f deploy/Dockerfile .` from repo root) |
| `helm/rec-server` | Deployment + Service + HPA + ConfigMap (engine.toml); artifacts from a PVC |

Requires Docker (Docker Desktop / OrbStack / Colima). `make infra-up && make infra-check`.
