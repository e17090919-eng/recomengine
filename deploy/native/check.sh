#!/usr/bin/env bash
# Smoke-check the native local infra (same checks as deploy/check-infra.sh, without docker).
set -u
P="${RECSYS_INFRA:-$HOME/opt/recsys-infra}"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export JAVA_HOME="$P/jdk/Contents/Home"
ok=0; fail=0
check() { if eval "$2" >/dev/null 2>&1; then echo "  ok    $1"; ok=$((ok+1)); else echo "  FAIL  $1"; fail=$((fail+1)); fi; }
check redis       "'$P/redis/bin/redis-cli' ping | grep -q PONG"
check postgres    "'$REPO/.venv/bin/python' -c \"import psycopg; psycopg.connect('postgresql://recsys:recsys@127.0.0.1:5432/recsys').execute('select count(*) from model_registry')\""
check kafka       "'$P/kafka/bin/kafka-topics.sh' --bootstrap-server localhost:9092 --list | grep -q behavior-events"
check clickhouse  "curl -sf 'http://recsys:recsys@127.0.0.1:8123/?query=SELECT%20count()%20FROM%20recsys.behavior'"
check s3          "curl -sf http://127.0.0.1:9100/recsys-lake"
check prometheus  "curl -sf http://127.0.0.1:9090/-/ready"
check grafana     "curl -sf http://127.0.0.1:3000/api/health"
echo "$ok ok, $fail failed"; [ "$fail" -eq 0 ]
