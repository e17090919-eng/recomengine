#!/usr/bin/env bash
# Smoke-check local infra started by deploy/docker-compose.yml.
set -u
ok=0; fail=0
check() { if eval "$2" >/dev/null 2>&1; then echo "  ok    $1"; ok=$((ok+1)); else echo "  FAIL  $1"; fail=$((fail+1)); fi; }
dc="docker compose -f $(dirname "$0")/docker-compose.yml"
check redis       "$dc exec -T redis redis-cli ping | grep -q PONG"
check postgres    "$dc exec -T postgres psql -U recsys -c 'select count(*) from model_registry'"
check kafka       "$dc exec -T kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:19092 --list | grep -q behavior-events"
check clickhouse  "curl -sf 'http://recsys:recsys@localhost:8123/?query=SELECT%20count()%20FROM%20recsys.behavior'"
check minio       "curl -sf http://localhost:9100/minio/health/live"
check prometheus  "curl -sf http://localhost:9090/-/ready"
check grafana     "curl -sf http://localhost:3000/api/health"
echo "$ok ok, $fail failed"; [ "$fail" -eq 0 ]
