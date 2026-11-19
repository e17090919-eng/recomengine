#!/usr/bin/env bash
# Stop the native local infra. --wipe also deletes its data (redis, postgres, kafka, clickhouse, prometheus, grafana).
set -uo pipefail
P="${RECSYS_INFRA:-$HOME/opt/recsys-infra}"; R="$P/run"
[ -f "$P/data/pg/postmaster.pid" ] && "$P/pgsql/bin/pg_ctl" -D "$P/data/pg" -m fast stop >/dev/null && echo "stopped postgres"
for n in grafana prometheus s3 clickhouse kafka redis; do
  f="$R/$n.pid"; [ -f "$f" ] || continue
  pid=$(cat "$f")
  if kill -0 "$pid" 2>/dev/null; then
    pkill -TERM -P "$pid" 2>/dev/null; kill -TERM "$pid" 2>/dev/null
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid"
    echo "stopped $n"
  fi
  rm -f "$f"
done
if [ "${1:-}" = "--wipe" ]; then rm -rf "$P/data" "$P/logs" "$R"; echo "data wiped"; fi
