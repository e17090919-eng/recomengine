#!/usr/bin/env bash
# Start the native local infra (installed by deploy/native/install.sh). Same ports as docker-compose:
#   redis 6379 · postgres 5432 · kafka 9092 · clickhouse 8123/9000 · s3 (moto) 9100
#   prometheus 9090 · grafana 3000 (admin/admin)
# Data and logs live in $RECSYS_INFRA/{data,logs}; `down.sh` stops everything, `down.sh --wipe` also deletes data.
set -euo pipefail
P="${RECSYS_INFRA:-$HOME/opt/recsys-infra}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
D="$P/data"; L="$P/logs"; R="$P/run"
mkdir -p "$D" "$L" "$R"
export JAVA_HOME="$P/jdk/Contents/Home" PATH="$P/jdk/Contents/Home/bin:$PATH"
[ -x "$JAVA_HOME/bin/java" ] || { echo "not installed: run deploy/native/install.sh"; exit 1; }

listening() { lsof -iTCP:"$1" -sTCP:LISTEN -n -P >/dev/null 2>&1; }
wait_port() { for _ in $(seq 1 "${2:-60}"); do listening "$1" && return 0; sleep 1; done; echo "  timeout waiting for :$1 (see $L)"; return 1; }
bg() { local name=$1; shift; nohup "$@" >"$L/$name.log" 2>&1 & echo $! >"$R/$name.pid"; }
py() { "$REPO/.venv/bin/python" "$@"; }

# ---- redis
if ! listening 6379; then
  echo "redis"; mkdir -p "$D/redis"
  "$P/redis/bin/redis-server" --port 6379 --bind 127.0.0.1 --dir "$D/redis" --appendonly yes \
    --maxmemory-policy allkeys-lru --daemonize yes --pidfile "$R/redis.pid" --logfile "$L/redis.log"
  wait_port 6379
fi

# ---- postgres (user/password/db = recsys, schema from deploy/postgres/init.sql)
if ! listening 5432; then
  echo "postgres"
  if [ ! -f "$D/pg/PG_VERSION" ]; then
    echo recsys >"$R/pgpass"
    "$P/pgsql/bin/initdb" -D "$D/pg" -U recsys --pwfile="$R/pgpass" -A scram-sha-256 -E UTF8 --locale=C >"$L/pg-init.log"
    rm -f "$R/pgpass"; fresh_pg=1
  fi
  "$P/pgsql/bin/pg_ctl" -D "$D/pg" -l "$L/postgres.log" -o "-p 5432 -k $R -h 127.0.0.1" -w start >/dev/null
  if [ "${fresh_pg:-0}" = 1 ]; then
    py - "$REPO/deploy/postgres/init.sql" <<'EOF'
import sys, psycopg
with psycopg.connect("postgresql://recsys:recsys@127.0.0.1:5432/postgres", autocommit=True) as c:
    c.execute("CREATE DATABASE recsys")
with psycopg.connect("postgresql://recsys:recsys@127.0.0.1:5432/recsys", autocommit=True) as c:
    c.execute(open(sys.argv[1]).read())
print("  schema loaded")
EOF
  fi
fi

# ---- kafka (KRaft, single node) + topics
if ! listening 9092; then
  echo "kafka"
  cfg="$R/kafka.properties"
  cat >"$cfg" <<EOF
node.id=1
process.roles=broker,controller
listeners=PLAINTEXT://127.0.0.1:9092,CONTROLLER://127.0.0.1:9093
advertised.listeners=PLAINTEXT://127.0.0.1:9092
controller.listener.names=CONTROLLER
controller.quorum.voters=1@127.0.0.1:9093
listener.security.protocol.map=CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT
log.dirs=$D/kafka
num.partitions=6
offsets.topic.replication.factor=1
transaction.state.log.replication.factor=1
transaction.state.log.min.isr=1
auto.create.topics.enable=false
EOF
  if [ ! -f "$D/kafka/meta.properties" ]; then
    "$P/kafka/bin/kafka-storage.sh" format -t "$("$P/kafka/bin/kafka-storage.sh" random-uuid)" -c "$cfg" >"$L/kafka-format.log"
  fi
  KAFKA_HEAP_OPTS="-Xmx512m -Xms256m" LOG_DIR="$L/kafka" bg kafka "$P/kafka/bin/kafka-server-start.sh" "$cfg"
  wait_port 9092 90
  for t in behavior-events:12 item-changes:3 user-changes:3 rec-log:6; do
    "$P/kafka/bin/kafka-topics.sh" --bootstrap-server localhost:9092 --create --if-not-exists \
      --topic "${t%%:*}" --partitions "${t##*:}" --replication-factor 1 >/dev/null
  done
  echo "  topics ready"
fi

# ---- clickhouse (user recsys/recsys; schema from deploy/clickhouse/init.sql, Kafka engine -> localhost:9092)
if ! listening 8123; then
  echo "clickhouse"; mkdir -p "$D/clickhouse"
  cat >"$R/clickhouse.xml" <<EOF
<clickhouse>
  <logger><level>warning</level><log>$L/clickhouse.log</log><errorlog>$L/clickhouse.err.log</errorlog></logger>
  <path>$D/clickhouse/</path>
  <tmp_path>$D/clickhouse/tmp/</tmp_path>
  <user_files_path>$D/clickhouse/user_files/</user_files_path>
  <listen_host>127.0.0.1</listen_host>
  <http_port>8123</http_port>
  <tcp_port>9000</tcp_port>
  <mysql_port remove="1"/><postgresql_port remove="1"/>
  <users>
    <default><password></password><networks><ip>127.0.0.1</ip></networks><profile>default</profile><quota>default</quota></default>
    <recsys><password>recsys</password><networks><ip>127.0.0.1</ip><ip>::1</ip></networks><profile>default</profile><quota>default</quota><access_management>1</access_management></recsys>
  </users>
  <profiles><default/></profiles>
  <quotas><default/></quotas>
</clickhouse>
EOF
  bg clickhouse "$P/bin/clickhouse" server --config-file="$R/clickhouse.xml"
  wait_port 8123 60
  sed 's/kafka:19092/localhost:9092/' "$REPO/deploy/clickhouse/init.sql" |
    "$P/bin/clickhouse" client --user recsys --password recsys --multiquery
  echo "  schema loaded"
fi

# ---- S3 stand-in (moto) + buckets
if ! listening 9100; then
  echo "s3 (moto)"
  bg s3 uvx --quiet --from 'moto[server]==5.0.28' moto_server -H 127.0.0.1 -p 9100
  wait_port 9100 120
  for b in recsys-lake recsys-models; do curl -sf -X PUT "http://127.0.0.1:9100/$b" >/dev/null; done
  echo "  buckets ready"
fi

# ---- prometheus (scrapes rec-server on :8000)
if ! listening 9090; then
  echo "prometheus"
  sed 's/host.docker.internal:8000/127.0.0.1:8000/' "$REPO/deploy/prometheus/prometheus.yml" >"$R/prometheus.yml"
  bg prometheus "$P/prometheus/prometheus" --config.file="$R/prometheus.yml" --storage.tsdb.path="$D/prometheus" \
    --storage.tsdb.retention.time=15d --web.listen-address=127.0.0.1:9090
  wait_port 9090
fi

# ---- grafana (datasource -> local prometheus, dashboards from deploy/grafana/dashboards)
if ! listening 3000; then
  echo "grafana"
  G="$R/grafana-provisioning"; mkdir -p "$G/datasources" "$G/dashboards" "$D/grafana"
  sed 's|http://prometheus:9090|http://127.0.0.1:9090|' "$REPO/deploy/grafana/provisioning/datasources/prometheus.yml" >"$G/datasources/prometheus.yml"
  sed "s|/var/lib/grafana/dashboards|$REPO/deploy/grafana/dashboards|" "$REPO/deploy/grafana/provisioning/dashboards/default.yml" >"$G/dashboards/default.yml"
  GF_PATHS_DATA="$D/grafana" GF_PATHS_LOGS="$L" GF_PATHS_PROVISIONING="$G" GF_SERVER_HTTP_ADDR=127.0.0.1 \
  GF_SECURITY_ADMIN_USER=admin GF_SECURITY_ADMIN_PASSWORD=admin GF_ANALYTICS_REPORTING_ENABLED=false \
  GF_ANALYTICS_CHECK_FOR_UPDATES=false \
    bg grafana "$P/grafana/bin/grafana" server --homepath "$P/grafana"
  wait_port 3000 90
fi

echo "infra up. engine against it: make serve-infra   (check: make infra-native-check)"
