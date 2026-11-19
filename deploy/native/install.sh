#!/usr/bin/env bash
# Install the local infra natively (no Docker, no sudo) under $RECSYS_INFRA (default ~/opt/recsys-infra).
# Mirrors deploy/docker-compose.yml: JDK 21 + Maven (Java SDK), Redis, PostgreSQL, Kafka (KRaft),
# ClickHouse, Prometheus, Grafana. MinIO's macOS binaries are no longer published, so the S3 stand-in
# is moto (python, run via uvx). Idempotent: skips anything already installed.
set -euo pipefail
P="${RECSYS_INFRA:-$HOME/opt/recsys-infra}"
mkdir -p "$P/dl" "$P/bin"
cd "$P"
fetch() { [ -s "dl/$2" ] || curl -fL --retry 3 -o "dl/$2" "$1"; }
step() { echo "==> $*"; }

if [ ! -x jdk/Contents/Home/bin/java ]; then
  step JDK 21
  fetch "https://api.adoptium.net/v3/binary/latest/21/ga/mac/aarch64/jdk/hotspot/normal/eclipse" jdk21.tar.gz
  rm -rf jdk && mkdir jdk && tar -xzf dl/jdk21.tar.gz -C jdk --strip-components 1
fi
if [ ! -x maven/bin/mvn ]; then
  step Maven
  fetch "https://archive.apache.org/dist/maven/maven-3/3.9.9/binaries/apache-maven-3.9.9-bin.tar.gz" maven.tar.gz
  rm -rf maven && mkdir maven && tar -xzf dl/maven.tar.gz -C maven --strip-components 1
fi
if [ ! -x redis/bin/redis-server ]; then
  step Redis "(compile)"
  fetch "https://download.redis.io/releases/redis-7.4.1.tar.gz" redis.tar.gz
  rm -rf redis-src && mkdir redis-src && tar -xzf dl/redis.tar.gz -C redis-src --strip-components 1
  make -C redis-src -j8 BUILD_TLS=no >/dev/null
  make -C redis-src PREFIX="$P/redis" install >/dev/null
  rm -rf redis-src
fi
if [ ! -x pgsql/bin/postgres ]; then
  step PostgreSQL 16
  fetch "https://repo1.maven.org/maven2/io/zonky/test/postgres/embedded-postgres-binaries-darwin-arm64v8/16.4.0/embedded-postgres-binaries-darwin-arm64v8-16.4.0.jar" pg.jar
  rm -rf pgsql pgtmp && mkdir pgsql pgtmp
  (cd pgtmp && unzip -q ../dl/pg.jar) && tar -xJf pgtmp/*.txz -C pgsql && rm -rf pgtmp
fi
if [ ! -x kafka/bin/kafka-server-start.sh ]; then
  step Kafka 3.8
  fetch "https://archive.apache.org/dist/kafka/3.8.0/kafka_2.13-3.8.0.tgz" kafka.tgz
  rm -rf kafka && mkdir kafka && tar -xzf dl/kafka.tgz -C kafka --strip-components 1
fi
if [ ! -x bin/clickhouse ]; then
  step ClickHouse 24.8
  fetch "https://github.com/ClickHouse/ClickHouse/releases/download/v24.8.4.13-lts/clickhouse-macos-aarch64" clickhouse
  cp dl/clickhouse bin/clickhouse && chmod +x bin/clickhouse
  xattr -d com.apple.quarantine bin/clickhouse 2>/dev/null || true
fi
if [ ! -x prometheus/prometheus ]; then
  step Prometheus
  fetch "https://github.com/prometheus/prometheus/releases/download/v2.54.1/prometheus-2.54.1.darwin-arm64.tar.gz" prom.tar.gz
  rm -rf prometheus && mkdir prometheus && tar -xzf dl/prom.tar.gz -C prometheus --strip-components 1
fi
if [ ! -x grafana/bin/grafana ]; then
  step Grafana 11.2
  fetch "https://dl.grafana.com/oss/release/grafana-11.2.0.darwin-arm64.tar.gz" grafana.tar.gz
  rm -rf grafana && mkdir grafana && tar -xzf dl/grafana.tar.gz -C grafana --strip-components 1
fi
step "done -> $P"
