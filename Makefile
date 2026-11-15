# Common developer tasks. `make help` lists them.
.DEFAULT_GOAL := help
UV ?= uv
RAW ?= raw
MAPPING ?= configs/mappings/sample_vendor.toml

help:  ## list targets
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

setup:  ## install all workspace packages + dev tools
	$(UV) sync --all-packages

lint:  ## ruff lint + format check
	$(UV) run ruff check .
	$(UV) run ruff format --check .

fmt:  ## auto-format and fix lint
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

typecheck:  ## mypy
	$(UV) run mypy

test:  ## pytest
	$(UV) run pytest

check: lint typecheck test  ## everything CI runs (except docker build)
	$(UV) run recsys config-check

docs:  ## regenerate docs/data-contract.md from the schema
	$(UV) run recsys schema-doc

raw:  ## generate a vendor-style raw export (stand-in for customer data)
	$(UV) run recsys simulate --raw $(RAW) --no-lake

backfill:  ## raw export -> canonical lake (+ reports/backfill.md)
	$(UV) run recsys backfill --mapping $(MAPPING) --raw $(RAW)

quality:  ## data-quality gate on the lake (+ reports/quality.md)
	$(UV) run recsys quality

baseline:  ## offline baseline report (+ reports/baseline.md)
	$(UV) run recsys baseline $(if $(wildcard $(RAW)/_truth.npz),--truth $(RAW)/_truth.npz,)

p0: raw backfill baseline  ## full P0 pipeline: raw -> backfill -> quality -> baseline

train:  ## train on the full lake -> artifacts/engine.pkl
	$(UV) run recsys train

serve: train  ## API + demo UI on :8000 (console: /admin)
	$(UV) run recsys serve

WORKERS ?= 8
serve-multi:  ## N workers (WORKERS=8); needs [state] backend = "redis" for real-time consistency
	$(UV) run recsys serve --workers $(WORKERS) --log-level warning

ab-sim: train  ## online A/B simulation through the production path + daily report
	$(UV) run recsys ab-sim --sessions 20000

DATE ?=
report:  ## T+1 daily report for DATE=YYYYMMDD on the lake
	$(UV) run recsys report --date $(DATE)

loadtest:  ## load test a running server (make serve-multi in another shell)
	$(UV) run recsys loadtest --concurrency 32,64,128 --rate 2000,3000 --duration 20

demo:  ## demo shop on :8100 driven by the engine on :8000 (Ctrl-C stops both); see docs/demo-script.md
	@test -f lake/item.parquet || $(MAKE) p0
	@test -f artifacts/engine.pkl || $(MAKE) train
	@rm -f artifacts/demo-config.db artifacts/demo-config.db-*
	@RECSYS_DEMO=1 RECSYS_INGEST_LAKE=lake_demo RECSYS_CONFIG_STORE=sqlite:///artifacts/demo-config.db \
	  $(UV) run recsys serve & ENGINE=$$!; trap 'kill $$ENGINE 2>/dev/null' EXIT INT TERM; \
	  echo "shop: http://127.0.0.1:8100   engine console: http://127.0.0.1:8000/admin"; \
	  $(UV) run demo-shop --engine http://127.0.0.1:8000 --lake lake

p1: p0 train ab-sim  ## P0 + train (ONNX export) + online A/B simulation

infra-up:  ## start local infra (docker compose)
	docker compose -f deploy/docker-compose.yml up -d

infra-down:  ## stop local infra
	docker compose -f deploy/docker-compose.yml down

infra-check:  ## verify local infra endpoints respond
	./deploy/check-infra.sh

.PHONY: help setup lint fmt typecheck test check docs raw backfill quality baseline p0 train serve serve-multi ab-sim report loadtest demo p1 infra-up infra-down infra-check
