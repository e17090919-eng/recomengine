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

serve: train  ## API + demo UI on :8000
	$(UV) run recsys serve

infra-up:  ## start local infra (docker compose)
	docker compose -f deploy/docker-compose.yml up -d

infra-down:  ## stop local infra
	docker compose -f deploy/docker-compose.yml down

infra-check:  ## verify local infra endpoints respond
	./deploy/check-infra.sh

.PHONY: help setup lint fmt typecheck test check docs raw backfill quality baseline p0 train serve infra-up infra-down infra-check
