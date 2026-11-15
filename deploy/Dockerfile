# rec-server image. Build from the repo root:  docker build -f deploy/Dockerfile -t rec-server .
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages ./packages
COPY tools ./tools
RUN uv sync --all-packages --frozen --no-dev

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build /app /app
COPY configs ./configs
ENV PATH="/app/.venv/bin:$PATH" RECSYS_HOME=/app PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
# artifacts/engine.pkl (+ ranker.onnx) is mounted (k8s: PVC or init container pulling from object storage).
# WORKERS > 1 needs [state] backend = "redis" (shared real-time state).
ENV WORKERS=1
# metrics of all workers are aggregated through this directory (cleared on start)
CMD ["sh", "-c", "rm -rf /tmp/prom && mkdir -p /tmp/prom && PROMETHEUS_MULTIPROC_DIR=/tmp/prom exec recsys serve --host 0.0.0.0 --port 8000 --workers ${WORKERS} --log-level warning"]
