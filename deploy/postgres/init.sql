-- Metadata store: versioned configuration, model registry, audit log.
-- (Operational config is edited in the console in P1; until then configs/engine.toml is the source.)

CREATE TABLE IF NOT EXISTS config_version (
    id          BIGSERIAL PRIMARY KEY,
    app_id      TEXT        NOT NULL,
    kind        TEXT        NOT NULL CHECK (kind IN ('engine', 'mapping')),
    content     JSONB       NOT NULL,
    comment     TEXT,
    created_by  TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    is_active   BOOLEAN     NOT NULL DEFAULT FALSE
);
CREATE UNIQUE INDEX IF NOT EXISTS config_version_one_active
    ON config_version (app_id, kind) WHERE is_active;

CREATE TABLE IF NOT EXISTS model_registry (
    id            BIGSERIAL PRIMARY KEY,
    app_id        TEXT        NOT NULL,
    name          TEXT        NOT NULL,           -- e.g. ranker, swing, embedding
    version       TEXT        NOT NULL,
    artifact_uri  TEXT        NOT NULL,           -- s3://recsys-models/...
    train_cutoff  TIMESTAMPTZ NOT NULL,
    metrics       JSONB       NOT NULL DEFAULT '{}',
    status        TEXT        NOT NULL DEFAULT 'candidate'
                  CHECK (status IN ('candidate', 'canary', 'live', 'retired', 'rejected')),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (app_id, name, version)
);

CREATE TABLE IF NOT EXISTS audit_log (
    id         BIGSERIAL PRIMARY KEY,
    actor      TEXT        NOT NULL,
    action     TEXT        NOT NULL,
    target     TEXT        NOT NULL,
    detail     JSONB       NOT NULL DEFAULT '{}',
    at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
