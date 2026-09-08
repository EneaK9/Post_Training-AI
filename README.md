# Outlier AI

Search for outlier ad campaigns, not above-average ones, and train a model that gets better at
finding them. Spec: [docs/spec_v4.md](docs/spec_v4.md). Reconstructed assumptions:
[docs/assumptions.md](docs/assumptions.md).

## Layout

```
packages/schemas    shared pydantic models, enums, typed config (no IO)
packages/backend    FastAPI backend, Loop A, evaluator integration, jobs, CLI `oai`
packages/trainer    Loop B trainer (GPU extras optional)
frontend/           Next.js (Phase 3+)
config/             config.yaml, category_medians.yaml
infra/              docker compose (Postgres 18 + pgvector, MinIO, optional vLLM)
tests/              pytest (unit, integration, slow, contract)
```

## Quick start

```
make install          # uv sync --all-packages into venv/
cp .env.example .env
make up               # postgres + minio
make migrate
make seed             # synthetic dataset
make test
```

The virtualenv is `venv/`, not `.venv/`. To use `uv run` directly instead of `make`, set
`export UV_PROJECT_ENVIRONMENT=venv` in your shell. Reason: on macOS with iCloud-synced folders,
dot-prefixed directories get the hidden flag and Python 3.12.12+ then ignores the `.pth` files
that wire up editable installs.

`DRY_RUN=true` is the default everywhere. The real Meta client is only constructed when
`DRY_RUN=false`, the kill switch is open, and the ad account is active.
