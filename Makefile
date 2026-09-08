.DEFAULT_GOAL := help
SHELL := /bin/bash

# The virtualenv lives in venv/ (no leading dot). On some macOS setups a background process
# marks dot-prefixed directories hidden, and Python 3.12.12+ skips hidden .pth files, which
# silently breaks editable installs of the workspace packages. Set the same variable in your
# shell to use plain `uv run` outside make:  export UV_PROJECT_ENVIRONMENT=venv
export UV_PROJECT_ENVIRONMENT ?= venv

COMPOSE := docker compose -f infra/docker-compose.yml

help: ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install all workspace packages and dev tools
	uv sync --all-packages

up: ## Start Postgres (pgvector) and MinIO
	$(COMPOSE) up -d postgres minio minio-init
	@echo "waiting for postgres..." && until $(COMPOSE) exec -T postgres pg_isready -U outlier -d outlier >/dev/null 2>&1; do sleep 1; done
	@echo "postgres ready on 127.0.0.1:5433, minio console on :9001"

down: ## Stop services (keeps volumes)
	$(COMPOSE) down

reset-db: ## Destroy volumes and restart Postgres from scratch
	$(COMPOSE) down -v
	$(MAKE) up

migrate: ## Apply Alembic migrations
	cd packages/backend && uv run alembic upgrade head

migration: ## Autogenerate a migration: make migration m="add foo"
	cd packages/backend && uv run alembic revision --autogenerate -m "$(m)"

seed: ## Seed the synthetic dataset
	uv run oai synth seed --briefs 20 --trajectories 400 --seed 1

lint: ## Ruff lint + format check
	uv run ruff check .
	uv run ruff format --check .

fmt: ## Ruff format + autofix
	uv run ruff format .
	uv run ruff check --fix .

typecheck: ## Pyright
	uv run pyright

test: ## Fast tests (unit + integration; integration skipped if TEST_DATABASE_URL unreachable)
	uv run pytest -m "not slow and not contract" -q

test-unit: ## Unit tests only
	uv run pytest tests/unit -q

test-slow: ## Simulator experiments and trainer smoke
	uv run pytest -m slow -q

api: ## Run the API on 127.0.0.1:8000 with reload
	uv run oai serve --reload

openapi: ## Export OpenAPI schema for the frontend
	uv run oai openapi

check: lint typecheck test ## Everything CI runs

.PHONY: help install up down reset-db migrate migration seed lint fmt typecheck test test-unit test-slow api openapi check
