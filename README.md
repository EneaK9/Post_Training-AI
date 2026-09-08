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
make api             # FastAPI on :8000
make web             # Next.js on :3000 (proxies /api to :8000)
make e2e             # Playwright against a fresh stack on :8100/:3100
```

Seeded users are `operator@example.com`, `researcher@example.com`, `expert@example.com`; set a
password with `uv run oai users set-password --email expert@example.com`.

The virtualenv is `venv/`, not `.venv/`. To use `uv run` directly instead of `make`, set
`export UV_PROJECT_ENVIRONMENT=venv` in your shell. Reason: on macOS with iCloud-synced folders,
dot-prefixed directories get the hidden flag and Python 3.12.12+ then ignores the `.pth` files
that wire up editable installs.

`DRY_RUN=true` is the default everywhere. The real Meta client is only constructed when
`DRY_RUN=false`, the kill switch is open, and the ad account is active.

## Loop B, reward model, eval (Phase 6)

Everything below runs against the synthetic archive; nothing here gates Loop A.

```bash
uv run oai archive export                      # Parquet snapshot -> object storage, row in archive_snapshots
uv run oai rm train                            # ensemble reward model (rm_cold from expert labels, rm_outcome once 50 tier 2+)
uv run oai rm gold-gap                         # proxy-vs-real drift; a trip pauses queued training runs
uv run oai train launch rft --dry              # snapshot + training_runs row + trainer subprocess (counts only)
uv run oai train launch rft --smoke            # a few real steps; needs the `train` extra (GPU box)
uv run oai train launch grpo_onpolicy --simulator   # the literal 7.3 loop, gated to the simulator
uv run oai eval launch --systems loop_a_fake,random_fake --briefs 3   # blind arms on held-out briefs
uv run oai feedback suggest-relations          # card pairs co-occurring in tier 2+ above chance -> pending relations
uv run oai feedback retrain-verifier           # classifier v2 once wrong_cards labels pass verifier.classifier_min_labels
```

Trainer stages (`packages/trainer/outlier_trainer`): `rft` (SFT on tier 2+), `dpo` (tier 2+ vs tier 0 pairs
on the same brief), `grpo_offpolicy` (custom loop over archived completions with the section 7.3
advantage transform, clipped sequence ratios, KL to a reference refreshed only at run boundaries),
`grpo_onpolicy` (TRL `GRPOTrainer` subclass; RM-shaped reward, so it only runs with `--simulator` or
`--allow-rm-reward`). Install GPU dependencies with `uv sync --package outlier-trainer --extra train`.
The Model screen exposes the same actions to researchers: Launch run (dry / smoke / full), Run eval,
Train reward model, plus the gold-gap badge.
