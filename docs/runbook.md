# Runbook

## Local development

```
make install && cp .env.example .env
make up && make migrate && make seed
make api          # :8000       make web   # :3000
make test         # unit + integration (Postgres on 127.0.0.1:5433)
make test-slow    # simulator experiments (minutes)
make e2e          # Playwright on its own outlier_e2e database (never the dev DB), API :8100 + Next :3100
```

Seeded users: `operator@example.com`, `researcher@example.com`, `expert@example.com`. Set passwords
with `uv run oai users set-password --email <email>`.

Useful commands:

```
uv run oai generate first --backend fake --k 8 --no-llm     # one generation pass
uv run oai loop-a run first --cap 5000 --backend fake       # whole simulated episode
uv run oai experiments loop-a-vs-random --briefs 6 --seeds 1,2
uv run oai outlier recompute                                # re-derive tiers from raw rows
uv run oai archive sample first                             # what the prompt would include
uv run oai prompt render first --k 4                        # the prompt itself
uv run oai fake advance --days 7                            # move the fake clock
uv run oai jobs run sync_insights | sync_comments | episode_tick | recompute_outcomes
uv run oai jobs worker                                      # long-running cadence
```

## Neon (cloud Postgres)

Neon's connection string uses libpq parameters (`sslmode=require&channel_binding=require`) that
asyncpg does not accept. Rewrite it as `postgresql+asyncpg://USER:PASSWORD@HOST/neondb?ssl=require`.

- `DATABASE_URL`: the `-pooler` host. The engine already sets `statement_cache_size=0`, which
  pgbouncer's transaction mode requires.
- `ALEMBIC_DATABASE_URL`: the direct host (same name without `-pooler`) for `alembic upgrade head`
  and other long sessions.
- `TEST_DATABASE_URL` stays on the local Docker Postgres: the test suite drops and recreates every
  table.
- pgvector 0.8.x and pg_trgm are available on Neon; Alembic creates both extensions.
- Turn scale-to-zero off for the production branch so the worker's daily cadence does not hit a
  cold start, and rotate the password if the connection string was ever pasted anywhere.

## Connecting a real Meta account (do not skip steps)

Before any of this: set `SIGNUP_ENABLED=false` so accounts on a deployment that can spend money are
created by an operator (`oai users create`) rather than by anyone who finds the sign-up page.

The Graph API version is pinned in `config.meta.api_version` (currently `v26.0`). Bumping it is a
deliberate change: run `pytest -m contract` against the sandbox first (`tests/contract/`).

Real money only after all of the following:

1. **Simulator gate.** `make test-slow` passes: Loop A finds tier 2 at least as often as random
   selection on the synthetic latent model, and every simulated episode reaches a terminal state.
2. **Contract tests.** `META_SANDBOX_TOKEN` set and `uv run pytest -m contract` green against a
   Meta sandbox ad account (recorded cassettes under `tests/contract/`).
3. **Meta access.** App Review approved for `ads_management`, `ads_read`, `pages_show_list`,
   `pages_read_user_content`; Business Verification complete; Full Access tier reached (500+
   successful calls in 15 days under 15 percent errors, made from the sandbox during Phase 4).
4. **Cards.** Real expert-authored style, principle, and mechanic cards are `active`; the synthetic
   placeholder cards (`contributed_by = synthetic`) are retired.
5. **Brand assets** uploaded to storage and referenced from the brief.
6. **Account registered** on the Model screen (or `POST /api/meta/accounts`) with encrypted tokens,
   the account's attribution setting, and a daily cap that matches what finance approved.
7. **Kill switch** verified: disable on the Model screen, confirm `POST /api/episodes/{id}/ship`
   returns 403, re-enable.
8. **`DRY_RUN=false` only in the production environment.** Everywhere else the real client is never
   constructed; the factory raises `SafetyError`.
9. **First episode**: small cap (one batch), `approval=manual`, an operator labels every idea, and
   the worker's daily cadence (`oai jobs worker`) handles review polling, insights, comments, and
   the controller. Watch the audit log for `trajectory.ship_refused` and `account.needs_reauth`.

## Operations

- **Token expired / API version retired**: the sync job marks the account `needs_reauth`, the
  Generate screen blocks Run for it, nothing else stops. Rotate the token with
  `PUT /api/meta/accounts/{id}` (`access_token`, `page_token`); status returns to `active`.
- **Threshold change**: apply on the Model screen; tiers are recomputed from raw daily rows and every
  derived row records the config hash it was computed with.
- **Runaway spend**: kill switch on the Model screen (or `PUT /api/meta/kill_switch`). Live ads are
  paused by the controller on the next tick when an episode ends; pause manually in Ads Manager if
  the worker is down.
- **Held-out briefs**: add campaign ids to `holdout_campaigns`; the archive sampler, reward-model
  training set, verifier few-shot, and baselines all exclude them.

## Loop B and the weekly feedback cadence

- The worker enqueues `suggest_relations`, `retrain_verifier`, `retrain_rm`, and `export_snapshot`
  once a week (`rl.feedback_weekday`, Sunday by default) after the insights sync. Each is idempotent
  on its `weekly:<ISO week>` key.
- Training runs are launched from the Model screen or `oai train launch <stage>`. A launch always
  exports a fresh snapshot (holdout excluded) and pins config hash, snapshot hash, RM version, and
  verifier version on the `training_runs` row. Full runs refuse to start below
  `rl.min_positives_to_train` tier 2+ trajectories; dry and smoke runs are always allowed.
- The `train` job runs `python -m outlier_trainer.run` as a subprocess, uploads the log and the
  LoRA adapter, and copies `run.json` into `training_runs.metrics.trainer`. Register the adapter
  with vLLM by pointing `generation.vllm_url` at a server started with `--enable-lora` and the
  checkpoint prefix; the `loop_b` eval system uses adapter name `loop_b`.
- The gold gap (`GET /api/stats/gold_gap`) compares the active reward model's mean score with the
  realized tier 2 rate across the two most recent windows of measured ideas. When it trips, queued
  runs are set to `paused` and new launches are refused until the RM is retrained. Loop A never
  pauses.
- On-policy GRPO is the only stage whose reward is not a real outcome. It runs only with
  `--simulator` (fake account) or an explicit `--allow-rm-reward`, and the API enforces the same.
- Online eval: `POST /api/eval/launch` picks held-out briefs, blocklists their campaigns, assigns
  blind labels, runs fake arms to completion and starts live arms as normal episodes. Refresh live
  runs with `POST /api/eval/{id}/refresh`. The success rule needs 20+ briefs and non-overlapping
  bootstrap intervals; below that the verdict is "no verdict", never "failed".
