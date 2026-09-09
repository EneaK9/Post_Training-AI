# Next steps

State on 2026-09-08: phases 0 through 6 of the plan are implemented, the fast suite (138 tests),
Playwright (10), lint, and type checks are green, and everything is committed and pushed to
`EneaK9/Post_Training-AI`. `ANTHROPIC_API_KEY` is set in `.env`. The items below are ordered by
what unblocks the most. Items 1 and 2 are cheap; 3 and 4 are the path to real money; 5 and 6
are product and infrastructure; 7 waits for data.

## 1. Move the repo out of iCloud

`~/Documents` is iCloud-synced. That is what forced the `venv/` workaround (macOS marks
dot-directories hidden and Python skips hidden `.pth` files), and syncing a live `.git`
directory is a known way to corrupt it. The remote protects the history, not the working copy.

- Move the checkout to a non-synced path such as `~/Projects/Post_Training-AI`.
- Remove `UV_PROJECT_ENVIRONMENT=venv` from the Makefile and pyright config, delete `venv/`,
  and let uv use a normal `.venv`. Re-run `make install && make check`.

## 2. Close the simulator gate (spec step 12)

Loop A currently ties random selection in the simulator: 24 runs per arm, 8% found rate each,
Loop A faster when it finds (23 vs 31 days). The gate before real spend is Loop A beating
random with separated intervals. The likely cause is thin synthetic history: about one seeded
trajectory in 120 is tier 2+, so the archive carries almost no signal.

Work, in order:
- Seed outlier-denser history: raise the tier 2 prevalence in `synthetic/latent.py` for the
  seeded history, or seed more trajectories, so the archive holds real signal per brief.
- Check that the fake "archive" policy (`generation/backends/fake_backend.py`) exploits the
  same niche projection the latent model rewards (`archive.niche_projection`, default
  `strategy_only`). If the latent rewards strategy x style but the archive keys on strategy
  only, the conditioning cannot help.
- Raise the per-run budget and `max_days` so more than ~8 ideas ship per run; the current
  runs end after 3 batches.
- Re-run: `oai experiments loop-a-vs-random --briefs 8 --seeds 1,2,3 --history 250 --cap 4000
  --max-days 150 --out /tmp/exp.json` against the `outlier_exp` database
  (`DATABASE_URL=postgresql+asyncpg://outlier:outlier@127.0.0.1:5433/outlier_exp`).
- Tighten the slow test to assert a margin once the arms separate; today it only asserts
  Loop A is not worse.

If the arms still do not separate after these, the archive conditioning itself needs work.
That is much cheaper to learn on the simulator than on Meta.

## 3. Real generator: first batch done, what it showed

`oai generate first --backend anthropic --k 8` ran on 2026-09-09: 8 ideas queued, 0 rejected,
$0.34, about 3.5 minutes. Every idea cited cards and signals, 7 of 8 cited history, copy lengths
were inside the Meta limits, and the typicality spread was 3 common / 3 uncommon / 2 rare.
Three things needed fixing and are fixed:

- **Reward model scores were all 0.** An ensemble trained on 43 near-uniform expert labels had
  been activated and scored out-of-distribution ideas at sigmoid(-30). Trained ensembles now need
  `reward_model.min_rows`, `min_per_class`, and a held-out AUC of at least `min_val_auc` before
  activation, and loading re-checks the guards; otherwise the Claude cold reward model is used.
  `oai rm rescore` re-scores queued ideas after the model changes. On the synthetic seed the
  held-out AUC is 0.30, so the guard correctly keeps the ensemble inactive.
- **The verifier over-tagged** (5 to 7 cards per idea, `bundle-offer` on all of them because the
  brief's offer line is in the copy). The prompt now tags only the cards that organize the
  execution, 2 to 4 of them; mean Jaccard against the written cards went from 0.47 to 0.54 on the
  same 8 ideas. Mismatch remains a signal for experts, not an error.
- **Claude invented social proof** ("41,000 kits shipped"). The pre-ship screen caught it, and
  the prompt now forbids figures not in the brief or history. A first version of that rule made Claude scatter `[number]` placeholders, nine in one idea and some in headlines; the rule now allows at most one placeholder per idea, never in the headline, and asks for claims that need no figure.

Next reads: run 3 to 5 more batches on different briefs and read them on the Generate screen with
"What the model saw" open. Watch whether the invented-figure rule holds, how often the pre-ship
screen flags health or competitor claims in skincare, and whether the rare ideas are actually rare.
Run `pytest -m contract` once so the Anthropic contract test is recorded green.

## 4. Start the Meta long-lead work now

App review, business verification, and the Full Access track record (500 successful API
calls in 15 days with under 15% errors) take weeks to months and are mostly waiting.

- Create the Meta app, request `ads_management`, `ads_read`, `pages_show_list`,
  `pages_read_user_content`.
- Create a sandbox ad account, set `META_SANDBOX_TOKEN` and `META_SANDBOX_ACCOUNT_ID`, and
  run `pytest -m contract` so the client is proven against live `v26.0` early. Add the
  missing calls to `tests/contract/test_meta_contract.py` as cassettes: upload image, create
  creative, create ad, insights with the four engagement action types, comments via
  `effective_object_story_id`.
- Follow `docs/runbook.md` "Connecting a real Meta account" in order. `DRY_RUN=false` only in
  prod, first episode with human approval and a small cap.

## 5. Get real cards

Only the 8 strategy cards are real. Every style, principle, and mechanic card in the playbook
is a synthetic placeholder marked `contributed_by = synthetic`. Nothing should ship on
placeholder cards.

- Have the experts author cards through the Data screen (draft, then activate) or the CSV
  import at `/api/import/cards`.
- Retire the synthetic style, principle, and mechanic cards before the first real episode.
- Import any practitioner trajectories with outcomes through the CSV importer so the archive
  and baselines start with real history.

## 6. Deploy

There is a compose file for local Postgres and MinIO but no Dockerfiles and no Neon project.

- Create the Neon project (Postgres 18, pgvector), run `alembic upgrade head` against the
  direct endpoint, turn scale-to-zero off in prod.
- Add `infra/Dockerfile.backend`, `Dockerfile.worker`, `Dockerfile.frontend`; the API is
  `oai serve`, the worker is `oai jobs worker`.
- S3 or R2 for object storage (`STORAGE_BACKEND=s3`), a real `FERNET_KEY`, `PII_HMAC_KEY`,
  and per-account daily caps set in `config.yaml` before any real account is added.

## 7. Loop B, when there is data

Real training waits for 50 tier 2+ outcomes (`rl.min_positives_to_train`), which only real
campaigns produce. Before then, prove the pipeline on a GPU box:

```bash
uv sync --all-packages --extra train
uv run --package outlier-trainer python -m outlier_trainer.run --stage rft \
  --snapshot tests/fixtures/mini_snapshot.parquet --out /tmp/rft --base-model Qwen/Qwen3-0.6B --smoke
```

Repeat for `dpo`, `grpo_offpolicy`, and `grpo_onpolicy --simulator`. The CI job
`trainer-smoke` does the same on a `gpu`-labeled PR. Then start vLLM with `--enable-lora`
and point `generation.vllm_url` at it so the `loop_a_local` and `loop_b` eval systems work.

## Suggested order

Do 1 and 2 this week. Kick off 4 immediately since it is mostly waiting. Run 3 and 5 in
parallel with 2, because reading real ideas and having real cards both change what the
simulator should be testing. 6 follows once 3 and 5 look right. 7 waits for data.
