# v3 reconstructions and other assumptions

Spec v4 says "unchanged from v3" for several parts. The v3 document is not available, so each
of those parts is reconstructed here from the v4 summaries. Every item is tagged
**[v3-assumed]** in code comments and in the plan. If v3 turns up, reconcile against this list.

## SearchEpisode and Batch [v3-assumed]

- `SearchEpisode`: `brief_id`, `ad_account_id`, `backend`, `budget_cap`, `spent`, `status`
  (`searching | outlier_found | budget_exhausted | stopped`), `keep_running_after_outlier`,
  `config_hash`, `created_by`, timestamps.
- `Batch` (not named in v4, implied by "batches"): `episode_id`, `index`, `state`
  (`proposed → approved → shipping → in_review → screening → scaling | stopped → measured`),
  `prompt_trace` (exactly what the model saw).
- Overlapping batches: the next batch may be generated as soon as the previous batch's
  screening resolves. v4 only says each batch is conditioned on the episode's own history
  including signals from failed batches, which screening already produces.

## Image rendering [v3-assumed]

- `ImageBackend.render(idea, brand_assets, size, seed) -> Render`.
- `brand_assets` mode is a Pillow compositor: brand image plus headline overlay templates for
  1080x1080 and 1080x1350; three seeds yield three layout variants.
- `generate` mode is a stub in v1.
- Meta copy length warnings at 125 / 40 / 30 characters (primary text / headline / description).

## Pre-ship checks [v3-assumed]

- Policy screen: LLM classification against Meta ad policy categories plus a keyword denylist.
- Brand constraints: `brief.constraints` checked against copy and visual brief.
- The human `run` review label is the approval; nothing ships without it.

## Reward model [v3-assumed]

- Three MLP heads on frozen text embeddings of (brief, idea, verified combination, cited
  confirmed signals, world state). `rm_score = mean - lambda_pess * std`. Focal loss.
- `rm_cold` trains on expert `run`/`skip` labels; `rm_outcome` on real tiers once 50 positives exist.
- Gold gap: proxy score trend vs realized tier rate; trips pause Loop B only.

## Meta campaign structure [v3-assumed]

- One campaign per episode, purchase objective, feed placement only.
- One ad set per ad, identical targeting from the brief, equal fixed daily budgets, no CBO.
- Names `OAI|<episode>|<trajectory>|<render>`.
- Screening pass raises the ad set daily budget to the scale budget; fail stops the ad.

## Evaluation [v3-assumed]

- Online eval on held-out briefs, equal budgets, blind arm labels, `tier2_rate` per arm with
  bootstrap intervals, success = non-overlapping intervals vs mean-objective RL on ≥ 20 briefs.
- Leakage blocklist by campaign id on retrieval, reward model, verifier few-shot, and baselines.

## Stack [v3-assumed]

- FastAPI, pydantic, Postgres + pgvector, TRL worker, Next.js, `facebook_business` behind one
  client with a fake, encrypted tokens, pytest and Playwright, synthetic generator with comments.

## Decisions not in the spec

| Topic | Decision |
|---|---|
| Loop B reward | Literal section 7.3 generates fresh rollouts that can never carry a real outcome, so it reduces to RL against the reward model. Loop B is built as an offline ladder (RFT → DPO → off-policy GRPO over the archive with real tiers). The literal on-policy loop exists as a mode gated behind `--simulator` or `--allow-rm-reward`. |
| Niche key | Section 5.3 samples best-per-combination. With dozens of cards that space is too sparse. `archive.niche_projection` defaults to `strategy_only`; `full_combination` reproduces the literal spec. |
| Ids in prompts | UUIDs everywhere in the database; prompts show 8-hex refs and `PromptTrace.ref_map` resolves citations. Cards are cited by slug. |
| Tag match | Set equality as specified, plus `tag_jaccard` recorded for calibration of `tag_penalty`. |
| Auth | Built-in email + password, roles `operator | researcher | expert`, single org, many ad accounts. |
| API model | Anthropic Claude behind provider-neutral interfaces. |
| Database | Postgres 18 + pgvector 0.8 on Neon (direct endpoint), same image locally. |
| Safety | `DRY_RUN=true` default, global kill switch, per-account daily hard cap, real client only constructed when all three allow. |

## Phase 6 decisions (reward model, Loop B, eval, feedback)

- **Screening failure is tier 0.** Spec 7.2 says the outcome reward is null only for ideas never
  shipped. A shipped idea whose screening window completed without a pass, with no render still
  live, therefore gets `outlier_tier = 0` (no `outcomes` row, since outcomes are scale-derived).
  Ideas still pending review, still screening, or at scale without a full window stay null.
  Consequence: DPO has tier 0 negatives, the novelty rule's "tier 0 with the same combination"
  clause has data, and the Loop A stats count screening failures as measured.
- **Gold gap trips on divergence in either direction.** `gap = (recent_score − reference_score) −
  (recent_rate − reference_rate)`; a gap above `reward_model.gold_gap_threshold` pauses queued
  training runs and refuses new launches. A proxy that holds steady while real outcomes collapse
  is treated the same as a proxy that rises while outcomes stay flat: both mean the reward model
  is over-scoring what ships. Loop A is never paused by it.
- **Eval verdicts below 20 briefs are "no verdict".** `success` is null, with a note, rather than
  false. `false` is reserved for runs with 20+ briefs whose intervals overlap.
- **Blind labels are random letters per run**, stored on `eval_arms.blind_label`; the system name
  is revealed on the Model screen only once the run is completed.
- **The trainer is a subprocess** (`python -m outlier_trainer.run`) launched by the `train` job.
  It reads a Parquet snapshot and writes `run.json`; it never touches the database. GPU
  dependencies live behind the `train` extra and are imported inside each stage, so `--dry`
  works everywhere and CI exercises the data path without a GPU.
- **On-policy GRPO is gated twice** (trainer flag and API validation) because its reward is
  reward-model shaping, exactly the failure mode section 6 warns about.
- **Verifier v2 is a nearest-centroid classifier** on the same hashed text embedding the archive
  uses, trained only on `wrong_cards` corrections and expert-tagged imports (holdout excluded). It
  replaces the LLM verifier automatically once `verifier.classifier_min_labels` labels exist.
- **Suggested relations** come from card pairs whose tier 2+ co-occurrence is at least 1.5x what
  the members' individual frequencies predict, with at least two observations; created as
  `card_relations(source=suggested, status=pending)` when an outlier lands and weekly.
- **Prompts are stored on the batch** (`batches.prompt_trace.prompt`) so the export ships the
  exact prompt the idea was generated from; older trajectories are re-rendered as of their
  creation time and marked as such.
