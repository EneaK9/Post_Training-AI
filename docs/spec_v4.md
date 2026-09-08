# Outlier AI: Post-Training Architecture and Product Spec (v4)

## 0. What changed from v3

| Change | Reason |
|---|---|
| Strategies and doctrine merged into one Playbook column | Top marketers combine "contrarian" and "Ogilvy long copy" the same way. Splitting them made one pickable and the other background reading. Now every card is pickable and every result attributes to a combination of cards. |
| History became a first-class column with Signals | The memory a marketer scans is not just tier results. It is comments, objections, jokes, misreadings, praise, what the audience said. A negative comment on a failed ad is raw material for the next combination. History now stores everything the campaign produced. |
| Two loops instead of one | Research on how AI actually finds outliers (FunSearch, AlphaEvolve) shows the working pattern is an LLM proposing, a hard evaluator scoring, an archive of ranked past attempts feeding the next prompt, and explicit diversity maintenance. No weight updates were needed to find new mathematics. This system runs that loop first with Facebook as the evaluator (Loop A), then trains weights on the archive it produces (Loop B). Loop B without Loop A has no data. |
| Diversity handled at generation time, not only in training | Aligned models mode-collapse because preference data favors typical text. Generation now uses distribution-style prompting, novelty rejection against the archive, and a base or lightly aligned model option. |
| Scenario walkthrough added (section 12) | Every failure path has a defined behavior. |

## 1. Purpose

Produce outlier ad campaigns, not above-average ones, and train a model that gets better at finding them.

Ad results are heavy-tailed. A typical account sees a 6 to 7 percent creative win rate, and roughly 5 percent of ads spend at least ten times the account median. Mean-optimizing systems (Meta AdLlama, RELATE) learn to avoid the zeros and converge on safe middle ideas. This system rewards only the tail and does not punish the swing.

The system produces ordinary results along the way and is not rewarded for them. A search for a brief is finished when a tier 2 outcome lands or the budget cap is hit.

Domain for v1: Meta feed image ads with a purchase objective. Architecture is domain-agnostic.

## 2. The three columns

Everything the model sees at generation time comes from three columns. This is the structure a top marketer uses and it is the structure of the prompt.

**Column 1, Playbook.** Expert-authored cards. Strategies, styles, principles, mechanics. All pickable, all combinable. The model's action is choosing a combination of cards.

**Column 2, Brief.** The company, product, offer, audience, constraints, and the world state right now.

**Column 3, History.** Every past attempt for this brief and for similar briefs: which cards were combined, what it produced (zero, winner, outlier), and everything the campaign threw off: comments, reactions, objections, praise, jokes, misreadings, marketer notes. Ranked and niched like an archive, so the prompt gets the best attempt per combination plus a few rare ones.

The move: read column 2, scan column 3 for similar situations and what combinations hit or missed and what the audience said, pick a new combination from column 1. Reward attaches to the combination.

## 3. Core objects

### 3.1 Card (Column 1)
```
Card {
  id, slug, name,
  kind: strategy | style | principle | mechanic,
  definition: str,                       # one paragraph
  qualifying_condition: str,             # when it applies, free text
  source: str | null,                    # e.g. "Ogilvy on Advertising, ch. 7" for style and principle cards
  contributed_by: expert_id,
  status: draft | active | retired,
  version: int,
  example_trajectory_ids: list[id]
}
CardRelation {
  from_id, to_id,
  kind: complements | conflicts | prerequisite,
  source: expert | suggested,
  status: pending | accepted | rejected,
  created_by
}
Combination {                            # derived, not authored; one row per distinct set of card ids ever used
  card_ids: sorted list[id],
  uses, tier_counts: {0,1,2,3}, tier2_rate, first_used, last_used, named_as_card_id | null
}
```
Seed cards (kind strategy): We're Killing X, Process Moats, Anti-Movements, Human Desires, 10x Product, Contrarian, Creativity and Humor, Objection Flip (take what critics say and make it the ad). Style, principle, and mechanic cards are written by the experts from their sources; none are shipped with the system.

### 3.2 Brief (Column 2)
```
Brief {
  id, created_by, created_at,
  company, product, offer, audience, goal_metric, channel,
  world_state: str, world_state_at: timestamp,
  constraints: list[str],
  brand_assets: list[asset_id],
  raw_text: str,
  meta: { ad_account_id, pixel_id, page_id, default_targeting_spec, landing_url }
}
```

### 3.3 Trajectory (Column 3, the attempt)
```
Trajectory {
  id, episode_id | null, brief_id, campaign_id, attempt_index,
  author_id, author_kind: human | model,
  card_ids: list[id],                    # as written by the author
  verified_card_ids: list[id],           # from the verifier (5.5)
  tag_source: expert | model | model_verified | verifier,
  tag_match: bool,
  reasoning: str, cited_ids: list[id],   # cards, trajectories, signals
  angle: str,
  copy: { primary_text, headline, description, cta },
  visual_brief: str,
  renders: list[Render],
  outcome: Outcome | null,               # idea-level, max across renders
  signals: list[signal_id],
  notes: list[Note],                     # marketer or expert free text, dated
  review: Review | null,
  rm_score: float | null, rm_version: str | null,
  library_version: int,
  created_at
}
```
Render, ScreeningStats, Outcome, Review are unchanged from v3 (see 3.6).

### 3.4 Signal (Column 3, what the campaign threw off)
```
Signal {
  id, trajectory_id, render_id | null,
  kind: comment_theme | objection | praise | joke | misreading | question | competitor_mention | quote | hook_stat | share_pattern | note,
  text: str,                             # the extracted theme or the quote, PII stripped
  evidence: list[{ source: meta_comment | reaction_stats | insight | manual, ref, excerpt }],
  sentiment: neg | neu | pos,
  count: int,                            # how many comments or events support it
  extracted_by: model | expert,
  status: proposed | confirmed | rejected,
  created_at
}
```
Signals are what make column 3 more than a scoreboard. "Everyone in the comments says the price is a joke" on a tier 0 ad is a proposed objection signal. Combined with the Objection Flip card it is a candidate for the next batch. Signals are citable in reasoning as `[signal:id]`.

### 3.5 SearchEpisode
Unchanged from v3: one per brief per search, budget cap, batches, status `searching | outlier_found | budget_exhausted | stopped`. Each batch is conditioned on the episode's own history, including signals from its failed batches.

### 3.6 Render, ScreeningStats, Outcome, Review
```
Render { id, image_url, image_backend, seed, size, meta_ad_id, meta_adset_id, meta_creative_id, effective_object_story_id,
         status: draft | pending_review | rejected | screening | scaled | stopped, screening: ScreeningStats | null, scale: Outcome | null }
ScreeningStats { impressions, link_clicks, spend, ctr, cpc, ctr_lower_bound, account_median_ctr_90d, screening_ratio, passed: bool,
                 engagement: { reactions, comments, shares, saves } }
Outcome { metric: roas | purchases_per_dollar, value, impressions, conversions, spend, revenue, days_at_scale,
          account_median_90d, account_median_human_90d | null, category_median, baseline: account | category_fallback,
          outlier_tier: 0|1|2|3, gates: { volume_ok, durability_ok, category_ok }, attribution_setting, measured_at }
Review { label: run | skip | wrong_cards, corrected_card_ids | null, reviewer_id, note, reviewed_at }
```

## 4. Outlier definition

Unchanged in substance from v3. Restated because everything depends on it.

- Two stages. Screening (CTR lower bound vs account CTR median) decides what gets scale budget and assigns no reward. Scale (ROAS vs account ROAS median) assigns the tier and is the only training reward.
- Baselines: account median over trailing 90 days, same channel and objective, needs at least 20 ads with scale data, else category fallback. Human-only baseline reported alongside. Category medians in config, updated quarterly.
- Tiers by multiple of account median ROAS at scale: below 1.5x tier 0 (reward 0); 1.5x to 3x tier 1 (reward 0 by default); 3x to 10x tier 2 (reward 1); 10x and up tier 3 (reward 1, capped).
- Gates: at least 30 purchases at scale and the bootstrap lower bound of the ROAS ratio clears the tier; tier held 7 consecutive days at scale spend; above category median.
- Screening pass: `ctr_lower_bound >= 1.5 x account_median_ctr_90d` after at least 5,000 impressions and the full screening window.
- All thresholds in `config.yaml` under `outlier:`, editable on the Model screen. Tiers are recomputed from raw daily rows when thresholds change.

## 5. Generation

### 5.1 Two loops
**Loop A, search.** No weight updates. The generator (any backend, including API models) proposes combinations and ads conditioned on the three columns. Facebook evaluates. The archive updates. Next batch is conditioned on the updated archive. This is FunSearch and AlphaEvolve mapped onto ads: LLM as proposal operator, hard evaluator, archive-based iteration, diversity control.

**Loop B, post-training.** Weight updates on the trainable backend using the trajectories Loop A produced, with real outcomes as reward. Starts when the archive has at least 50 tier 2+ trajectories with verified cards. Section 7.

Loop A produces outliers on its own if the loop works, and it produces the data Loop B needs. Loop B is what makes the generator better at proposing without needing the archive in the prompt for everything it has learned.

### 5.2 Backends
```
GeneratorBackend (interface)  generate(prompt, n) -> list[RawOutput];  trainable: bool
```
- `LocalPolicy`: open-weight, initial choice Qwen3-8B. Config chooses `base` or `instruct` weights. Base weights are less mode-collapsed; instruct weights follow format better. Both are tried in the first runs.
- `APIModel`: Claude, Kimi, or any chat API. Not trainable. Used for Loop A, cold start, baselines, comparison.

### 5.3 Prompt rendering
```
render(brief, archive_sample, playbook, episode_history) -> str
```
- `playbook`: all active cards, grouped by kind, with definitions and qualifying conditions.
- `archive_sample` (the FunSearch pattern): from history for this brief and similar briefs, take the best trajectory per combination niche (highest tier, then recency) for the top `n_elite` niches, plus `n_rare` trajectories from rarely used niches, plus every tier 2+ trajectory in the last 90 days across the account. Each with its cards, angle, copy, tier, and confirmed signals. Held-out campaigns excluded.
- `episode_history`: every prior batch in this episode with screening stats, tier, and proposed and confirmed signals, so the model knows what already failed here and what the audience said.
- Ask: propose `K` ideas as a set, each with a different combination, and label each `common | uncommon | rare` relative to what the account has run. Distribution-style prompting reduces mode collapse without training (Verbalized Sampling, Zhang et al. 2025).

### 5.4 Output format
Strict, one block per idea. Grammar-parsed. Malformed ideas get zero reward and a format penalty.
```
<idea>
<cards>card ids, two or more</cards>
<typicality>common | uncommon | rare</typicality>
<reasoning>Cites [card:id] [history:id] [signal:id]. States why this combination fits this brief and world state and what in history it responds to.</reasoning>
<angle>hook / enemy / promise, two lines</angle>
<copy>primary_text: ... headline: ... description: ... cta: ...</copy>
<visual_brief>one paragraph</visual_brief>
</idea>
```
Reasoning receives no direct reward. It exists for the expert audit and for the verifier.

### 5.5 Card verifier
Independent of the generator. Reads angle, copy, and visual_brief only, predicts card ids from the active playbook. v1 an APIModel judge with expert-tagged trajectories as few-shot; v2 a classifier once expert labels exceed 500. `tag_match` is set equality. Mismatch: `tag_penalty` in Loop B, and the uniqueness bonus uses `verified_card_ids`. Expert `wrong_cards` labels train the verifier.

### 5.6 Novelty rejection
Before an idea reaches the review queue: reject if its verified combination plus angle embedding is within `novelty_threshold` of any trajectory already shipped for this brief, or of any tier 0 trajectory in the account with the same combination in the last 90 days. Rejected ideas are stored with `review.label = skip` and `note = novelty_reject`, never shipped. This is the deduplication step every evolutionary loop needs, and it saves paid evaluations.

### 5.7 Image rendering
Unchanged from v3: `ImageBackend`, 3 renders per idea, `image_mode: generate | brand_assets`, brand assets mode for the first measured runs, sizes 1080x1080 and 1080x1350, Meta length warnings.

### 5.8 Pre-ship checks
Unchanged from v3: policy screen, brand constraints, human `run` label as approval.

## 6. Reward model
Unchanged from v3 in structure: pessimistic three-head ensemble, `rm_score = mean - lambda_pess * std`, focal loss, calibration on tier 2+, `rm_cold` (expert labels) until `rm_outcome` (real tiers) has 50 positives, gold gap monitor that pauses Loop B.

Role in Loop A: ranks candidates for the review queue and picks the top `batch_size` when the expert delegates selection. Role in Loop B: bounded shaping only.

Inputs now include the verified combination and the confirmed signals the idea cites, so the model can learn that "objection flip on a price objection" scores differently from "objection flip" alone.

## 7. Loop B: post-training

### 7.1 Data
All trajectories with a scale outcome and verified cards, from Loop A and from practitioner imports. Zero-tier trajectories are kept; they are the misses that define the search.

### 7.2 Reward
```
reward = outcome_reward (1 if tier >= 2 else 0; null if never shipped)
       + shaping_weight * rm_score (only when outcome is null; decays to 0 as outcome positives grow)
       - tag_penalty * (1 - tag_match) - format_penalty * (1 - format_ok)
```

### 7.3 Loop
```
for batch of briefs with archive data:
    prompt = render(...)
    outs   = policy.generate(prompt, n=K)            # K 32
    outs   = parse(outs); tags = verifier.tag(outs); r = reward(outs)
    u      = risk_transform(r, tau)                  # exponential utility; tau annealed low to target
    A      = (1 - eps_mean) * (u - mean(u)) + eps_mean * (r - mean(r))
    A      = A * (1 / cluster_size(verified_card_ids)) ** beta
    loss   = grpo_clipped_loss(policy, ref, prompt, outs, A) + kl_coef * KL(policy, ref)
    guards: stop on swing_rate drop, entropy floor, gold gap trip; steps between outcome batches capped
```
Pluggable: risk transform (exponential now; pass@k analytical and unlikeliness reward reserved), KL variant, reference refresh only at run boundaries. All values in `config.yaml` are starting points.

### 7.4 What Loop B should show if it works
The trained policy proposes tier 2+ combinations at a higher rate than the same backend run in Loop A with the same archive in the prompt. That comparison is in the online eval (section 11) as `loop_b_vs_loop_a`.

## 8. Facebook Ads integration
Unchanged from v3 in setup, campaign structure (one ad per ad set, identical targeting, equal fixed budgets, no CBO, feed only), shipping via `/adimages`, `/adcreatives`, ad sets and ads named `OAI|episode|trajectory|render`, ad review polling, daily insights ingestion with `use_unified_attribution_setting=true`, episode controller, budget cap, and CSV manual path. Additions:

### 8.1 Engagement and comments
- Insights pull adds engagement action types (post reactions, comments, shares, post saves) per ad per day into `ScreeningStats.engagement`.
- Comments: each render stores `effective_object_story_id`; a daily job reads comments on that post through the Graph API with the Page token (permissions pinned per API version in config). Stored raw with commenter identity hashed and any names, emails, or phone numbers stripped before storage. Abuse and spam filtered by the same policy screen used pre-ship.
- Signal extraction: an APIModel pass groups comments into themes with counts, evidence excerpts, and sentiment, and writes `Signal` rows with `status = proposed`. Experts confirm or reject on the Data screen. Only confirmed signals enter prompts and reward model inputs; proposed signals are visible to experts.
- Manual signals: any expert or operator can add a `note` kind signal with free text.

### 8.2 Cost
Screening 4 ideas x 3 renders x `screening_budget_per_ad` x 7 days is about 1,700 dollars per batch at 20 dollars per ad per day, before scale. Novelty rejection and reward model ranking exist to make each paid evaluation count. The loop only makes economic sense on accounts already spending at that level.

## 9. UI

Roles: `operator`, `researcher`, `expert`. Shared left nav, global search over cards, briefs, trajectories, signals, and episodes, and a drawer that opens any object by id from any citation.

### 9.1 Generate (`/generate`)
- Left: brief form or load, backend, K (default 8), image mode, renders per idea, size, episode (new with budget cap, or attach).
- Right: one card per idea, laid out as a Meta feed ad mock with render tabs. Shows `rm_score` and active reward model kind, written cards and verified cards side by side with mismatches flagged, typicality label, angle, reasoning with clickable citations to cards, past attempts, and signals.
- "What the model saw" panel: the archive sample (best per niche and rare picks), the episode history, and the confirmed signals, exactly as rendered into the prompt.
- Actions per idea: Run, Skip, Wrong Cards, Edit Copy, Regenerate. Novelty-rejected ideas appear collapsed with the reason.
- Episode strip: cap, spent, live ads with screening ratio and scale tier, new signals since last batch, status. Flips to Outlier Found on tier 2.
- Compare mode: same brief on two backends, or Loop A prompt vs Loop B checkpoint.

### 9.2 Model (`/model`)
- Live architecture diagram from config: Loop A components (backend, archive sampling settings, novelty threshold, verifier, image mode), Loop B components (policy checkpoint, transform, KL variant, reward model kind and version), evaluator (Meta account set, outlier thresholds). Every node opens its config.
- Config editor with validation and diff against last run: `rl:`, `outlier:`, `episode:`, `archive:` (n_elite, n_rare, novelty_threshold), `generation:` (K, typicality prompting on/off, base vs instruct).
- Loop A stats: ideas generated, novelty rejected, shipped, screening pass rate, tier 2 rate per backend, per combination, per typicality label. This table answers whether "rare" ideas actually hit more.
- Loop B runs: table with metrics, stop reason, logs, launch button pinning versions. Reward model panel with calibration and gold gap. Verifier panel with confusion matrix.
- Online eval table per eval id: systems side by side, tier 2 rate with intervals, plus `loop_b_vs_loop_a`.
- Review stats per backend: run, skip, wrong cards, novelty rejects.

### 9.3 Data (`/data`)
Default: three-column board.
- Column 1 Playbook: cards grouped by kind. Each shows name, definition, qualifying condition, source, contributor, status, version, stats (uses, tier 2+ count and rate, verifier agreement). Relations drawn between cards, suggested ones dashed. A Combinations sub-tab lists every combination ever used with its tier counts, sortable, with a Name As Card button that creates a new draft card from a combination.
- Column 2 Briefs: one card per brief with world state and constraints; expanding shows its episodes.
- Column 3 History: trajectories newest first, filterable by brief. Each shows cards written and verified, angle, copy preview, render thumbnails with status, screening ratio, tier badge or pending, review label, and a signals strip: confirmed signals as chips, proposed ones in a lighter style with confirm and reject. Expanding shows comments grouped by theme with evidence excerpts, engagement numbers, notes, and the reasoning that produced the idea with its citations.
- Add card in any column; drafts excluded from prompts, verifier, and training until active. Versioning on edit. Expert layers and side-by-side diffs. Filters by status, expert, tier, card, combination, mismatch, signal kind, episode status, typicality.
- Review Queue tab: unlabeled ideas sorted by `rm_score`, with the Generate actions. Signals Queue tab: proposed signals awaiting confirmation, grouped by trajectory.
- Graph tab: cards as nodes sized by tier 2+ count, edges from relations, combinations as hyperedges, trajectories hanging off combinations, signals hanging off trajectories.
- Audit tab. CSV import for cards, practitioner trajectories (verifier proposes tags, expert confirms), outcomes, and comments.

## 10. API
As v3, with `doctrine` removed, `strategies` renamed `cards`, and additions:
```
cards         GET list, GET id, POST, PUT, POST id/retire, GET id/versions
combinations  GET list, POST name_as_card
signals       GET list, GET id, POST, PUT id/confirm, PUT id/reject, POST extract (trajectory_id)
archive       GET sample (brief_id) -> exactly what render() would include
meta          POST accounts, GET accounts, POST sync, POST sync_comments
eval          POST launch { kind: online | loop_b_vs_loop_a }, GET id
```

## 11. Evaluation
As v3: online eval on held-out briefs, equal budgets under the fixed campaign structure, blind, `tier2_rate` per system with bootstrap intervals, success when Outlier AI beats mean-objective RL with non-overlapping intervals on at least 20 briefs. Offline proxies reported but never used as success. Leakage blocklist on retrieval, reward model, verifier, and baselines.

Added system rows: `loop_a_api` (API model in Loop A), `loop_a_local` (same open model in Loop A), `loop_b` (trained checkpoint, same prompt). `loop_b_vs_loop_a` is the number that says whether post-training added anything beyond the search loop.

## 12. Scenario walkthrough
| Scenario | System behavior |
|---|---|
| New account, no 90-day baseline | Category fallback for gating, `baseline = category_fallback` on outcomes, Model screen shows the account as fallback until 20 ads have scale data |
| All three renders rejected by Meta review | Trajectory `outcome = null`, excluded from outcome reward, rejection reasons on the renders, reviewer notified, episode requests the next batch without counting the spend |
| Screening passes, scale fails a gate | Tier stays 0, gate flags stored, ad stopped at end of scale window, signals still extracted from comments |
| Tier 2 on batch 1 | Episode `outlier_found`, remaining ads stopped unless operator keeps them, the combination's stats update, a suggested relation is created for any card pair in it that co-occurs in tier 2+ above chance |
| Budget exhausted with no outlier | Episode `budget_exhausted`, all trajectories and signals kept in history, nothing rewarded, Loop A stats record the miss |
| Expert disagrees with the verifier | `wrong_cards` sets corrected ids, trajectory shows both chip sets, label enters the verifier's next training set, uniqueness bonus uses the corrected ids after retrain |
| Two experts edit the same card | Two versions, side by side diff, both remain, runs pin the version they used |
| Policy hacks the reward model | Gold gap trips, Loop B pauses, Model screen shows the trip, Loop A continues unaffected |
| Policy collapses onto one combination | Entropy floor or swing rate guard stops the run; Loop A novelty rejection prevents shipping duplicates regardless |
| Seasonal or cultural shift makes an old winner stale | `world_state_at` and recency ordering in the archive sample; reward model inputs include world state; 90-day windows on baselines and history |
| Comments contain PII or abuse | Hashed identity, PII stripped before storage, abuse filtered, only themes and excerpts reach experts |
| Meta token expires or API version changes | Client raises a typed error, sync job marks the account `needs_reauth`, Generate screen blocks Run for that account, nothing else stops |
| Ad fatigue during scale | Frequency tracked daily; durability gate is on consecutive days, so a decaying ad that drops below the tier boundary fails durability |
| Attribution setting differs between accounts | Stored on every outcome; tiers only compared within the same setting; online eval fixes one setting for all systems |
| Practitioner history arrives without card tags | Verifier proposes, expert confirms, `tag_source` recorded; untagged trajectories stay out of training until confirmed |
| Lead-gen account instead of purchases | `goal_metric = leads`, metric falls back to leads per dollar with the same tier multiples; flagged as a different metric family and never pooled with ROAS accounts |
| Video ads requested | Out of scope for v1; the brief form rejects video channel with a message |
| Held-out brief accidentally similar to a training brief | Blocklist is by campaign id; similarity is not a leakage criterion, only shared campaigns are |

## 13. Research check

The Model tab's choices were checked against what is known about making AI produce outliers.

- How AI has actually produced outliers: FunSearch paired an LLM with a programmatic evaluator to discover novel solutions to open problems in mathematics, and AlphaEvolve extended it to full codebases with MAP-Elites. Both are LLM-as-proposer, hard evaluator, ranked archive in the prompt, diversity via islands or niches, with no weight updates required. Loop A is that pattern with Facebook as the evaluator and card combinations as the niches. Later work adds test-time RL on top of the evolve loop (ThetaEvolve, TTT-Discover), which is what Loop B is.
- Why aligned models struggle to propose outliers: typicality bias in preference data, where annotators favor familiar text, is a pervasive cause of mode collapse, and distribution-style prompting increases creative diversity 1.6 to 2.1x over direct prompting. This is why 5.3 asks for a labeled set of ideas and why base weights are an option.
- Why the objective is risk-seeking: pass@k and max@k training objectives (2025) and RS-GRPO show that optimizing the best of a batch rather than the mean preserves exploration; Rewarding the Unlikely (He, Fried, Welleck 2025) shows GRPO's rank bias otherwise only reinforces already probable outputs.
- Why diversity is rewarded at the strategy level: Uniqueness-Aware RL (2026) clusters rollouts by high-level strategy and gives more signal to correct but rare ones. Verified card combinations replace the LLM judge.
- Why the reward model is not the reward: Gao et al. 2023 showed proxy reward rises while true reward peaks and declines; Coste et al. 2024 showed pessimistic ensembles with a small KL penalty prevent it. Hence the ensemble, the gold gap, and real outcomes as the only reward.
- Why the tiers are multiples of medians: creative performance is heavy-tailed with roughly 5 percent of ads at 10x the account median; mean and standard deviation are dominated by the tail.

Honest gap: none of the cited systems ran against an evaluator that costs money per evaluation, returns results a week later, and shifts with culture. Sample efficiency (novelty rejection, reward model ranking, the cheap screening stage as a first evaluator cascade) is where this design differs from the math and code precedents, and it is unproven. Section 11 decides.

## 14. Stack, repo, build order
Stack as v3 (FastAPI, pydantic, Postgres, pgvector, TRL or verl worker, Next.js, `facebook_business` behind one client with a fake for tests, encrypted tokens, pytest and Playwright, synthetic generator now also emitting comments and signals).

Repo layout as v3 with `doctrine/` removed and `signals/` (extraction, PII stripping, confirmation) and `archive/` (niche sampling, novelty rejection, combination stats) added.

Build order:
1. Schemas incl. Card, Combination, Signal; database; synthetic generator incl. comments
2. Outlier module with tests
3. Archive: combination stats, niche sampling, novelty rejection
4. Prompt renderer with distribution-style prompting, strict multi-idea parser
5. Backends, image backend with brand-assets mode
6. Verifier and pre-ship checks
7. Signals: comment ingestion, PII stripping, extraction, confirmation flow
8. Reward ensemble, calibration, gold gap
9. Meta client, fake, campaign structure, ship, insights and comments sync, episode controller
10. API and audit
11. Data screen, then Generate, then Model
12. Loop A end-to-end on synthetic data with the fake Meta client
13. Loop B module, guards, worker, smoke on synthetic archive
14. Online eval harness incl. `loop_b_vs_loop_a`
15. Feedback: retrain schedules, suggested relations, combination naming

Real cards, real practitioner history, a real Meta account, and real budgets connect after step 12 passes on synthetic data and step 9 passes against the fake client. Loop B is not launched until the archive holds 50 tier 2+ trajectories with verified cards.
