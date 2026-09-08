"""Seed a synthetic dataset: cards, briefs, historical trajectories with raw daily rows,
comments, signals, reviews, and combinations.

Derived objects (screening stats, outcomes, tiers, combination tier counts) are NOT written
here. They come from `outlier.recompute` (Phase 1), exactly as they would for real data.
The seed does decide which renders reached scale, using the same Wilson rule the screening
module applies, because scale rows only exist for ads that passed screening.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.archive.combinations import combination_key, niche_key
from outlier_ai.archive.novelty import embed_combo_angle
from outlier_ai.core.config import ConfigStore, load_file_config
from outlier_ai.core.crypto import hash_identity
from outlier_ai.core.embeddings import HashEmbedder
from outlier_ai.models import Base
from outlier_ai.models.auth import User
from outlier_ai.models.briefs import Brief
from outlier_ai.models.cards import Card, CardVersion, Combination
from outlier_ai.models.meta import AdAccount, Comment, DailyInsight
from outlier_ai.models.ops import KillSwitch
from outlier_ai.models.signals import Signal
from outlier_ai.models.trajectories import Note, Render, Review, Trajectory
from outlier_ai.outlier.stats import wilson_lower_bound
from outlier_ai.synthetic.briefs import CATEGORIES, WORLD_STATES, brief_schedule, make_brief
from outlier_ai.synthetic.cards import ALL_CARDS, CardSpec, cards_by_kind
from outlier_ai.synthetic.comments import generate_comments, summarize_signals
from outlier_ai.synthetic.latent import AccountProfile, LatentOutcomeModel
from outlier_ai.synthetic.trajectories import compose_idea, pick_cards, pick_typicality
from outlier_schemas.config import AppConfig
from outlier_schemas.enums import (
    AuthorKind,
    CardKind,
    RenderStatus,
    ReviewLabel,
    SignalStatus,
    TagSource,
)

FAKE_ACCOUNT_ID = "act_fake_1"
FAKE_PAGE_ID = "page_fake_1"


@dataclass
class SeedSummary:
    cards: int = 0
    briefs: int = 0
    trajectories: int = 0
    shipped: int = 0
    renders: int = 0
    scaled_renders: int = 0
    daily_rows: int = 0
    comments: int = 0
    signals: int = 0
    combinations: int = 0
    hot_pairs: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


async def truncate_all(session: AsyncSession) -> None:
    names = ", ".join(f'"{t.name}"' for t in reversed(Base.metadata.sorted_tables))
    await session.execute(text(f"TRUNCATE {names} RESTART IDENTITY CASCADE"))


def _card_row(spec: CardSpec) -> tuple[Card, CardVersion]:
    cid = uuid.uuid4()
    card = Card(
        id=cid,
        slug=spec.slug,
        name=spec.name,
        kind=spec.kind.value,
        definition=spec.definition,
        qualifying_condition=spec.qualifying_condition,
        source=spec.source,
        contributed_by=spec.contributed_by,
        status="active",
        version=1,
        example_trajectory_ids=[],
    )
    version = CardVersion(
        card_id=cid,
        version=1,
        name=spec.name,
        kind=spec.kind.value,
        definition=spec.definition,
        qualifying_condition=spec.qualifying_condition,
        source=spec.source,
        status="active",
        edited_by=spec.contributed_by,
    )
    return card, version


async def seed(
    session: AsyncSession,
    *,
    n_briefs: int = 20,
    n_trajectories: int = 400,
    seed: int = 1,
    days_back: int = 120,
    truncate: bool = True,
    config: AppConfig | None = None,
    now: datetime | None = None,
) -> SeedSummary:
    rng = np.random.default_rng(seed)
    cfg = config or load_file_config()
    now = now or datetime.now(UTC)
    summary = SeedSummary()
    embedder = HashEmbedder(cfg.archive.embedding_dims)

    if truncate:
        await truncate_all(session)

    await ConfigStore(session).ensure_bootstrapped("seed")
    session.add(KillSwitch(id=1, shipping_enabled=True, changed_by="seed", reason="seed default"))
    for role in ("operator", "researcher", "expert"):
        session.add(
            User(
                email=f"{role}@example.com",
                display_name=role.title(),
                password_hash="$unset$",
                role=role,
            )
        )

    account = AdAccount(
        id=uuid.uuid4(),
        name="Synthetic Account",
        meta_account_id=FAKE_ACCOUNT_ID,
        page_id=FAKE_PAGE_ID,
        status="active",
        attribution_setting=cfg.meta.attribution_setting,
        api_version=cfg.meta.api_version,
        category=None,
        daily_cap_usd=cfg.meta.daily_account_cap_usd,
        is_fake=True,
    )
    session.add(account)
    await session.flush()

    # Cards -------------------------------------------------------------------------------
    card_rows: dict[str, Card] = {}
    for spec in ALL_CARDS:
        card, version = _card_row(spec)
        card_rows[spec.slug] = card
        session.add_all([card, version])
    cards_by_id = {c.id: c for c in card_rows.values()}
    summary.cards = len(card_rows)
    by_kind = cards_by_kind()

    # Briefs ------------------------------------------------------------------------------
    categories = list(CATEGORIES)
    brief_rows: list[Brief] = []
    brief_meta: dict[uuid.UUID, dict] = {}
    for i, created in enumerate(brief_schedule(rng, n_briefs, days_back, now)):
        category = categories[i % len(categories)]
        data = make_brief(rng, i, category, created)
        b = Brief(
            id=uuid.uuid4(),
            created_by="seed",
            created_at=created,
            ad_account_id=account.id,
            company=data["company"],
            product=data["product"],
            offer=data["offer"],
            audience=data["audience"],
            goal_metric="purchases",
            channel="meta_feed_image",
            category=category,
            world_state=data["world_state"],
            world_state_at=created,
            constraints=data["constraints"],
            brand_assets=[],
            raw_text=data["raw_text"],
            embedding=embedder.embed([data["raw_text"]])[0].tolist(),
            # world_state_tag is what the latent model keys on; the fake client reads it here.
            meta={**data["meta"], "world_state_tag": data["world_state_tag"]},
        )
        brief_rows.append(b)
        brief_meta[b.id] = data
        session.add(b)
    summary.briefs = len(brief_rows)

    # Latent truth ------------------------------------------------------------------------
    latent = LatentOutcomeModel(
        seed=seed,
        categories=categories,
        strategy_slugs=[c.slug for c in by_kind[CardKind.strategy]],
        world_tags=[t for t, _ in WORLD_STATES],
    )
    summary.hot_pairs = len(latent.hot_pairs())
    account_profile = AccountProfile(
        attribution_setting=cfg.meta.attribution_setting,
    )

    screening_days = cfg.outlier.screening.window_days
    scale_days = cfg.episode.scale_days
    screening_budget = cfg.episode.screening_budget_per_ad_usd
    scale_budget = cfg.episode.scale_budget_per_ad_usd
    ctr_multiple = cfg.outlier.screening.ctr_multiple
    ctr_conf = cfg.outlier.screening.confidence
    min_impr = cfg.outlier.screening.min_impressions

    combos: dict[tuple[uuid.UUID, ...], Combination] = {}
    comment_counter = 0
    total_window_days = screening_days + scale_days

    for t_index in range(n_trajectories):
        brief = brief_rows[int(rng.integers(0, len(brief_rows)))]
        bdata = brief_meta[brief.id]
        latest_start = now - timedelta(days=total_window_days + 1)
        if brief.created_at >= latest_start:
            start = brief.created_at
        else:
            span = (latest_start - brief.created_at).total_seconds()
            start = brief.created_at + timedelta(seconds=float(rng.uniform(0, span)))

        specs = pick_cards(rng, by_kind)
        card_ids = [card_rows[s.slug].id for s in specs]
        niche = niche_key(card_ids, cards_by_id, cfg.archive.niche_projection)
        idea = compose_idea(rng, bdata, specs)
        author_kind = AuthorKind.model if rng.random() < 0.6 else AuthorKind.human
        shipped = rng.random() > 0.15

        verified = list(card_ids)
        tag_match = True
        if rng.random() < 0.10 and len(verified) > 2:
            verified = verified[:-1]
            tag_match = False

        traj = Trajectory(
            id=uuid.uuid4(),
            brief_id=brief.id,
            campaign_id=f"camp_hist_{brief.id.hex[:8]}",
            attempt_index=t_index,
            author_id="synthetic-model"
            if author_kind == AuthorKind.model
            else "expert@example.com",
            author_kind=author_kind.value,
            backend="fake" if author_kind == AuthorKind.model else None,
            card_ids=card_ids,
            verified_card_ids=verified,
            tag_source=TagSource.verifier.value,
            tag_match=tag_match,
            tag_jaccard=len(set(verified) & set(card_ids)) / len(set(verified) | set(card_ids)),
            typicality=pick_typicality(rng).value,
            format_ok=True,
            format_errors=[],
            reasoning=idea["reasoning"],
            cited_ids=[card_rows[s.slug].id for s in specs],
            angle=idea["angle"],
            ad_copy=idea["copy"],
            visual_brief=idea["visual_brief"],
            combo_angle_embedding=embed_combo_angle(
                embedder,
                [sp.slug for sp in specs if card_rows[sp.slug].id in verified],
                idea["angle"],
            ),
            library_version=1,
            config_hash=cfg.hash,
            created_at=start,
        )
        session.add(traj)
        summary.trajectories += 1

        key = combination_key(card_ids)
        combo = combos.get(key)
        if combo is None:
            combo = Combination(
                card_ids=list(key), niche_key=niche, uses=0, first_used=start, last_used=start
            )
            combos[key] = combo
            session.add(combo)
        combo.uses += 1
        combo.first_used = min(combo.first_used, start) if combo.first_used else start
        combo.last_used = max(combo.last_used, start) if combo.last_used else start

        if not shipped:
            session.add(
                Review(
                    trajectory_id=traj.id,
                    label=ReviewLabel.skip.value,
                    reviewer_id="expert@example.com",
                    note="synthetic skip",
                    reviewed_at=start,
                )
            )
            for seed_i in range(cfg.episode.renders_per_idea):
                session.add(
                    Render(
                        trajectory_id=traj.id,
                        image_backend="synthetic",
                        seed=seed_i,
                        width=1080,
                        height=1080,
                        status=RenderStatus.draft.value,
                    )
                )
                summary.renders += 1
            continue

        label = ReviewLabel.wrong_cards if not tag_match and rng.random() < 0.5 else ReviewLabel.run
        session.add(
            Review(
                trajectory_id=traj.id,
                label=label.value,
                corrected_card_ids=card_ids if label == ReviewLabel.wrong_cards else None,
                reviewer_id="expert@example.com",
                note="",
                reviewed_at=start,
            )
        )
        summary.shipped += 1

        truth = latent.sample_truth(niche, brief.category, bdata["world_state_tag"], rng)
        traj_comments: list = []

        for seed_i in range(cfg.episode.renders_per_idea):
            render = Render(
                id=uuid.uuid4(),
                trajectory_id=traj.id,
                image_backend="synthetic",
                seed=seed_i,
                width=1080,
                height=1080 if seed_i < 2 else 1350,
                meta_ad_id=f"fake_ad_{traj.id.hex[:8]}_{seed_i}",
                meta_adset_id=f"fake_adset_{traj.id.hex[:8]}_{seed_i}",
                meta_creative_id=f"fake_cr_{traj.id.hex[:8]}_{seed_i}",
                effective_object_story_id=f"{FAKE_PAGE_ID}_{traj.id.hex[:8]}{seed_i}",
                status=RenderStatus.stopped.value,
                shipped_at=start,
            )
            session.add(render)
            summary.renders += 1

            # per-render variation of the idea truth
            r_truth = type(truth)(
                roas_multiple=truth.roas_multiple * float(np.exp(0.10 * rng.normal())),
                ctr_multiple=truth.ctr_multiple * float(np.exp(0.10 * rng.normal())),
                polarity=truth.polarity,
            )
            impressions = clicks = 0
            comments_total = 0
            for d in range(screening_days):
                row = latent.simulate_day(
                    r_truth, screening_budget, account_profile, d, "screening", rng
                )
                impressions += row.impressions
                clicks += row.link_clicks
                comments_total += row.comments
                session.add(
                    _insight(
                        render.id,
                        start + timedelta(days=d),
                        "screening",
                        row,
                        screening_budget,
                        cfg,
                    )
                )
                summary.daily_rows += 1

            passed = impressions >= min_impr and wilson_lower_bound(
                clicks, impressions, ctr_conf
            ) >= (ctr_multiple * account_profile.median_ctr)
            if passed:
                for d in range(scale_days):
                    row = latent.simulate_day(
                        r_truth, scale_budget, account_profile, d, "scale", rng
                    )
                    comments_total += row.comments
                    session.add(
                        _insight(
                            render.id,
                            start + timedelta(days=screening_days + d),
                            "scale",
                            row,
                            scale_budget,
                            cfg,
                        )
                    )
                    summary.daily_rows += 1
                render.status = RenderStatus.scaled.value
                summary.scaled_renders += 1
                render.stopped_at = start + timedelta(days=total_window_days)
            else:
                render.stopped_at = start + timedelta(days=screening_days)

            n_comments = min(40, int(comments_total * 0.3))
            window_hours = 24.0 * (total_window_days if passed else screening_days)
            for c in generate_comments(
                rng, n_comments, r_truth.roas_multiple, r_truth.polarity, window_hours
            ):
                comment_counter += 1
                session.add(
                    Comment(
                        render_id=render.id,
                        external_id=f"fake_c_{comment_counter}",
                        commenter_hash=hash_identity(c.commenter_ext_id),
                        text=c.clean_text,
                        created_time=start + timedelta(hours=c.hours_after_launch),
                        like_count=c.like_count,
                        pii_removed=c.raw_text != c.clean_text,
                    )
                )
                summary.comments += 1
                traj_comments.append((render.id, c))

        for draft in summarize_signals(
            [c for _, c in traj_comments], cfg.signals.min_comments_for_theme
        ):
            status = SignalStatus.confirmed if rng.random() < 0.6 else SignalStatus.proposed
            session.add(
                Signal(
                    trajectory_id=traj.id,
                    render_id=None,
                    kind=draft.kind.value,
                    text=draft.text,
                    evidence=[
                        {"source": "meta_comment", "ref": "synthetic", "excerpt": e}
                        for e in draft.excerpts
                    ],
                    sentiment=draft.sentiment.value,
                    count=draft.count,
                    extracted_by=AuthorKind.model.value,
                    status=status.value,
                    decided_by="expert@example.com" if status == SignalStatus.confirmed else None,
                    decided_at=start + timedelta(days=total_window_days + 1)
                    if status == SignalStatus.confirmed
                    else None,
                    created_at=start + timedelta(days=screening_days),
                )
            )
            summary.signals += 1

        if rng.random() < 0.2:
            session.add(
                Note(
                    trajectory_id=traj.id,
                    author_id="operator@example.com",
                    text="Synthetic marketer note: watch the price objections here.",
                    created_at=start + timedelta(days=3),
                )
            )

        if t_index % 50 == 49:
            await session.flush()

    await session.flush()
    summary.combinations = len(combos)
    return summary


def _insight(
    render_id: uuid.UUID, day: datetime, phase: str, row, budget: float, cfg: AppConfig
) -> DailyInsight:
    return DailyInsight(
        render_id=render_id,
        day=day.date(),
        phase=phase,
        impressions=row.impressions,
        link_clicks=row.link_clicks,
        spend=row.spend,
        purchases=row.purchases,
        revenue=row.revenue,
        frequency=row.frequency,
        reactions=row.reactions,
        comments=row.comments,
        shares=row.shares,
        saves=row.saves,
        daily_budget=budget,
        attribution_setting=cfg.meta.attribution_setting,
        source="fake",
    )
