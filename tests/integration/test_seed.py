import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.models.cards import Card, Combination
from outlier_ai.models.meta import Comment, DailyInsight
from outlier_ai.models.ops import ConfigVersion, KillSwitch
from outlier_ai.models.signals import Signal
from outlier_ai.models.trajectories import Render, Trajectory
from outlier_ai.synthetic.seed import seed

pytestmark = pytest.mark.integration


async def _count(session: AsyncSession, model) -> int:
    return int((await session.execute(select(func.count()).select_from(model))).scalar_one())


async def test_seed_populates_every_column(db_session: AsyncSession):
    summary = await seed(db_session, n_briefs=4, n_trajectories=40, seed=7, days_back=90)
    await db_session.commit()

    assert summary.cards == 28
    assert await _count(db_session, Card) == 28
    assert await _count(db_session, Trajectory) == 40
    assert await _count(db_session, Render) == 40 * 3
    assert summary.shipped > 0 and summary.daily_rows > 0
    assert await _count(db_session, DailyInsight) == summary.daily_rows
    assert await _count(db_session, Comment) == summary.comments
    assert await _count(db_session, Signal) == summary.signals
    assert await _count(db_session, Combination) == summary.combinations
    assert await _count(db_session, ConfigVersion) == 1
    ks = (await db_session.execute(select(KillSwitch))).scalar_one()
    assert ks.shipping_enabled is True


async def test_seed_is_reproducible_and_truncates(db_session: AsyncSession):
    first = await seed(db_session, n_briefs=3, n_trajectories=25, seed=11, days_back=90)
    await db_session.commit()
    second = await seed(db_session, n_briefs=3, n_trajectories=25, seed=11, days_back=90)
    await db_session.commit()
    assert first.as_dict() == second.as_dict()
    assert await _count(db_session, Trajectory) == 25


async def test_scaled_renders_have_scale_rows_only_after_screening(db_session: AsyncSession):
    await seed(db_session, n_briefs=3, n_trajectories=60, seed=3, days_back=90)
    await db_session.commit()
    scaled = (
        (await db_session.execute(select(Render).where(Render.status == "scaled"))).scalars().all()
    )
    assert scaled, "with 60 trajectories some renders should pass screening"
    rows = (
        (
            await db_session.execute(
                select(DailyInsight)
                .where(DailyInsight.render_id == scaled[0].id)
                .order_by(DailyInsight.day)
            )
        )
        .scalars()
        .all()
    )
    phases = [r.phase for r in rows]
    assert phases[:7] == ["screening"] * 7
    assert set(phases[7:]) == {"scale"}
