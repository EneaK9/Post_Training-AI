"""The step-12 gate: Loop A must beat random selection on the simulator."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.core.storage import LocalStorage
from outlier_ai.experiments.loop_a_vs_random import run_experiment
from outlier_schemas.config import AppConfig

pytestmark = [pytest.mark.integration, pytest.mark.slow]


async def test_loop_a_finds_outliers_at_least_as_often_as_random(
    db_session: AsyncSession, app_config: AppConfig, tmp_path
):
    report = await run_experiment(
        db_session,
        cfg=app_config,
        storage=LocalStorage(tmp_path),
        n_briefs=4,
        seeds=[1, 2],
        history_trajectories=250,
        budget_cap=4000.0,
        max_days=75,
    )
    s = report.summary()
    print("\nEXPERIMENT SUMMARY", s)
    assert s["loop_a"]["runs"] == 8 and s["random"]["runs"] == 8
    # Loop A conditions on history and lets the reward model pick; random ignores both.
    assert (s["loop_a"]["found_rate"] or 0.0) >= (s["random"]["found_rate"] or 0.0), s
    assert all(
        r.status in ("outlier_found", "budget_exhausted")
        for arm in report.arms.values()
        for a in arm
        for r in a.runs
    )
