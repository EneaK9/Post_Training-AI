"""Training-run launcher: export a snapshot, create the `training_runs` row, run the trainer as a
subprocess from a job, and copy `run.json` back into the row.

The trainer never touches the database; everything it needs is the Parquet snapshot on disk and
the pinned versions in its arguments. The gold-gap trip (spec section 6) pauses queued runs.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.archive.export import export_snapshot
from outlier_ai.core.settings import Settings, get_settings
from outlier_ai.core.storage import get_storage, key_from_uri
from outlier_ai.jobs.registry import enqueue
from outlier_ai.models.ml import ArchiveSnapshot, RewardModelVersion, TrainingRun, VerifierVersion
from outlier_ai.models.ops import Job
from outlier_ai.reward.train import gold_gap_check
from outlier_schemas.config import AppConfig
from outlier_schemas.enums import TrainingRunStatus, TrainingStage


async def _active_version(session: AsyncSession, model: Any) -> str | None:
    row = (
        (
            await session.execute(
                select(model).where(model.is_active.is_(True)).order_by(model.created_at.desc())
            )
        )
        .scalars()
        .first()
    )
    return row.version if row else None


async def pause_if_gold_gap_tripped(session: AsyncSession, cfg: AppConfig) -> dict[str, Any]:
    gg = await gold_gap_check(session, cfg)
    if gg.get("tripped"):
        queued = (
            (
                await session.execute(
                    select(TrainingRun).where(TrainingRun.status == TrainingRunStatus.queued.value)
                )
            )
            .scalars()
            .all()
        )
        for run in queued:
            run.status = TrainingRunStatus.paused.value
            run.stop_reason = f"gold gap tripped: {gg.get('detail', '')}"[:255]
        await session.flush()
    return gg


async def queue_training_run(
    session: AsyncSession,
    *,
    cfg: AppConfig,
    stage: TrainingStage,
    base_model: str,
    created_by: str,
    smoke: bool = False,
    dry: bool = False,
    simulator: bool = False,
    allow_rm_reward: bool = False,
    adapter_from: UUID | None = None,
    settings: Settings | None = None,
) -> tuple[TrainingRun, Job]:
    s = settings or get_settings()
    gg = await pause_if_gold_gap_tripped(session, cfg)
    if gg.get("tripped"):
        raise ValueError("gold gap tripped; retrain the reward model before launching a run")
    snap = await export_snapshot(session, cfg=cfg, storage=get_storage(s))
    if snap.n_tier2 < cfg.rl.min_positives_to_train and not (smoke or dry):
        raise ValueError(
            f"{snap.n_tier2} tier 2+ trajectories < rl.min_positives_to_train "
            f"({cfg.rl.min_positives_to_train}); use smoke=true to force a tiny run"
        )
    adapter_uri: str | None = None
    if adapter_from is not None:
        prev = await session.get(TrainingRun, adapter_from)
        if prev is None or not prev.checkpoint_uri:
            raise ValueError("adapter_from run has no checkpoint")
        adapter_uri = prev.checkpoint_uri
    run = TrainingRun(
        stage=stage.value,
        status=TrainingRunStatus.queued.value,
        config_hash=cfg.hash,
        snapshot_hash=snap.hash,
        rm_version=await _active_version(session, RewardModelVersion),
        verifier_version=await _active_version(session, VerifierVersion),
        base_model=base_model,
        metrics={
            "requested": {
                "smoke": smoke,
                "dry": dry,
                "simulator": simulator,
                "allow_rm_reward": allow_rm_reward,
                "adapter_from": str(adapter_from) if adapter_from else None,
            },
            "snapshot": {
                "n_trajectories": snap.n_trajectories,
                "n_tier2": snap.n_tier2,
                "n_with_prompt": snap.n_with_prompt,
            },
        },
        created_by=created_by,
    )
    session.add(run)
    await session.flush()
    job = await enqueue(
        session,
        "train",
        key=f"train:{run.id}",
        payload={
            "run_id": str(run.id),
            "snapshot_uri": snap.uri,
            "adapter_uri": adapter_uri,
            "smoke": smoke,
            "dry": dry,
            "simulator": simulator,
            "allow_rm_reward": allow_rm_reward,
            "gold_gap": gg.get("gap"),
        },
    )
    return run, job


def _stage_inputs(
    storage: Any, snapshot_uri: str, adapter_uri: str | None
) -> tuple[Path, Path, str | None]:
    workdir = Path(tempfile.mkdtemp(prefix="oai-train-"))
    snapshot_path = workdir / "archive.parquet"
    snapshot_path.write_bytes(storage.get(key_from_uri(snapshot_uri)))
    adapter_dir: str | None = None
    if adapter_uri:
        adapter_dir = str(workdir / "adapter_in")
        Path(adapter_dir).mkdir()
        prefix = key_from_uri(adapter_uri).rstrip("/") + "/"
        for key in storage.list(prefix):
            target = Path(adapter_dir) / key[len(prefix) :]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(storage.get(key))
    return workdir, snapshot_path, adapter_dir


def _read_report(out_dir: Path) -> dict[str, Any]:
    run_json = out_dir / "run.json"
    return json.loads(run_json.read_text()) if run_json.exists() else {}


def trainer_command(
    *,
    stage: str,
    snapshot_path: Path,
    out_dir: Path,
    base_model: str,
    config_path: str | None,
    adapter: str | None,
    smoke: bool,
    dry: bool,
    simulator: bool,
    allow_rm_reward: bool,
    gold_gap: float | None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "outlier_trainer.run",
        "--stage",
        stage,
        "--snapshot",
        str(snapshot_path),
        "--out",
        str(out_dir),
        "--base-model",
        base_model,
    ]
    if config_path:
        cmd += ["--config", config_path]
    if adapter:
        cmd += ["--adapter", adapter]
    if smoke:
        cmd.append("--smoke")
    if dry:
        cmd.append("--dry")
    if simulator:
        cmd.append("--simulator")
    if allow_rm_reward:
        cmd.append("--allow-rm-reward")
    if gold_gap is not None:
        cmd += ["--gold-gap", str(gold_gap)]
    return cmd


async def run_training_job(
    session: AsyncSession,
    payload: dict[str, Any],
    *,
    cfg: AppConfig,
    settings: Settings | None = None,
    timeout_seconds: float = 6 * 3600,
) -> dict[str, Any]:
    s = settings or get_settings()
    run = await session.get(TrainingRun, UUID(payload["run_id"]))
    if run is None:
        raise ValueError("training run not found")
    if run.status != TrainingRunStatus.queued.value:
        return {"skipped": f"run is {run.status}"}
    storage = get_storage(s)
    run.status = TrainingRunStatus.running.value
    run.started_at = datetime.now(UTC)
    await session.flush()
    workdir, snapshot_path, adapter_dir = await asyncio.to_thread(
        _stage_inputs, storage, payload["snapshot_uri"], payload.get("adapter_uri")
    )
    out_dir = workdir / "out"
    cmd = trainer_command(
        stage=run.stage,
        snapshot_path=snapshot_path,
        out_dir=out_dir,
        base_model=run.base_model or "Qwen/Qwen3-8B",
        config_path=str(s.config_path) if Path(s.config_path).exists() else None,  # noqa: ASYNC240
        adapter=adapter_dir,
        smoke=bool(payload.get("smoke")),
        dry=bool(payload.get("dry")),
        simulator=bool(payload.get("simulator")),
        allow_rm_reward=bool(payload.get("allow_rm_reward")),
        gold_gap=payload.get("gold_gap"),
    )
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
    except TimeoutError:
        proc.kill()
        run.status = TrainingRunStatus.failed.value
        run.stop_reason = "trainer timed out"
        run.finished_at = datetime.now(UTC)
        return {"status": "failed", "error": "timeout"}
    logs_uri = storage.put(
        f"training/{run.id}/log.txt",
        (stdout or b"") + b"\n--- stderr ---\n" + (stderr or b""),
        "text/plain",
    )
    run.logs_uri = logs_uri
    report: dict[str, Any] = await asyncio.to_thread(_read_report, out_dir)
    adapter_out = out_dir / "adapter"
    if await asyncio.to_thread(adapter_out.is_dir):
        from outlier_trainer.export_checkpoint import export_adapter

        uris = export_adapter(adapter_out, storage.put)
        if uris:
            run.checkpoint_uri = f"{uris[0].rsplit('/', 1)[0]}/"
    status_ = report.get("status")
    if proc.returncode == 0 and status_ in ("completed", "skipped", None):
        run.status = TrainingRunStatus.completed.value
        run.stop_reason = report.get("stop_reason") or report.get("skipped")
    else:
        run.status = TrainingRunStatus.failed.value
        run.stop_reason = (report.get("error") or (stderr or b"").decode()[-255:])[:255]
    run.metrics = {
        **(run.metrics or {}),
        "trainer": {k: v for k, v in report.items() if k != "history"},
    }
    run.finished_at = datetime.now(UTC)
    await session.flush()
    return {"status": run.status, "returncode": proc.returncode, "stop_reason": run.stop_reason}


async def latest_snapshot(session: AsyncSession) -> ArchiveSnapshot | None:
    return (
        (await session.execute(select(ArchiveSnapshot).order_by(ArchiveSnapshot.created_at.desc())))
        .scalars()
        .first()
    )
