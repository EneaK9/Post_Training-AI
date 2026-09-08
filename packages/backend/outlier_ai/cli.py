"""`oai` command line."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from outlier_ai.core import db as dbmod
from outlier_ai.core.config import ConfigStore
from outlier_ai.core.settings import get_settings

app = typer.Typer(no_args_is_help=True, help="Outlier AI command line")
synth_app = typer.Typer(no_args_is_help=True, help="Synthetic data")
db_app = typer.Typer(no_args_is_help=True, help="Database")
config_app = typer.Typer(no_args_is_help=True, help="Configuration")
app.add_typer(synth_app, name="synth")
app.add_typer(db_app, name="db")
app.add_typer(config_app, name="config")

console = Console()
BACKEND_DIR = Path(__file__).resolve().parent.parent


def _alembic_config():
    from alembic.config import Config

    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    return cfg


@db_app.command("migrate")
def db_migrate(revision: str = "head") -> None:
    """Apply Alembic migrations."""
    from alembic import command

    command.upgrade(_alembic_config(), revision)
    console.print(f"[green]migrated to {revision}[/green]")


@db_app.command("create-all")
def db_create_all(database_url: str | None = None) -> None:
    """Create tables directly from the ORM (dev/test only; prefer migrate)."""
    from sqlalchemy import text

    from outlier_ai.models import Base

    async def _run() -> None:
        engine = dbmod.make_engine(database_url) if database_url else dbmod.get_engine()
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)
        await engine.dispose()

    asyncio.run(_run())
    console.print("[green]tables created[/green]")


@synth_app.command("seed")
def synth_seed(
    briefs: int = typer.Option(20, help="number of briefs"),
    trajectories: int = typer.Option(400, help="number of historical trajectories"),
    seed: int = typer.Option(1, help="random seed"),
    days_back: int = typer.Option(120, help="history horizon in days"),
    no_truncate: bool = typer.Option(False, help="append instead of truncating all tables"),
    database_url: str | None = typer.Option(None, help="override DATABASE_URL"),
) -> None:
    """Seed the synthetic dataset (truncates all tables unless --no-truncate)."""
    from outlier_ai.synthetic.seed import seed as run_seed

    async def _run():
        if database_url:
            dbmod.configure(dbmod.make_engine(database_url))
        async with dbmod.session_scope() as session:
            return await run_seed(
                session,
                n_briefs=briefs,
                n_trajectories=trajectories,
                seed=seed,
                days_back=days_back,
                truncate=not no_truncate,
            )

    summary = asyncio.run(_run())
    table = Table(title="synthetic seed")
    table.add_column("object")
    table.add_column("count", justify="right")
    for k, v in summary.as_dict().items():
        table.add_row(k, str(v))
    console.print(table)


@config_app.command("show")
def config_show(path: Path | None = None) -> None:
    """Print the effective file config as YAML with its hash."""
    from outlier_ai.core.config import config_to_yaml, load_file_config

    cfg = load_file_config(path)
    console.print(f"# hash: {cfg.hash}")
    console.print(config_to_yaml(cfg))


@config_app.command("hash")
def config_hash(path: Path | None = None) -> None:
    from outlier_ai.core.config import load_file_config

    console.print(load_file_config(path).hash)


@config_app.command("validate")
def config_validate(path: Path | None = None) -> None:
    from outlier_ai.core.config import load_file_config

    cfg = load_file_config(path)
    console.print(f"[green]valid[/green] hash={cfg.hash} categories={len(cfg.category_medians)}")


outlier_app = typer.Typer(no_args_is_help=True, help="Outlier definition: recompute derived stats")
archive_app = typer.Typer(no_args_is_help=True, help="Archive sampling")
prompt_app = typer.Typer(no_args_is_help=True, help="Prompt rendering")
app.add_typer(outlier_app, name="outlier")
app.add_typer(archive_app, name="archive")
app.add_typer(prompt_app, name="prompt")


async def _resolve_brief_id(session, brief: str):
    import uuid as _uuid

    from sqlalchemy import select

    from outlier_ai.models.briefs import Brief

    if brief == "first":
        row = (
            (await session.execute(select(Brief).order_by(Brief.created_at).limit(1)))
            .scalars()
            .first()
        )
        if row is None:
            raise typer.BadParameter("no briefs in the database")
        return row.id
    return _uuid.UUID(brief)


@outlier_app.command("recompute")
def outlier_recompute(
    account: str | None = typer.Option(None, help="meta account id; default: every account"),
) -> None:
    """Recompute screening stats, outcomes, tiers, and combination counts from raw rows."""
    from sqlalchemy import select

    from outlier_ai.core.config import load_file_config
    from outlier_ai.models.meta import AdAccount
    from outlier_ai.outlier.recompute import recompute_account, recompute_all

    async def _run():
        cfg = load_file_config()
        async with dbmod.session_scope() as session:
            if account:
                row = (
                    await session.execute(
                        select(AdAccount).where(AdAccount.meta_account_id == account)
                    )
                ).scalar_one()
                return [await recompute_account(session, row.id, cfg)]
            return await recompute_all(session, cfg)

    table = Table(title="recompute")
    for col in ("account", "renders", "screening", "outcomes", "tiered", "tiers", "fallbacks"):
        table.add_column(col)
    for s in asyncio.run(_run()):
        table.add_row(
            str(s.account_id)[:8],
            str(s.renders),
            str(s.screening_rows),
            str(s.outcome_rows),
            str(s.trajectories_tiered),
            str(dict(sorted((s.tier_counts or {}).items()))),
            str(s.baseline_fallbacks),
        )
    console.print(table)


@archive_app.command("sample")
def archive_sample(brief: str = typer.Argument("first", help="brief uuid or 'first'")) -> None:
    """Show exactly what the archive sampler would put in the prompt for a brief."""
    from outlier_ai.archive.sampling import sample_archive
    from outlier_ai.core.config import load_file_config

    async def _run():
        cfg = load_file_config()
        async with dbmod.session_scope() as session:
            bid = await _resolve_brief_id(session, brief)
            return await sample_archive(session, bid, cfg)

    sample = asyncio.run(_run())
    table = Table(
        title=f"archive sample for {sample.brief_id} (holdout excluded: {sample.excluded_holdout})"
    )
    for col in ("ref", "reason", "niche", "tier", "ratio", "typicality", "signals", "angle"):
        table.add_column(col)
    for i in sample.items:
        table.add_row(
            i.ref,
            i.reason,
            i.niche,
            "-" if i.tier is None else str(i.tier),
            "-" if i.ratio is None else f"{i.ratio:.2f}",
            i.typicality or "-",
            str(len(i.signals)),
            i.angle.split("\n")[0][:60],
        )
    console.print(table)


@prompt_app.command("render")
def prompt_render(
    brief: str = typer.Argument("first", help="brief uuid or 'first'"),
    episode: str | None = typer.Option(None, help="episode uuid to include its batches"),
    k: int | None = typer.Option(None, help="ideas to ask for"),
    trace: bool = typer.Option(False, help="print the prompt trace as JSON instead of the prompt"),
) -> None:
    """Render the Loop A prompt for a brief exactly as the generator would see it."""
    import json
    import uuid as _uuid

    from outlier_ai.core.config import load_file_config
    from outlier_ai.generation.assemble import build_prompt

    async def _run():
        cfg = load_file_config()
        async with dbmod.session_scope() as session:
            bid = await _resolve_brief_id(session, brief)
            return await build_prompt(
                session, bid, cfg, episode_id=_uuid.UUID(episode) if episode else None, k=k
            )

    prompt, tr, _, _ = asyncio.run(_run())
    if trace:
        console.print_json(json.dumps(tr.as_dict()))
    else:
        print(prompt)
        console.print(
            console.print(
                f"[dim]~{tr.token_estimate} tokens, {len(tr.archive_refs)} history items, "
                f"trimmed: {tr.trimmed}[/dim]"
            )
        )


@app.command("generate")
def generate(
    brief: str = typer.Argument("first", help="brief uuid or 'first'"),
    backend: str = typer.Option("fake", help="fake | anthropic | local"),
    k: int = typer.Option(8, help="ideas per set"),
    episode: str | None = typer.Option(None, help="episode uuid to attach the batch to"),
    renders: int = typer.Option(3, help="renders per idea"),
    no_llm: bool = typer.Option(False, help="force heuristic verifier and reward model"),
    seed: int = typer.Option(0, help="seed for the fake backend"),
) -> None:
    """Run the generation pipeline once: prompt, backend, parse, verify, novelty, score, renders."""
    import uuid as _uuid

    from outlier_ai.core.config import load_file_config
    from outlier_ai.generation.factory import build_service
    from outlier_schemas.enums import BackendKind
    from outlier_schemas.models import GenerationRequest

    async def _run():
        cfg = load_file_config()
        async with dbmod.session_scope() as session:
            bid = await _resolve_brief_id(session, brief)
            service = await build_service(
                session,
                cfg,
                backend_kind=BackendKind(backend),
                use_llm_judges=not no_llm,
                seed=seed,
            )
            req = GenerationRequest(
                brief_id=bid,
                episode_id=_uuid.UUID(episode) if episode else None,
                backend=BackendKind(backend),
                k=k,
                renders_per_idea=renders,
            )
            return await service.generate_batch(session, req)

    result = asyncio.run(_run())
    table = Table(title=f"generated for brief {result.brief_id} ({result.raw.backend})")
    for col in (
        "status",
        "cards (written)",
        "verified",
        "typ",
        "rm",
        "novelty",
        "preship",
        "headline",
    ):
        table.add_column(col)
    for idea in result.ideas:
        t = idea.trajectory
        table.add_row(
            idea.status,
            ", ".join(idea.parsed.card_slugs),
            ", ".join(idea.verifier.card_slugs),
            t.typicality or "-",
            f"{t.rm_score:.2f}" if t.rm_score is not None else "-",
            idea.novelty.reason if idea.novelty else "-",
            "ok"
            if idea.preship and idea.preship.clean
            else ("-" if idea.preship is None else "flag"),
            idea.parsed.copy.headline[:32],
        )
    console.print(table)
    console.print(
        f"[dim]{len(result.queued)} queued, {len(result.rejected)} rejected; "
        f"~{result.raw.input_tokens} in / {result.raw.output_tokens} out tokens; "
        f"cost ${result.raw.cost_usd if result.raw.cost_usd is not None else 'n/a'}; "
        f"prompt ~{result.trace.token_estimate} tokens, trimmed={result.trace.trimmed}[/dim]"
    )


users_app = typer.Typer(no_args_is_help=True, help="Users and roles")
app.add_typer(users_app, name="users")


@users_app.command("create")
def users_create(
    email: str = typer.Option(..., help="login email"),
    password: str = typer.Option(..., help="at least 8 characters", prompt=True, hide_input=True),
    role: str = typer.Option("expert", help="operator | researcher | expert"),
    display_name: str = typer.Option("", help="shown in the UI"),
) -> None:
    """Create a user."""
    from outlier_ai.core.auth import create_user
    from outlier_schemas.enums import Role

    async def _run():
        async with dbmod.session_scope() as session:
            return await create_user(session, email, password, Role(role), display_name)

    user = asyncio.run(_run())
    console.print(f"[green]created[/green] {user.email} ({user.role})")


@users_app.command("set-password")
def users_set_password(
    email: str = typer.Option(...),
    password: str = typer.Option(..., prompt=True, hide_input=True),
) -> None:
    """Set a user's password (also fixes seeded users whose hash is unset)."""
    from outlier_ai.core.auth import set_password

    async def _run():
        async with dbmod.session_scope() as session:
            return await set_password(session, email, password)

    user = asyncio.run(_run())
    console.print(f"[green]password set[/green] for {user.email}")


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False, help="auto-reload on code changes"),
) -> None:
    """Run the API with uvicorn."""
    import uvicorn

    uvicorn.run("outlier_ai.api.app:app", host=host, port=port, reload=reload)


@app.command("openapi")
def openapi(
    out: Path = typer.Option(Path("frontend/openapi.json"), help="where to write the schema"),
) -> None:
    """Export the OpenAPI schema (the frontend generates its TypeScript types from it)."""
    import json

    from outlier_ai.api.app import app as fastapi_app

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fastapi_app.openapi(), indent=2))
    console.print(f"[green]wrote[/green] {out}")


jobs_app = typer.Typer(no_args_is_help=True, help="Jobs and the worker")
episodes_app = typer.Typer(no_args_is_help=True, help="Episodes: ship, tick")
fake_app = typer.Typer(no_args_is_help=True, help="Fake Meta account controls")
app.add_typer(jobs_app, name="jobs")
app.add_typer(episodes_app, name="episodes")
app.add_typer(fake_app, name="fake")


@jobs_app.command("run")
def jobs_run(
    kind: str = typer.Argument(
        ..., help="sync_insights | sync_comments | episode_tick | recompute_outcomes"
    ),
) -> None:
    """Run one job kind immediately (enqueue + process)."""
    from outlier_ai.jobs.registry import enqueue
    from outlier_ai.jobs.worker import run_once

    async def _run():
        async with dbmod.session_scope() as session:
            await enqueue(session, kind, key=f"cli:{kind}", payload={})
        return await run_once()

    n = asyncio.run(_run())
    console.print(f"[green]processed {n} job(s)[/green]")


@jobs_app.command("worker")
def jobs_worker(poll_seconds: float = typer.Option(30.0)) -> None:
    """Run the worker loop (daily cadence + due jobs)."""
    from outlier_ai.jobs.worker import serve

    asyncio.run(serve(poll_seconds))


@jobs_app.command("list")
def jobs_list(limit: int = 20) -> None:
    from sqlalchemy import select

    from outlier_ai.models.ops import Job

    async def _run():
        async with dbmod.session_scope() as session:
            return (
                (await session.execute(select(Job).order_by(Job.scheduled_for.desc()).limit(limit)))
                .scalars()
                .all()
            )

    table = Table(title="jobs")
    for col in ("kind", "key", "state", "attempts", "scheduled", "error"):
        table.add_column(col)
    for j in asyncio.run(_run()):
        table.add_row(
            j.kind,
            j.key,
            j.state,
            str(j.attempts),
            j.scheduled_for.isoformat(timespec="minutes"),
            (j.last_error or "")[:60],
        )
    console.print(table)


@episodes_app.command("ship")
def episodes_ship(
    batch: str = typer.Argument(..., help="batch uuid; ships every run-labeled idea"),
) -> None:
    """Ship a batch's approved ideas through the account's client (fake or real)."""
    import uuid as _uuid

    from outlier_ai.core.config import load_file_config
    from outlier_ai.core.storage import get_storage
    from outlier_ai.meta.factory import client_for_account
    from outlier_ai.meta.ship import ship_batch
    from outlier_ai.models.episodes import Batch, SearchEpisode
    from outlier_ai.models.meta import AdAccount

    async def _run():
        cfg = load_file_config()
        async with dbmod.session_scope() as session:
            b = await session.get(Batch, _uuid.UUID(batch))
            if b is None:
                raise typer.BadParameter("batch not found")
            ep = await session.get(SearchEpisode, b.episode_id)
            acc = (
                await session.get(AdAccount, ep.ad_account_id) if ep and ep.ad_account_id else None
            )
            if ep is None or acc is None:
                raise typer.BadParameter("episode has no ad account")
            client = await client_for_account(session, acc, cfg, require_shippable=True)
            return await ship_batch(
                session, b.id, client=client, storage=get_storage(), cfg=cfg, actor="cli"
            )

    results = asyncio.run(_run())
    for r in results:
        console.print(f"{r.trajectory_id}: shipped {len(r.shipped_render_ids)} renders {r.skipped}")


@episodes_app.command("tick")
def episodes_tick() -> None:
    """Run the controller once for every searching episode."""
    from outlier_ai.core.config import load_file_config
    from outlier_ai.jobs.handlers import run_episode_tick

    async def _run():
        async with dbmod.session_scope() as session:
            return await run_episode_tick(session, load_file_config())

    for r in asyncio.run(_run()):
        console.print(r)


@fake_app.command("advance")
def fake_advance(days: int = typer.Option(1), account: str = typer.Option("act_fake_1")) -> None:
    """Advance the fake Meta clock (resolves reviews, emits insights and comments)."""
    from sqlalchemy import select

    from outlier_ai.core.config import load_file_config
    from outlier_ai.meta.factory import client_for_account
    from outlier_ai.models.meta import AdAccount

    async def _run():
        cfg = load_file_config()
        async with dbmod.session_scope() as session:
            acc = (
                await session.execute(select(AdAccount).where(AdAccount.meta_account_id == account))
            ).scalar_one()
            client = await client_for_account(session, acc, cfg)
            return await client.advance_days(days)  # type: ignore[attr-defined]

    console.print(f"fake clock now {asyncio.run(_run())}")


loop_app = typer.Typer(no_args_is_help=True, help="Loop A")
experiments_app = typer.Typer(no_args_is_help=True, help="Simulator experiments")
app.add_typer(loop_app, name="loop-a")
app.add_typer(experiments_app, name="experiments")


@loop_app.command("run")
def loop_a_run(
    brief: str = typer.Argument("first", help="brief uuid or 'first'"),
    cap: float = typer.Option(5000.0, help="episode budget cap in USD"),
    backend: str = typer.Option("fake", help="fake | anthropic | local"),
    approval: str = typer.Option("top_rm", help="top_rm | random | manual"),
    max_days: int = typer.Option(120),
    no_llm: bool = typer.Option(True, help="use heuristic verifier/RM (default)"),
    seed: int = typer.Option(0),
) -> None:
    """Run a whole simulated episode on the fake account (or one step on a real one)."""
    from outlier_ai.core.config import load_file_config
    from outlier_ai.core.storage import get_storage
    from outlier_ai.episodes.loop_a import LoopARunner
    from outlier_ai.generation.factory import build_service
    from outlier_ai.meta.factory import client_for_account
    from outlier_ai.meta.fake import FakeMetaClient
    from outlier_ai.models.briefs import Brief
    from outlier_ai.models.episodes import SearchEpisode
    from outlier_ai.models.meta import AdAccount
    from outlier_schemas.enums import BackendKind

    async def _run():
        cfg = load_file_config()
        async with dbmod.session_scope() as session:
            bid = await _resolve_brief_id(session, brief)
            b = await session.get(Brief, bid)
            acc = await session.get(AdAccount, b.ad_account_id) if b and b.ad_account_id else None
            if b is None or acc is None:
                raise typer.BadParameter("brief has no ad account")
            ep = SearchEpisode(
                brief_id=b.id,
                ad_account_id=acc.id,
                backend=backend,
                budget_cap=cap,
                created_by="cli",
                config_hash=cfg.hash,
            )
            session.add(ep)
            await session.flush()
            client = await client_for_account(
                session, acc, cfg, require_shippable=True, fake_seed=seed
            )
            service = await build_service(
                session,
                cfg,
                backend_kind=BackendKind(backend),
                use_llm_judges=not no_llm,
                seed=seed,
            )
            runner = LoopARunner(
                session=session,
                cfg=cfg,
                service=service,
                client=client,
                storage=get_storage(),
                approval=approval,
                actor="cli",
                seed=seed,
            )  # type: ignore[arg-type]
            if isinstance(client, FakeMetaClient):
                return await runner.simulate(ep, max_days=max_days)
            return await runner.step(ep)

    res = asyncio.run(_run())
    console.print(res)


@experiments_app.command("loop-a-vs-random")
def experiment_loop_a(
    briefs: int = typer.Option(6),
    seeds: str = typer.Option("1", help="comma-separated seeds"),
    history: int = typer.Option(120, help="historical trajectories to seed"),
    cap: float = typer.Option(4000.0),
    max_days: int = typer.Option(90),
    out: Path | None = typer.Option(None, help="write the full report as JSON"),
) -> None:
    """Re-seeds the database per (seed, arm) and reports found rate and cost for both arms."""
    import json

    from outlier_ai.core.config import load_file_config
    from outlier_ai.core.storage import get_storage
    from outlier_ai.experiments.loop_a_vs_random import run_experiment

    async def _run():
        cfg = load_file_config()
        async with dbmod.session_scope() as session:
            return await run_experiment(
                session,
                cfg=cfg,
                storage=get_storage(),
                n_briefs=briefs,
                seeds=[int(x) for x in seeds.split(",") if x],
                history_trajectories=history,
                budget_cap=cap,
                max_days=max_days,
            )

    report = asyncio.run(_run())
    table = Table(title=f"loop A vs random: {briefs} briefs x seeds {seeds}")
    for col in (
        "arm",
        "runs",
        "found rate",
        "mean batches",
        "mean spend",
        "days to outlier",
        "ideas shipped",
    ):
        table.add_column(col)
    for arm, sm in report.summary().items():
        table.add_row(
            arm,
            f"{sm['runs']:.0f}",
            f"{sm['found_rate']:.2f}",
            f"{sm['mean_batches']:.1f}",
            f"${sm['mean_spend']:.0f}",
            "-" if sm["mean_days_to_outlier"] is None else f"{sm['mean_days_to_outlier']:.0f}",
            f"{sm['mean_ideas_shipped']:.1f}",
        )
    console.print(table)
    if out:
        out.write_text(json.dumps(report.as_dict(), indent=2, default=str))
        console.print(f"[dim]wrote {out}[/dim]")


rm_app = typer.Typer(no_args_is_help=True, help="Reward model")
train_app = typer.Typer(no_args_is_help=True, help="Loop B training runs")
eval_app = typer.Typer(no_args_is_help=True, help="Online eval")
feedback_app = typer.Typer(no_args_is_help=True, help="Feedback loops")
app.add_typer(rm_app, name="rm")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(feedback_app, name="feedback")


@archive_app.command("export")
def archive_export(include_unmeasured: bool = typer.Option(False)) -> None:
    """Export the archive as a Parquet snapshot (holdout excluded) to object storage."""

    async def _run() -> None:
        from outlier_ai.archive.export import export_snapshot
        from outlier_ai.core.storage import get_storage

        async with dbmod.session_scope() as session:
            cfg = await ConfigStore(session).current()
            snap = await export_snapshot(
                session, cfg=cfg, storage=get_storage(), include_unmeasured=include_unmeasured
            )
        typer.echo(json.dumps(snap.__dict__, indent=2))

    asyncio.run(_run())


@rm_app.command("train")
def rm_train(
    kind: str | None = typer.Option(None, help="rm_cold | rm_outcome (auto when omitted)"),
    activate: bool = typer.Option(True),
    seed: int = typer.Option(0),
) -> None:
    """Train the ensemble reward model on the current archive."""

    async def _run() -> None:
        from outlier_ai.core.embeddings import get_embedder
        from outlier_ai.core.storage import get_storage
        from outlier_ai.reward.train import train_reward_model
        from outlier_schemas.enums import RewardModelKind

        s = get_settings()
        async with dbmod.session_scope() as session:
            cfg = await ConfigStore(session).current()
            embedder = get_embedder(
                s, dims=cfg.archive.embedding_dims, model_name=cfg.archive.embedding_model
            )
            res = await train_reward_model(
                session,
                cfg=cfg,
                embedder=embedder,
                storage=get_storage(s),
                kind=RewardModelKind(kind) if kind else None,
                activate=activate,
                seed=seed,
            )
        if res is None:
            typer.echo("not enough labeled rows to train (need 20 rows with 3+ of each class)")
            raise typer.Exit(1)
        typer.echo(
            json.dumps(
                {
                    "version": res.version,
                    "kind": res.kind.value,
                    "n_rows": res.n_rows,
                    "n_positive": res.n_positive,
                    "train": res.report.__dict__,
                    "calibration": res.calibration.__dict__,
                },
                indent=2,
                default=str,
            )
        )

    asyncio.run(_run())


@rm_app.command("gold-gap")
def rm_gold_gap(window: int = typer.Option(100)) -> None:
    async def _run() -> None:
        from outlier_ai.reward.train import gold_gap_check

        async with dbmod.session_scope() as session:
            cfg = await ConfigStore(session).current()
            typer.echo(json.dumps(await gold_gap_check(session, cfg, window=window), indent=2))

    asyncio.run(_run())


@train_app.command("launch")
def train_launch(
    stage: str = typer.Argument(..., help="rft | dpo | grpo_offpolicy | grpo_onpolicy"),
    base_model: str = typer.Option("Qwen/Qwen3-8B"),
    smoke: bool = typer.Option(False, help="tiny run"),
    dry: bool = typer.Option(False, help="load data and report counts only"),
    simulator: bool = typer.Option(False),
    allow_rm_reward: bool = typer.Option(False),
    run_now: bool = typer.Option(
        True, help="run the job immediately instead of waiting for the worker"
    ),
) -> None:
    """Snapshot the archive, create a training run, and run the trainer job."""

    async def _run() -> None:
        from outlier_ai.jobs.training import queue_training_run, run_training_job
        from outlier_ai.models.ml import TrainingRun
        from outlier_schemas.enums import TrainingStage

        async with dbmod.session_scope() as session:
            cfg = await ConfigStore(session).current()
            run, job = await queue_training_run(
                session,
                cfg=cfg,
                stage=TrainingStage(stage),
                base_model=base_model,
                created_by="cli",
                smoke=smoke,
                dry=dry,
                simulator=simulator,
                allow_rm_reward=allow_rm_reward,
            )
            run_id, payload = run.id, dict(job.payload or {})
        typer.echo(f"training run {run_id} queued (job {job.id})")
        if run_now:
            async with dbmod.session_scope() as session:
                cfg = await ConfigStore(session).current()
                result = await run_training_job(session, payload, cfg=cfg)
                job_row = await session.get(type(job), job.id)
                if job_row is not None:
                    job_row.state = "done"
            typer.echo(json.dumps(result, indent=2, default=str))
            async with dbmod.session_scope() as session:
                run = await session.get(TrainingRun, run_id)
                assert run is not None
                typer.echo(
                    json.dumps(
                        {
                            "status": run.status,
                            "stop_reason": run.stop_reason,
                            "checkpoint": run.checkpoint_uri,
                            "metrics": run.metrics,
                        },
                        indent=2,
                        default=str,
                    )
                )

    asyncio.run(_run())


@train_app.command("list")
def train_list(limit: int = 20) -> None:
    async def _run() -> None:
        from sqlalchemy import select

        from outlier_ai.models.ml import TrainingRun

        async with dbmod.session_scope() as session:
            rows = (
                (
                    await session.execute(
                        select(TrainingRun).order_by(TrainingRun.created_at.desc()).limit(limit)
                    )
                )
                .scalars()
                .all()
            )
            for r in rows:
                typer.echo(
                    f"{r.id}  {r.stage:15} {r.status:10} snapshot={r.snapshot_hash} "
                    f"{r.stop_reason or ''}"
                )

    asyncio.run(_run())


@eval_app.command("launch")
def eval_launch(
    systems: str = typer.Option("loop_a_fake,random_fake", help="comma-separated systems"),
    briefs: int = typer.Option(3),
    cap: float = typer.Option(2000.0),
    seed: int = typer.Option(0),
    max_days: int = typer.Option(120),
    kind: str = typer.Option("online"),
) -> None:
    """Run the online eval harness (fake systems run to completion; live ones start episodes)."""

    async def _run() -> None:
        from outlier_ai.core.storage import get_storage
        from outlier_ai.eval.harness import run_eval
        from outlier_schemas.enums import EvalKind

        async with dbmod.session_scope() as session:
            cfg = await ConfigStore(session).current()
            result = await run_eval(
                session,
                cfg=cfg,
                storage=get_storage(),
                systems=[x.strip() for x in systems.split(",") if x.strip()],
                n_briefs=briefs,
                budget_cap=cap,
                created_by="cli",
                kind=EvalKind(kind),
                seed=seed,
                max_days=max_days,
            )
        typer.echo(
            json.dumps({"eval_id": str(result.eval_id), **result.summary()}, indent=2, default=str)
        )

    asyncio.run(_run())


@feedback_app.command("suggest-relations")
def feedback_suggest() -> None:
    async def _run() -> None:
        from outlier_ai.jobs.feedback import suggest_relations

        async with dbmod.session_scope() as session:
            out = await suggest_relations(session, actor="cli")
        for r in out:
            typer.echo(
                f"{r.a} + {r.b}: observed {r.observed}, expected {r.expected:.2f}, "
                f"lift {r.lift:.2f}"
            )
        typer.echo(f"{len(out)} suggested relations")

    asyncio.run(_run())


@feedback_app.command("retrain-verifier")
def feedback_retrain_verifier(min_labels: int | None = typer.Option(None)) -> None:
    async def _run() -> None:
        from outlier_ai.core.embeddings import get_embedder
        from outlier_ai.core.storage import get_storage
        from outlier_ai.jobs.feedback import retrain_verifier

        s = get_settings()
        async with dbmod.session_scope() as session:
            cfg = await ConfigStore(session).current()
            embedder = get_embedder(
                s, dims=cfg.archive.embedding_dims, model_name=cfg.archive.embedding_model
            )
            vv = await retrain_verifier(
                session,
                cfg=cfg,
                embedder=embedder,
                storage=get_storage(s),
                min_labels=min_labels,
                actor="cli",
            )
        if vv is None:
            typer.echo("not enough expert labels")
            raise typer.Exit(1)
        typer.echo(
            json.dumps(
                {"version": vv.version, "metrics": vv.metrics, "labels": vv.trained_on_labels},
                indent=2,
            )
        )

    asyncio.run(_run())


@app.command("env")
def env() -> None:
    """Show non-secret settings."""
    s = get_settings()
    console.print(
        {
            "env": s.env,
            "dry_run": s.dry_run,
            "database_url": s.database_url.split("@")[-1],
            "config_path": str(s.config_path),
            "storage_backend": s.storage_backend,
        }
    )


if __name__ == "__main__":
    app()
