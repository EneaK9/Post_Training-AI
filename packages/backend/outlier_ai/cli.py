"""`oai` command line."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from outlier_ai.core import db as dbmod
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
