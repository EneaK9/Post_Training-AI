"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from outlier_ai import __version__
from outlier_ai.api.routers import (
    archive,
    audit,
    auth,
    briefs,
    cards,
    combinations,
    config,
    episodes,
    eval,
    generate,
    imports,
    meta,
    reviews,
    search,
    signals,
    trajectories,
)
from outlier_ai.core import db as dbmod
from outlier_ai.core.config import ConfigStore
from outlier_ai.core.errors import (
    ConfigError,
    NotFoundError,
    OutlierError,
    SafetyError,
    ValidationError,
)
from outlier_ai.core.settings import get_settings
from outlier_ai.generation.backends.base import BackendError
from outlier_ai.meta.errors import MetaError

ROUTERS = [
    auth.router,
    cards.router,
    combinations.router,
    briefs.router,
    trajectories.router,
    reviews.router,
    signals.router,
    archive.router,
    episodes.router,
    generate.router,
    meta.router,
    config.router,
    search.router,
    audit.router,
    eval.router,
    imports.router,
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        async with dbmod.session_scope() as session:
            await ConfigStore(session).ensure_bootstrapped("api-boot")
    except Exception as e:
        app.state.boot_error = repr(e)
    yield
    await dbmod.dispose()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Outlier AI", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for r in ROUTERS:
        app.include_router(r, prefix="/api")

    @app.get("/api/health")
    async def health() -> dict:
        return {
            "ok": True,
            "version": __version__,
            "dry_run": settings.dry_run,
            "env": settings.env,
        }

    @app.exception_handler(NotFoundError)
    async def _nf(_: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(ValidationError)
    async def _val(_: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.exception_handler(SafetyError)
    async def _safe(_: Request, exc: SafetyError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=403)

    @app.exception_handler(ConfigError)
    async def _cfg(_: Request, exc: ConfigError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=500)

    @app.exception_handler(BackendError)
    async def _backend(_: Request, exc: BackendError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=502)

    @app.exception_handler(MetaError)
    async def _meta(_: Request, exc: MetaError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=502)

    @app.exception_handler(OutlierError)
    async def _other(_: Request, exc: OutlierError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    return app


app = create_app()
