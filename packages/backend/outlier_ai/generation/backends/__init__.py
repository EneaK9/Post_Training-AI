"""Generator and judge backends (spec section 5.2)."""

from __future__ import annotations

from outlier_ai.core.settings import Settings, get_settings
from outlier_ai.generation.backends.base import (
    BackendError,
    BackendUnavailableError,
    GeneratorBackend,
    JudgeBackend,
    RawOutput,
)
from outlier_schemas.config import AppConfig
from outlier_schemas.enums import BackendKind


def get_generator(
    kind: BackendKind,
    cfg: AppConfig,
    settings: Settings | None = None,
    *,
    seed: int = 0,
    adapter: str | None = None,
) -> GeneratorBackend:
    s = settings or get_settings()
    if kind == BackendKind.fake:
        from outlier_ai.generation.backends.fake_backend import FakeBackend

        return FakeBackend(seed=seed)
    if kind == BackendKind.anthropic:
        from outlier_ai.generation.backends.anthropic_backend import AnthropicBackend

        return AnthropicBackend(
            cfg.generation.anthropic_model,
            api_key=s.anthropic_api_key,
            effort=cfg.generation.anthropic_effort,
            use_fallbacks=cfg.generation.anthropic_fallbacks,
        )
    if kind == BackendKind.local:
        from outlier_ai.generation.backends.local_policy import LocalPolicyBackend

        model = (
            cfg.generation.local_base_model
            if cfg.generation.weights == "base"
            else cfg.generation.local_model
        )
        return LocalPolicyBackend(
            cfg.generation.vllm_url,
            model,
            adapter=adapter,
            temperature=cfg.generation.temperature,
        )
    raise BackendUnavailableError(f"unknown backend kind {kind}")


def get_judge(
    cfg: AppConfig, settings: Settings | None = None, *, model: str | None = None
) -> JudgeBackend | None:
    """A Claude judge for the verifier, extraction, policy screen, and cold reward model.

    Returns None when no API key is configured so callers fall back to heuristics.
    """
    s = settings or get_settings()
    if not s.anthropic_api_key:
        return None
    try:
        from outlier_ai.generation.backends.anthropic_backend import AnthropicBackend

        return AnthropicBackend(
            model or cfg.verifier.model, api_key=s.anthropic_api_key, effort="low"
        )
    except BackendUnavailableError:
        return None


__all__ = [
    "BackendError",
    "BackendUnavailableError",
    "GeneratorBackend",
    "JudgeBackend",
    "RawOutput",
    "get_generator",
    "get_judge",
]
