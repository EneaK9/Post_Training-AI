"""Backend interfaces. `GeneratorBackend.generate(prompt, n) -> list[RawOutput]`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from outlier_ai.core.errors import OutlierError
from outlier_schemas.enums import BackendKind


class BackendError(OutlierError):
    """A backend call failed after the SDK's own retries."""


class BackendUnavailableError(BackendError):
    """Missing SDK, key, or server for the requested backend."""


@dataclass(frozen=True)
class RawOutput:
    text: str
    backend: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float | None = None
    stop_reason: str | None = None
    refused: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


class GeneratorBackend(Protocol):
    name: str
    kind: BackendKind
    model: str
    trainable: bool

    async def generate(
        self, prompt: str, n: int = 1, *, max_tokens: int = 16000
    ) -> list[RawOutput]: ...


class JudgeBackend(Protocol):
    """Structured-output helper calls: verifier, signal extraction, policy screen, cold RM."""

    name: str
    model: str

    async def judge(self, system: str, user: str, *, max_tokens: int = 2000) -> str: ...
