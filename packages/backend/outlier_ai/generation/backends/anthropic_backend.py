"""Claude via the official Anthropic SDK (APIModel in the spec).

Generator: one user message per call, `output_config.effort` from config, no sampling
parameters (Claude Opus 5 and Sonnet 5 reject `temperature`). Diversity comes from the
distribution-style prompt. Judge: system + user, used for the verifier, signal extraction,
the policy screen, and the cold reward model.

Refusals are surfaced, not swallowed: `RawOutput.refused` is set when `stop_reason` is
"refusal". For Opus 5 and Fable models the server-side fallback parameter is on by default so
a policy decline re-runs on a fallback model inside the same call.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

from outlier_ai.generation.backends.base import BackendError, BackendUnavailableError, RawOutput
from outlier_schemas.enums import BackendKind

# USD per 1M tokens (input, output), Anthropic first-party rates.
PRICING: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    for prefix, (pin, pout) in PRICING.items():
        if model.startswith(prefix):
            return input_tokens / 1e6 * pin + output_tokens / 1e6 * pout
    return None


class AnthropicBackend:
    kind = BackendKind.anthropic
    trainable = False

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        effort: str = "high",
        use_fallbacks: bool = True,
        timeout: float = 600.0,
        max_retries: int = 3,
    ) -> None:
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover - exercised only without the extra
            raise BackendUnavailableError(
                "the anthropic SDK is not installed; install outlier-ai[llm]"
            ) from e
        self._anthropic = anthropic
        # api_key None lets the SDK resolve ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / profile.
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key, timeout=timeout, max_retries=max_retries
        )
        self.model = model
        self.name = f"anthropic:{model}"
        self.effort = effort
        self.use_fallbacks = use_fallbacks

    def _wants_fallbacks(self) -> bool:
        return self.use_fallbacks and self.model.startswith(("claude-opus-5", "claude-fable-5"))

    async def _create(self, *, system: str | None, user: str, max_tokens: int) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user}],
            "output_config": {"effort": self.effort},
        }
        if system:
            kwargs["system"] = system
        a = self._anthropic
        try:
            if self._wants_fallbacks():
                return await self.client.beta.messages.create(
                    betas=[FALLBACK_BETA], fallbacks="default", **kwargs
                )
            return await self.client.messages.create(**kwargs)
        except a.RateLimitError as e:
            raise BackendError(f"anthropic rate limited: {e.message}") from e
        except a.APIStatusError as e:
            raise BackendError(f"anthropic api error {e.status_code}: {e.message}") from e
        except a.APIConnectionError as e:
            raise BackendError(f"anthropic connection error: {e}") from e

    @staticmethod
    def _text_and_refusal(response: Any) -> tuple[str, bool]:
        refused = getattr(response, "stop_reason", None) == "refusal"
        text = "".join(
            getattr(b, "text", "") for b in response.content if getattr(b, "type", "") == "text"
        )
        return text, refused

    async def generate(
        self, prompt: str, n: int = 1, *, max_tokens: int = 16000
    ) -> list[RawOutput]:
        outs: list[RawOutput] = []
        for _ in range(n):
            t0 = perf_counter()
            resp = await self._create(system=None, user=prompt, max_tokens=max_tokens)
            text, refused = self._text_and_refusal(resp)
            usage = resp.usage
            outs.append(
                RawOutput(
                    text=text,
                    backend=self.name,
                    model=str(resp.model),
                    input_tokens=int(usage.input_tokens),
                    output_tokens=int(usage.output_tokens),
                    latency_ms=(perf_counter() - t0) * 1000,
                    cost_usd=estimate_cost_usd(
                        str(resp.model), int(usage.input_tokens), int(usage.output_tokens)
                    ),
                    stop_reason=resp.stop_reason,
                    refused=refused,
                    meta={
                        "request_id": getattr(resp, "_request_id", None),
                        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
                    },
                )
            )
        return outs

    async def judge(self, system: str, user: str, *, max_tokens: int = 2000) -> str:
        resp = await self._create(system=system, user=user, max_tokens=max_tokens)
        text, refused = self._text_and_refusal(resp)
        if refused:
            raise BackendError("anthropic judge call was refused")
        return text
