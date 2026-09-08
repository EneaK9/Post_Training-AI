"""LocalPolicy: an open-weight model served by vLLM's OpenAI-compatible HTTP API.

The trainable backend. Loop B checkpoints are LoRA adapters registered with the same vLLM
server; `adapter` selects one by name.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import httpx

from outlier_ai.generation.backends.base import BackendError, RawOutput
from outlier_schemas.enums import BackendKind


class LocalPolicyBackend:
    kind = BackendKind.local
    trainable = True

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        adapter: str | None = None,
        temperature: float = 1.0,
        timeout: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.adapter = adapter
        self.temperature = temperature
        self.timeout = timeout
        self.name = f"local:{adapter or model}"

    async def _chat(
        self, messages: list[dict[str, str]], n: int, max_tokens: int
    ) -> dict[str, Any]:
        payload = {
            "model": self.adapter or self.model,
            "messages": messages,
            "n": n,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(f"{self.base_url}/chat/completions", json=payload)
                r.raise_for_status()
                return r.json()
        except httpx.HTTPError as e:
            raise BackendError(f"vllm request failed: {e}") from e

    async def generate(self, prompt: str, n: int = 1, *, max_tokens: int = 8000) -> list[RawOutput]:
        t0 = perf_counter()
        data = await self._chat([{"role": "user", "content": prompt}], n, max_tokens)
        usage = data.get("usage", {})
        latency = (perf_counter() - t0) * 1000
        choices = data.get("choices", [])
        return [
            RawOutput(
                text=ch.get("message", {}).get("content", "") or "",
                backend=self.name,
                model=str(data.get("model", self.model)),
                input_tokens=int(usage.get("prompt_tokens", 0)),
                output_tokens=int(usage.get("completion_tokens", 0)) // max(1, len(choices)),
                latency_ms=latency,
                cost_usd=0.0,
                stop_reason=ch.get("finish_reason"),
            )
            for ch in choices
        ]

    async def judge(self, system: str, user: str, *, max_tokens: int = 2000) -> str:
        data = await self._chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            1,
            max_tokens,
        )
        choices = data.get("choices", [])
        return choices[0].get("message", {}).get("content", "") if choices else ""
