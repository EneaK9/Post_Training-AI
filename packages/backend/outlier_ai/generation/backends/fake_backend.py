"""Deterministic generator for tests and the simulator.

Reads the playbook, brief, and history refs straight out of the rendered prompt, then
composes K well-formed idea blocks with distinct card combinations, typicality labels
(at least one rare), and citations that resolve. `malformed_rate` injects broken blocks
so the parser and the format penalty get exercised.
"""

from __future__ import annotations

import re
from time import perf_counter

import numpy as np

from outlier_ai.generation.backends.base import RawOutput
from outlier_schemas.enums import BackendKind

_SECTION = re.compile(r"^\[(STRATEGY|STYLE|PRINCIPLE|MECHANIC)\]\s*$", re.M)
_CARD_LINE = re.compile(r"^- card:([a-z0-9-]+) \| (.+?)\s*$", re.M)
_HISTORY_REF = re.compile(r"history:([0-9a-f]{8})")
_SIGNAL_REF = re.compile(r"signal:([0-9a-f]{8})")
_K = re.compile(r"Propose exactly (\d+) ideas")
_FIELD = {
    "company": re.compile(r"^Company: (.+)$", re.M),
    "product": re.compile(r"^Product: (.+)$", re.M),
    "offer": re.compile(r"^Offer: (.+)$", re.M),
}

HOOKS = [
    "Stop doing it the old way.",
    "Nobody tells you this about {product}.",
    "We measured it. The difference is not small.",
    "The complaint everyone makes is exactly the point.",
    "This is the opposite of what the category says.",
    "Made slowly, on purpose.",
]
ENEMIES = ["the incumbent", "shortcuts", "received wisdom", "boring ads", "the usual routine"]
CTAS = ["Shop Now", "Learn More", "Get Offer"]


class FakeBackend:
    name = "fake"
    kind = BackendKind.fake
    model = "fake-composer-v1"
    trainable = False

    def __init__(self, seed: int = 0, malformed_rate: float = 0.0) -> None:
        self.rng = np.random.default_rng(seed)
        self.malformed_rate = malformed_rate

    @staticmethod
    def parse_playbook(prompt: str) -> dict[str, list[tuple[str, str]]]:
        playbook: dict[str, list[tuple[str, str]]] = {}
        end = prompt.find("COLUMN 2: BRIEF")
        section_text = prompt[: end if end > 0 else len(prompt)]
        current: str | None = None
        for line in section_text.splitlines():
            m = _SECTION.match(line)
            if m:
                kind = m.group(1).lower()
                current = kind
                playbook.setdefault(kind, [])
                continue
            c = _CARD_LINE.match(line)
            if c and current:
                playbook[current].append((c.group(1), c.group(2)))
        return playbook

    def _pick(self, pool: list[tuple[str, str]], n: int) -> list[tuple[str, str]]:
        if not pool or n <= 0:
            return []
        idx = self.rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        return [pool[int(i)] for i in idx]

    def compose(self, prompt: str) -> str:
        playbook = self.parse_playbook(prompt)
        strategies = playbook.get("strategy", [])
        others = [c for k, cs in playbook.items() if k != "strategy" for c in cs]
        all_cards = strategies + others
        km = _K.search(prompt)
        k = int(km.group(1)) if km else 8
        fields = {
            key: (m.group(1).strip() if (m := rx.search(prompt)) else key)
            for key, rx in _FIELD.items()
        }
        history = sorted(set(_HISTORY_REF.findall(prompt)))
        signals = sorted(set(_SIGNAL_REF.findall(prompt)))

        blocks: list[str] = []
        used: set[tuple[str, ...]] = set()
        labels = ["common", "uncommon", "rare"]
        for i in range(k):
            combo: list[tuple[str, str]] = []
            for _ in range(12):
                n_strat = 1 if not strategies or self.rng.random() < 0.7 else 2
                combo = self._pick(strategies, n_strat) + self._pick(
                    others, int(self.rng.integers(1, 3))
                )
                if len(combo) < 2:
                    combo = self._pick(all_cards, 2)
                key = tuple(sorted(s for s, _ in combo))
                if key not in used:
                    used.add(key)
                    break
            typ = "rare" if i == k - 1 else labels[int(self.rng.integers(0, 3))]
            strat_name = combo[0][1] if combo else "Idea"
            other_names = ", ".join(n for _, n in combo[1:])
            hook = str(self.rng.choice(HOOKS)).format(product=fields["product"])
            enemy = str(self.rng.choice(ENEMIES))
            cites = " ".join(f"[card:{s}]" for s, _ in combo)
            if history:
                cites += f" [history:{self.rng.choice(history)}]"
            if signals and self.rng.random() < 0.7:
                cites += f" [signal:{self.rng.choice(signals)}]"
            headline = f"{strat_name}: {fields['product']}"[:38]
            primary = (
                f"{hook} {fields['company']} makes {fields['product']}. "
                f"Executed as {other_names or strat_name}. {fields['offer']}."
            )
            reasoning = (
                f"{strat_name} fits because the brief and history point at {enemy}. "
                f"Responds to what the audience said. {cites}"
            )
            visual = (
                f"{fields['product']} shown in a real setting with a {other_names or 'plain'} "
                f"treatment; text overlay reads '{headline}'."
            )
            cards_line = ", ".join(s for s, _ in combo)
            block = (
                "<idea>\n"
                f"<cards>{cards_line}</cards>\n"
                f"<typicality>{typ}</typicality>\n"
                f"<reasoning>{reasoning}</reasoning>\n"
                f"<angle>Hook: {hook} / Enemy: {enemy}\nPromise: {fields['offer']}</angle>\n"
                f"<copy>primary_text: {primary}\nheadline: {headline}\n"
                f"description: {fields['offer'][:28]}\ncta: {self.rng.choice(CTAS)}</copy>\n"
                f"<visual_brief>{visual}</visual_brief>\n"
                "</idea>"
            )
            if self.rng.random() < self.malformed_rate:
                block = block.replace("</cards>", "").replace(
                    "<typicality>rare", "<typicality>wild"
                )
            blocks.append(block)
        return "\n\n".join(blocks)

    async def generate(
        self, prompt: str, n: int = 1, *, max_tokens: int = 16000
    ) -> list[RawOutput]:
        outs: list[RawOutput] = []
        for _ in range(n):
            t0 = perf_counter()
            text = self.compose(prompt)
            outs.append(
                RawOutput(
                    text=text,
                    backend=self.name,
                    model=self.model,
                    input_tokens=len(prompt) // 4,
                    output_tokens=len(text) // 4,
                    latency_ms=(perf_counter() - t0) * 1000,
                    cost_usd=0.0,
                    stop_reason="end_turn",
                )
            )
        return outs

    async def judge(self, system: str, user: str, *, max_tokens: int = 2000) -> str:
        """Minimal judge so the fake can stand in for Claude in tests: echoes an empty JSON."""
        return "{}"
