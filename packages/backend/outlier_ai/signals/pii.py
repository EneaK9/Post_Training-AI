"""PII stripping before any comment text is stored.

Regex pass for emails, phone numbers, @handles, URLs, and "my name is X" patterns. When
`presidio-analyzer` is installed (extra `signals`) it runs first for names and other entity
types; the regex pass always runs after it as a floor.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+", re.I)
_PHONE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)")
_HANDLE = re.compile(r"(?<![\w])@[A-Za-z0-9_.]{2,}")
_NAME_INTRO = re.compile(
    r"\b(my name is|i am|i'm|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b"
)
_URL = re.compile(r"https?://\S+", re.I)
_ENGINE: Any = None


@dataclass(frozen=True)
class PiiResult:
    text: str
    removed: bool
    kinds: tuple[str, ...]


def _presidio(text: str) -> tuple[str, list[str]]:
    global _ENGINE
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore[import-not-found]
    except Exception:
        return text, []
    if _ENGINE is None:
        _ENGINE = AnalyzerEngine()
    results = _ENGINE.analyze(
        text=text, entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION"], language="en"
    )
    kinds: list[str] = []
    out = text
    for r in sorted(results, key=lambda r: r.start, reverse=True):
        if r.score < 0.6:
            continue
        out = out[: r.start] + f"[{r.entity_type.lower()}]" + out[r.end :]
        kinds.append(r.entity_type.lower())
    return out, kinds


def strip_pii(text: str, *, use_presidio: bool = True) -> PiiResult:
    original = text
    kinds: list[str] = []
    if use_presidio:
        text, k = _presidio(text)
        kinds.extend(k)
    for rx, label in ((_URL, "url"), (_EMAIL, "email"), (_PHONE, "phone"), (_HANDLE, "handle")):
        text, n = rx.subn(f"[{label}]", text)
        if n:
            kinds.append(label)
    text, n = _NAME_INTRO.subn(lambda m: f"{m.group(1)} [name]", text)
    if n:
        kinds.append("name")
    return PiiResult(text=text.strip(), removed=text != original, kinds=tuple(dict.fromkeys(kinds)))
