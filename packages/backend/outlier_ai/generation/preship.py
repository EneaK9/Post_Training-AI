"""Pre-ship checks (spec section 5.8) [v3-assumed].

Policy screen (keyword denylist plus optional Claude classification), brand constraints from
the brief, and Meta copy-length warnings. Produces a `PreshipReport`; the human `run` label
remains the approval and nothing ships while `policy_ok` or `brand_ok` is false.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from outlier_ai.generation.backends.base import JudgeBackend
from outlier_schemas.config import ImagesConfig
from outlier_schemas.models import AdCopy, PreshipReport

# Meta advertising policy themes, v1 keyword screen. The LLM screen catches the rest.
POLICY_DENYLIST: dict[str, tuple[str, ...]] = {
    "unrealistic_outcomes": (
        "guaranteed results",
        "miracle",
        "cures ",
        "cure for",
        "lose 10 pounds",
        "lose weight fast",
        "overnight results",
        "100% safe",
        "no side effects",
    ),
    "personal_attributes": (
        "are you fat",
        "are you depressed",
        "your disease",
        "people like you who",
    ),
    "misleading_claims": ("fda approved", "clinically proven", "doctors hate", "#1 rated"),
    "prohibited_content": ("get rich quick", "crypto giveaway", "free money"),
}

_STOP = {
    "a",
    "an",
    "the",
    "of",
    "in",
    "on",
    "for",
    "to",
    "and",
    "or",
    "with",
    "photos",
    "photo",
    "images",
    "image",
    "real",
}
_WORD = re.compile(r"[a-z0-9/]+")
_JSON_OBJ = re.compile(r"\{.*\}", re.S)


def combined_text(copy: AdCopy, angle: str, visual_brief: str) -> str:
    return "\n".join(
        [angle, copy.primary_text, copy.headline, copy.description, copy.cta, visual_brief]
    ).lower()


def check_policy_keywords(text: str) -> list[str]:
    flags: list[str] = []
    for theme, phrases in POLICY_DENYLIST.items():
        for p in phrases:
            if p in text:
                flags.append(f"{theme}: '{p.strip()}'")
    return flags


def check_brand_constraints(constraints: Sequence[str], text: str) -> list[str]:
    """`no X` constraints flag when X's key words all appear; `include X` flags when absent."""
    flags: list[str] = []
    words = set(_WORD.findall(text))
    for c in constraints:
        lc = c.strip().lower()
        if lc.startswith("no "):
            phrase = lc[3:].strip()
            keys = [w for w in _WORD.findall(phrase) if w not in _STOP and len(w) > 2]
            if phrase and (phrase in text or (keys and all(k in words for k in keys))):
                flags.append(f"constraint violated: {c}")
        elif lc.startswith("include "):
            phrase = lc[8:].strip()
            keys = [w for w in _WORD.findall(phrase) if w not in _STOP and len(w) > 2]
            if phrase and not (phrase in text or (keys and all(k in words for k in keys))):
                flags.append(f"required element missing: {c}")
    return flags


def check_lengths(copy: AdCopy, cfg: ImagesConfig) -> list[str]:
    warnings: list[str] = []
    if len(copy.primary_text) > cfg.primary_text_warn_chars:
        warnings.append(
            f"primary_text {len(copy.primary_text)} chars > "
            f"{cfg.primary_text_warn_chars} (truncated in feed)"
        )
    if len(copy.headline) > cfg.headline_warn_chars:
        warnings.append(f"headline {len(copy.headline)} chars > {cfg.headline_warn_chars}")
    if len(copy.description) > cfg.description_warn_chars:
        warnings.append(f"description {len(copy.description)} chars > {cfg.description_warn_chars}")
    return warnings


POLICY_SYSTEM = (
    "You review Meta (Facebook and Instagram) feed ads for policy problems before they run. "
    "Flag: personal attributes, unrealistic or guaranteed outcomes, misleading or unsubstantiated "
    "health claims, prohibited products, adult content, hate, and sensational or shocking content. "
    'Return JSON only: {"ok": true|false, "flags": ["short reason", ...]}. Be specific and brief.'
)


async def llm_policy_screen(judge: JudgeBackend, text: str) -> tuple[bool, list[str]]:
    raw = await judge.judge(POLICY_SYSTEM, f"Ad text:\n{text}\n\nJSON:", max_tokens=300)
    m = _JSON_OBJ.search(raw or "")
    if not m:
        return True, []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return True, []
    ok = bool(data.get("ok", True))
    flags = [str(f) for f in data.get("flags", []) if f]
    return ok and not flags, [f"llm: {f}" for f in flags]


async def preship_check(
    *,
    copy: AdCopy,
    angle: str,
    visual_brief: str,
    constraints: Sequence[str],
    images_cfg: ImagesConfig,
    judge: JudgeBackend | None = None,
) -> PreshipReport:
    text = combined_text(copy, angle, visual_brief)
    policy_flags = check_policy_keywords(text)
    if judge is not None:
        ok, llm_flags = await llm_policy_screen(judge, text)
        if not ok:
            policy_flags.extend(llm_flags or ["llm: flagged"])
    brand_flags = check_brand_constraints(constraints, text)
    return PreshipReport(
        policy_ok=not policy_flags,
        policy_flags=policy_flags,
        brand_ok=not brand_flags,
        brand_flags=brand_flags,
        length_warnings=check_lengths(copy, images_cfg),
    )
