"""Group stored comments into proposed signals.

With a Claude judge: one pass returns themes with counts, sentiment, kind, and evidence
excerpts. Without one: a keyword-theme fallback that mirrors the synthetic theme bank so the
pipeline runs end to end in tests. Signals are written with `status = proposed`; experts
confirm or reject on the Data screen. Only confirmed signals enter prompts.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from outlier_ai.generation.backends.base import JudgeBackend
from outlier_ai.models.meta import Comment
from outlier_ai.models.signals import Signal
from outlier_ai.models.trajectories import Render
from outlier_schemas.enums import AuthorKind, SignalKind, SignalStatus

_JSON = re.compile(r"\[.*\]|\{.*\}", re.S)

# (kind, sentiment, summary, keyword regex)
KEYWORD_THEMES: list[tuple[SignalKind, str, str, re.Pattern[str]]] = [
    (
        SignalKind.objection,
        "neg",
        "Price is seen as too high",
        re.compile(r"\b(price|overpriced|expensive|\$\$\$|pays? this much|cost)\b", re.I),
    ),
    (
        SignalKind.objection,
        "neg",
        "Audience says it is nothing new",
        re.compile(r"\b(seen this|every brand|same ad|nothing new)\b", re.I),
    ),
    (
        SignalKind.objection,
        "neg",
        "Suspicion the offer is a scam",
        re.compile(r"\b(scam|too good to be true|reported)\b", re.I),
    ),
    (
        SignalKind.objection,
        "neg",
        "Complaints about shipping or delivery",
        re.compile(r"\b(shipping|arrive|delivery|weeks)\b", re.I),
    ),
    (
        SignalKind.misreading,
        "neg",
        "Offer is misread as something else",
        re.compile(r"\b(subscription\?\?|not actually free|i thought this was)\b", re.I),
    ),
    (
        SignalKind.praise,
        "pos",
        "Praise for product quality",
        re.compile(r"\b(best i've tried|love it|quality is|unreal)\b", re.I),
    ),
    (
        SignalKind.question,
        "pos",
        "People asking where to buy",
        re.compile(r"\b(where can i|link\?|do you ship)\b", re.I),
    ),
    (
        SignalKind.share_pattern,
        "pos",
        "Tagging friends",
        re.compile(r"(@\w+|\[handle\]|tagging my)", re.I),
    ),
    (
        SignalKind.quote,
        "pos",
        "Testimonial-style comments",
        re.compile(r"\b(fixed my|ordered twice|converted a skeptic)\b", re.I),
    ),
    (
        SignalKind.question,
        "neu",
        "Questions about details",
        re.compile(r"\b(what sizes|is it vegan|how long does)\b", re.I),
    ),
    (
        SignalKind.joke,
        "neu",
        "Jokes riffing on the ad",
        re.compile(r"\b(wallet|left the chat|that hook got me|better than the product)\b", re.I),
    ),
    (
        SignalKind.competitor_mention,
        "neu",
        "Competitor mentions",
        re.compile(r"\b(brand[xy]|does this cheaper|how is this different)\b", re.I),
    ),
]

EXTRACT_SYSTEM = (
    "You read audience comments on a paid social ad and group them into themes a marketer "
    "can act on. Kinds: comment_theme, objection, praise, joke, misreading, question, "
    "competitor_mention, quote, hook_stat, share_pattern. "
    'Return JSON only: a list of {"kind": ..., "sentiment": "neg|neu|pos", "text": one-line theme, '
    '"count": number of comments supporting it, "excerpts": [up to 3 short verbatim quotes]}. '
    "Ignore spam. Never include names, emails, phone numbers, or handles."
)


def keyword_extract(comments: list[Comment], min_count: int) -> list[dict]:
    groups: dict[int, list[Comment]] = defaultdict(list)
    for c in comments:
        if c.filtered_reason:
            continue
        for idx, (_, _, _, rx) in enumerate(KEYWORD_THEMES):
            if rx.search(c.text):
                groups[idx].append(c)
                break
    out: list[dict] = []
    for idx, items in groups.items():
        if len(items) < min_count:
            continue
        kind, sentiment, summary, _ = KEYWORD_THEMES[idx]
        excerpts = list(dict.fromkeys(c.text for c in items))[:3]
        out.append(
            {
                "kind": kind.value,
                "sentiment": sentiment,
                "text": summary,
                "count": len(items),
                "excerpts": excerpts,
                "refs": [str(c.id) for c in items[:3]],
            }
        )
    out.sort(key=lambda d: -d["count"])
    return out


async def llm_extract(judge: JudgeBackend, comments: list[Comment], min_count: int) -> list[dict]:
    body = "\n".join(f"- {c.text}" for c in comments if not c.filtered_reason)[:20000]
    raw = await judge.judge(EXTRACT_SYSTEM, f"Comments:\n{body}\n\nJSON:", max_tokens=1500)
    m = _JSON.search(raw or "")
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    items = (
        data if isinstance(data, list) else data.get("themes", []) if isinstance(data, dict) else []
    )
    valid_kinds = {k.value for k in SignalKind}
    out: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        kind = str(it.get("kind", "comment_theme"))
        if kind not in valid_kinds:
            kind = "comment_theme"
        count = int(it.get("count", 1) or 1)
        if count < min_count:
            continue
        out.append(
            {
                "kind": kind,
                "sentiment": str(it.get("sentiment", "neu"))
                if it.get("sentiment") in ("neg", "neu", "pos")
                else "neu",
                "text": str(it.get("text", ""))[:300],
                "count": count,
                "excerpts": [str(e)[:200] for e in (it.get("excerpts") or [])][:3],
                "refs": [],
            }
        )
    return out


def _evidence(theme: dict) -> list[dict]:
    refs = list(theme.get("refs") or []) + [""] * 3
    return [
        {"source": "meta_comment", "ref": ref or "comments", "excerpt": e}
        for e, ref in zip(theme["excerpts"], refs, strict=False)
    ]


async def extract_for_trajectory(
    session: AsyncSession,
    trajectory_id: UUID,
    *,
    actor: str = "system",
    judge: JudgeBackend | None = None,
    min_count: int = 2,
) -> list[Signal]:
    render_ids = [
        r
        for (r,) in (
            await session.execute(select(Render.id).where(Render.trajectory_id == trajectory_id))
        ).all()
    ]
    if not render_ids:
        return []
    comments = (
        (
            await session.execute(
                select(Comment)
                .where(Comment.render_id.in_(render_ids))
                .order_by(Comment.created_time)
            )
        )
        .scalars()
        .all()
    )
    if not comments:
        return []
    themes = (
        await llm_extract(judge, list(comments), min_count)
        if judge
        else keyword_extract(list(comments), min_count)
    )
    existing = {
        (s.kind, s.text)
        for s in (
            await session.execute(select(Signal).where(Signal.trajectory_id == trajectory_id))
        )
        .scalars()
        .all()
    }
    created: list[Signal] = []
    for t in themes:
        if (t["kind"], t["text"]) in existing:
            continue
        s = Signal(
            trajectory_id=trajectory_id,
            render_id=None,
            kind=t["kind"],
            text=t["text"],
            evidence=_evidence(t),
            sentiment=t["sentiment"],
            count=t["count"],
            extracted_by=AuthorKind.model.value,
            status=SignalStatus.proposed.value,
        )
        session.add(s)
        created.append(s)
    await session.flush()
    return created
