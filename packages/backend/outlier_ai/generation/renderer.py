"""Prompt renderer: three columns in, one prompt and a trace out (spec section 5.3).

The trace records exactly what went into the prompt (card slugs, history refs, signal refs,
the ref -> id map, and what was trimmed to fit the token budget) so the Generate screen can
show "what the model saw" and the parser can resolve citations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import UUID

from jinja2 import Environment, PackageLoader, select_autoescape

from outlier_ai.generation.views import BatchView, BriefView, CardView, HistoryItemView
from outlier_schemas.config import AppConfig

_env = Environment(
    loader=PackageLoader("outlier_ai.generation", "templates"),
    autoescape=select_autoescape(default=False, default_for_string=False),
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=True,
)

KIND_ORDER = ("strategy", "style", "principle", "mechanic")
CHARS_PER_TOKEN = 4.0
COPY_CHARS_FULL = 400
COPY_CHARS_TRIMMED = 160


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


@dataclass
class PromptTrace:
    brief_id: UUID
    config_hash: str
    k: int
    card_slugs: list[str]
    archive_refs: list[dict[str, Any]]
    episode_refs: list[dict[str, Any]]
    signal_refs: list[str]
    ref_map: dict[str, str]
    token_estimate: int
    trimmed: list[str] = field(default_factory=list)
    template: str = "loop_a.j2"

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["brief_id"] = str(self.brief_id)
        return d


def render_prompt(
    *,
    brief: BriefView,
    playbook: list[CardView],
    archive_items: list[HistoryItemView],
    episode_batches: list[BatchView],
    niche_uses: dict[str, int],
    ref_map: dict[str, UUID],
    cfg: AppConfig,
    k: int | None = None,
) -> tuple[str, PromptTrace]:
    k = k or cfg.generation.k
    template = _env.get_template("loop_a.j2")
    playbook_by_kind: dict[str, list[CardView]] = {}
    for kind in KIND_ORDER:
        cards = sorted((c for c in playbook if c.kind == kind), key=lambda c: c.slug)
        if cards:
            playbook_by_kind[kind] = cards
    for c in playbook:
        if c.kind not in playbook_by_kind and c.kind not in KIND_ORDER:
            playbook_by_kind.setdefault(c.kind, []).append(c)

    archive = list(archive_items)
    batches = list(episode_batches)
    trimmed: list[str] = []
    copy_chars = COPY_CHARS_FULL

    def render(items: list[HistoryItemView], bs: list[BatchView], chars: int) -> str:
        return template.render(
            playbook_by_kind=playbook_by_kind,
            brief=brief,
            archive_items=items,
            episode_batches=bs,
            niche_uses=dict(sorted(niche_uses.items(), key=lambda kv: -kv[1])[:24]),
            k=k,
            min_cards=cfg.generation.min_cards_per_idea,
            tier1_multiple=cfg.outlier.tier_multiples.tier1,
            tier2_multiple=cfg.outlier.tier_multiples.tier2,
            tier3_multiple=cfg.outlier.tier_multiples.tier3,
            screening_multiple=cfg.outlier.screening.ctr_multiple,
            headline_chars=cfg.images.headline_warn_chars,
            description_chars=cfg.images.description_warn_chars,
            copy_chars=chars,
        )

    prompt = render(archive, batches, copy_chars)
    budget = cfg.generation.max_prompt_tokens

    # Trim order: rare picks, then elites from the bottom, then oldest episode batches,
    # then shorten copy. Tier 2+ recent items and the newest batch are kept as long as possible.
    while estimate_tokens(prompt) > budget:
        rares = [i for i in archive if i.reason == "rare"]
        elites = [i for i in archive if i.reason == "elite"]
        if rares:
            drop = rares[-1]
            archive.remove(drop)
            trimmed.append(f"history:{drop.ref} (rare)")
        elif len(elites) > 1:
            drop = elites[-1]
            archive.remove(drop)
            trimmed.append(f"history:{drop.ref} (elite)")
        elif len(batches) > 1:
            drop_b = batches.pop(0)
            trimmed.append(f"batch:{drop_b.index}")
        elif copy_chars > COPY_CHARS_TRIMMED:
            copy_chars = COPY_CHARS_TRIMMED
            trimmed.append("copy shortened")
        else:
            others = [i for i in archive if i.reason == "tier2_recent"]
            if others:
                drop = others[-1]
                archive.remove(drop)
                trimmed.append(f"history:{drop.ref} (tier2_recent)")
            else:
                break
        prompt = render(archive, batches, copy_chars)

    kept_refs = {i.ref for i in archive} | {i.ref for b in batches for i in b.items}
    kept_signal_refs = {s.ref for i in archive for s in i.signals} | {
        s.ref for b in batches for i in b.items for s in i.signals
    }
    trace = PromptTrace(
        brief_id=brief.id,
        config_hash=cfg.hash,
        k=k,
        card_slugs=[c.slug for cards in playbook_by_kind.values() for c in cards],
        archive_refs=[
            {"ref": i.ref, "reason": i.reason, "niche": i.niche, "tier": i.tier} for i in archive
        ],
        episode_refs=[
            {"batch": b.index, "ref": i.ref, "tier": i.tier, "screening_ratio": i.screening_ratio}
            for b in batches
            for i in b.items
        ],
        signal_refs=sorted(kept_signal_refs),
        ref_map={r: str(u) for r, u in ref_map.items() if r in kept_refs or r in kept_signal_refs},
        token_estimate=estimate_tokens(prompt),
        trimmed=trimmed,
    )
    return prompt, trace
