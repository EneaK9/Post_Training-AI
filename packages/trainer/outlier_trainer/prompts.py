"""Chat formatting for the open model. The prompt is one user turn; the completion is the
assistant turn holding the idea blocks."""

from __future__ import annotations

from typing import Any


def messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def render_chat(
    tokenizer: Any,
    prompt: str,
    completion: str | None = None,
    *,
    add_generation_prompt: bool = True,
) -> str:
    """Apply the tokenizer's chat template. Falls back to a plain format for base weights."""
    msgs = messages(prompt)
    if completion is not None:
        msgs.append({"role": "assistant", "content": completion})
    template = getattr(tokenizer, "chat_template", None)
    if template:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=add_generation_prompt and completion is None
        )
    text = f"### Instruction\n{prompt}\n\n### Response\n"
    return text + (completion or "")
