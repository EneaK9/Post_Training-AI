from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from outlier_ai.core.storage import Storage


@dataclass(frozen=True)
class RenderedImage:
    uri: str
    width: int
    height: int
    backend: str
    seed: int


@dataclass(frozen=True)
class ImageSpec:
    headline: str
    primary_text: str
    brand_name: str
    visual_brief: str
    size: tuple[int, int]
    seed: int
    brand_asset: bytes | None = None


class ImageBackend(Protocol):
    name: str

    def render(self, spec: ImageSpec, storage: Storage, key_prefix: str) -> RenderedImage: ...
