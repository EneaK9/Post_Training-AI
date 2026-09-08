"""Image rendering (spec section 5.7) [v3-assumed]: brand-assets compositor first."""

from __future__ import annotations

from outlier_ai.images.base import ImageBackend, RenderedImage
from outlier_ai.images.brand_assets import BrandAssetsCompositor
from outlier_ai.images.generative_stub import GenerativeImageStub
from outlier_schemas.enums import ImageMode


def get_image_backend(mode: ImageMode) -> ImageBackend:
    if mode == ImageMode.generate:
        return GenerativeImageStub()
    return BrandAssetsCompositor()


__all__ = [
    "BrandAssetsCompositor",
    "GenerativeImageStub",
    "ImageBackend",
    "RenderedImage",
    "get_image_backend",
]
