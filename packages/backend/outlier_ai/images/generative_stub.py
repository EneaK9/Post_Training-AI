"""Placeholder for a generative image backend. Produces a labeled solid image so the
pipeline runs end to end; swap for a real model behind the same interface."""

from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageFont

from outlier_ai.core.storage import Storage
from outlier_ai.images.base import ImageSpec, RenderedImage


class GenerativeImageStub:
    name = "generate_stub"

    def render(self, spec: ImageSpec, storage: Storage, key_prefix: str) -> RenderedImage:
        w, h = spec.size
        img = Image.new("RGB", (w, h), (40 + spec.seed * 20 % 100, 60, 90))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default(size=40)
        except TypeError:
            font = ImageFont.load_default()
        draw.text((60, 60), "GENERATIVE STUB", font=font, fill=(255, 255, 255))
        draw.text((60, 130), spec.headline[:40], font=font, fill=(255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        uri = storage.put(f"{key_prefix}/{spec.seed}_{w}x{h}.png", buf.getvalue(), "image/png")
        return RenderedImage(uri=uri, width=w, height=h, backend=self.name, seed=spec.seed)
