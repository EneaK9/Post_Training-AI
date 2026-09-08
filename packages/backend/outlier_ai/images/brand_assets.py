"""Pillow compositor: brand asset (or a seeded solid background) + headline overlay.

Three layout variants keyed by seed so the three renders per idea differ: headline at the
top, centered, or at the bottom band. Sizes 1080x1080 and 1080x1350.
"""

from __future__ import annotations

import hashlib
import io
import textwrap

from PIL import Image, ImageDraw, ImageFont

from outlier_ai.core.storage import Storage
from outlier_ai.images.base import ImageSpec, RenderedImage

PALETTE = [
    (24, 40, 72),
    (72, 36, 24),
    (18, 66, 54),
    (60, 30, 78),
    (90, 74, 20),
    (30, 30, 30),
]


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def _background(spec: ImageSpec) -> Image.Image:
    w, h = spec.size
    if spec.brand_asset:
        try:
            img = Image.open(io.BytesIO(spec.brand_asset)).convert("RGB")
            scale = max(w / img.width, h / img.height)
            img = img.resize((int(img.width * scale) + 1, int(img.height * scale) + 1))
            left = (img.width - w) // 2
            top = (img.height - h) // 2
            return img.crop((left, top, left + w, top + h))
        except Exception:
            pass
    idx = int(hashlib.blake2b(spec.brand_name.encode(), digest_size=2).hexdigest(), 16) % len(
        PALETTE
    )
    return Image.new("RGB", (w, h), PALETTE[(idx + spec.seed) % len(PALETTE)])


class BrandAssetsCompositor:
    name = "brand_assets"

    def render(self, spec: ImageSpec, storage: Storage, key_prefix: str) -> RenderedImage:
        w, h = spec.size
        img = _background(spec)
        draw = ImageDraw.Draw(img, "RGBA")
        layout = spec.seed % 3
        band_h = int(h * 0.34)
        top = {0: 0, 1: (h - band_h) // 2, 2: h - band_h}[layout]
        draw.rectangle([0, top, w, top + band_h], fill=(0, 0, 0, 150))

        headline = spec.headline.strip() or spec.brand_name
        size = 72 if len(headline) <= 24 else 56
        font = _font(size)
        lines = textwrap.wrap(headline, width=22 if size == 72 else 30)[:3]
        y = top + 40
        for line in lines:
            draw.text((60, y), line, font=font, fill=(255, 255, 255, 255))
            y += size + 12
        sub = textwrap.shorten(spec.primary_text.replace("\n", " "), width=90, placeholder="…")
        draw.text((60, y + 8), sub, font=_font(28), fill=(230, 230, 230, 255))
        draw.text((60, h - 70), spec.brand_name.upper(), font=_font(26), fill=(255, 255, 255, 220))

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        key = f"{key_prefix}/{spec.seed}_{w}x{h}.png"
        uri = storage.put(key, buf.getvalue(), "image/png")
        return RenderedImage(uri=uri, width=w, height=h, backend=self.name, seed=spec.seed)
