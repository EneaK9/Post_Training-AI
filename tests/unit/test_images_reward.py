import asyncio
import io
from dataclasses import replace
from pathlib import Path

from PIL import Image

from outlier_ai.core.storage import LocalStorage, key_from_uri
from outlier_ai.images import get_image_backend
from outlier_ai.images.base import ImageSpec
from outlier_ai.reward.base import IdeaFeatures
from outlier_ai.reward.cold import HeuristicColdRewardModel
from outlier_schemas.enums import ImageMode
from outlier_schemas.models import AdCopy


def _spec(seed: int, size=(1080, 1080), asset: bytes | None = None) -> ImageSpec:
    return ImageSpec(
        headline="Stop doing it the old way",
        primary_text="Lumen Labs makes a serum.",
        brand_name="Lumen Labs",
        visual_brief="serum",
        size=size,
        seed=seed,
        brand_asset=asset,
    )


def test_compositor_renders_png_of_right_size_with_layout_variants(tmp_path: Path):
    st = LocalStorage(tmp_path)
    backend = get_image_backend(ImageMode.brand_assets)
    outs = [backend.render(_spec(s), st, "renders/t") for s in range(3)]
    assert len({o.uri for o in outs}) == 3
    imgs = [Image.open(io.BytesIO(st.get(key_from_uri(o.uri)))) for o in outs]
    assert all(i.size == (1080, 1080) for i in imgs)
    assert len({im.tobytes() for im in imgs}) == 3, "seeds must produce different layouts"
    tall = backend.render(_spec(0, (1080, 1350)), st, "renders/t2")
    assert (tall.width, tall.height) == (1080, 1350)


def test_compositor_uses_brand_asset_and_survives_garbage(tmp_path: Path):
    st = LocalStorage(tmp_path)
    buf = io.BytesIO()
    Image.new("RGB", (400, 300), (200, 20, 20)).save(buf, format="PNG")
    backend = get_image_backend(ImageMode.brand_assets)
    with_asset = backend.render(_spec(1, asset=buf.getvalue()), st, "a")
    img = Image.open(io.BytesIO(st.get(key_from_uri(with_asset.uri))))
    px = img.getpixel((1000, 1000))
    assert isinstance(px, tuple) and px[0] > 150  # red asset shows through outside the band
    garbage = backend.render(_spec(1, asset=b"not an image"), st, "b")
    assert garbage.width == 1080
    stub = get_image_backend(ImageMode.generate).render(_spec(0), st, "c")
    assert stub.backend == "generate_stub"


BASE = IdeaFeatures(
    brief_text="b",
    world_state="",
    angle="a",
    copy=AdCopy(),
    visual_brief="v",
    verified_card_slugs=["a", "b"],
    typicality="common",
    novelty_distance=0.3,
    cited_signal_count=0,
    format_ok=True,
    tag_match=True,
)


def _features(**kw) -> IdeaFeatures:
    return replace(BASE, **kw)


def test_heuristic_cold_reward_orders_sensibly():
    rm = HeuristicColdRewardModel()
    rare = asyncio.run(
        rm.score(_features(typicality="rare", novelty_distance=0.6, cited_signal_count=2))
    )
    common = asyncio.run(rm.score(_features()))
    dup = asyncio.run(rm.score(_features(novelty_distance=0.01)))
    broken = asyncio.run(rm.score(_features(format_ok=False)))
    assert rare.mean > common.mean > dup.mean > broken.mean
    assert 0.0 <= broken.mean <= 1.0 and rare.mean <= 1.0
    assert rare.rm_score(1.0) == rare.mean - rare.std
