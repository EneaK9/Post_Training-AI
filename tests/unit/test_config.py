import pytest
from pydantic import ValidationError

from outlier_schemas.config import AppConfig, TierMultiples
from outlier_schemas.enums import NicheProjection


def test_file_config_loads_with_category_medians(app_config: AppConfig):
    assert app_config.outlier.tier_multiples.tier2 == 3.0
    assert "skincare" in app_config.category_medians
    assert app_config.category_medians["default"].roas > 0
    assert app_config.archive.niche_projection == NicheProjection.strategy_only


def test_hash_is_stable_and_sensitive(app_config: AppConfig):
    again = AppConfig.from_dict(app_config.model_dump(mode="json"))
    assert again.hash == app_config.hash
    changed = app_config.model_copy(deep=True)
    changed.outlier.tier_multiples.tier2 = 3.5
    assert changed.hash != app_config.hash
    assert changed.diff(app_config) == {"outlier.tier_multiples.tier2": (3.5, 3.0)}


def test_tier_multiples_must_increase():
    with pytest.raises(ValidationError):
        TierMultiples(tier1=3.0, tier2=2.0, tier3=10.0)


def test_unknown_keys_rejected():
    with pytest.raises(ValidationError):
        AppConfig.from_dict({"outlier": {"nope": 1}})


def test_defaults_match_spec_section_4():
    cfg = AppConfig()
    assert cfg.outlier.min_purchases_at_scale == 30
    assert cfg.outlier.durability_days == 7
    assert cfg.outlier.baseline_min_ads == 20
    assert cfg.outlier.screening.min_impressions == 5000
    assert cfg.outlier.screening.ctr_multiple == 1.5
    assert cfg.outlier.tier_rewards.tier1 == 0.0
    assert cfg.outlier.tier_rewards.tier2 == 1.0
