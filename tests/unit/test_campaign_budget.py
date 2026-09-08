from types import SimpleNamespace

from outlier_ai.meta.base import ad_object_name
from outlier_ai.meta.campaign import attribution_spec, cta_type
from outlier_ai.meta.client import map_facebook_error
from outlier_ai.meta.errors import (
    MetaApiVersionError,
    MetaAuthError,
    MetaError,
    MetaObjectNotFoundError,
    MetaRateLimitError,
    MetaValidationError,
)


def test_attribution_spec_parses_settings():
    assert attribution_spec("7d_click_1d_view") == [
        {"event_type": "CLICK_THROUGH", "window_days": 7},
        {"event_type": "VIEW_THROUGH", "window_days": 1},
    ]
    assert attribution_spec("1d_click") == [{"event_type": "CLICK_THROUGH", "window_days": 1}]
    assert attribution_spec("garbage") == [{"event_type": "CLICK_THROUGH", "window_days": 7}]


def test_cta_mapping_and_names():
    assert cta_type("Shop Now") == "SHOP_NOW"
    assert cta_type("something else") == "LEARN_MORE"
    assert ad_object_name("e", "t", "r") == "OAI|e|t|r"


def _err(code, message="", subcode=None):
    return SimpleNamespace(
        api_error_code=lambda: code,
        api_error_subcode=lambda: subcode,
        api_error_message=lambda: message,
    )


def test_facebook_error_mapping():
    assert isinstance(map_facebook_error(_err(190, "token expired")), MetaAuthError)
    assert isinstance(map_facebook_error(_err(17, "user request limit")), MetaRateLimitError)
    assert isinstance(map_facebook_error(_err(2635, "version deprecated")), MetaApiVersionError)
    assert isinstance(
        map_facebook_error(_err(803, "object does not exist")), MetaObjectNotFoundError
    )
    assert isinstance(map_facebook_error(_err(100, "invalid parameter")), MetaValidationError)
    assert type(map_facebook_error(_err(999, "weird"))) is MetaError
