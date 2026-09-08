from pathlib import Path

import pytest

from outlier_ai.core.crypto import TokenCipher, hash_identity
from outlier_ai.core.settings import Settings
from outlier_ai.core.storage import LocalStorage, key_from_uri


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("false", False),
        ("FALSE", False),
        ("true", True),
        ("0", True),
        ("no", True),
        ("", True),
        ("f", True),
    ],
)
def test_dry_run_only_disabled_by_literal_false(
    raw: str, expected: bool, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("DRY_RUN", raw)
    assert Settings(_env_file=None).dry_run is expected  # type: ignore[call-arg]


def test_local_storage_round_trip(tmp_path: Path):
    st = LocalStorage(tmp_path)
    uri = st.put("renders/a/b.png", b"bytes", "image/png")
    assert uri == "storage://renders/a/b.png"
    assert st.exists(key_from_uri(uri))
    assert st.get("renders/a/b.png") == b"bytes"
    assert st.public_url("renders/a/b.png").startswith("file://")


def test_local_storage_rejects_escape(tmp_path: Path):
    st = LocalStorage(tmp_path)
    with pytest.raises(ValueError):
        st.put("../outside.txt", b"x")


def test_token_cipher_round_trip():
    from cryptography.fernet import Fernet

    cipher = TokenCipher(Fernet.generate_key().decode())
    assert cipher.decrypt(cipher.encrypt("EAAB.token")) == "EAAB.token"


def test_hash_identity_is_keyed_and_stable(monkeypatch: pytest.MonkeyPatch):
    s1 = Settings(_env_file=None, pii_hmac_key="k1")  # type: ignore[call-arg]
    s2 = Settings(_env_file=None, pii_hmac_key="k2")  # type: ignore[call-arg]
    assert hash_identity("user_1", s1) == hash_identity("user_1", s1)
    assert hash_identity("user_1", s1) != hash_identity("user_1", s2)
    assert len(hash_identity("user_1", s1)) == 64
