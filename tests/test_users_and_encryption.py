"""Multi-user identity + at-rest encryption tests."""
import os
from unittest.mock import patch

import pytest
from cryptography.fernet import Fernet

from vessel.users import derive_user_id, is_valid_token


def test_user_id_is_deterministic():
    a = derive_user_id("hello-this-is-a-token-of-sufficient-length")
    b = derive_user_id("hello-this-is-a-token-of-sufficient-length")
    assert a == b
    assert len(a) == 64  # sha256 hex


def test_user_id_changes_with_token():
    a = derive_user_id("token-one-of-sufficient-length")
    b = derive_user_id("token-two-of-sufficient-length")
    assert a != b


def test_token_min_length():
    assert not is_valid_token(None)
    assert not is_valid_token("")
    assert not is_valid_token("short")
    assert is_valid_token("x" * 16)
    assert is_valid_token("x" * 100)


def test_encryption_roundtrip(monkeypatch):
    key = Fernet.generate_key().decode()
    with patch.dict(
        os.environ,
        {
            "DATABASE_URL": "postgresql://fake/fake",
            "VESSEL_ENCRYPTION_KEY": key,
        },
        clear=False,
    ):
        import vessel.config as cfg
        import vessel.encryption as enc

        cfg._settings = None
        enc.reset()

        cipher = enc.encrypt_text("hello world")
        assert isinstance(cipher, bytes)
        assert b"hello world" not in cipher  # actually encrypted
        assert enc.decrypt_text(cipher) == "hello world"

        payload = {"projects": [{"id": "p1", "name": "Home"}], "n": 7}
        cblob = enc.encrypt_json(payload)
        assert enc.decrypt_json(cblob) == payload


def test_encryption_requires_key():
    """Without VESSEL_ENCRYPTION_KEY, Settings() must refuse to construct.
    We disable env_file loading so the project's .env doesn't sneak the
    key in via pydantic-settings's file fallback."""
    with patch.dict(
        os.environ,
        {"DATABASE_URL": "postgresql://fake/fake"},
        clear=True,
    ):
        import vessel.config as cfg
        import vessel.encryption as enc

        cfg._settings = None
        enc.reset()
        with patch.object(cfg.Settings, "model_config", {"extra": "ignore"}):
            with pytest.raises(Exception):
                cfg.get_settings()  # missing VESSEL_ENCRYPTION_KEY → validation error
