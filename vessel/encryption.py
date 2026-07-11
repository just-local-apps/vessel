"""Server-side master-key encryption.

Vessel encrypts every user-scoped artifact at rest with a Fernet key:

- `vessel.state.encrypted_data`
- `vessel.events.encrypted_payload`
- `vessel.invocations.encrypted_state_before` / `encrypted_state_after` /
  `encrypted_rendered_prompt` / `encrypted_raw_response`

The key lives in `VESSEL_ENCRYPTION_KEY` (44-char base64 from
`Fernet.generate_key()`). The scheduler loop runs autonomously, so the
encryption key has to be available without the user's token — this is
not zero-knowledge encryption, but it does protect the database at rest.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from cryptography.fernet import Fernet

from .config import get_settings

_fernet: Optional[Fernet] = None


def _get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        key = get_settings().vessel_encryption_key
        if not key:
            raise RuntimeError(
                "VESSEL_ENCRYPTION_KEY not configured. "
                "Generate one with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
            )
        _fernet = Fernet(key.encode("utf-8") if isinstance(key, str) else key)
    return _fernet


def reset() -> None:
    """Test helper — re-reads the key on next call."""
    global _fernet
    _fernet = None


def encrypt_text(text: str) -> bytes:
    return _get_fernet().encrypt(text.encode("utf-8"))


def decrypt_text(blob: bytes) -> str:
    return _get_fernet().decrypt(blob).decode("utf-8")


def encrypt_json(value: Any) -> bytes:
    return encrypt_text(json.dumps(value, separators=(",", ":")))


def decrypt_json(blob: bytes) -> Any:
    return json.loads(decrypt_text(blob))
