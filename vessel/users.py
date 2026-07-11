"""Per-user identification.

Vessel does not have a registration step. Any non-empty token is a valid
identity; `user_id = sha256(token).hex()` is deterministic. The first time
a user shows up with a new token, the scheduler/MCP path lazily creates
their state row.

Tokens are never stored — only their hash. If you lose your token, you lose
access to that user's data; nothing on the server can recover it.
"""
from __future__ import annotations

import hashlib
from typing import Optional


MIN_TOKEN_LENGTH = 16


def derive_user_id(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def is_valid_token(token: Optional[str]) -> bool:
    if not token:
        return False
    return len(token) >= MIN_TOKEN_LENGTH
