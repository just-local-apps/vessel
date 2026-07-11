"""Token-derived per-user auth.

Vessel does not have a registration step. The `Authorization: Bearer <token>`
header (or `?token=<token>` query param) is hashed to derive `user_id`. The
first time a user shows up, the scheduler/MCP path lazily creates their state
row.

There is no central "the auth token" anymore — anyone with a 16+ character
token is a valid identity for that token. Lose the token, lose the data.
"""
from typing import Optional

from fastapi import Header, HTTPException, Query, status

from .users import derive_user_id, is_valid_token


def _extract_token(authorization: Optional[str], token_query: Optional[str]) -> Optional[str]:
    if authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer" and value:
            return value.strip()
    if token_query:
        return token_query.strip()
    return None


def require_user_id(
    authorization: Optional[str] = Header(default=None),
    token: Optional[str] = Query(default=None),
) -> str:
    presented = _extract_token(authorization, token)
    if not is_valid_token(presented):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid token (need >= 16 chars)",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return derive_user_id(presented)  # type: ignore[arg-type]
