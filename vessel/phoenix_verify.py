"""Tiny client for the Phoenix Cloud REST API used to verify that traces
actually land in Phoenix after a deploy.

Phoenix exposes `GET /v1/projects/{project}/spans` (see
https://arize.com/docs/phoenix/sdk-api-reference/rest-api). We use it to
poll for a recently-emitted span by name and assert that the OTel
ingest path is healthy end-to-end.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from urllib.parse import urlencode

import httpx


def _base_url() -> str:
    ep = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    if not ep:
        raise RuntimeError(
            "PHOENIX_COLLECTOR_ENDPOINT is not set — cannot query Phoenix"
        )
    return ep.rstrip("/")


def _api_key() -> str:
    key = os.getenv("PHOENIX_API_KEY")
    if not key:
        raise RuntimeError("PHOENIX_API_KEY is not set — cannot query Phoenix")
    return key


def list_spans(
    project: str,
    *,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 200,
    client: Optional[httpx.Client] = None,
) -> list[dict[str, Any]]:
    """Return spans for `project` between `start_time` and `end_time`.

    Times default to "the last 5 minutes" — this endpoint requires a time
    window to be useful, and tests almost always want recent activity.
    """
    if start_time is None:
        start_time = datetime.now(timezone.utc) - timedelta(minutes=5)
    params: dict[str, Any] = {"limit": limit, "start_time": start_time.isoformat()}
    if end_time is not None:
        params["end_time"] = end_time.isoformat()

    url = f"{_base_url()}/v1/projects/{project}/spans?{urlencode(params)}"
    headers = {"Authorization": f"Bearer {_api_key()}"}
    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=15.0)
    try:
        r = client.get(url, headers=headers)
        r.raise_for_status()
        return r.json().get("data") or []
    finally:
        if own_client:
            client.close()


def wait_for_span(
    project: str,
    span_name: str,
    *,
    since: Optional[datetime] = None,
    timeout_s: float = 60.0,
    poll_interval_s: float = 3.0,
) -> dict[str, Any]:
    """Poll Phoenix until a span with `span_name` appears, then return it.

    Raises TimeoutError if the span never lands. Used by the integration
    test that runs after deploy to assert the ingest path is alive.
    """
    if since is None:
        since = datetime.now(timezone.utc) - timedelta(minutes=2)
    deadline = time.time() + timeout_s
    last_count = 0
    while time.time() < deadline:
        spans = list_spans(project, start_time=since)
        last_count = len(spans)
        for s in spans:
            if s.get("name") == span_name:
                return s
        time.sleep(poll_interval_s)
    raise TimeoutError(
        f"Span {span_name!r} did not appear in Phoenix project "
        f"{project!r} within {timeout_s:.0f}s (saw {last_count} other spans)"
    )
