"""Live MCP integration test against the deployed Fly instance.

Runs only when both env vars are set:

    VESSEL_FLY_URL=https://vessel-ravi.fly.dev \
    VESSEL_FLY_TOKEN=<your token> \
    uv run pytest -q tests/test_fly_live.py -s

Without them the test is skipped, so the default test suite stays hermetic.

These tests hit a real Anthropic Claude call via Vessel's deployed MCP
server. They cost a few cents per run.
"""
from __future__ import annotations

import json
import os
from typing import Any

import httpx
import pytest
from mcp import ClientSession
from mcp.client.sse import sse_client

URL = os.getenv("VESSEL_FLY_URL", "").rstrip("/")
TOKEN = os.getenv("VESSEL_FLY_TOKEN", "")

pytestmark = pytest.mark.skipif(
    not (URL and TOKEN),
    reason="set VESSEL_FLY_URL and VESSEL_FLY_TOKEN to run live tests",
)


def _sse_url() -> str:
    return f"{URL}/mcp/sse?token={TOKEN}"


def _text(result: Any) -> str:
    parts = []
    for block in result.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts)


async def _cleanup_marker(marker: str) -> None:
    """Delete every project / task / calendar entry created during a
    probe run that carries `marker` in its id, title, or notes. Logs
    the outcome and never raises — failures here can't be allowed to
    mask a real assertion failure in the calling test. Also retires
    the "Deploy Probes" project (used by the Phoenix probe test) if
    it ends up empty.

    The marker arrives as `<prefix>-<hex>` but the intake agent invents
    slug ids using underscores (e.g. `task_gym_<hex>` from a
    `deploy-probe-<hex>` marker), so a literal substring match misses
    the very thing we just created. We feed BOTH forms into the
    instruction so the LLM matches either."""
    hex_suffix = marker.rsplit("-", 1)[-1]
    instruction = (
        f"Cleanup task: delete every project, task, and calendar entry "
        f"whose id, title, or notes contains either the substring "
        f"{marker!r} OR the substring {hex_suffix!r}. ALSO delete any "
        "project named 'Deploy Probes' or 'Phoenix Probe' if it has no "
        "remaining tasks. Apply immediately. Do not ask clarifying "
        "questions. If nothing matches, return state unchanged."
    )
    try:
        async with sse_client(_sse_url()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                res = await session.call_tool(
                    "apply_instruction", {"text": instruction}
                )
        txt = _text(res)
        try:
            data = json.loads(txt)
        except Exception:  # noqa: BLE001
            print(f"[cleanup {marker}] non-JSON response: {txt[:200]}")
            return
        diff = data.get("diff") or {}
        removed = sum(
            len((diff.get(b) or {}).get("removed") or [])
            for b in ("projects", "tasks", "calendar")
        )
        print(f"[cleanup {marker}] applied={data.get('applied')} removed={removed}")
    except Exception as exc:  # noqa: BLE001 — cleanup is best-effort
        print(f"[cleanup {marker}] FAILED: {exc!r}")


@pytest.mark.asyncio
async def test_health_endpoint_is_up():
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{URL}/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("ok") is True
    # /health now also surfaces tracing pipeline status.
    assert "tracing" in body


@pytest.mark.asyncio
async def test_unauthenticated_mcp_is_rejected():
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{URL}/mcp/sse?token=wrong", timeout=5.0)
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_get_state_round_trip():
    async with sse_client(_sse_url()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            tool_names = {t.name for t in tools.tools}
            assert {"get_state", "apply_instruction"} <= tool_names

            result = await session.call_tool("get_state", {})
    text = _text(result)
    data = json.loads(text)
    for key in ("projects", "tasks", "calendar", "priority_ranking"):
        assert key in data, f"missing key {key} in state"


@pytest.mark.asyncio
async def test_apply_instruction_adds_calendar_entry():
    """Issue an `apply_instruction` and confirm the deploy mutated state.

    We use a unique marker so the LLM is forced to add something new
    rather than declaring an existing entry "already there". The test
    accepts mutation in either `tasks` or `calendar` because the agent
    legitimately decides between the two based on whether a precise time
    was specified."""
    import uuid as _uuid

    marker = f"deploy-probe-{_uuid.uuid4().hex[:8]}"
    instruction = (
        f"Add a calendar entry for tomorrow at 8am for one hour titled "
        f"'Go to the gym ({marker})'. If no Health project exists, also add one."
    )
    try:
        async with sse_client(_sse_url()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "apply_instruction", {"text": instruction}
                )
        assert not result.isError, f"intake call failed: {_text(result)}"
        data = json.loads(_text(result))

        # apply_instruction returns either {applied, diff} or {applied, clarifications}.
        if not data.get("applied", False):
            assert "clarifications" in data and data["clarifications"]
            return

        diff = data["diff"]
        # Look for the marker anywhere in calendar OR tasks — the agent
        # legitimately picks the better bucket, and we don't want to fail
        # the deploy gate on that judgment call.
        candidates: list[dict] = []
        for bucket in ("calendar", "tasks"):
            section = diff.get(bucket) or {}
            candidates.extend(section.get("added") or [])
            candidates.extend(c.get("after") for c in (section.get("changed") or []))
        candidates = [c for c in candidates if c]

        blob = json.dumps(candidates).lower()
        assert marker in blob, (
            f"no entry carrying marker {marker!r} in diff: "
            f"{json.dumps(candidates)[:400]}"
        )
    finally:
        await _cleanup_marker(marker)
