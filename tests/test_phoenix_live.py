"""Live post-deploy verification for Phoenix Cloud tracing.

Drives the **deployed** Vessel MCP server with a real `apply_instruction`
call (which fans out to Claude through the auto-instrumented SDK) and
then queries the Phoenix REST API to confirm the resulting LLM span
actually landed.

This is the canary you run after every deploy: if it passes, the full
ingest path is alive end-to-end:

    MCP client → Vessel HTTP → chat tool-loop → Groq (instrumented)
                 → OTLP exporter → Phoenix Cloud → REST API

Required env vars:
    VESSEL_FLY_URL=https://<host>
    VESSEL_FLY_TOKEN=<auth token>
    PHOENIX_API_KEY=<phoenix cloud key>
    PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com/s/<workspace>
    PHOENIX_PROJECT=vessel  (optional; defaults to "vessel")

Run:
    uv run pytest -q tests/test_phoenix_live.py -s

Without the env vars the test is skipped, so the default suite stays
hermetic.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pytest
from mcp import ClientSession
from mcp.client.sse import sse_client

from vessel.phoenix_verify import list_spans


URL = os.getenv("VESSEL_FLY_URL", "").rstrip("/")
TOKEN = os.getenv("VESSEL_FLY_TOKEN", "")
PHX_KEY = os.getenv("PHOENIX_API_KEY", "")
PHX_ENDPOINT = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "")
PHX_PROJECT = os.getenv("PHOENIX_PROJECT", "vessel")


pytestmark = pytest.mark.skipif(
    not (URL and TOKEN and PHX_KEY and PHX_ENDPOINT),
    reason=(
        "set VESSEL_FLY_URL, VESSEL_FLY_TOKEN, PHOENIX_API_KEY, "
        "PHOENIX_COLLECTOR_ENDPOINT to run the post-deploy Phoenix check"
    ),
)


def _sse_url() -> str:
    return f"{URL}/mcp/sse?token={TOKEN}"


def _text(result: Any) -> str:
    return "\n".join(b.text for b in result.content if hasattr(b, "text"))


async def _cleanup_marker(marker: str) -> None:
    """Cleanup of the state mutation made by a probe run. Logs the
    outcome so we can tell whether a leftover probe is from this run
    or pre-existing — but never raises, so it can't mask a real
    assertion failure in the calling test.

    We extract the unique hex suffix from the marker because the intake
    agent invents slug ids with underscores ('task_phoenix_probe_<hex>')
    while the marker carries hyphens ('phoenix-probe-<hex>'). Matching
    on the bare hex matches both forms — without that fix every probe
    leaked a task into live state on every deploy."""
    # Marker shape: "phoenix-probe-<10 hex chars>". Pull the suffix.
    hex_suffix = marker.rsplit("-", 1)[-1]
    instruction = (
        f"Cleanup task: delete every project, task, and calendar entry "
        f"whose id, title, or notes contains either the substring "
        f"{marker!r} OR the substring {hex_suffix!r}. ALSO delete any "
        "task whose id starts with 'task_phoenix_probe_' AND any "
        "project named 'Deploy Probes' or 'Phoenix Probe' if no other "
        "tasks reference it. Apply immediately. Do not ask clarifying "
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
async def test_health_reports_phoenix_configured():
    """Before exercising MCP, confirm the deployed app booted with the
    Phoenix tracer wired up. If `tracer_provider_set` is False, no MCP
    activity will ever produce spans — fail fast with a useful message."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(f"{URL}/health")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    tracing = body.get("tracing") or {}
    assert tracing.get("configured") is True, (
        f"Deployed app missing Phoenix env vars: {tracing}"
    )
    assert tracing.get("tracer_provider_set") is True, tracing
    assert tracing.get("anthropic_instrumented") is True, tracing


@pytest.mark.asyncio
async def test_apply_instruction_emits_span_to_phoenix():
    """Drive the deployed MCP server with a real `apply_instruction` call
    that includes a unique marker, then poll the Phoenix REST API until a
    span carrying that marker shows up."""
    marker = f"phoenix-probe-{uuid.uuid4().hex[:10]}"
    instruction = (
        f"Note: this instruction is a deploy probe (marker={marker}). "
        f"Add a low-importance task called {marker!r} due tomorrow under "
        "any existing project, or create a 'Deploy Probes' project."
    )
    since = datetime.now(timezone.utc) - timedelta(seconds=30)

    try:
        async with sse_client(_sse_url()) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "apply_instruction", {"text": instruction}
                )
        assert not result.isError, f"MCP call failed: {_text(result)}"
        # Sanity check: response is a JSON envelope, regardless of applied vs
        # clarifications path. Either path runs Claude → emits an LLM span.
        json.loads(_text(result))

        # Poll Phoenix for any span recorded after `since` whose attributes
        # contain our marker. The Anthropic instrumentor stores the prompt
        # under `input.value` (or nested under `llm.input_messages.*`), so we
        # search broadly on the serialized attribute payload.
        # Claude Opus + Phoenix ingest can together take 2–3 minutes per
        # call; budget accordingly. The bottleneck is the deployed app's
        # `apply_instruction` returning, then the OTLP batch flushing,
        # then Phoenix making the span queryable via REST.
        deadline = time.time() + 240.0
        matched: dict | None = None
        last_count = 0
        while time.time() < deadline:
            spans = list_spans(PHX_PROJECT, start_time=since, limit=500)
            last_count = len(spans)
            for s in spans:
                blob = json.dumps(s.get("attributes") or {})
                if marker in blob:
                    matched = s
                    break
            if matched:
                break
            time.sleep(4)

        assert matched is not None, (
            f"No Phoenix span carrying marker {marker!r} appeared in "
            f"project {PHX_PROJECT!r} within 240s "
            f"(saw {last_count} other spans during the window). "
            "Either the deploy is not exporting traces, the project name in "
            "PHOENIX_PROJECT differs from what the deploy registered, or "
            "Phoenix ingest is lagging."
        )
        # The matched span must carry an LLM span kind — that's what proves
        # the Anthropic auto-instrumentor is the producer (vs. some unrelated
        # span we accidentally tagged). Phoenix surfaces this as `span_kind`
        # at the top of the span dict; older clients put it under
        # `attributes["openinference.span.kind"]`.
        attrs = matched.get("attributes") or {}
        kind = (
            matched.get("span_kind")
            or attrs.get("openinference.span.kind")
            or attrs.get("openinference", {}).get("span", {}).get("kind")
        )
        assert str(kind).upper() == "LLM", (
            f"Matched span is not an LLM span — got kind={kind!r}. "
            "Auto-instrumentation may have regressed."
        )
        # Sanity-check the auto-instrumentor's signature attributes.
        # Either provider (anthropic / openai+groq) is acceptable — we
        # just need a populated model name to prove instrumentation is
        # firing.
        assert attrs.get("llm.model_name"), "missing llm.model_name on LLM span"
        provider = (attrs.get("llm.provider") or "").lower()
        system = (attrs.get("llm.system") or "").lower()
        assert provider in {"anthropic", "openai"} or system in {"anthropic", "openai"}, (
            f"unexpected llm.provider={provider!r} system={system!r}"
        )
    finally:
        await _cleanup_marker(marker)
