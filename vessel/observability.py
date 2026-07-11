"""Phoenix Cloud / OpenInference tracing setup.

Auto-instruments the `anthropic` SDK so every Claude call shows up in
Phoenix Cloud (app.phoenix.arize.com) as an LLM span with the prompt,
completion, model, and latency.

If `PHOENIX_API_KEY` and `PHOENIX_COLLECTOR_ENDPOINT` aren't set, this
module is a no-op — Vessel runs without tracing.

`get_status()` returns a JSON-serializable dict the /health endpoint and
tests can use to assert the pipeline is wired correctly end-to-end.
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


_state: dict[str, Any] = {
    "initialized": False,
    "configured": False,         # API key + endpoint present
    "tracer_provider_set": False,
    "anthropic_instrumented": False,
    "openai_instrumented": False,   # also covers Groq (OpenAI-compatible)
    "project_name": None,
    "endpoint": None,
    "error": None,
}


def get_status() -> dict[str, Any]:
    """Snapshot of the tracing pipeline's state. Safe to call any time."""
    return dict(_state)


def init() -> None:
    if _state["initialized"]:
        return
    _state["initialized"] = True

    api_key = os.getenv("PHOENIX_API_KEY")
    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    project_name = os.getenv("PHOENIX_PROJECT", "vessel")
    _state["project_name"] = project_name
    _state["endpoint"] = endpoint

    if not api_key or not endpoint:
        logger.info(
            "Phoenix tracing disabled (set PHOENIX_API_KEY and "
            "PHOENIX_COLLECTOR_ENDPOINT to enable)"
        )
        return
    _state["configured"] = True

    # phoenix.otel reads PHOENIX_API_KEY/PHOENIX_COLLECTOR_ENDPOINT from
    # the environment automatically, but we pass them explicitly so the
    # call site is self-documenting and unit tests can monkeypatch the env.
    try:
        from phoenix.otel import register
    except ImportError as exc:
        msg = f"arize-phoenix-otel not installed — tracing disabled: {exc}"
        logger.warning(msg)
        _state["error"] = msg
        return

    try:
        tracer_provider = register(
            project_name=project_name,
            endpoint=f"{endpoint.rstrip('/')}/v1/traces",
            headers={"authorization": f"Bearer {api_key}"},
            batch=True,
        )
        _state["tracer_provider_set"] = True
    except Exception as exc:  # noqa: BLE001
        logger.exception("Phoenix register() failed — tracing disabled")
        _state["error"] = f"register failed: {exc}"
        return

    try:
        from openinference.instrumentation.anthropic import AnthropicInstrumentor

        AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)
        _state["anthropic_instrumented"] = True
    except ImportError as exc:
        msg = (
            "openinference-instrumentation-anthropic not installed — "
            f"Anthropic auto-tracing disabled: {exc}"
        )
        logger.warning(msg)
        _state["error"] = msg
    except Exception as exc:  # noqa: BLE001
        logger.exception("Anthropic OpenInference instrument() failed")
        _state["error"] = f"anthropic instrument failed: {exc}"

    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor

        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
        _state["openai_instrumented"] = True
    except ImportError as exc:
        msg = (
            "openinference-instrumentation-openai not installed — "
            f"OpenAI/Groq auto-tracing disabled: {exc}"
        )
        logger.warning(msg)
        _state["error"] = msg
    except Exception as exc:  # noqa: BLE001
        logger.exception("OpenAI OpenInference instrument() failed")
        _state["error"] = f"openai instrument failed: {exc}"

    if _state["anthropic_instrumented"] or _state["openai_instrumented"]:
        logger.info(
            "Phoenix OpenInference tracing active for project %s @ %s "
            "(anthropic=%s, openai/groq=%s)",
            project_name,
            endpoint,
            _state["anthropic_instrumented"],
            _state["openai_instrumented"],
        )


def reset() -> None:
    """Test-only: clear cached state so init() can re-run."""
    for key in list(_state.keys()):
        if key in ("project_name", "endpoint", "error"):
            _state[key] = None
        else:
            _state[key] = False
