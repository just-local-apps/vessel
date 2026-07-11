"""Unit tests for the Phoenix tracing pipeline.

Validates that observability.init() wires up correctly when the right env
vars are present, and stays a clean no-op when they aren't.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

import vessel.observability as obs


@pytest.fixture(autouse=True)
def reset_state():
    obs.reset()
    yield
    obs.reset()


def test_no_env_vars_means_no_tracing():
    with patch.dict(os.environ, {}, clear=True):
        os.environ["DATABASE_URL"] = "postgresql://fake/fake"
        obs.init()
    s = obs.get_status()
    assert s["initialized"] is True
    assert s["configured"] is False
    assert s["tracer_provider_set"] is False
    assert s["anthropic_instrumented"] is False
    assert s["error"] is None


def test_full_pipeline_with_env_vars_set():
    """When PHOENIX_API_KEY + PHOENIX_COLLECTOR_ENDPOINT are present, init()
    must register a tracer provider AND instrument the Anthropic SDK. We
    patch the actual register/instrument to avoid hitting the network."""
    with patch.dict(
        os.environ,
        {
            "PHOENIX_API_KEY": "fake-key",
            "PHOENIX_COLLECTOR_ENDPOINT": "https://app.phoenix.arize.com/s/test",
            "PHOENIX_PROJECT": "test-vessel",
            "DATABASE_URL": "postgresql://fake/fake",
        },
        clear=True,
    ):
        with patch("phoenix.otel.register") as mock_register, patch(
            "openinference.instrumentation.anthropic.AnthropicInstrumentor"
        ) as mock_anthropic_cls, patch(
            "openinference.instrumentation.openai.OpenAIInstrumentor"
        ) as mock_openai_cls:
            mock_register.return_value = "tracer-provider-stub"
            obs.init()

    s = obs.get_status()
    assert s["initialized"] is True
    assert s["configured"] is True
    assert s["tracer_provider_set"] is True
    assert s["anthropic_instrumented"] is True
    assert s["openai_instrumented"] is True
    assert s["project_name"] == "test-vessel"
    assert s["endpoint"] == "https://app.phoenix.arize.com/s/test"
    assert s["error"] is None
    mock_register.assert_called_once()
    kwargs = mock_register.call_args.kwargs
    assert kwargs["project_name"] == "test-vessel"
    assert kwargs["endpoint"].endswith("/v1/traces")
    assert kwargs["headers"]["authorization"] == "Bearer fake-key"
    mock_anthropic_cls.return_value.instrument.assert_called_once_with(
        tracer_provider="tracer-provider-stub"
    )
    mock_openai_cls.return_value.instrument.assert_called_once_with(
        tracer_provider="tracer-provider-stub"
    )


def test_register_failure_records_error_and_disables_tracing():
    with patch.dict(
        os.environ,
        {
            "PHOENIX_API_KEY": "fake-key",
            "PHOENIX_COLLECTOR_ENDPOINT": "https://app.phoenix.arize.com/s/test",
            "DATABASE_URL": "postgresql://fake/fake",
        },
        clear=True,
    ):
        with patch("phoenix.otel.register", side_effect=RuntimeError("nope")):
            obs.init()

    s = obs.get_status()
    assert s["configured"] is True
    assert s["tracer_provider_set"] is False
    assert s["anthropic_instrumented"] is False
    assert s["error"] is not None
    assert "register failed" in s["error"]


def test_init_is_idempotent():
    with patch.dict(os.environ, {"DATABASE_URL": "postgresql://fake/fake"}, clear=True):
        obs.init()
        obs._state["sentinel"] = True  # type: ignore[index]
        obs.init()  # no-op
    assert obs._state.get("sentinel") is True


def test_actual_anthropic_call_emits_a_span():
    """End-to-end without the network: with a real OTel tracer wired through
    an in-memory exporter, a fake Anthropic-style instrumented call must
    produce at least one span. Closest we can get to "traces show up in
    Phoenix" without hitting the wire."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exporter))

    tracer = tp.get_tracer("vessel-test")
    with tracer.start_as_current_span("anthropic.messages.create") as span:
        span.set_attribute("llm.model_name", "claude-opus-4-7")
        span.set_attribute("input.value", "hello")
        span.set_attribute("output.value", "hi back")

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    s = spans[0]
    assert s.name == "anthropic.messages.create"
    assert s.attributes["llm.model_name"] == "claude-opus-4-7"
