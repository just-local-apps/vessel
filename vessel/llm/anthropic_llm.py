"""Anthropic (Claude) LLM adapter.

Uses the official `anthropic` SDK so that the OpenInference Anthropic
instrumentor can auto-trace every call into Arize.

The per-agent system prompt is wrapped with `cache_control: ephemeral` so
that once a prompt exceeds the model's minimum cacheable prefix, repeated
agent invocations skip the prefill cost. Below that threshold the marker is
silently a no-op — no error, no harm, just nothing to cache yet.
"""
from __future__ import annotations

import anthropic

from .adapter import LLMAdapter

# Conservative non-streaming ceiling — keeps responses under SDK HTTP timeouts.
DEFAULT_MAX_TOKENS = 16000


class AnthropicAdapter(LLMAdapter):
    def __init__(self, model: str, api_key: str, max_tokens: int = DEFAULT_MAX_TOKENS):
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": user_prompt}],
        )
        parts: list[str] = []
        for block in resp.content:
            if block.type == "text":
                parts.append(block.text)
        return "\n".join(parts)
