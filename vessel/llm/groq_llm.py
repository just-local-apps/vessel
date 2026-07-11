"""Groq LLM adapter (OpenAI-compatible API).

Groq exposes Llama / Mixtral / OpenAI OSS / Qwen models over an
OpenAI-compatible REST API at https://api.groq.com/openai/v1, so we use
the official `openai` SDK with a custom `base_url`. The
OpenInference OpenAI instrumentor traces every call into Phoenix the
same way the Anthropic instrumentor does for Claude.

Per-agent system prompts are sent as a `system` message; Groq supports
prompt caching transparently when the same prefix shows up across
requests, so we don't need explicit cache-control markers.
"""
from __future__ import annotations

from openai import AsyncOpenAI

from .adapter import LLMAdapter

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MAX_TOKENS = 16000


class GroqAdapter(LLMAdapter):
    def __init__(
        self,
        model: str,
        api_key: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        base_url: str = GROQ_BASE_URL,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        resp = await self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        choice = resp.choices[0]
        return choice.message.content or ""
