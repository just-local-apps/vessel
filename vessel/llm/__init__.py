from .adapter import LLMAdapter
from .anthropic_llm import AnthropicAdapter
from .groq_llm import GroqAdapter

__all__ = [
    "LLMAdapter",
    "AnthropicAdapter",
    "GroqAdapter",
    "get_default_adapter",
]


def get_default_adapter() -> LLMAdapter:
    """Pick the adapter the agents should use, based on env config.

    `LLM_PROVIDER` chooses between `groq` (default — cheap, fast, Llama /
    OSS family on Groq Cloud) and `anthropic` (the original Claude path,
    kept as a fallback). The provider's API key + model are read from
    its own dedicated env vars so both can stay configured side-by-side.
    """
    from ..config import get_settings

    settings = get_settings()
    provider = (settings.llm_provider or "").strip().lower()
    if provider == "anthropic":
        if not settings.claude_api_key:
            raise RuntimeError("CLAUDE_API_KEY not configured")
        return AnthropicAdapter(
            model=settings.claude_model, api_key=settings.claude_api_key
        )

    # Default = Groq.
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY not configured (set LLM_PROVIDER=anthropic to "
            "use Claude instead)"
        )
    return GroqAdapter(model=settings.groq_model, api_key=settings.groq_api_key)
