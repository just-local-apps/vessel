"""FastAPI entrypoint. Wires DB, state_manager, observability,
PWA router, and the MCP server together.
"""
import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from . import observability
from .config import get_settings
from .db import close_pool, get_pool, run_migrations
from .mcp_routes import build_mcp_routes
from .mcp_server import build_mcp_server
from .models import StateData
from .pwa.routes import mount_static, router as pwa_router
from .scheduler import state_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger("vessel")

# Initialize tracing as early as possible so OpenInference can patch the
# `anthropic` SDK before any LLM client is constructed.
observability.init()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("vessel starting on %s:%s", settings.host, settings.port)

    pool = await get_pool()
    await run_migrations(pool)

    app.state.pool = pool

    try:
        yield
    finally:
        logger.info("vessel shutting down")
        await close_pool()


app = FastAPI(title="Vessel", lifespan=lifespan)
app.include_router(pwa_router)
mount_static(app)


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "tracing": observability.get_status()}


@app.get("/")
async def root() -> RedirectResponse:
    return RedirectResponse(url="/pwa/")


# ---------- MCP wiring ----------

def _active_model_name() -> str:
    settings = get_settings()
    if (settings.llm_provider or "").lower() == "anthropic":
        return settings.claude_model
    return settings.groq_model


def _build_chat_client() -> Any:
    """Build the OpenAI-shaped client the chat tool-loop drives.

    Uses Groq's OpenAI-compatible endpoint by default (the only LLM
    surface vessel exposes after the chat/skip/MCP unification).
    Returns a stub that raises on use when no key is configured —
    the MCP server still imports cleanly, so tests and the PWA
    routes that don't go through `apply_instruction` keep working.
    """
    settings = get_settings()
    if not settings.groq_api_key:
        class _NoOpChat:
            class _Completions:
                async def create(self, **_kwargs):
                    raise RuntimeError("GROQ_API_KEY not configured")

            class _Chat:
                completions = _NoOpChat._Completions()

            chat = _Chat()

        return _NoOpChat()

    from openai import AsyncOpenAI as _AsyncOpenAI

    return _AsyncOpenAI(
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )


def _mcp_server_factory(user_id: str):
    """Build an MCP Server scoped to a single user_id.

    The closures below bind every read/write to that user_id; the MCP server
    object itself is otherwise identical to the single-user version.
    """

    async def read_state() -> StateData:
        return await state_manager.read(await get_pool(), user_id)

    async def write_state(state: StateData) -> None:
        await state_manager.write(await get_pool(), user_id, state)

    return build_mcp_server(
        read_state=read_state,
        write_state=write_state,
        chat_client=_build_chat_client(),
        chat_model=_active_model_name(),
    )


for _route in build_mcp_routes(_mcp_server_factory):
    app.router.routes.append(_route)


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "vessel.main:app",
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
