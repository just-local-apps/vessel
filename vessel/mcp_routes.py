"""SSE transport for the Vessel MCP server, mounted on FastAPI.

Authentication: `?token=<your-token>` on the SSE GET. The token is hashed
to a `user_id` and a per-user MCP server is built; every tool call inside
the session is scoped to that user.
"""
from __future__ import annotations

import logging
from typing import Callable

from mcp.server import Server
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Mount, Route

from mcp.server.sse import SseServerTransport

from .users import derive_user_id, is_valid_token

logger = logging.getLogger(__name__)


def build_mcp_routes(server_factory: Callable[[str], Server]) -> list:
    """`server_factory(user_id)` builds an MCP Server scoped to that user."""
    transport = SseServerTransport("/mcp/messages/")

    async def handle_sse(request: Request) -> Response:
        provided = request.query_params.get("token", "")
        if not is_valid_token(provided):
            return Response(status_code=401, content="invalid or missing token")

        user_id = derive_user_id(provided)
        logger.info("MCP session opening for user %s...", user_id[:8])
        server = server_factory(user_id)
        async with transport.connect_sse(
            request.scope, request.receive, request._send  # type: ignore[attr-defined]
        ) as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
        return Response()

    return [
        Route("/mcp/sse", endpoint=handle_sse, methods=["GET"]),
        Mount("/mcp/messages/", app=transport.handle_post_message),
    ]
