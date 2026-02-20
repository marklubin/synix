"""Authentication middleware and helpers for mesh API."""

from __future__ import annotations

import logging
import secrets

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Endpoints that bypass authentication
PUBLIC_PATHS = frozenset({"/api/v1/health"})

TOKEN_PREFIX = "msh_"


def generate_token() -> str:
    """Generate a new mesh token with msh_ prefix + 32-byte random hex."""
    return f"{TOKEN_PREFIX}{secrets.token_hex(32)}"


def auth_headers(token: str, node_name: str = "", term_counter: int | None = None) -> dict[str, str]:
    """Build request headers for authenticated mesh API calls."""
    headers = {"Authorization": f"Bearer {token}"}
    if node_name:
        headers["X-Mesh-Node"] = node_name
    if term_counter is not None:
        headers["X-Mesh-Term"] = str(term_counter)
    return headers


class AuthMiddleware(BaseHTTPMiddleware):
    """Starlette middleware: validates Bearer token on all non-public endpoints."""

    def __init__(self, app, token: str):
        super().__init__(app)
        self.token = token

    async def dispatch(self, request: Request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            client_host = request.client.host if request.client else "unknown"
            logger.warning("Missing or malformed Authorization header from %s", client_host)
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        provided_token = auth_header[len("Bearer ") :]
        if provided_token != self.token:
            logger.warning("Invalid token from %s", request.client.host if request.client else "unknown")
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        return await call_next(request)
