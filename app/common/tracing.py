import contextvars
from collections.abc import Awaitable, Callable
from logging import getLogger
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import config

logger = getLogger(__name__)

ctx_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id")
ctx_request: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar("request")
ctx_response: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "response"
)


# Inbound HTTP requests on the platform will have a `x-cdp-request-id` header.
# This can be used to follow a single request across multiple services.
# TraceIdMiddleware handles extracting the tracing header and persisting it
# for the duration of the request in the ContextVar `ctx_trace_id`.
class TraceIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        req_trace_id = request.headers.get(config.tracing_header, None)
        if req_trace_id:
            ctx_trace_id.set(req_trace_id)

        ctx_request.set({"url": str(request.url), "method": request.method})

        response = await call_next(request)
        ctx_response.set({"status_code": response.status_code})
        return response
