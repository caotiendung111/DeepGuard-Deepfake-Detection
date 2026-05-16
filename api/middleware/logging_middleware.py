"""
Request logging middleware with request IDs.

Logs one structured line per request with request/response sizes, status, and
duration. The request ID is propagated through the response header so API users
can correlate client errors with server logs.
"""
import time
import uuid
from contextvars import ContextVar

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware


request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")


def get_request_id() -> str:
    return request_id_ctx.get()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = request_id_ctx.set(request_id)
        started = time.perf_counter()
        request_size = int(request.headers.get("content-length") or 0)
        response_size = 0
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            response_size = int(response.headers.get("content-length") or 0)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            duration_ms = (time.perf_counter() - started) * 1000
            client = request.client.host if request.client else "unknown"
            logger.bind(request_id=request_id).info(
                "request completed | "
                f"method={request.method} path={request.url.path} "
                f"status={status_code} duration_ms={duration_ms:.2f} "
                f"request_bytes={request_size} response_bytes={response_size} "
                f"client={client}"
            )
            request_id_ctx.reset(token)
