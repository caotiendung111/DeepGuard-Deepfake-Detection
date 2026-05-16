"""
DeepGuard — Simple Rate Limiter Middleware
Limits requests per IP to prevent abuse.
"""
import time
from collections import defaultdict
from typing import Dict, List

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Token bucket rate limiter.

    Args:
        calls: Maximum number of calls per period.
        period: Time window in seconds.
    """

    def __init__(self, app, calls: int = 60, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self._requests: Dict[str, List[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip health endpoint
        if request.url.path in ("/health", "/", "/docs", "/redoc", "/openapi.json"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Remove expired timestamps
        self._requests[client_ip] = [
            t for t in self._requests[client_ip] if now - t < self.period
        ]

        if len(self._requests[client_ip]) >= self.calls:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Max {self.calls} requests per {self.period}s",
                    "retry_after": self.period,
                },
                headers={"Retry-After": str(self.period)},
            )

        self._requests[client_ip].append(now)
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.calls)
        response.headers["X-RateLimit-Remaining"] = str(
            self.calls - len(self._requests[client_ip])
        )
        return response
