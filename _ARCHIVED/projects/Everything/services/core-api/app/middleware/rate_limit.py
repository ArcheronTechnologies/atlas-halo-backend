from __future__ import annotations

import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ..core.config import settings
from redis import asyncio as aioredis


class RateLimiter(BaseHTTPMiddleware):
    def __init__(self, app, rate_per_minute: int):
        super().__init__(app)
        self.rate = max(1, rate_per_minute)
        self.window = 60
        self._store: dict[str, tuple[int, int]] = {}
        self._redis = None
        if settings.redis_url:
            try:
                self._redis = aioredis.from_url(settings.redis_url)
            except Exception:
                self._redis = None

    async def dispatch(self, request: Request, call_next: Callable):
        key = request.headers.get("Authorization") or (request.client.host if request.client else None) or "anon"
        now = int(time.time())
        window_start = now - (now % self.window)
        reset = window_start + self.window
        remaining = None
        if self._redis:
            try:
                rkey = f"rl:{key}:{window_start}"
                pipe = self._redis.pipeline()
                pipe.incr(rkey)
                pipe.expire(rkey, self.window)
                count, _ = await pipe.execute()
                remaining = max(0, self.rate - int(count))
                if int(count) > self.rate:
                    return Response(
                        status_code=429,
                        content='{"error": {"code": "RATE_LIMITED", "message": "Too many requests"}}',
                        media_type="application/json",
                        headers={
                            "X-RateLimit-Limit": str(self.rate),
                            "X-RateLimit-Remaining": "0",
                            "X-RateLimit-Reset": str(reset),
                        },
                    )
            except Exception:
                # Fall back to memory store on redis failure
                self._redis = None

        if self._redis is None:
            count, ts = self._store.get(key, (0, window_start))
            if ts != window_start:
                count = 0
                ts = window_start
            count += 1
            self._store[key] = (count, ts)
            remaining = max(0, self.rate - count)
            if count > self.rate:
                return Response(
                    status_code=429,
                    content='{"error": {"code": "RATE_LIMITED", "message": "Too many requests"}}',
                    media_type="application/json",
                    headers={
                        "X-RateLimit-Limit": str(self.rate),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(reset),
                    },
                )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.rate)
        response.headers["X-RateLimit-Remaining"] = str(remaining if remaining is not None else "")
        response.headers["X-RateLimit-Reset"] = str(reset)
        return response
