from __future__ import annotations

import time
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ..observability import metrics as prom_metrics


class PrometheusRequestMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to record per-request Prometheus metrics."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.time()
        response: Response
        try:
            response = await call_next(request)
            return response
        finally:
            duration = time.time() - start
            if prom_metrics:
                try:
                    path = request.url.path
                    method = request.method
                    status = getattr(response, 'status_code', 500)
                    prom_metrics.record_http_request(method, path, status, duration)
                except Exception:
                    pass

