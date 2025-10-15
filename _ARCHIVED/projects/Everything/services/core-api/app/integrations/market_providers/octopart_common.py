from __future__ import annotations

import os
import asyncio
import json
import hashlib
import logging
import random
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
import time

from ...cache.redis_cache import RedisCache
from ...observability import metrics as prom_metrics
from ...observability.tracing import create_async_span

logger = logging.getLogger(__name__)


OCTOPART_ENDPOINT = os.getenv("OCTOPART_ENDPOINT", "https://api.octopart.com/v4/graphql")
OCTOPART_API_KEY = os.getenv("OCTOPART_API_KEY")
OCTOPART_TIMEOUT_SECONDS = float(os.getenv("OCTOPART_TIMEOUT_SECONDS", "15"))
OCTOPART_MAX_CONCURRENCY = int(os.getenv("OCTOPART_MAX_CONCURRENCY", "5"))
OCTOPART_CACHE_TTL_SECONDS = int(os.getenv("OCTOPART_CACHE_TTL_SECONDS", "300"))
OCTOPART_RATE_LIMIT_PER_MIN = int(os.getenv("OCTOPART_RATE_LIMIT_PER_MIN", "0"))  # 0 disables

_sem = asyncio.Semaphore(OCTOPART_MAX_CONCURRENCY)
_cache = RedisCache()


def _hash_params(params: Dict[str, Any]) -> str:
    try:
        return hashlib.md5(json.dumps(params, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    except Exception:
        return hashlib.md5(str(params).encode("utf-8")).hexdigest()


async def _rate_limit_allow(op: str) -> bool:
    if OCTOPART_RATE_LIMIT_PER_MIN <= 0:
        return True
    # Use minute bucket key
    minute_key = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
    key = f"octopart:{op}:{minute_key}"
    try:
        count = await _cache.incr("rate_limits", key, amount=1, ttl=60)
        allowed = count <= OCTOPART_RATE_LIMIT_PER_MIN
        if not allowed:
            logger.warning(f"Octopart rate limit exceeded for op={op} count={count}")
        return allowed
    except Exception:
        # If cache unavailable, fail open
        return True


async def graphql_request(
    op: str,
    query: str,
    variables: Dict[str, Any],
    *,
    use_cache: bool = True,
    cache_key_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute an Octopart GraphQL request with concurrency limit, retry/backoff, and optional Redis cache."""
    if os.getenv("SCIP_SKIP_NETWORK") == "1":
        return {"data": {}}

    if not OCTOPART_API_KEY:
        raise RuntimeError("OCTOPART_API_KEY not configured")

    # Cache lookup
    cache_key = None
    if use_cache:
        try:
            params = cache_key_params or variables
            cache_key = f"octopart:{op}:{_hash_params(params)}"
            cached = await _cache.get("api_responses", cache_key)
            if cached:
                return cached  # type: ignore[return-value]
        except Exception:
            pass

    # Rate limit
    if not await _rate_limit_allow(op):
        raise RuntimeError("octopart_rate_limited")

    headers = {"Content-Type": "application/json", "token": OCTOPART_API_KEY}
    payload = {"query": query, "variables": variables}

    # Retry with exponential backoff
    attempts = int(os.getenv("OCTOPART_RETRY_ATTEMPTS", "3"))
    base_delay = float(os.getenv("OCTOPART_RETRY_BASE_DELAY_SEC", "0.5"))
    last_exc: Optional[Exception] = None

    async with _sem:
        for i in range(attempts):
            start = time.time()
            try:
                async with create_async_span(
                    name=f"octopart.{op}",
                    attributes={"endpoint": OCTOPART_ENDPOINT, "attempt": i + 1}
                ):
                    async with httpx.AsyncClient(timeout=OCTOPART_TIMEOUT_SECONDS) as client:
                        resp = await client.post(OCTOPART_ENDPOINT, json=payload, headers=headers)
                        resp.raise_for_status()
                        data = resp.json()

                duration = time.time() - start
                if prom_metrics:
                    try:
                        prom_metrics.record_processing_time(f"octopart.{op}", duration, complexity="network")
                        prom_metrics.update_dependency_health("octopart", OCTOPART_ENDPOINT, True)
                    except Exception:
                        pass

                # GraphQL errors handling
                if isinstance(data, dict) and data.get("errors"):
                    logger.warning(f"Octopart GraphQL errors for op={op}: {data.get('errors')}")
                else:
                    # Cache set
                    if use_cache and cache_key:
                        try:
                            await _cache.set("api_responses", cache_key, data, ttl=OCTOPART_CACHE_TTL_SECONDS, serialize_as="json")
                        except Exception:
                            pass
                    return data
            except Exception as e:
                duration = time.time() - start
                if prom_metrics:
                    try:
                        prom_metrics.record_processing_time(f"octopart.{op}", duration, complexity="network")
                        prom_metrics.update_dependency_health("octopart", OCTOPART_ENDPOINT, False)
                    except Exception:
                        pass
                last_exc = e
                delay = (2 ** i) * base_delay + random.uniform(0, 0.1)
                await asyncio.sleep(delay)

    # Exhausted retries
    assert last_exc is not None
    raise last_exc
