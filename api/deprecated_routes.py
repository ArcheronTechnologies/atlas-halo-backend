"""
Deprecated API Routes - Backward Compatibility Layer
Provides 6-month deprecation period for old routes

MIGRATION TIMELINE:
- Created: 2025-10-03
- Deprecation warnings: 2025-10-03 to 2026-04-03
- Full removal: 2026-04-03

All clients should migrate to /api/v1/* pattern
"""

from fastapi import APIRouter, Request, Response, status
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

# Create deprecation router
deprecated_router = APIRouter(tags=["deprecated"])


@deprecated_router.api_route("/mobile/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def deprecated_mobile_routes(request: Request, path: str):
    """
    Redirect old /mobile/* routes to /api/v1/mobile/*

    Returns 308 Permanent Redirect with deprecation warning header
    """

    # Log deprecation usage
    logger.warning(
        f"DEPRECATED: Client using old route /mobile/{path} "
        f"from {request.client.host}. "
        f"Should migrate to /api/v1/mobile/{path}"
    )

    # Build new URL
    new_url = f"/api/v1/mobile/{path}"
    if request.url.query:
        new_url += f"?{request.url.query}"

    # Return redirect response with deprecation header
    return JSONResponse(
        status_code=status.HTTP_308_PERMANENT_REDIRECT,
        headers={
            "Location": new_url,
            "X-Deprecated-Route": "true",
            "X-Deprecation-Warning": "Route /mobile/* is deprecated. Use /api/v1/mobile/* instead. Removal date: 2026-04-03",
            "X-Migration-Guide": "https://docs.atlas-ai.com/api-migration-v1"
        },
        content={
            "error": "Deprecated Route",
            "message": f"Route /mobile/{path} is deprecated. Please use /api/v1/mobile/{path} instead.",
            "deprecation_date": "2025-10-03",
            "removal_date": "2026-04-03",
            "new_url": new_url
        }
    )


@deprecated_router.get("/mobile")
async def deprecated_mobile_root():
    """Redirect /mobile to /api/v1/mobile"""
    return JSONResponse(
        status_code=status.HTTP_308_PERMANENT_REDIRECT,
        headers={
            "Location": "/api/v1/mobile",
            "X-Deprecated-Route": "true"
        },
        content={
            "message": "Route /mobile is deprecated. Use /api/v1/mobile instead.",
            "removal_date": "2026-04-03"
        }
    )


# Track deprecation usage statistics
deprecation_stats = {
    "total_calls": 0,
    "unique_clients": set(),
    "endpoints_used": {}
}


def get_deprecation_stats() -> dict:
    """Get statistics on deprecated route usage"""
    return {
        "total_calls": deprecation_stats["total_calls"],
        "unique_clients": len(deprecation_stats["unique_clients"]),
        "endpoints": deprecation_stats["endpoints_used"]
    }


@deprecated_router.get("/_deprecation_stats")
async def show_deprecation_stats():
    """
    Admin endpoint to view deprecation usage statistics
    Helps track which clients need to migrate
    """
    return get_deprecation_stats()
