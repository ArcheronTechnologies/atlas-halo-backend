from fastapi import APIRouter
import os

router = APIRouter(prefix="/debug", tags=["debug"])

@router.get("/env")
async def get_env_vars():
    """Debug endpoint to check environment variables"""
    return {
        "POSTGRES_HOST": os.getenv('POSTGRES_HOST', 'NOT_SET'),
        "POSTGRES_PORT": os.getenv('POSTGRES_PORT', 'NOT_SET'),
        "POSTGRES_DB": os.getenv('POSTGRES_DB', 'NOT_SET'),
        "POSTGRES_USER": os.getenv('POSTGRES_USER', 'NOT_SET'),
        "password_length": len(os.getenv('POSTGRES_PASSWORD', '')) if os.getenv('POSTGRES_PASSWORD') else 0,
        "ATLAS_API_URL": os.getenv('ATLAS_API_URL', 'NOT_SET'),
        "ENVIRONMENT": os.getenv('ENVIRONMENT', 'NOT_SET'),
        "SENTRY_DSN_SET": "YES" if os.getenv('SENTRY_DSN') else "NO",
    }

@router.get("/sentry-debug")
async def trigger_sentry_error():
    """Test endpoint to trigger a Sentry error - should appear in Sentry dashboard"""
    division_by_zero = 1 / 0
    return {"status": "This will never be returned"}
