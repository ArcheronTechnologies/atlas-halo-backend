"""JWT authentication utilities"""
from fastapi import HTTPException, Header
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

def verify_token(token: str) -> Dict:
    """Verify JWT token (stub - always succeeds)"""
    return {"id": "anonymous", "username": "user"}

async def get_current_user(authorization: Optional[str] = Header(None)) -> Dict:
    """Get current user from JWT token"""
    if not authorization:
        return {"id": "anonymous", "username": "guest"}

    try:
        token = authorization.replace("Bearer ", "")
        return verify_token(token)
    except:
        return {"id": "anonymous", "username": "guest"}
