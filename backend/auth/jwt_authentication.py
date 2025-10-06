"""JWT authentication utilities"""
from fastapi import HTTPException, Header, Depends
from typing import Optional, Dict, Any
import logging
import jwt
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

JWT_SECRET = os.getenv("JWT_SECRET_KEY", "dev-secret-key")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

class AuthenticationService:
    """Authentication service for JWT token management"""

    def __init__(self):
        self.secret_key = JWT_SECRET
        self.algorithm = JWT_ALGORITHM
        self.expiration_minutes = JWT_EXPIRATION_MINUTES

    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.expiration_minutes)
        to_encode.update({"exp": expire})

        try:
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            return encoded_jwt
        except Exception as e:
            logger.error(f"Error creating token: {e}")
            raise HTTPException(status_code=500, detail="Could not create token")

    def verify_token(self, token: str) -> Dict:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")

    async def get_current_user(self, token: str) -> Dict:
        """Get current user from token"""
        return self.verify_token(token)

# Singleton instance
auth_service = AuthenticationService()

def verify_token(token: str) -> Dict:
    """Verify JWT token (legacy function)"""
    return auth_service.verify_token(token)

async def get_current_user(authorization: Optional[str] = Header(None)) -> Dict:
    """Get current user from JWT token"""
    if not authorization:
        return {"id": "anonymous", "username": "guest"}

    try:
        token = authorization.replace("Bearer ", "")
        return verify_token(token)
    except:
        return {"id": "anonymous", "username": "guest"}
