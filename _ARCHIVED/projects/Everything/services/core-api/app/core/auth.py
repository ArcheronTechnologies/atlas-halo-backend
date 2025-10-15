from __future__ import annotations

from typing import List

from fastapi import Depends, HTTPException, status, Request
import os
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .security import verify_jwt
from ..db.session import get_session
from ..repositories.users_repo import UsersRepository
from sqlalchemy.orm import Session
from ..db.models import RevokedJTI


bearer_scheme = HTTPBearer(auto_error=False)


def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    session: Session = Depends(get_session),
) -> dict:
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    token = credentials.credentials
    try:
        claims = verify_jwt(token) or {}
        # Normalize permissions from common claim shapes
        perms = claims.get("permissions")
        if not perms:
            scope = claims.get("scope")
            if isinstance(scope, str):
                perms = scope.split()
        if not perms:
            scp = claims.get("scp")  # Azure AD
            if isinstance(scp, list):
                perms = scp
        if not perms:
            roles = claims.get("roles")
            if isinstance(roles, list):
                perms = roles
        if perms:
            claims["permissions"] = perms
        # Enforce JTI blacklist if present
        jti = claims.get("jti")
        if jti and session.get(RevokedJTI, jti) is not None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token revoked")
        return claims
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")


def require_scopes(scopes: List[str]):
    def dependency(claims: dict = Depends(require_api_key_or_bearer)) -> dict:
        perms = set((claims or {}).get("permissions") or [])
        missing = [s for s in scopes if s not in perms]
        if missing:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Missing scopes: {', '.join(missing)}")
        return claims

    return dependency


def require_api_key_or_bearer(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    session: Session = Depends(get_session),
) -> dict:
    auth_mode = os.getenv("SCIP_AUTH_MODE", "mixed").lower()  # 'mixed' or 'bearer_only'
    # Prefer bearer JWT if present (always accepted)
    if credentials is not None:
        return require_auth(credentials, session)
    # In bearer_only mode, reject API keys
    if auth_mode == "bearer_only":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bearer token required")
    # Otherwise accept X-API-Key
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing credentials")
    repo = UsersRepository(session)
    user = repo.verify_api_key(api_key)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    perms = repo.permissions_for_user(user.id)
    return {"sub": user.id, "email": user.email, "permissions": perms or ["api:access"]}


def require_api_key(api_key: str | None = Depends(lambda request: request.headers.get("X-API-Key") if hasattr(request := request, 'headers') else None)):
    # Placeholder: FastAPI can't inject Request via lambda; implement a small wrapper in routers instead if needed.
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Use bearer auth for now")


# Alias for get_current_user - used by notifications router
get_current_user = require_api_key_or_bearer
