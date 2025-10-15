import os
import secrets
import hashlib
import base64
from urllib.parse import urlencode

from fastapi import APIRouter, Response, Request, HTTPException, Depends
from sqlalchemy.orm import Session
import httpx
from ..db.session import get_session
from ..repositories.tokens_repo import TokensRepository
from ..core.security import generate_jwt
from ..core.security import generate_jwt, dummy_permissions
from ..models.auth_models import LoginRequest, LoginResponse, RefreshRequest, UserInfo


router = APIRouter()


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest) -> LoginResponse:
    # Handle test credentials - in production this would be real authentication
    if body.email == "user@admin" and body.password == "dev":
        # Convert test email to valid format
        email = "admin@example.com"
        role = "admin"
    elif body.email == "user@analyst" and body.password == "dev":
        email = "analyst@example.com"
        role = "analyst"
    else:
        # For now, accept any valid email format and determine role
        email = body.email
        role = "analyst"
        if "@admin" in body.email or "admin" in body.email.lower():
            role = "admin"
    
    user = UserInfo(id="uuid", email=email, role=role, permissions=dummy_permissions(role))
    access = generate_jwt({
        "sub": user.id,
        "email": user.email,
        "role": user.role,
        "permissions": user.permissions,
    }, expires_in=3600)
    refresh = generate_jwt({"sub": user.id, "type": "refresh"}, expires_in=3600 * 24 * 30)
    return LoginResponse(accessToken=access, refreshToken=refresh, expiresIn=3600, user=user)


@router.post("/refresh")
def refresh(_: RefreshRequest) -> dict:
    # In MVP, accept any refresh token and mint a new access token
    access = generate_jwt({"sub": "uuid"}, expires_in=3600)
    return {"accessToken": access, "expiresIn": 3600}


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


@router.get("/oidc/authorize")
def oidc_authorize(response: Response, redirect_uri: str):
    """Build OIDC authorization URL with PKCE and set state/verifier cookies"""
    auth_url = os.getenv("OIDC_AUTH_URL")
    client_id = os.getenv("OIDC_CLIENT_ID")
    scope = os.getenv("OIDC_SCOPE", "openid email profile")
    if not auth_url or not client_id:
        raise HTTPException(400, detail="OIDC not configured")
    state = secrets.token_urlsafe(16)
    verifier = _b64url(secrets.token_bytes(32))
    challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
    response.set_cookie("oidc_state", state, httponly=True, samesite="lax")
    response.set_cookie("oidc_verifier", verifier, httponly=True, samesite="lax")
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scope,
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    return {"authorizationUrl": f"{auth_url}?{urlencode(params)}", "state": state}


@router.get("/oidc/callback")
def oidc_callback(request: Request, code: str | None = None, state: str | None = None, redirect_uri: str | None = None, id_token: str | None = None):
    """Exchange authorization code for tokens (or accept id_token directly), then mint API access token."""
    # Validate state
    cookie_state = request.cookies.get("oidc_state")
    verifier = request.cookies.get("oidc_verifier")
    if not state or state != cookie_state:
        raise HTTPException(400, detail="Invalid state")

    email = None
    oidc_issuer = os.getenv("OIDC_ISSUER")
    token_url = os.getenv("OIDC_TOKEN_URL")
    client_id = os.getenv("OIDC_CLIENT_ID")
    client_secret = os.getenv("OIDC_CLIENT_SECRET")
    # Try code exchange if configured
    if code and token_url and client_id and verifier and redirect_uri:
        try:
            data = {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
                "client_id": client_id,
                "code_verifier": verifier,
            }
            if client_secret:
                data["client_secret"] = client_secret
            with httpx.Client(timeout=5.0) as client:
                resp = client.post(token_url, data=data)
                resp.raise_for_status()
                payload = resp.json()
                id_token = payload.get("id_token") or id_token
        except Exception as e:
            raise HTTPException(400, detail=f"Token exchange failed: {e}")

    # Derive email from id_token if available
    if id_token:
        # Let verify_jwt handle RS256 verification via JWKS
        from ..core.security import verify_jwt

        try:
            claims = verify_jwt(id_token)
            email = claims.get("email") or claims.get("upn") or claims.get("preferred_username")
        except Exception as e:
            raise HTTPException(400, detail=f"Invalid id_token: {e}")

    if not email:
        raise HTTPException(400, detail="Missing email")

    # Map to role (simple default)
    role = "analyst"
    if email.endswith("@admin") or email.startswith("admin"):
        role = "admin"
    user = UserInfo(id="uuid", email=email, role=role, permissions=dummy_permissions(role))
    access = generate_jwt({
        "sub": user.id,
        "email": user.email,
        "role": user.role,
        "permissions": user.permissions,
    }, expires_in=3600)
    refresh = generate_jwt({"sub": user.id, "type": "refresh"}, expires_in=3600 * 24 * 30)
    return LoginResponse(accessToken=access, refreshToken=refresh, expiresIn=3600, user=user)


@router.post("/logout")
def logout(request: Request, session: Session = Depends(get_session)):
    repo = TokensRepository(session, os.getenv("SCIP_SECRET_KEY", "dev"))
    # Try to revoke provided refresh token in JSON body
    try:
        data = request.json()
    except Exception:
        data = None
    if isinstance(data, dict) and data.get("refreshToken"):
        repo.revoke_refresh(data["refreshToken"])
    return {"revoked": True}


@router.post("/token")
def token_refresh(body: dict, session: Session = Depends(get_session)):
    refresh = body.get("refreshToken")
    sub = body.get("sub") or body.get("userId") or "uuid"
    if not refresh:
        raise HTTPException(status_code=400, detail="Missing refreshToken")
    t = TokensRepository(session, os.getenv("SCIP_SECRET_KEY", "dev"))
    new_refresh = t.rotate_refresh(refresh, sub)
    if not new_refresh:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    access = generate_jwt({"sub": sub}, expires_in=3600)
    return {"accessToken": access, "refreshToken": new_refresh, "expiresIn": 3600}
