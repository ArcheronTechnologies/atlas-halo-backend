import base64
import hashlib
import hmac
import json
import time
import uuid
from typing import Any, Dict

from .config import settings
import os
import secrets
import hashlib as _hashlib


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def generate_jwt(payload: Dict[str, Any], expires_in: int = 3600) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    payload = {**payload, "iat": now, "exp": now + expires_in, "jti": str(uuid.uuid4())}

    header_b64 = _b64url(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = _b64url(json.dumps(payload, separators=(",", ":")).encode())
    to_sign = f"{header_b64}.{payload_b64}".encode()
    signature = hmac.new(settings.secret_key.encode(), to_sign, hashlib.sha256).digest()
    token = f"{header_b64}.{payload_b64}.{_b64url(signature)}"
    return token


def get_permissions_for_role(role: str, custom_permissions: list[str] = None) -> list[str]:
    """
    Get permissions for a role with support for custom permissions and hierarchical roles.
    
    Args:
        role: User role (e.g., 'admin', 'analyst', 'user', 'supplier', 'procurement')
        custom_permissions: Additional custom permissions to grant
    
    Returns:
        List of permissions
    """
    # Base permission sets
    permission_sets = {
        "read_basic": [
            "read:components", "read:companies", "read:inventory"
        ],
        "read_advanced": [
            "read:rfqs", "read:purchase_orders", "read:audit", "read:graph", 
            "read:market_intelligence", "read:notifications", "read:integrations"
        ],
        "write_basic": [
            "write:components", "write:rfqs", "write:inventory"
        ],
        "write_advanced": [
            "write:companies", "write:purchase_orders", "write:graph",
            "write:notifications", "write:market_intelligence", "write:integrations"
        ],
        "admin_only": [
            "read:users", "write:users", "read:system", "write:system",
            "read:audit_full", "manage:roles", "manage:permissions"
        ],
        "supplier_specific": [
            "update:own_products", "read:own_rfqs", "write:quotes",
            "read:own_orders", "update:delivery_status"
        ],
        "procurement_specific": [
            "approve:purchase_orders", "manage:suppliers", "read:cost_analytics",
            "write:procurement_policies", "approve:budget"
        ]
    }
    
    # Role hierarchy definitions
    role_permissions = {
        "admin": (
            permission_sets["read_basic"] + 
            permission_sets["read_advanced"] +
            permission_sets["write_basic"] +
            permission_sets["write_advanced"] +
            permission_sets["admin_only"]
        ),
        "procurement_manager": (
            permission_sets["read_basic"] +
            permission_sets["read_advanced"] +
            permission_sets["write_basic"] +
            permission_sets["procurement_specific"]
        ),
        "analyst": (
            permission_sets["read_basic"] +
            permission_sets["read_advanced"] +
            permission_sets["write_basic"]
        ),
        "procurement_user": (
            permission_sets["read_basic"] +
            permission_sets["read_advanced"] +
            ["write:rfqs", "write:purchase_orders"]
        ),
        "supplier": (
            permission_sets["read_basic"] +
            permission_sets["supplier_specific"] +
            ["read:own_notifications"]
        ),
        "user": permission_sets["read_basic"],
        "readonly": ["read:components", "read:companies"]
    }
    
    # Get base permissions for role
    base_permissions = role_permissions.get(role.lower(), permission_sets["read_basic"])
    
    # Add custom permissions if provided
    if custom_permissions:
        base_permissions = list(set(base_permissions + custom_permissions))
    
    return base_permissions


def validate_permission(required_permission: str, user_permissions: list[str]) -> bool:
    """
    Validate if user has required permission with support for wildcard patterns.
    
    Args:
        required_permission: Permission to check (e.g., 'read:components')
        user_permissions: List of user's permissions
    
    Returns:
        True if permission is granted
    """
    # Direct match
    if required_permission in user_permissions:
        return True
    
    # Check for wildcard permissions
    permission_parts = required_permission.split(':')
    if len(permission_parts) == 2:
        action, resource = permission_parts
        
        # Check for action wildcards (e.g., '*:components')
        if f"*:{resource}" in user_permissions:
            return True
            
        # Check for resource wildcards (e.g., 'read:*')
        if f"{action}:*" in user_permissions:
            return True
            
        # Check for full wildcard
        if "*:*" in user_permissions:
            return True
    
    return False


# Backward compatibility
def dummy_permissions(role: str) -> list[str]:
    """Backward compatibility wrapper"""
    return get_permissions_for_role(role)


def _b64url_decode(data: str) -> bytes:
    pad = '=' * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + pad)


def verify_jwt(token: str) -> Dict[str, Any]:
    """Verify JWT.

    If OIDC issuer is configured, verify with RS256 using JWKS from file or URL (with optional discovery).
    Otherwise, fall back to HS256 using the local secret (dev mode).
    """
    oidc_issuer = os.getenv("OIDC_ISSUER")
    oidc_aud = os.getenv("OIDC_AUDIENCE")
    jwks_path = os.getenv("OIDC_JWKS_PATH")
    jwks_url = os.getenv("OIDC_JWKS_URL")
    discovery_url = os.getenv("OIDC_DISCOVERY_URL")
    require_oidc = os.getenv("SCIP_REQUIRE_OIDC") == "1" or os.getenv("SCIP_ENV") == "production"
    if oidc_issuer and (jwks_path or jwks_url or discovery_url):
        try:
            from jose import jwt
            import json as _json

            # simple in-process cache
            cache = getattr(verify_jwt, "_jwks_cache", {"jwks": None, "ts": 0})
            now = int(time.time())
            jwks = cache["jwks"] if (cache["jwks"] and now - cache["ts"] < 3600) else None

            if not jwks:
                if jwks_path:
                    with open(jwks_path, "r") as f:
                        jwks = _json.load(f)
                else:
                    try:
                        import httpx

                        if discovery_url:
                            r = httpx.get(discovery_url, timeout=5.0)
                            r.raise_for_status()
                            jwks_uri = r.json().get("jwks_uri")
                        else:
                            jwks_uri = jwks_url
                        if jwks_uri:
                            r = httpx.get(jwks_uri, timeout=5.0)
                            r.raise_for_status()
                            jwks = r.json()
                    except Exception:
                        jwks = None
                setattr(verify_jwt, "_jwks_cache", {"jwks": jwks, "ts": now})

            if jwks:
                unverified = jwt.get_unverified_header(token)
                kid = unverified.get("kid")
                key = None
                for k in jwks.get("keys", []):
                    if not kid or k.get("kid") == kid:
                        key = k
                        break
                if key is None and jwks.get("keys"):
                    key = jwks["keys"][0]
                if key:
                    claims = jwt.decode(
                        token,
                        key,
                        algorithms=["RS256"],
                        audience=oidc_aud,
                        issuer=oidc_issuer,
                        options={"verify_at_hash": False},
                    )
                    return claims  # type: ignore[return-value]
        except Exception as e:
            # If OIDC is required, do not fall back
            if require_oidc:
                raise ValueError(f"oidc verification failed: {e}")
            # else fall through to HS256 for dev
    elif require_oidc:
        # OIDC required but not properly configured
        raise ValueError("oidc required but not configured (set JWKS or discovery)")

    # HS256 fallback (dev)
    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
    except ValueError:
        raise ValueError("invalid token format")
    to_sign = f"{header_b64}.{payload_b64}".encode()
    expected = hmac.new(settings.secret_key.encode(), to_sign, hashlib.sha256).digest()
    if not hmac.compare_digest(expected, _b64url_decode(sig_b64)):
        raise ValueError("invalid signature")
    payload = json.loads(_b64url_decode(payload_b64))
    leeway = getattr(settings, "jwt_leeway", 0)
    if "exp" in payload and int(payload["exp"]) + int(leeway) < int(time.time()):
        raise ValueError("token expired")
    # Check if jti is revoked (optional, requires DB session; defer to caller dependency for now)
    return payload


def generate_api_key() -> tuple[str, str]:
    raw = secrets.token_urlsafe(32)
    key_id = secrets.token_hex(8)
    api_key = f"sk_{key_id}_{raw}"
    return key_id, api_key


def hash_api_key(api_key: str) -> str:
    return _hashlib.sha256((settings.secret_key + ":" + api_key).encode()).hexdigest()
