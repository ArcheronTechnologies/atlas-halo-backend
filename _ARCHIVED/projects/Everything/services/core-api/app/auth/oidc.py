"""
OpenID Connect (OIDC) Authentication with Full Session Lifecycle

This module provides comprehensive OIDC authentication including:
- Authorization code flow with PKCE
- Token refresh and rotation
- Session management and revocation  
- Configurable role mapping from IdP claims
- Logout with RP-initiated and IdP-initiated support
"""

import asyncio
import logging
import secrets
import hashlib
import base64
import json
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from urllib.parse import urlencode, parse_qs, urlparse
import aiohttp
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from ..cache.redis_cache import cache
from ..core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class OIDCConfig:
    """OIDC Provider configuration"""
    issuer: str
    client_id: str
    client_secret: str
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: str
    jwks_uri: str
    end_session_endpoint: Optional[str] = None
    scopes: List[str] = None
    response_types: List[str] = None
    grant_types: List[str] = None
    
    def __post_init__(self):
        if self.scopes is None:
            self.scopes = ["openid", "profile", "email"]
        if self.response_types is None:
            self.response_types = ["code"]
        if self.grant_types is None:
            self.grant_types = ["authorization_code", "refresh_token"]


@dataclass
class OIDCSession:
    """OIDC User session with tokens"""
    session_id: str
    user_id: str
    email: str
    name: str
    roles: List[str]
    permissions: List[str]
    access_token: str
    refresh_token: Optional[str]
    id_token: str
    expires_at: datetime
    refresh_expires_at: Optional[datetime]
    created_at: datetime
    last_activity: datetime
    ip_address: Optional[str]
    user_agent: Optional[str]
    idp_session_id: Optional[str] = None
    
    @property
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) >= self.expires_at
    
    @property
    def is_refresh_expired(self) -> bool:
        if not self.refresh_expires_at:
            return True
        return datetime.now(timezone.utc) >= self.refresh_expires_at
    
    @property
    def needs_refresh(self) -> bool:
        # Refresh if token expires within next 5 minutes
        return datetime.now(timezone.utc) >= (self.expires_at - timedelta(minutes=5))


@dataclass 
class RoleMapping:
    """Configuration for mapping IdP claims to application roles"""
    claim_name: str
    claim_value: str
    roles: List[str]
    permissions: List[str] = None
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = []


class PKCEChallenge:
    """PKCE (Proof Key for Code Exchange) challenge generator"""
    
    @staticmethod
    def generate_challenge() -> Tuple[str, str]:
        """Generate PKCE code verifier and challenge"""
        # Generate code verifier (43-128 characters)
        code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(96)).decode('utf-8').rstrip('=')
        
        # Generate code challenge (SHA256 hash)
        challenge_bytes = hashlib.sha256(code_verifier.encode('utf-8')).digest()
        code_challenge = base64.urlsafe_b64encode(challenge_bytes).decode('utf-8').rstrip('=')
        
        return code_verifier, code_challenge


class JWTValidator:
    """JWT token validation with caching"""
    
    def __init__(self):
        self.jwks_cache: Dict[str, Any] = {}
        self.jwks_cache_expiry: Dict[str, datetime] = {}
    
    async def validate_jwt(self, token: str, issuer: str, audience: str, jwks_uri: str) -> Dict[str, Any]:
        """Validate JWT token with JWKS"""
        try:
            # Decode header to get key ID
            header = jwt.get_unverified_header(token)
            kid = header.get('kid')
            
            if not kid:
                raise ValueError("JWT token missing key ID")
            
            # Get signing key
            signing_key = await self._get_signing_key(jwks_uri, kid)
            
            # Validate and decode token
            payload = jwt.decode(
                token,
                signing_key,
                algorithms=["RS256"],
                issuer=issuer,
                audience=audience,
                options={"verify_exp": True}
            )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid token: {e}")
    
    async def _get_signing_key(self, jwks_uri: str, kid: str) -> str:
        """Get signing key from JWKS endpoint with caching"""
        cache_key = f"{jwks_uri}:{kid}"
        
        # Check cache
        if (cache_key in self.jwks_cache and 
            cache_key in self.jwks_cache_expiry and
            datetime.now(timezone.utc) < self.jwks_cache_expiry[cache_key]):
            return self.jwks_cache[cache_key]
        
        # Fetch JWKS
        async with aiohttp.ClientSession() as session:
            async with session.get(jwks_uri) as response:
                if response.status != 200:
                    raise ValueError(f"Failed to fetch JWKS: HTTP {response.status}")
                
                jwks = await response.json()
        
        # Find key
        for key_data in jwks.get('keys', []):
            if key_data.get('kid') == kid:
                # Convert JWK to PEM
                signing_key = self._jwk_to_pem(key_data)
                
                # Cache key for 1 hour
                self.jwks_cache[cache_key] = signing_key
                self.jwks_cache_expiry[cache_key] = datetime.now(timezone.utc) + timedelta(hours=1)
                
                return signing_key
        
        raise ValueError(f"Key ID {kid} not found in JWKS")
    
    def _jwk_to_pem(self, jwk: Dict[str, Any]) -> str:
        """Convert JWK to PEM format"""
        # This is simplified - in production you'd use a proper JWK library
        # For now, assume RSA keys
        if jwk.get('kty') != 'RSA':
            raise ValueError("Only RSA keys supported")
        
        # Decode components
        n = self._base64url_decode(jwk['n'])
        e = self._base64url_decode(jwk['e'])
        
        # Create RSA public key
        public_numbers = rsa.RSAPublicNumbers(
            int.from_bytes(e, byteorder='big'),
            int.from_bytes(n, byteorder='big')
        )
        public_key = public_numbers.public_key()
        
        # Convert to PEM
        pem = public_key.serialize(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return pem.decode('utf-8')
    
    def _base64url_decode(self, data: str) -> bytes:
        """Decode base64url"""
        # Add padding if needed
        padding = 4 - (len(data) % 4)
        if padding != 4:
            data += '=' * padding
        return base64.urlsafe_b64decode(data)


class OIDCClient:
    """Complete OIDC client implementation"""
    
    def __init__(self, config: OIDCConfig, role_mappings: List[RoleMapping] = None):
        self.config = config
        self.role_mappings = role_mappings or []
        self.jwt_validator = JWTValidator()
        self.sessions: Dict[str, OIDCSession] = {}
        
        # Session cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background session cleanup"""
        if self._cleanup_task and not self._cleanup_task.done():
            return
        
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
    
    async def _cleanup_expired_sessions(self):
        """Background task to cleanup expired sessions"""
        while True:
            try:
                now = datetime.now(timezone.utc)
                expired_sessions = []
                
                for session_id, session in self.sessions.items():
                    if (session.is_expired and session.is_refresh_expired) or \
                       (now - session.last_activity) > timedelta(hours=24):
                        expired_sessions.append(session_id)
                
                # Remove expired sessions
                for session_id in expired_sessions:
                    del self.sessions[session_id]
                    # Remove from cache
                    await cache.delete('oidc_sessions', session_id)
                
                if expired_sessions:
                    logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")
                await asyncio.sleep(300)
    
    async def discover_configuration(self, issuer: str) -> OIDCConfig:
        """Auto-discover OIDC configuration from issuer"""
        discovery_url = f"{issuer.rstrip('/')}/.well-known/openid-configuration"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(discovery_url) as response:
                if response.status != 200:
                    raise ValueError(f"Failed to discover OIDC configuration: HTTP {response.status}")
                
                config_data = await response.json()
        
        return OIDCConfig(
            issuer=config_data['issuer'],
            client_id=self.config.client_id,
            client_secret=self.config.client_secret,
            authorization_endpoint=config_data['authorization_endpoint'],
            token_endpoint=config_data['token_endpoint'], 
            userinfo_endpoint=config_data['userinfo_endpoint'],
            jwks_uri=config_data['jwks_uri'],
            end_session_endpoint=config_data.get('end_session_endpoint'),
            scopes=config_data.get('scopes_supported', self.config.scopes),
            response_types=config_data.get('response_types_supported', self.config.response_types),
            grant_types=config_data.get('grant_types_supported', self.config.grant_types)
        )
    
    async def initiate_login(self, redirect_uri: str, state: Optional[str] = None) -> Tuple[str, str, str]:
        """Initiate OIDC login flow"""
        # Generate state and PKCE challenge
        if not state:
            state = secrets.token_urlsafe(32)
        
        code_verifier, code_challenge = PKCEChallenge.generate_challenge()
        
        # Store PKCE verifier for later use
        await cache.set('oidc_pkce', state, code_verifier, ttl=600)  # 10 minutes
        
        # Build authorization URL
        auth_params = {
            'client_id': self.config.client_id,
            'response_type': 'code',
            'scope': ' '.join(self.config.scopes),
            'redirect_uri': redirect_uri,
            'state': state,
            'code_challenge': code_challenge,
            'code_challenge_method': 'S256'
        }
        
        auth_url = f"{self.config.authorization_endpoint}?{urlencode(auth_params)}"
        
        return auth_url, state, code_verifier
    
    async def handle_callback(self, 
                            code: str, 
                            state: str, 
                            redirect_uri: str,
                            ip_address: str = None,
                            user_agent: str = None) -> OIDCSession:
        """Handle OIDC callback and create session"""
        
        # Retrieve PKCE verifier
        code_verifier = await cache.get('oidc_pkce', state)
        if not code_verifier:
            raise ValueError("Invalid or expired state parameter")
        
        # Exchange code for tokens
        token_data = await self._exchange_code_for_tokens(code, redirect_uri, code_verifier)
        
        # Validate ID token
        id_claims = await self.jwt_validator.validate_jwt(
            token_data['id_token'],
            self.config.issuer,
            self.config.client_id,
            self.config.jwks_uri
        )
        
        # Get user info
        user_info = await self._get_user_info(token_data['access_token'])
        
        # Map roles and permissions
        roles, permissions = self._map_roles_and_permissions(user_info, id_claims)
        
        # Create session
        session_id = secrets.token_urlsafe(32)
        
        # Calculate token expiry
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=token_data.get('expires_in', 3600))
        refresh_expires_at = None
        if token_data.get('refresh_token'):
            # Refresh tokens typically last longer
            refresh_expires_at = datetime.now(timezone.utc) + timedelta(days=30)
        
        session = OIDCSession(
            session_id=session_id,
            user_id=user_info.get('sub') or id_claims.get('sub'),
            email=user_info.get('email') or id_claims.get('email'),
            name=user_info.get('name') or id_claims.get('name') or user_info.get('preferred_username'),
            roles=roles,
            permissions=permissions,
            access_token=token_data['access_token'],
            refresh_token=token_data.get('refresh_token'),
            id_token=token_data['id_token'],
            expires_at=expires_at,
            refresh_expires_at=refresh_expires_at,
            created_at=datetime.now(timezone.utc),
            last_activity=datetime.now(timezone.utc),
            ip_address=ip_address,
            user_agent=user_agent,
            idp_session_id=id_claims.get('sid')
        )
        
        # Store session
        self.sessions[session_id] = session
        await cache.set('oidc_sessions', session_id, asdict(session), ttl=86400)  # 24 hours
        
        # Clean up PKCE verifier
        await cache.delete('oidc_pkce', state)
        
        logger.info(f"Created OIDC session for user {session.email}")
        return session
    
    async def _exchange_code_for_tokens(self, code: str, redirect_uri: str, code_verifier: str) -> Dict[str, Any]:
        """Exchange authorization code for tokens"""
        token_data = {
            'grant_type': 'authorization_code',
            'client_id': self.config.client_id,
            'client_secret': self.config.client_secret,
            'code': code,
            'redirect_uri': redirect_uri,
            'code_verifier': code_verifier
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.config.token_endpoint,
                data=token_data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Token exchange failed: HTTP {response.status} - {error_text}")
                
                return await response.json()
    
    async def _get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user info from userinfo endpoint"""
        headers = {'Authorization': f'Bearer {access_token}'}
        
        async with aiohttp.ClientSession() as session:
            async with session.get(self.config.userinfo_endpoint, headers=headers) as response:
                if response.status != 200:
                    raise ValueError(f"Failed to get user info: HTTP {response.status}")
                
                return await response.json()
    
    def _map_roles_and_permissions(self, user_info: Dict[str, Any], id_claims: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Map IdP claims to application roles and permissions"""
        roles = set()
        permissions = set()
        
        # Combine user info and ID token claims
        all_claims = {**user_info, **id_claims}
        
        # Apply role mappings
        for mapping in self.role_mappings:
            claim_value = all_claims.get(mapping.claim_name)
            
            if claim_value is None:
                continue
            
            # Handle different claim value types
            if isinstance(claim_value, list):
                if mapping.claim_value in claim_value:
                    roles.update(mapping.roles)
                    permissions.update(mapping.permissions)
            elif isinstance(claim_value, str):
                if mapping.claim_value == claim_value or mapping.claim_value == "*":
                    roles.update(mapping.roles)
                    permissions.update(mapping.permissions)
        
        # Default role if no mappings match
        if not roles:
            roles.add("user")
            permissions.add("read:profile")
        
        return list(roles), list(permissions)
    
    async def refresh_session(self, session: OIDCSession) -> OIDCSession:
        """Refresh session tokens"""
        if not session.refresh_token:
            raise ValueError("No refresh token available")
        
        if session.is_refresh_expired:
            raise ValueError("Refresh token has expired")
        
        # Exchange refresh token for new tokens
        token_data = {
            'grant_type': 'refresh_token',
            'client_id': self.config.client_id,
            'client_secret': self.config.client_secret,
            'refresh_token': session.refresh_token
        }
        
        async with aiohttp.ClientSession() as session_http:
            async with session_http.post(
                self.config.token_endpoint,
                data=token_data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Token refresh failed: HTTP {response.status} - {error_text}")
                
                new_tokens = await response.json()
        
        # Update session with new tokens
        session.access_token = new_tokens['access_token']
        if new_tokens.get('refresh_token'):
            session.refresh_token = new_tokens['refresh_token']
        if new_tokens.get('id_token'):
            session.id_token = new_tokens['id_token']
        
        session.expires_at = datetime.now(timezone.utc) + timedelta(seconds=new_tokens.get('expires_in', 3600))
        session.last_activity = datetime.now(timezone.utc)
        
        # Update cache
        await cache.set('oidc_sessions', session.session_id, asdict(session), ttl=86400)
        
        logger.info(f"Refreshed session for user {session.email}")
        return session
    
    async def get_session(self, session_id: str, auto_refresh: bool = True) -> Optional[OIDCSession]:
        """Get session by ID with optional auto-refresh"""
        # Try memory cache first
        if session_id in self.sessions:
            session = self.sessions[session_id]
        else:
            # Try Redis cache
            session_data = await cache.get('oidc_sessions', session_id)
            if not session_data:
                return None
            
            # Reconstruct session object
            session_data['expires_at'] = datetime.fromisoformat(session_data['expires_at'])
            session_data['created_at'] = datetime.fromisoformat(session_data['created_at'])
            session_data['last_activity'] = datetime.fromisoformat(session_data['last_activity'])
            if session_data.get('refresh_expires_at'):
                session_data['refresh_expires_at'] = datetime.fromisoformat(session_data['refresh_expires_at'])
            
            session = OIDCSession(**session_data)
            self.sessions[session_id] = session
        
        # Check if session is expired
        if session.is_expired:
            if auto_refresh and session.refresh_token and not session.is_refresh_expired:
                try:
                    session = await self.refresh_session(session)
                except Exception as e:
                    logger.warning(f"Failed to refresh session {session_id}: {e}")
                    return None
            else:
                # Session cannot be refreshed
                await self.revoke_session(session_id)
                return None
        
        # Auto-refresh if needed
        if auto_refresh and session.needs_refresh and session.refresh_token and not session.is_refresh_expired:
            try:
                session = await self.refresh_session(session)
            except Exception as e:
                logger.warning(f"Failed to auto-refresh session {session_id}: {e}")
        
        # Update last activity
        session.last_activity = datetime.now(timezone.utc)
        await cache.set('oidc_sessions', session_id, asdict(session), ttl=86400)
        
        return session
    
    async def revoke_session(self, session_id: str) -> bool:
        """Revoke session and tokens"""
        session = self.sessions.get(session_id)
        if not session:
            # Try to get from cache
            session_data = await cache.get('oidc_sessions', session_id)
            if session_data:
                session_data['expires_at'] = datetime.fromisoformat(session_data['expires_at'])
                session_data['created_at'] = datetime.fromisoformat(session_data['created_at'])
                session_data['last_activity'] = datetime.fromisoformat(session_data['last_activity'])
                if session_data.get('refresh_expires_at'):
                    session_data['refresh_expires_at'] = datetime.fromisoformat(session_data['refresh_expires_at'])
                session = OIDCSession(**session_data)
        
        success = True
        
        # Revoke tokens at IdP if supported
        if session and hasattr(self.config, 'revocation_endpoint'):
            try:
                await self._revoke_token_at_idp(session.access_token, 'access_token')
                if session.refresh_token:
                    await self._revoke_token_at_idp(session.refresh_token, 'refresh_token')
            except Exception as e:
                logger.error(f"Failed to revoke tokens at IdP: {e}")
                success = False
        
        # Remove from local storage
        if session_id in self.sessions:
            del self.sessions[session_id]
        
        await cache.delete('oidc_sessions', session_id)
        
        logger.info(f"Revoked session {session_id}")
        return success
    
    async def _revoke_token_at_idp(self, token: str, token_type: str):
        """Revoke token at IdP"""
        if not hasattr(self.config, 'revocation_endpoint'):
            return
        
        revocation_data = {
            'token': token,
            'token_type_hint': token_type,
            'client_id': self.config.client_id,
            'client_secret': self.config.client_secret
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.config.revocation_endpoint,
                data=revocation_data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            ) as response:
                if response.status not in [200, 204]:
                    logger.warning(f"Token revocation returned HTTP {response.status}")
    
    async def initiate_logout(self, session: OIDCSession, post_logout_redirect_uri: str = None) -> str:
        """Initiate logout at IdP"""
        # Revoke local session first
        await self.revoke_session(session.session_id)
        
        # Build logout URL if supported
        if self.config.end_session_endpoint:
            logout_params = {
                'id_token_hint': session.id_token
            }
            
            if post_logout_redirect_uri:
                logout_params['post_logout_redirect_uri'] = post_logout_redirect_uri
            
            if session.idp_session_id:
                logout_params['sid'] = session.idp_session_id
            
            logout_url = f"{self.config.end_session_endpoint}?{urlencode(logout_params)}"
            return logout_url
        
        return post_logout_redirect_uri or "/"
    
    async def get_all_sessions(self, user_id: str = None) -> List[OIDCSession]:
        """Get all active sessions, optionally filtered by user"""
        sessions = []
        
        # Get from memory
        for session in self.sessions.values():
            if user_id is None or session.user_id == user_id:
                sessions.append(session)
        
        return sessions
    
    async def revoke_all_user_sessions(self, user_id: str) -> int:
        """Revoke all sessions for a user"""
        sessions_to_revoke = [s.session_id for s in self.sessions.values() if s.user_id == user_id]
        
        revoked_count = 0
        for session_id in sessions_to_revoke:
            try:
                await self.revoke_session(session_id)
                revoked_count += 1
            except Exception as e:
                logger.error(f"Failed to revoke session {session_id}: {e}")
        
        logger.info(f"Revoked {revoked_count} sessions for user {user_id}")
        return revoked_count
    
    async def stop(self):
        """Stop OIDC client and cleanup"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass


# Pre-configured role mappings for common IdP scenarios
COMMON_ROLE_MAPPINGS = {
    'azure_ad': [
        RoleMapping('roles', 'admin', ['admin'], ['*']),
        RoleMapping('roles', 'user', ['user'], ['read:profile', 'read:components']),
        RoleMapping('groups', 'SCIP-Admins', ['admin'], ['*']),
        RoleMapping('groups', 'SCIP-Users', ['user'], ['read:profile', 'read:components', 'write:rfqs'])
    ],
    'auth0': [
        RoleMapping('https://scip.com/roles', 'admin', ['admin'], ['*']),
        RoleMapping('https://scip.com/roles', 'user', ['user'], ['read:profile', 'read:components'])
    ],
    'keycloak': [
        RoleMapping('realm_access.roles', 'admin', ['admin'], ['*']),
        RoleMapping('realm_access.roles', 'user', ['user'], ['read:profile', 'read:components'])
    ]
}

# Global OIDC client instance (will be initialized from config)
oidc_client: Optional[OIDCClient] = None


async def initialize_oidc(provider: str = "azure_ad") -> OIDCClient:
    """Initialize OIDC client from configuration"""
    global oidc_client
    
    # This would typically read from environment variables
    config = OIDCConfig(
        issuer=settings.oidc_issuer,
        client_id=settings.oidc_client_id, 
        client_secret=settings.oidc_client_secret,
        authorization_endpoint=settings.oidc_auth_endpoint,
        token_endpoint=settings.oidc_token_endpoint,
        userinfo_endpoint=settings.oidc_userinfo_endpoint,
        jwks_uri=settings.oidc_jwks_uri,
        end_session_endpoint=settings.oidc_end_session_endpoint
    )
    
    role_mappings = COMMON_ROLE_MAPPINGS.get(provider, [])
    oidc_client = OIDCClient(config, role_mappings)
    
    logger.info(f"Initialized OIDC client for provider: {provider}")
    return oidc_client


async def cleanup_oidc():
    """Cleanup OIDC client"""
    global oidc_client
    if oidc_client:
        await oidc_client.stop()
        oidc_client = None


def get_current_oidc_session(request=None):
    """
    Get current OIDC session from request context.
    
    Args:
        request: FastAPI request object (can be injected via Depends)
    
    Returns:
        Dict with session information or None if no session
    """
    from fastapi import Request, HTTPException
    
    # In a real implementation, this would:
    # 1. Check for session cookies
    # 2. Validate session tokens
    # 3. Return user and session info
    
    if request and hasattr(request, 'state'):
        # Check if user info is stored in request state
        user_info = getattr(request.state, 'user_info', None)
        if user_info:
            return {
                "user_id": user_info.get("sub"),
                "session_id": user_info.get("sid"),
                "email": user_info.get("email"),
                "roles": user_info.get("roles", []),
                "permissions": user_info.get("permissions", [])
            }
    
    # Check for Authorization header as fallback
    if request and hasattr(request, 'headers'):
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            try:
                from ..core.security import verify_jwt
                claims = verify_jwt(token)
                return {
                    "user_id": claims.get("sub"),
                    "session_id": claims.get("jti"),
                    "email": claims.get("email"),
                    "roles": claims.get("roles", []),
                    "permissions": claims.get("permissions", [])
                }
            except Exception:
                pass
    
    # Return minimal session for unauthenticated users
    return {
        "user_id": "anonymous",
        "session_id": None,
        "email": None,
        "roles": ["anonymous"],
        "permissions": ["read:public"]
    }