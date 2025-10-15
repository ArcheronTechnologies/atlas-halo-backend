"""
OIDC Authentication Router

Provides endpoints for OpenID Connect authentication including login, callback,
refresh, logout, and session management.
"""

from fastapi import APIRouter, HTTPException, Request, Response, Query, Depends
from fastapi.responses import RedirectResponse
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timezone

from ..auth.oidc import oidc_client, initialize_oidc, OIDCSession
from ..core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/login")
async def oidc_login(
    request: Request,
    redirect_uri: str = Query(..., description="Callback URL after authentication"),
    state: Optional[str] = Query(None, description="State parameter for CSRF protection")
):
    """Initiate OIDC login flow"""
    try:
        if not oidc_client:
            await initialize_oidc(settings.oidc_provider)
        
        if not oidc_client.config.issuer:
            raise HTTPException(
                status_code=500,
                detail="OIDC not configured. Please set OIDC_* environment variables."
            )
        
        # Generate authorization URL
        auth_url, state_value, code_verifier = await oidc_client.initiate_login(
            redirect_uri=redirect_uri,
            state=state
        )
        
        return {
            "authorization_url": auth_url,
            "state": state_value
        }
        
    except Exception as e:
        logger.error(f"Error initiating OIDC login: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to initiate login"
        )


@router.get("/callback")
async def oidc_callback(
    request: Request,
    response: Response,
    code: str = Query(..., description="Authorization code from IdP"),
    state: str = Query(..., description="State parameter"),
    redirect_uri: str = Query(..., description="Callback URL"),
    error: Optional[str] = Query(None, description="Error from IdP"),
    error_description: Optional[str] = Query(None, description="Error description")
):
    """Handle OIDC callback from IdP"""
    try:
        if error:
            logger.error(f"OIDC callback error: {error} - {error_description}")
            raise HTTPException(
                status_code=400,
                detail=f"Authentication failed: {error_description or error}"
            )
        
        if not oidc_client:
            raise HTTPException(
                status_code=500,
                detail="OIDC client not initialized"
            )
        
        # Get client info
        client_ip = getattr(request.client, 'host', None) if request.client else None
        user_agent = request.headers.get('user-agent')
        
        # Handle callback and create session
        session = await oidc_client.handle_callback(
            code=code,
            state=state, 
            redirect_uri=redirect_uri,
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        # Set secure session cookie
        response.set_cookie(
            key="oidc_session",
            value=session.session_id,
            httponly=True,
            secure=True,  # HTTPS only in production
            samesite="lax",
            max_age=86400  # 24 hours
        )
        
        return {
            "success": True,
            "session_id": session.session_id,
            "user": {
                "id": session.user_id,
                "email": session.email,
                "name": session.name,
                "roles": session.roles,
                "permissions": session.permissions
            },
            "expires_at": session.expires_at.isoformat()
        }
        
    except ValueError as e:
        logger.error(f"OIDC callback validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error handling OIDC callback: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process authentication callback"
        )


@router.post("/refresh")
async def refresh_token(request: Request):
    """Refresh access token using refresh token"""
    try:
        if not oidc_client:
            raise HTTPException(status_code=500, detail="OIDC client not initialized")
        
        # Get session from cookie
        session_id = request.cookies.get("oidc_session")
        if not session_id:
            raise HTTPException(status_code=401, detail="No active session")
        
        # Get current session
        session = await oidc_client.get_session(session_id, auto_refresh=False)
        if not session:
            raise HTTPException(status_code=401, detail="Invalid or expired session")
        
        # Refresh the session
        refreshed_session = await oidc_client.refresh_session(session)
        
        return {
            "success": True,
            "access_token": refreshed_session.access_token,
            "expires_at": refreshed_session.expires_at.isoformat(),
            "user": {
                "id": refreshed_session.user_id,
                "email": refreshed_session.email,
                "name": refreshed_session.name,
                "roles": refreshed_session.roles,
                "permissions": refreshed_session.permissions
            }
        }
        
    except ValueError as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error refreshing token: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to refresh token"
        )


@router.get("/session")
async def get_current_session(request: Request):
    """Get current session information"""
    try:
        if not oidc_client:
            raise HTTPException(status_code=500, detail="OIDC client not initialized")
        
        # Get session from cookie
        session_id = request.cookies.get("oidc_session")
        if not session_id:
            raise HTTPException(status_code=401, detail="No active session")
        
        # Get current session (with auto-refresh)
        session = await oidc_client.get_session(session_id, auto_refresh=True)
        if not session:
            raise HTTPException(status_code=401, detail="Invalid or expired session")
        
        return {
            "session_id": session.session_id,
            "user": {
                "id": session.user_id,
                "email": session.email,
                "name": session.name,
                "roles": session.roles,
                "permissions": session.permissions
            },
            "expires_at": session.expires_at.isoformat(),
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "ip_address": session.ip_address,
            "needs_refresh": session.needs_refresh
        }
        
    except Exception as e:
        logger.error(f"Error getting session: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to get session information"
        )


@router.post("/logout") 
async def logout(
    request: Request,
    response: Response,
    post_logout_redirect_uri: Optional[str] = None
):
    """Logout user and revoke session"""
    try:
        if not oidc_client:
            raise HTTPException(status_code=500, detail="OIDC client not initialized")
        
        # Get session from cookie
        session_id = request.cookies.get("oidc_session")
        if not session_id:
            # No active session, just clear cookie
            response.delete_cookie("oidc_session")
            return {"success": True, "message": "No active session to logout"}
        
        # Get current session
        session = await oidc_client.get_session(session_id, auto_refresh=False)
        
        # Clear session cookie
        response.delete_cookie("oidc_session")
        
        logout_url = "/"
        if session:
            # Initiate logout at IdP
            logout_url = await oidc_client.initiate_logout(
                session,
                post_logout_redirect_uri
            )
        
        return {
            "success": True,
            "logout_url": logout_url,
            "message": "Session terminated"
        }
        
    except Exception as e:
        logger.error(f"Error during logout: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to logout"
        )


@router.delete("/sessions/{session_id}")
async def revoke_session(
    session_id: str,
    current_session: OIDCSession = Depends(get_current_oidc_session)
):
    """Revoke a specific session (admin only or own session)"""
    try:
        if not oidc_client:
            raise HTTPException(status_code=500, detail="OIDC client not initialized")
        
        # Check permissions
        if (session_id != current_session.session_id and 
            'admin' not in current_session.roles):
            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions to revoke other sessions"
            )
        
        success = await oidc_client.revoke_session(session_id)
        
        return {
            "success": success,
            "session_id": session_id,
            "message": "Session revoked"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error revoking session {session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to revoke session"
        )


@router.get("/sessions")
async def list_sessions(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    current_session: OIDCSession = Depends(get_current_oidc_session)
):
    """List active sessions (admin only or own sessions)"""
    try:
        if not oidc_client:
            raise HTTPException(status_code=500, detail="OIDC client not initialized")
        
        # Check permissions
        if user_id and user_id != current_session.user_id and 'admin' not in current_session.roles:
            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions to view other user sessions"
            )
        
        # Default to current user if not admin
        if 'admin' not in current_session.roles:
            user_id = current_session.user_id
        
        sessions = await oidc_client.get_all_sessions(user_id)
        
        # Format response
        session_list = []
        for session in sessions:
            session_list.append({
                "session_id": session.session_id,
                "user_id": session.user_id,
                "email": session.email,
                "name": session.name,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "expires_at": session.expires_at.isoformat(),
                "ip_address": session.ip_address,
                "user_agent": session.user_agent[:100] if session.user_agent else None,
                "is_current": session.session_id == current_session.session_id
            })
        
        return {
            "sessions": session_list,
            "total": len(session_list)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to list sessions"
        )


@router.delete("/users/{user_id}/sessions")
async def revoke_all_user_sessions(
    user_id: str,
    current_session: OIDCSession = Depends(get_current_oidc_session)
):
    """Revoke all sessions for a user (admin only or own sessions)"""
    try:
        if not oidc_client:
            raise HTTPException(status_code=500, detail="OIDC client not initialized")
        
        # Check permissions
        if user_id != current_session.user_id and 'admin' not in current_session.roles:
            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions to revoke other user sessions"
            )
        
        revoked_count = await oidc_client.revoke_all_user_sessions(user_id)
        
        return {
            "success": True,
            "user_id": user_id,
            "revoked_sessions": revoked_count,
            "message": f"Revoked {revoked_count} sessions"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error revoking all sessions for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to revoke user sessions"
        )


@router.get("/configuration")
async def get_oidc_configuration():
    """Get OIDC client configuration (public info only)"""
    try:
        if not oidc_client:
            await initialize_oidc(settings.oidc_provider)
        
        if not oidc_client.config.issuer:
            return {
                "configured": False,
                "message": "OIDC not configured"
            }
        
        return {
            "configured": True,
            "issuer": oidc_client.config.issuer,
            "authorization_endpoint": oidc_client.config.authorization_endpoint,
            "scopes": oidc_client.config.scopes,
            "response_types": oidc_client.config.response_types,
            "provider": settings.oidc_provider
        }
        
    except Exception as e:
        logger.error(f"Error getting OIDC configuration: {e}")
        return {
            "configured": False,
            "error": str(e)
        }


# Dependency for getting current OIDC session
async def get_current_oidc_session(request: Request) -> OIDCSession:
    """Dependency to get current OIDC session"""
    if not oidc_client:
        raise HTTPException(status_code=500, detail="OIDC client not initialized")
    
    session_id = request.cookies.get("oidc_session")
    if not session_id:
        raise HTTPException(status_code=401, detail="No active session")
    
    session = await oidc_client.get_session(session_id, auto_refresh=True)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    
    return session


# Dependency for requiring specific roles
def require_oidc_roles(required_roles: List[str]):
    """Dependency factory for requiring specific OIDC roles"""
    async def dependency(session: OIDCSession = Depends(get_current_oidc_session)):
        if not any(role in session.roles for role in required_roles):
            raise HTTPException(
                status_code=403,
                detail=f"Required roles: {required_roles}"
            )
        return session
    return dependency


# Dependency for requiring specific permissions
def require_oidc_permissions(required_permissions: List[str]):
    """Dependency factory for requiring specific OIDC permissions"""
    async def dependency(session: OIDCSession = Depends(get_current_oidc_session)):
        if not any(perm in session.permissions for perm in required_permissions):
            raise HTTPException(
                status_code=403,
                detail=f"Required permissions: {required_permissions}"
            )
        return session
    return dependency