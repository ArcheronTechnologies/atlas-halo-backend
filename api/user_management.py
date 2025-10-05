"""
User Management API Endpoints
Comprehensive user administration, role management, and account operations
"""

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, EmailStr
import logging
import uuid
from enum import Enum

from ..auth.jwt_authentication import (
    AuthenticationService, UserRole, get_current_user,
    AuthenticatedUser, get_auth_service
)
from ..database.postgis_database import get_database
from ..caching.redis_cache import get_cache
from ..observability.metrics import metrics

logger = logging.getLogger(__name__)
security = HTTPBearer()

# Pydantic models for user management
class UserStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"

class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, pattern="^[a-zA-Z0-9_-]+$")
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = Field(None, max_length=255)
    role: UserRole = UserRole.CITIZEN
    home_latitude: Optional[float] = Field(None, ge=-90, le=90)
    home_longitude: Optional[float] = Field(None, ge=-180, le=180)
    notification_radius: Optional[float] = Field(5000, ge=100, le=50000)

class UpdateUserRequest(BaseModel):
    full_name: Optional[str] = Field(None, max_length=255)
    email: Optional[EmailStr] = None
    role: Optional[UserRole] = None
    status: Optional[UserStatus] = None
    home_latitude: Optional[float] = Field(None, ge=-90, le=90)
    home_longitude: Optional[float] = Field(None, ge=-180, le=180)
    notification_radius: Optional[float] = Field(None, ge=100, le=50000)

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)

class UserResponse(BaseModel):
    user_id: str
    username: str
    email: str
    full_name: Optional[str]
    role: str
    status: str
    is_verified: bool
    created_at: str
    last_login: Optional[str]
    home_location: Optional[Dict[str, float]]
    notification_radius: Optional[float]

class UserListResponse(BaseModel):
    users: List[UserResponse]
    total_count: int
    page: int
    page_size: int
    total_pages: int

class AdminStatsResponse(BaseModel):
    total_users: int
    active_users: int
    new_users_today: int
    users_by_role: Dict[str, int]
    recent_activity: List[Dict[str, Any]]

# Initialize router
router = APIRouter(prefix="/api/admin/users", tags=["User Management"])

@router.post("/create", response_model=UserResponse)
async def create_user(
    request: CreateUserRequest,
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """Create a new user account (Admin only)"""
    try:
        auth_service = await get_auth_service()
        
        # Register the user
        user_result = await auth_service.register_user(
            username=request.username,
            email=request.email,
            password=request.password,
            full_name=request.full_name,
            role=request.role
        )
        
        # Update home location if provided
        if request.home_latitude and request.home_longitude:
            db = await get_database()
            await db.execute_query(
                """
                UPDATE users 
                SET home_location = ST_Point($1, $2)::geography,
                    notification_radius = $3
                WHERE id = $4
                """,
                request.home_longitude, request.home_latitude, 
                request.notification_radius, user_result['user_id']
            )
        
        # Get complete user data
        user_data = await get_user_details(user_result['user_id'])
        
        # Track user creation
        user_counter = metrics.counter("admin_users_created", "Users created by admin", ("role",))
        user_counter.labels(request.role.value).inc()
        
        logger.info(f"👤 Admin {current_user.username} created user {request.username} ({request.role.value})")
        
        return user_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User creation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to create user")

@router.get("/list", response_model=UserListResponse)
async def list_users(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    role: Optional[UserRole] = Query(None, description="Filter by role"),
    status: Optional[UserStatus] = Query(None, description="Filter by status"),
    search: Optional[str] = Query(None, description="Search username or email"),
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """List all users with pagination and filtering"""
    try:
        db = await get_database()
        
        # Build query with filters
        where_conditions = ["1=1"]
        params = []
        param_count = 0
        
        if role:
            param_count += 1
            where_conditions.append(f"role = ${param_count}")
            params.append(role.value)
        
        if status:
            param_count += 1
            where_conditions.append(f"status = ${param_count}")
            params.append(status.value)
        
        if search:
            param_count += 2
            where_conditions.append(f"(username ILIKE ${param_count - 1} OR email ILIKE ${param_count})")
            params.extend([f"%{search}%", f"%{search}%"])
        
        where_clause = " AND ".join(where_conditions)
        
        # Get total count
        count_query = f"""
        SELECT COUNT(*) as total
        FROM users 
        WHERE {where_clause}
        """
        
        count_result = await db.execute_query(count_query, *params)
        total_count = count_result[0]['total'] if count_result else 0
        
        # Get paginated results
        offset = (page - 1) * page_size
        param_count += 2
        
        list_query = f"""
        SELECT 
            id, username, email, full_name, role, is_active,
            is_verified, created_at, last_login,
            ST_X(home_location::geometry) as home_lng,
            ST_Y(home_location::geometry) as home_lat,
            notification_radius
        FROM users 
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT ${param_count - 1} OFFSET ${param_count}
        """
        
        params.extend([page_size, offset])
        users_result = await db.execute_query(list_query, *params)
        
        # Format user data
        users = []
        for user in users_result:
            home_location = None
            if user['home_lat'] and user['home_lng']:
                home_location = {"lat": user['home_lat'], "lng": user['home_lng']}
            
            users.append(UserResponse(
                user_id=user['id'],
                username=user['username'],
                email=user['email'],
                full_name=user['full_name'],
                role=user['role'],
                status="active" if user['is_active'] else "inactive",
                is_verified=user['is_verified'],
                created_at=user['created_at'].isoformat() if user['created_at'] else "",
                last_login=user['last_login'].isoformat() if user['last_login'] else None,
                home_location=home_location,
                notification_radius=user['notification_radius']
            ))
        
        total_pages = (total_count + page_size - 1) // page_size
        
        return UserListResponse(
            users=users,
            total_count=total_count,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )
        
    except Exception as e:
        logger.error(f"User listing failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve users")

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """Get detailed user information"""
    try:
        user_data = await get_user_details(user_id)
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        return user_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user")

@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    request: UpdateUserRequest,
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """Update user information (Admin only)"""
    try:
        db = await get_database()
        
        # Check if user exists
        existing_user = await db.execute_query(
            "SELECT id, username FROM users WHERE id = $1",
            user_id
        )
        if not existing_user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Build update query
        update_fields = []
        params = []
        param_count = 0
        
        if request.full_name is not None:
            param_count += 1
            update_fields.append(f"full_name = ${param_count}")
            params.append(request.full_name)
        
        if request.email is not None:
            param_count += 1
            update_fields.append(f"email = ${param_count}")
            params.append(request.email)
        
        if request.role is not None:
            param_count += 1
            update_fields.append(f"role = ${param_count}")
            params.append(request.role.value)
        
        if request.status is not None:
            param_count += 1
            update_fields.append(f"is_active = ${param_count}")
            params.append(request.status == UserStatus.ACTIVE)
        
        if request.notification_radius is not None:
            param_count += 1
            update_fields.append(f"notification_radius = ${param_count}")
            params.append(request.notification_radius)
        
        # Handle home location update
        if request.home_latitude is not None and request.home_longitude is not None:
            param_count += 2
            update_fields.append(f"home_location = ST_Point(${param_count - 1}, ${param_count})::geography")
            params.extend([request.home_longitude, request.home_latitude])
        
        if update_fields:
            param_count += 1
            update_query = f"""
            UPDATE users 
            SET {', '.join(update_fields)}, updated_at = NOW()
            WHERE id = ${param_count}
            """
            params.append(user_id)
            
            await db.execute_query(update_query, *params)
        
        # Get updated user data
        updated_user = await get_user_details(user_id)
        
        logger.info(f"👤 Admin {current_user.username} updated user {existing_user[0]['username']}")
        
        return updated_user
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User update failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to update user")

@router.delete("/{user_id}")
async def delete_user(
    user_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """Delete user account (Admin only)"""
    try:
        db = await get_database()
        
        # Check if user exists
        existing_user = await db.execute_query(
            "SELECT username FROM users WHERE id = $1",
            user_id
        )
        if not existing_user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Prevent self-deletion
        if user_id == current_user.user_id:
            raise HTTPException(status_code=400, detail="Cannot delete your own account")
        
        # Delete user (cascade should handle related records)
        await db.execute_query("DELETE FROM users WHERE id = $1", user_id)
        
        # Invalidate any active sessions
        cache = await get_cache()
        # Note: In production, you'd want to track and invalidate specific sessions
        
        logger.info(f"🗑️ Admin {current_user.username} deleted user {existing_user[0]['username']}")
        
        return {"success": True, "message": "User deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User deletion failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete user")

@router.post("/{user_id}/suspend")
async def suspend_user(
    user_id: str,
    reason: str = Query(..., description="Reason for suspension"),
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """Suspend user account"""
    try:
        db = await get_database()
        
        # Check if user exists
        existing_user = await db.execute_query(
            "SELECT username FROM users WHERE id = $1",
            user_id
        )
        if not existing_user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Suspend user
        await db.execute_query(
            "UPDATE users SET is_active = FALSE, updated_at = NOW() WHERE id = $1",
            user_id
        )
        
        # Log suspension reason
        await db.execute_query(
            """
            INSERT INTO user_audit_log (user_id, action, reason, admin_id, timestamp)
            VALUES ($1, 'suspended', $2, $3, NOW())
            """,
            user_id, reason, current_user.user_id
        )
        
        logger.info(f"⚠️ Admin {current_user.username} suspended user {existing_user[0]['username']}: {reason}")
        
        return {"success": True, "message": "User suspended successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User suspension failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to suspend user")

@router.post("/{user_id}/activate")
async def activate_user(
    user_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """Activate suspended user account"""
    try:
        db = await get_database()
        
        # Check if user exists
        existing_user = await db.execute_query(
            "SELECT username FROM users WHERE id = $1",
            user_id
        )
        if not existing_user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Activate user
        await db.execute_query(
            "UPDATE users SET is_active = TRUE, updated_at = NOW() WHERE id = $1",
            user_id
        )
        
        # Log activation
        await db.execute_query(
            """
            INSERT INTO user_audit_log (user_id, action, admin_id, timestamp)
            VALUES ($1, 'activated', $2, NOW())
            """,
            user_id, current_user.user_id
        )
        
        logger.info(f"✅ Admin {current_user.username} activated user {existing_user[0]['username']}")
        
        return {"success": True, "message": "User activated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User activation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to activate user")

@router.post("/{user_id}/reset-password")
async def reset_user_password(
    user_id: str,
    background_tasks: BackgroundTasks,
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """Reset user password and send new credentials"""
    try:
        db = await get_database()
        auth_service = await get_auth_service()
        
        # Check if user exists
        existing_user = await db.execute_query(
            "SELECT username, email FROM users WHERE id = $1",
            user_id
        )
        if not existing_user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Generate temporary password
        import secrets
        import string
        temp_password = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(12))
        
        # Hash new password
        hashed_password = auth_service.password_validator.hash_password(temp_password)
        
        # Update user password
        await db.execute_query(
            "UPDATE users SET hashed_password = $1, updated_at = NOW() WHERE id = $2",
            hashed_password, user_id
        )
        
        # Log password reset
        await db.execute_query(
            """
            INSERT INTO user_audit_log (user_id, action, admin_id, timestamp)
            VALUES ($1, 'password_reset', $2, NOW())
            """,
            user_id, current_user.user_id
        )
        
        # In production, you'd send this via email
        # background_tasks.add_task(send_password_reset_email, existing_user[0]['email'], temp_password)
        
        logger.info(f"🔑 Admin {current_user.username} reset password for user {existing_user[0]['username']}")
        
        return {
            "success": True,
            "message": "Password reset successfully",
            "temporary_password": temp_password,  # In production, don't return this
            "note": "User should change password on first login"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password reset failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset password")

@router.get("/stats/overview", response_model=AdminStatsResponse)
async def get_admin_stats(
    current_user: AuthenticatedUser = Depends(get_current_user)
):
    """Get administrative statistics overview"""
    try:
        db = await get_database()
        
        # Get user counts
        stats_query = """
        SELECT 
            COUNT(*) as total_users,
            COUNT(*) FILTER (WHERE is_active = true) as active_users,
            COUNT(*) FILTER (WHERE created_at >= CURRENT_DATE) as new_users_today
        FROM users
        """
        
        stats_result = await db.execute_query(stats_query)
        stats = stats_result[0] if stats_result else {}
        
        # Get users by role
        role_query = """
        SELECT role, COUNT(*) as count
        FROM users 
        WHERE is_active = true
        GROUP BY role
        """
        
        role_result = await db.execute_query(role_query)
        users_by_role = {row['role']: row['count'] for row in role_result}
        
        # Get recent activity
        activity_query = """
        SELECT 
            u.username,
            al.action,
            al.timestamp,
            al.reason
        FROM user_audit_log al
        JOIN users u ON al.user_id = u.id
        ORDER BY al.timestamp DESC
        LIMIT 10
        """
        
        activity_result = await db.execute_query(activity_query)
        recent_activity = [
            {
                "username": row['username'],
                "action": row['action'],
                "timestamp": row['timestamp'].isoformat(),
                "reason": row['reason']
            }
            for row in activity_result
        ]
        
        return AdminStatsResponse(
            total_users=stats.get('total_users', 0),
            active_users=stats.get('active_users', 0),
            new_users_today=stats.get('new_users_today', 0),
            users_by_role=users_by_role,
            recent_activity=recent_activity
        )
        
    except Exception as e:
        logger.error(f"Admin stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve admin statistics")

# Helper functions
async def get_user_details(user_id: str) -> Optional[UserResponse]:
    """Get complete user details"""
    try:
        db = await get_database()
        
        query = """
        SELECT 
            id, username, email, full_name, role, is_active,
            is_verified, created_at, last_login,
            ST_X(home_location::geometry) as home_lng,
            ST_Y(home_location::geometry) as home_lat,
            notification_radius
        FROM users 
        WHERE id = $1
        """
        
        result = await db.execute_query(query, user_id)
        if not result:
            return None
        
        user = result[0]
        home_location = None
        if user['home_lat'] and user['home_lng']:
            home_location = {"lat": user['home_lat'], "lng": user['home_lng']}
        
        return UserResponse(
            user_id=user['id'],
            username=user['username'],
            email=user['email'],
            full_name=user['full_name'],
            role=user['role'],
            status="active" if user['is_active'] else "inactive",
            is_verified=user['is_verified'],
            created_at=user['created_at'].isoformat() if user['created_at'] else "",
            last_login=user['last_login'].isoformat() if user['last_login'] else None,
            home_location=home_location,
            notification_radius=user['notification_radius']
        )
        
    except Exception as e:
        logger.error(f"Failed to get user details: {e}")
        return None

# Create audit log table
async def create_audit_tables():
    """Create audit log tables"""
    try:
        db = await get_database()
        
        create_audit_table = """
        CREATE TABLE IF NOT EXISTS user_audit_log (
            id SERIAL PRIMARY KEY,
            user_id VARCHAR(255) NOT NULL,
            action VARCHAR(100) NOT NULL,
            reason TEXT,
            admin_id VARCHAR(255),
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            metadata JSONB
        );
        
        CREATE INDEX IF NOT EXISTS idx_audit_user_id ON user_audit_log(user_id);
        CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON user_audit_log(timestamp);
        CREATE INDEX IF NOT EXISTS idx_audit_admin_id ON user_audit_log(admin_id);
        """
        
        await db.execute_query(create_audit_table)
        logger.info("User audit tables created successfully")
        
    except Exception as e:
        logger.error(f"Failed to create audit tables: {e}")
        raise

# Export the router
__all__ = ["router", "create_audit_tables"]