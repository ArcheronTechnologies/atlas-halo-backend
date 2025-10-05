"""
Authentication API endpoints
User registration, login, and token management
"""

from fastapi import APIRouter, HTTPException, Depends, status, Request
from pydantic import BaseModel, Field
from typing import Dict, Any
import logging

from ..auth.jwt_authentication import AuthenticationService, get_current_user
from ..auth.bankid_authentication import bankid_service
from ..database.mobile_database_manager import get_mobile_database, MobileDatabaseManager
from ..utils.validation import validate_email, validate_phone_number, validate_user_type
from ..security.rate_limiting import check_rate_limit
from ..security.input_validation import SecurityValidator, SecureBaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])


class UserRegistration(BaseModel):
    """User registration model"""
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: str = Field(..., description="Email address")
    password: str = Field(..., min_length=8, max_length=128, description="Password")
    user_type: str = Field(default="citizen", description="User type")
    phone_number: str = Field(None, description="Phone number (optional)")


class UserLogin(BaseModel):
    """User login model"""
    email: str = Field(..., description="Email address")
    password: str = Field(..., description="Password")


class TokenResponse(BaseModel):
    """Token response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 1800  # 30 minutes
    user: Dict[str, Any]


class BankIDStartRequest(BaseModel):
    """BankID authentication start request"""
    personal_number: str = Field(None, description="Swedish personal number (optional)")
    user_visible_data: str = Field(None, description="Message shown to user")


class BankIDStartResponse(BaseModel):
    """BankID authentication start response"""
    order_ref: str
    auto_start_token: str
    qr_start_token: str = ""
    qr_start_secret: str = ""


class BankIDCollectRequest(BaseModel):
    """BankID collect request"""
    order_ref: str = Field(..., description="Order reference from start request")


class BankIDCollectResponse(BaseModel):
    """BankID collect response"""
    status: str
    hint_code: str = ""
    order_ref: str
    user: Dict[str, Any] = None


@router.post("/register", response_model=TokenResponse)
async def register_user(
    user_data: UserRegistration,
    db: MobileDatabaseManager = Depends(get_mobile_database)
):
    """Register a new user account"""
    try:
        validate_email(user_data.email)
        validate_user_type(user_data.user_type)
        if user_data.phone_number:
            validate_phone_number(user_data.phone_number)

        existing_user = await db.get_user_by_email(user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        password_hash = AuthenticationService.hash_password(user_data.password)

        user_id = await db.create_user(
            username=user_data.username,
            email=user_data.email,
            password_hash=password_hash,
            user_type=user_data.user_type,
            phone_number=user_data.phone_number
        )

        user = await db.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user"
            )

        token_data = {"sub": str(user_id), "email": user_data.email, "type": user_data.user_type}
        access_token = AuthenticationService.create_access_token(token_data)
        refresh_token = AuthenticationService.create_refresh_token(token_data)

        await db.log_user_activity(
            user_id=user_id,
            activity_type="account_created",
            description="User account created",
            metadata={"user_type": user_data.user_type}
        )

        user_response = {k: v for k, v in user.items() if k != 'password_hash'}

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            user=user_response
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=TokenResponse)
async def login_user(
    login_data: UserLogin,
    db: MobileDatabaseManager = Depends(get_mobile_database)
):
    """User login with email and password"""
    try:
        user = await db.get_user_by_email(login_data.email)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )

        if not AuthenticationService.verify_password(login_data.password, user['password_hash']):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )

        if not user.get('is_active', False):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is disabled"
            )

        token_data = {
            "sub": str(user['id']),
            "email": user['email'],
            "type": user['user_type']
        }
        access_token = AuthenticationService.create_access_token(token_data)
        refresh_token = AuthenticationService.create_refresh_token(token_data)

        await db.update_user_last_login(user['id'])

        await db.log_user_activity(
            user_id=user['id'],
            activity_type="login",
            description="User logged in",
            metadata={"method": "email_password"}
        )

        user_response = {k: v for k, v in user.items() if k != 'password_hash'}

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            user=user_response
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.get("/me", response_model=Dict[str, Any])
async def get_current_user_info(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get current user information"""
    user_response = {k: v for k, v in current_user.items() if k != 'password_hash'}
    return user_response


@router.post("/bankid/start", response_model=BankIDStartResponse)
async def start_bankid_auth(
    request: Request,
    bankid_request: BankIDStartRequest,
):
    """
    Start BankID authentication

    Initiates Mobile BankID authentication process for Swedish users.
    Returns order reference and tokens needed for the BankID app.
    """
    try:
        # Apply rate limiting
        await check_rate_limit(request, "auth_login")

        # Start BankID authentication
        result = await bankid_service.start_auth(
            personal_number=bankid_request.personal_number,
            user_visible_data=bankid_request.user_visible_data or "Atlas AI Login"
        )

        return BankIDStartResponse(
            order_ref=result["orderRef"],
            auto_start_token=result["autoStartToken"],
            qr_start_token=result.get("qrStartToken", ""),
            qr_start_secret=result.get("qrStartSecret", "")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"BankID start error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to start BankID authentication"
        )


@router.post("/bankid/collect", response_model=BankIDCollectResponse)
async def collect_bankid_result(
    request: Request,
    collect_request: BankIDCollectRequest,
    db: MobileDatabaseManager = Depends(get_mobile_database)
):
    """
    Collect BankID authentication result

    Polls the BankID service for authentication completion.
    Returns user data and creates/logs in user when completed.
    """
    try:
        # Apply rate limiting
        await check_rate_limit(request, "auth_login")

        # Collect BankID result
        result = await bankid_service.collect_auth_result(collect_request.order_ref)

        response = BankIDCollectResponse(
            status=result["status"],
            hint_code=result.get("hintCode", ""),
            order_ref=result["orderRef"]
        )

        # If authentication completed, handle user registration/login
        if result["status"] == "complete" and "user" in result:
            bankid_user = result["user"]
            personal_number = bankid_user["personalNumber"]
            name = bankid_user["name"]

            # Check if user exists
            existing_user = await db.get_user_by_personal_number(personal_number)

            if existing_user:
                # User exists, log them in
                user = existing_user

                # Update last login
                await db.update_user_last_login(user['id'])

                # Log activity
                await db.log_user_activity(
                    user_id=user['id'],
                    activity_type="bankid_login",
                    description="User logged in with BankID",
                    metadata={"method": "bankid", "device_ip": bankid_user.get("deviceIpAddress")}
                )

            else:
                # Create new user from BankID data
                username = f"bankid_{personal_number[-4:]}"  # Use last 4 digits for username
                email = f"{personal_number}@bankid.temp"  # Temporary email, user can update

                # Create user account
                user_id = await db.create_user_with_bankid(
                    username=username,
                    email=email,
                    personal_number=personal_number,
                    full_name=name,
                    given_name=bankid_user.get("givenName"),
                    surname=bankid_user.get("surname"),
                    user_type="citizen"
                )

                # Get the created user
                user = await db.get_user_by_id(user_id)

                # Log activity
                await db.log_user_activity(
                    user_id=user_id,
                    activity_type="bankid_register",
                    description="User registered with BankID",
                    metadata={"method": "bankid", "device_ip": bankid_user.get("deviceIpAddress")}
                )

            # Create JWT tokens
            token_data = {
                "sub": str(user['id']),
                "email": user['email'],
                "type": user['user_type'],
                "auth_method": "bankid"
            }

            access_token = AuthenticationService.create_access_token(token_data)
            refresh_token = AuthenticationService.create_refresh_token(token_data)

            # Include user data and tokens in response
            user_response = {k: v for k, v in user.items() if k != 'password_hash'}
            user_response["tokens"] = {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "expires_in": 1800
            }

            response.user = user_response

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"BankID collect error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to collect BankID authentication result"
        )


@router.post("/bankid/cancel")
async def cancel_bankid_auth(
    request: Request,
    collect_request: BankIDCollectRequest
):
    """Cancel BankID authentication"""
    try:
        # Apply rate limiting
        await check_rate_limit(request, "auth_login")

        success = await bankid_service.cancel_auth(collect_request.order_ref)

        return {
            "success": success,
            "message": "BankID authentication cancelled" if success else "Session not found"
        }

    except Exception as e:
        logger.error(f"BankID cancel error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to cancel BankID authentication"
        )