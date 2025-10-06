"""Input validation utilities"""
from fastapi import Request, HTTPException
from pydantic import BaseModel, validator
from typing import Any, Optional
import re
import logging

logger = logging.getLogger(__name__)

class SecurityValidator:
    """Security validation utilities"""

    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize string input"""
        if not value:
            return ""

        # Truncate to max length
        value = value[:max_length]

        # Remove null bytes and control characters
        value = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', value)

        return value.strip()

    @staticmethod
    def validate_coordinates(lat: float, lon: float) -> bool:
        """Validate geographic coordinates"""
        return -90 <= lat <= 90 and -180 <= lon <= 180

    @staticmethod
    def validate_email(email: str) -> bool:
        """Basic email validation"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    @staticmethod
    def validate_phone(phone: str) -> bool:
        """Swedish phone number validation"""
        # Remove common formatting characters
        clean_phone = re.sub(r'[\s\-\(\)]', '', phone)

        # Swedish phone patterns
        patterns = [
            r'^\+46\d{9}$',  # +46701234567
            r'^0\d{9}$',     # 0701234567
            r'^46\d{9}$',    # 46701234567
        ]

        return any(re.match(pattern, clean_phone) for pattern in patterns)

class SecureBaseModel(BaseModel):
    """Base model with built-in security validations"""

    class Config:
        # Validate assignment
        validate_assignment = True
        # Allow arbitrary types
        arbitrary_types_allowed = False
        # Use enum values
        use_enum_values = True

def add_security_headers(response):
    """Add security headers to response"""
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response
