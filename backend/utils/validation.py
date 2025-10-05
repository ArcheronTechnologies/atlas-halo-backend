"""
Validation utilities for Atlas AI
"""

from typing import Any


def validate_coordinates(latitude: float, longitude: float) -> None:
    """Validate latitude and longitude coordinates"""
    if not isinstance(latitude, (int, float)):
        raise ValueError("Latitude must be a number")
    if not isinstance(longitude, (int, float)):
        raise ValueError("Longitude must be a number")

    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90")
    if not -180 <= longitude <= 180:
        raise ValueError("Longitude must be between -180 and 180")


def validate_severity_level(severity: str) -> None:
    """Validate severity level"""
    valid_levels = ['safe', 'low', 'moderate', 'high', 'critical']
    if severity not in valid_levels:
        raise ValueError(f"Severity must be one of: {', '.join(valid_levels)}")


def validate_incident_type(incident_type: str) -> None:
    """Validate incident type"""
    valid_types = [
        'theft', 'robbery', 'assault', 'violence', 'vandalism',
        'drug_activity', 'traffic_accident', 'fire', 'suspicious_activity',
        'public_disturbance', 'border_control', 'other'
    ]
    if incident_type not in valid_types:
        raise ValueError(f"Incident type must be one of: {', '.join(valid_types)}")


def validate_user_type(user_type: str) -> None:
    """Validate user type"""
    valid_types = ['citizen', 'officer', 'admin', 'analyst']
    if user_type not in valid_types:
        raise ValueError(f"User type must be one of: {', '.join(valid_types)}")


def validate_email(email: str) -> None:
    """Basic email validation"""
    if not email or '@' not in email or '.' not in email:
        raise ValueError("Invalid email format")


def validate_phone_number(phone: str) -> None:
    """Basic phone number validation for Swedish numbers"""
    if not phone:
        return  # Optional field

    # Remove common separators
    clean_phone = phone.replace(' ', '').replace('-', '').replace('+', '')

    # Check if it's all digits
    if not clean_phone.isdigit():
        raise ValueError("Phone number must contain only digits, spaces, dashes, or +")

    # Check length (Swedish numbers)
    if len(clean_phone) < 8 or len(clean_phone) > 15:
        raise ValueError("Phone number must be between 8 and 15 digits")