"""
Atlas AI API Validation and Testing Utilities
Comprehensive validation for API requests and responses
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import time

from fastapi import HTTPException, Request
from pydantic import BaseModel, ValidationError as PydanticValidationError

from ..utils.error_handling import ValidationError, AtlasAIException


@dataclass
class ValidationResult:
    """Result of API validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    performance_ms: float


class APIValidator:
    """Comprehensive API validation utilities."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Geographic bounds for Sweden
        self.sweden_bounds = {
            "min_lat": 55.3617,
            "max_lat": 69.0599,
            "min_lng": 11.0273,
            "max_lng": 24.1687
        }
    
    async def validate_location(self, latitude: float, longitude: float) -> ValidationResult:
        """Validate geographic coordinates."""
        start_time = time.time()
        errors = []
        warnings = []
        
        # Basic coordinate validation
        if not (-90 <= latitude <= 90):
            errors.append(f"Latitude {latitude} out of range (-90 to 90)")
        
        if not (-180 <= longitude <= 180):
            errors.append(f"Longitude {longitude} out of range (-180 to 180)")
        
        # Sweden-specific validation (warning only)
        if not (self.sweden_bounds["min_lat"] <= latitude <= self.sweden_bounds["max_lat"]):
            warnings.append("Location is outside Swedish latitude range")
        
        if not (self.sweden_bounds["min_lng"] <= longitude <= self.sweden_bounds["max_lng"]):
            warnings.append("Location is outside Swedish longitude range")
        
        # Check for obviously invalid coordinates (e.g., 0,0)
        if latitude == 0 and longitude == 0:
            warnings.append("Coordinates (0,0) may indicate GPS failure")
        
        performance_ms = (time.time() - start_time) * 1000
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            performance_ms=performance_ms
        )
    
    async def validate_timestamp(self, timestamp: Optional[datetime]) -> ValidationResult:
        """Validate timestamp parameters."""
        start_time = time.time()
        errors = []
        warnings = []
        
        if timestamp is None:
            # No timestamp is valid (use current time)
            pass
        else:
            now = datetime.utcnow()
            
            # Check if timestamp is too far in the past (more than 10 years)
            if timestamp < now - timedelta(days=365 * 10):
                errors.append("Timestamp is too far in the past (>10 years)")
            
            # Check if timestamp is too far in the future (more than 1 year)
            if timestamp > now + timedelta(days=365):
                errors.append("Timestamp is too far in the future (>1 year)")
            
            # Warning for very recent past (potential timezone issues)
            if now - timedelta(minutes=5) <= timestamp <= now:
                warnings.append("Timestamp is very recent - check timezone")
            
            # Warning for future timestamps
            if timestamp > now:
                warnings.append("Future timestamp provided - using for prediction")
        
        performance_ms = (time.time() - start_time) * 1000
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            performance_ms=performance_ms
        )
    
    async def validate_area_request(self, center_lat: float, center_lng: float, 
                                   radius_km: float, grid_size: int) -> ValidationResult:
        """Validate area analysis request parameters."""
        start_time = time.time()
        errors = []
        warnings = []
        
        # Validate center coordinates
        location_result = await self.validate_location(center_lat, center_lng)
        errors.extend(location_result.errors)
        warnings.extend(location_result.warnings)
        
        # Validate radius
        if radius_km <= 0:
            errors.append("Radius must be positive")
        elif radius_km > 10:
            errors.append("Radius cannot exceed 10km for performance reasons")
        elif radius_km > 5:
            warnings.append("Large radius may result in slower processing")
        
        # Validate grid size
        if grid_size < 5:
            errors.append("Grid size must be at least 5")
        elif grid_size > 50:
            errors.append("Grid size cannot exceed 50 for performance reasons")
        elif grid_size > 30:
            warnings.append("Large grid size may result in slower processing")
        
        # Calculate total analysis points
        total_points = grid_size * grid_size
        if total_points > 1000:
            warnings.append(f"Analysis will process {total_points} points - may be slow")
        
        performance_ms = (time.time() - start_time) * 1000
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            performance_ms=performance_ms
        )
    
    async def validate_batch_request(self, locations: List[Dict[str, Any]]) -> ValidationResult:
        """Validate batch location request."""
        start_time = time.time()
        errors = []
        warnings = []
        
        # Check batch size
        if len(locations) == 0:
            errors.append("Batch request cannot be empty")
        elif len(locations) > 50:
            errors.append("Batch size cannot exceed 50 locations")
        elif len(locations) > 20:
            warnings.append("Large batch size may result in slower processing")
        
        # Validate each location
        for i, location in enumerate(locations):
            if 'latitude' not in location or 'longitude' not in location:
                errors.append(f"Location {i} missing required coordinates")
                continue
            
            try:
                lat = float(location['latitude'])
                lng = float(location['longitude'])
                
                location_result = await self.validate_location(lat, lng)
                
                # Add location index to errors/warnings
                for error in location_result.errors:
                    errors.append(f"Location {i}: {error}")
                for warning in location_result.warnings:
                    warnings.append(f"Location {i}: {warning}")
                    
            except (ValueError, TypeError):
                errors.append(f"Location {i} has invalid coordinate format")
        
        performance_ms = (time.time() - start_time) * 1000
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            performance_ms=performance_ms
        )
    
    async def validate_file_upload(self, file_data: bytes, content_type: str, 
                                  max_size_mb: int = 100) -> ValidationResult:
        """Validate file upload parameters."""
        start_time = time.time()
        errors = []
        warnings = []
        
        # Check file size
        file_size_mb = len(file_data) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            errors.append(f"File size {file_size_mb:.1f}MB exceeds limit of {max_size_mb}MB")
        elif file_size_mb > max_size_mb * 0.8:
            warnings.append(f"Large file size {file_size_mb:.1f}MB may slow processing")
        
        # Check content type
        allowed_video_types = [
            'video/mp4', 'video/avi', 'video/mov', 'video/quicktime', 
            'video/x-msvideo', 'video/webm'
        ]
        allowed_audio_types = [
            'audio/mp3', 'audio/wav', 'audio/mpeg', 'audio/x-wav',
            'audio/mp4', 'audio/aac'
        ]
        allowed_image_types = [
            'image/jpeg', 'image/png', 'image/gif', 'image/webp'
        ]
        
        all_allowed_types = allowed_video_types + allowed_audio_types + allowed_image_types
        
        if content_type not in all_allowed_types:
            errors.append(f"Unsupported content type: {content_type}")
        
        # Check for empty file
        if len(file_data) == 0:
            errors.append("File is empty")
        
        performance_ms = (time.time() - start_time) * 1000
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            performance_ms=performance_ms
        )
    
    async def validate_pydantic_model(self, model_class: BaseModel, 
                                     data: Dict[str, Any]) -> ValidationResult:
        """Validate data against Pydantic model."""
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            # Attempt to create model instance
            model_instance = model_class(**data)
            
            # Check for potential data quality issues
            if hasattr(model_instance, 'latitude') and hasattr(model_instance, 'longitude'):
                location_result = await self.validate_location(
                    model_instance.latitude, 
                    model_instance.longitude
                )
                warnings.extend(location_result.warnings)
            
        except PydanticValidationError as e:
            for error in e.errors():
                field = '.'.join(str(loc) for loc in error['loc'])
                message = error['msg']
                errors.append(f"Field '{field}': {message}")
        
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        performance_ms = (time.time() - start_time) * 1000
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            performance_ms=performance_ms
        )


class RequestValidator:
    """FastAPI request validation middleware helper."""
    
    def __init__(self):
        self.validator = APIValidator()
        self.logger = logging.getLogger(__name__)
    
    async def validate_request(self, request: Request, 
                              validation_rules: Dict[str, Callable]) -> Optional[ValidationError]:
        """
        Validate FastAPI request against custom rules.
        
        Args:
            request: FastAPI request object
            validation_rules: Dictionary of validation functions
            
        Returns:
            ValidationError if validation fails, None if valid
        """
        try:
            # Get request data
            if request.method in ["POST", "PUT", "PATCH"]:
                try:
                    request_data = await request.json()
                except Exception:
                    return ValidationError("Invalid JSON in request body")
            else:
                request_data = dict(request.query_params)
            
            # Apply validation rules
            all_errors = []
            all_warnings = []
            
            for rule_name, rule_func in validation_rules.items():
                try:
                    result = await rule_func(request_data)
                    if isinstance(result, ValidationResult):
                        all_errors.extend(result.errors)
                        all_warnings.extend(result.warnings)
                    elif result is False:
                        all_errors.append(f"Validation rule '{rule_name}' failed")
                except Exception as e:
                    all_errors.append(f"Validation rule '{rule_name}' error: {str(e)}")
            
            # Log warnings
            for warning in all_warnings:
                self.logger.warning(f"Request validation warning: {warning}")
            
            # Return error if any validation failed
            if all_errors:
                return ValidationError(
                    "Request validation failed",
                    context={
                        "validation_errors": all_errors,
                        "validation_warnings": all_warnings,
                        "request_path": str(request.url.path),
                        "request_method": request.method
                    }
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Request validation error: {e}")
            return ValidationError(f"Validation system error: {str(e)}")


# Pre-defined validation rule sets
class ValidationRules:
    """Common validation rule sets for different endpoints."""
    
    @staticmethod
    async def location_endpoint_rules(data: Dict[str, Any]) -> ValidationResult:
        """Validation rules for location-based endpoints."""
        validator = APIValidator()
        
        lat = data.get('latitude')
        lng = data.get('longitude')
        
        if lat is None or lng is None:
            return ValidationResult(
                is_valid=False,
                errors=["Missing required latitude and longitude"],
                warnings=[],
                performance_ms=0
            )
        
        try:
            lat = float(lat)
            lng = float(lng)
        except (ValueError, TypeError):
            return ValidationResult(
                is_valid=False,
                errors=["Latitude and longitude must be numbers"],
                warnings=[],
                performance_ms=0
            )
        
        return await validator.validate_location(lat, lng)
    
    @staticmethod
    async def area_endpoint_rules(data: Dict[str, Any]) -> ValidationResult:
        """Validation rules for area analysis endpoints."""
        validator = APIValidator()
        
        required_fields = ['center_latitude', 'center_longitude', 'radius_km', 'grid_size']
        for field in required_fields:
            if field not in data:
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Missing required field: {field}"],
                    warnings=[],
                    performance_ms=0
                )
        
        try:
            center_lat = float(data['center_latitude'])
            center_lng = float(data['center_longitude'])
            radius_km = float(data['radius_km'])
            grid_size = int(data['grid_size'])
        except (ValueError, TypeError):
            return ValidationResult(
                is_valid=False,
                errors=["Invalid data types for area parameters"],
                warnings=[],
                performance_ms=0
            )
        
        return await validator.validate_area_request(center_lat, center_lng, radius_km, grid_size)
    
    @staticmethod
    async def batch_endpoint_rules(data: Dict[str, Any]) -> ValidationResult:
        """Validation rules for batch processing endpoints."""
        validator = APIValidator()
        
        if 'locations' not in data:
            return ValidationResult(
                is_valid=False,
                errors=["Missing required 'locations' field"],
                warnings=[],
                performance_ms=0
            )
        
        locations = data['locations']
        if not isinstance(locations, list):
            return ValidationResult(
                is_valid=False,
                errors=["'locations' must be a list"],
                warnings=[],
                performance_ms=0
            )
        
        return await validator.validate_batch_request(locations)


# Example usage and testing utilities
async def test_validation_system():
    """Test the validation system with sample data."""
    validator = APIValidator()
    
    # Test location validation
    print("Testing location validation...")
    result = await validator.validate_location(59.3293, 18.0686)
    print(f"Stockholm validation: Valid={result.is_valid}, Errors={result.errors}, Warnings={result.warnings}")
    
    # Test invalid coordinates
    result = await validator.validate_location(200, 300)
    print(f"Invalid coords validation: Valid={result.is_valid}, Errors={result.errors}")
    
    # Test area validation
    print("\nTesting area validation...")
    result = await validator.validate_area_request(59.3293, 18.0686, 2.5, 25)
    print(f"Area validation: Valid={result.is_valid}, Errors={result.errors}, Warnings={result.warnings}")
    
    # Test batch validation
    print("\nTesting batch validation...")
    locations = [
        {"latitude": 59.3293, "longitude": 18.0686},
        {"latitude": 59.3320, "longitude": 18.0649}
    ]
    result = await validator.validate_batch_request(locations)
    print(f"Batch validation: Valid={result.is_valid}, Errors={result.errors}, Warnings={result.warnings}")


if __name__ == "__main__":
    asyncio.run(test_validation_system())