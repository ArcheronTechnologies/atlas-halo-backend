"""
Atlas AI Comprehensive Error Handling System
Production-ready error handling with proper logging, monitoring, and user feedback
"""

import logging
import traceback
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class ErrorCategory(str, Enum):
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    RATE_LIMIT = "rate_limit"
    INTERNAL = "internal"
    EXTERNAL_SERVICE = "external_service"
    DATABASE = "database"
    CACHE = "cache"
    AI_MODEL = "ai_model"
    FILE_PROCESSING = "file_processing"
    NETWORK = "network"


class ErrorSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorDetail:
    """Detailed error information."""
    error_id: str
    error_code: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    user_message: str
    http_status: int
    timestamp: datetime
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    context: Dict[str, Any] = None
    stacktrace: Optional[str] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}
        if self.error_id is None:
            self.error_id = str(uuid.uuid4())


class AtlasAIException(Exception):
    """Base exception for Atlas AI with detailed error information."""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        http_status: int = 500,
        user_message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.category = category
        self.severity = severity
        self.http_status = http_status
        self.user_message = user_message or self._get_default_user_message()
        self.context = context or {}
        self.error_id = str(uuid.uuid4())
        self.timestamp = datetime.now()
        
        super().__init__(self.message)
    
    def _get_default_user_message(self) -> str:
        """Get default user-friendly message based on category."""
        messages = {
            ErrorCategory.AUTHENTICATION: "Authentication required. Please log in.",
            ErrorCategory.AUTHORIZATION: "You don't have permission to access this resource.",
            ErrorCategory.VALIDATION: "The provided data is invalid. Please check your input.",
            ErrorCategory.NOT_FOUND: "The requested resource was not found.",
            ErrorCategory.CONFLICT: "This operation conflicts with existing data.",
            ErrorCategory.RATE_LIMIT: "Too many requests. Please try again later.",
            ErrorCategory.INTERNAL: "An internal error occurred. Please try again.",
            ErrorCategory.EXTERNAL_SERVICE: "External service is unavailable. Please try again later.",
            ErrorCategory.DATABASE: "Database operation failed. Please try again.",
            ErrorCategory.CACHE: "Cache operation failed. Please try again.",
            ErrorCategory.AI_MODEL: "AI processing failed. Please try again.",
            ErrorCategory.FILE_PROCESSING: "File processing failed. Please check the file format.",
            ErrorCategory.NETWORK: "Network error. Please check your connection."
        }
        return messages.get(self.category, "An error occurred. Please try again.")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            "error_id": self.error_id,
            "error_code": self.error_code,
            "category": self.category.value,
            "severity": self.severity.value,
            "message": self.user_message,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context
        }
    
    def to_error_detail(self, request: Optional[Request] = None) -> ErrorDetail:
        """Convert to ErrorDetail object."""
        request_id = None
        user_id = None
        endpoint = None
        method = None
        ip_address = None
        user_agent = None
        
        if request:
            request_id = getattr(request.state, 'request_id', None)
            user_id = getattr(request.state, 'user_id', None) 
            endpoint = str(request.url.path)
            method = request.method
            ip_address = request.client.host if request.client else None
            user_agent = request.headers.get('user-agent')
        
        return ErrorDetail(
            error_id=self.error_id,
            error_code=self.error_code,
            category=self.category,
            severity=self.severity,
            message=self.message,
            user_message=self.user_message,
            http_status=self.http_status,
            timestamp=self.timestamp,
            request_id=request_id,
            user_id=user_id,
            endpoint=endpoint,
            method=method,
            ip_address=ip_address,
            user_agent=user_agent,
            context=self.context,
            stacktrace=traceback.format_exc()
        )


# Specific exception classes
class AuthenticationError(AtlasAIException):
    """Authentication related errors."""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(
            message=message,
            error_code="AUTH_001",
            category=ErrorCategory.AUTHENTICATION,
            http_status=401,
            **kwargs
        )


class AuthorizationError(AtlasAIException):
    """Authorization related errors."""
    
    def __init__(self, message: str = "Access denied", **kwargs):
        super().__init__(
            message=message,
            error_code="AUTH_002", 
            category=ErrorCategory.AUTHORIZATION,
            http_status=403,
            **kwargs
        )


class ValidationError(AtlasAIException):
    """Input validation errors."""
    
    def __init__(self, message: str = "Validation failed", field: Optional[str] = None, **kwargs):
        context = kwargs.get('context', {})
        if field:
            context['field'] = field
        
        super().__init__(
            message=message,
            error_code="VAL_001",
            category=ErrorCategory.VALIDATION,
            http_status=422,
            context=context,
            **kwargs
        )


class NotFoundError(AtlasAIException):
    """Resource not found errors."""
    
    def __init__(self, resource: str = "Resource", resource_id: Optional[str] = None, **kwargs):
        message = f"{resource} not found"
        if resource_id:
            message += f" (ID: {resource_id})"
        
        context = kwargs.get('context', {})
        context.update({'resource': resource, 'resource_id': resource_id})
        
        super().__init__(
            message=message,
            error_code="NOT_001",
            category=ErrorCategory.NOT_FOUND,
            http_status=404,
            context=context,
            **kwargs
        )


class ConflictError(AtlasAIException):
    """Resource conflict errors."""
    
    def __init__(self, message: str = "Resource conflict", **kwargs):
        super().__init__(
            message=message,
            error_code="CON_001",
            category=ErrorCategory.CONFLICT,
            http_status=409,
            **kwargs
        )


class RateLimitError(AtlasAIException):
    """Rate limiting errors."""
    
    def __init__(self, limit: int, window: str = "minute", **kwargs):
        message = f"Rate limit exceeded: {limit} requests per {window}"
        context = kwargs.get('context', {})
        context.update({'limit': limit, 'window': window})
        
        super().__init__(
            message=message,
            error_code="RATE_001",
            category=ErrorCategory.RATE_LIMIT,
            http_status=429,
            context=context,
            **kwargs
        )


class DatabaseError(AtlasAIException):
    """Database operation errors."""
    
    def __init__(self, operation: str = "Database operation", **kwargs):
        super().__init__(
            message=f"{operation} failed",
            error_code="DB_001",
            category=ErrorCategory.DATABASE,
            severity=ErrorSeverity.HIGH,
            context={'operation': operation},
            **kwargs
        )


class CacheError(AtlasAIException):
    """Cache operation errors."""
    
    def __init__(self, operation: str = "Cache operation", **kwargs):
        super().__init__(
            message=f"{operation} failed",
            error_code="CACHE_001",
            category=ErrorCategory.CACHE,
            severity=ErrorSeverity.LOW,
            http_status=200,  # Don't fail request for cache errors
            context={'operation': operation},
            **kwargs
        )


class AIModelError(AtlasAIException):
    """AI model processing errors."""
    
    def __init__(self, model: str = "AI model", operation: str = "prediction", **kwargs):
        super().__init__(
            message=f"{model} {operation} failed",
            error_code="AI_001",
            category=ErrorCategory.AI_MODEL,
            severity=ErrorSeverity.HIGH,
            context={'model': model, 'operation': operation},
            **kwargs
        )


class FileProcessingError(AtlasAIException):
    """File processing errors."""
    
    def __init__(self, filename: str, operation: str = "processing", **kwargs):
        super().__init__(
            message=f"File {operation} failed: {filename}",
            error_code="FILE_001",
            category=ErrorCategory.FILE_PROCESSING,
            context={'filename': filename, 'operation': operation},
            **kwargs
        )


class ExternalServiceError(AtlasAIException):
    """External service errors."""
    
    def __init__(self, service: str, operation: str = "request", **kwargs):
        super().__init__(
            message=f"{service} {operation} failed",
            error_code="EXT_001",
            category=ErrorCategory.EXTERNAL_SERVICE,
            severity=ErrorSeverity.MEDIUM,
            http_status=503,
            context={'service': service, 'operation': operation},
            **kwargs
        )


class ErrorLogger:
    """Centralized error logging system."""
    
    def __init__(self):
        self.logger = logging.getLogger('atlas_ai.errors')
        
        # Create formatter for error logs
        formatter = logging.Formatter(
            '%(asctime)s - ERROR - %(name)s - %(message)s'
        )
        
        # File handler for error logs
        try:
            import os
            os.makedirs('logs', exist_ok=True)
            file_handler = logging.FileHandler('logs/errors.log')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.ERROR)
            self.logger.addHandler(file_handler)
        except Exception:
            pass  # Fallback to console logging only
    
    def log_error(self, error_detail: ErrorDetail):
        """Log error with structured information."""
        
        # Create structured log entry
        log_data = {
            'error_id': error_detail.error_id,
            'error_code': error_detail.error_code,
            'category': error_detail.category.value,
            'severity': error_detail.severity.value,
            'message': error_detail.message,
            'http_status': error_detail.http_status,
            'timestamp': error_detail.timestamp.isoformat(),
            'request_id': error_detail.request_id,
            'user_id': error_detail.user_id,
            'endpoint': error_detail.endpoint,
            'method': error_detail.method,
            'ip_address': error_detail.ip_address,
            'context': error_detail.context
        }
        
        # Log with appropriate level based on severity
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(error_detail.severity, logging.ERROR)
        
        self.logger.log(log_level, json.dumps(log_data, default=str))
        
        # Log stacktrace for high severity errors
        if error_detail.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] and error_detail.stacktrace:
            self.logger.error(f"Stacktrace for {error_detail.error_id}:\\n{error_detail.stacktrace}")
    
    def log_exception(self, exc: Exception, request: Optional[Request] = None, context: Optional[Dict] = None):
        """Log a generic exception."""
        
        if isinstance(exc, AtlasAIException):
            error_detail = exc.to_error_detail(request)
        else:
            # Convert generic exception to AtlasAIException
            atlas_exc = AtlasAIException(
                message=str(exc),
                error_code="GEN_001",
                category=ErrorCategory.INTERNAL,
                severity=ErrorSeverity.HIGH,
                context=context or {}
            )
            error_detail = atlas_exc.to_error_detail(request)
        
        self.log_error(error_detail)
        return error_detail


# Global error logger instance
error_logger = ErrorLogger()


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for global error handling."""
    
    async def dispatch(self, request: Request, call_next):
        # Add request ID for tracking
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        try:
            response = await call_next(request)
            return response
            
        except AtlasAIException as exc:
            # Log the error
            error_detail = exc.to_error_detail(request)
            error_logger.log_error(error_detail)
            
            # Return structured error response
            return JSONResponse(
                status_code=exc.http_status,
                content={
                    "error": exc.to_dict(),
                    "request_id": request_id
                }
            )
            
        except HTTPException as exc:
            # Handle FastAPI HTTP exceptions
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": {
                        "error_id": str(uuid.uuid4()),
                        "error_code": f"HTTP_{exc.status_code}",
                        "category": "validation" if exc.status_code == 422 else "internal",
                        "message": exc.detail,
                        "timestamp": datetime.now().isoformat()
                    },
                    "request_id": request_id
                }
            )
            
        except Exception as exc:
            # Handle all other exceptions
            error_detail = error_logger.log_exception(exc, request)
            
            # Don't expose internal error details in production
            from ..config.production_settings import is_production
            
            if is_production():
                user_message = "An internal error occurred. Please try again later."
            else:
                user_message = str(exc)
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "error_id": error_detail.error_id,
                        "error_code": "INTERNAL_001",
                        "category": "internal",
                        "message": user_message,
                        "timestamp": datetime.now().isoformat()
                    },
                    "request_id": request_id
                }
            )


# Utility functions for common error patterns
def handle_database_operation(operation_name: str):
    """Decorator for database operations with error handling."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                raise DatabaseError(operation_name, context={'args': str(args), 'kwargs': str(kwargs)})
        return wrapper
    return decorator


def handle_cache_operation(operation_name: str):
    """Decorator for cache operations with error handling."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # Log cache error but don't fail the request
                error_logger.log_exception(CacheError(operation_name))
                return None  # Return None for failed cache operations
        return wrapper
    return decorator


def handle_ai_operation(model_name: str, operation: str):
    """Decorator for AI operations with error handling."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                raise AIModelError(model_name, operation, context={'error': str(e)})
        return wrapper
    return decorator


def validate_required_fields(data: Dict, required_fields: List[str]):
    """Validate required fields in request data."""
    missing_fields = [field for field in required_fields if field not in data or data[field] is None]
    
    if missing_fields:
        raise ValidationError(
            f"Missing required fields: {', '.join(missing_fields)}",
            context={'missing_fields': missing_fields}
        )


def validate_file_upload(file, allowed_types: List[str], max_size_mb: int = 50):
    """Validate uploaded file."""
    if not file:
        raise ValidationError("File upload required")
    
    if file.content_type not in allowed_types:
        raise ValidationError(
            f"Invalid file type. Allowed types: {', '.join(allowed_types)}",
            context={'provided_type': file.content_type, 'allowed_types': allowed_types}
        )
    
    # Check file size if available
    if hasattr(file, 'size') and file.size > max_size_mb * 1024 * 1024:
        raise ValidationError(
            f"File too large. Maximum size: {max_size_mb}MB",
            context={'file_size_mb': file.size / (1024 * 1024), 'max_size_mb': max_size_mb}
        )


# Error metrics tracking
class ErrorMetrics:
    """Track error metrics for monitoring."""
    
    def __init__(self):
        self.error_counts = {}
        self.error_rates = {}
    
    def record_error(self, error_detail: ErrorDetail):
        """Record error for metrics."""
        key = f"{error_detail.category.value}_{error_detail.error_code}"
        self.error_counts[key] = self.error_counts.get(key, 0) + 1
        
        # Record by endpoint
        if error_detail.endpoint:
            endpoint_key = f"{error_detail.endpoint}_{error_detail.category.value}"
            self.error_counts[endpoint_key] = self.error_counts.get(endpoint_key, 0) + 1
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary for monitoring."""
        return {
            'total_errors': sum(self.error_counts.values()),
            'error_counts': self.error_counts,
            'timestamp': datetime.now().isoformat()
        }


# Global error metrics instance
error_metrics = ErrorMetrics()


# Example usage and testing
if __name__ == "__main__":
    # Demo error handling
    try:
        raise ValidationError("Invalid email format", field="email")
    except AtlasAIException as e:
        print(f"Caught Atlas AI Exception: {e.to_dict()}")
    
    try:
        raise DatabaseError("user insertion")
    except AtlasAIException as e:
        error_detail = e.to_error_detail()
        error_logger.log_error(error_detail)
        print(f"Logged database error: {error_detail.error_id}")
    
    print("Error handling system demo completed")