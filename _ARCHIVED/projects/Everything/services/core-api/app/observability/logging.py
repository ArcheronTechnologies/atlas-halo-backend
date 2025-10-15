"""
Structured Logging

Production-ready structured logging with correlation IDs,
request tracing, and comprehensive log aggregation support.
"""

import logging
import logging.config
import json
import traceback
import sys
import asyncio
import time
from typing import Dict, Any, Optional, Union
from datetime import datetime, timezone
from contextvars import ContextVar
from functools import wraps
import uuid

try:
    from opentelemetry import trace
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None

from ..core.config import settings

# Context variables for request tracking
correlation_id_var: ContextVar[str] = ContextVar('correlation_id', default='')
user_id_var: ContextVar[str] = ContextVar('user_id', default='')
session_id_var: ContextVar[str] = ContextVar('session_id', default='')


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging"""
    
    def __init__(self, service_name: str = "scip-api", include_trace_info: bool = True):
        super().__init__()
        self.service_name = service_name
        self.include_trace_info = include_trace_info
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        
        # Base log structure
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "service": self.service_name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add context information
        correlation_id = correlation_id_var.get()
        if correlation_id:
            log_entry["correlation_id"] = correlation_id
        
        user_id = user_id_var.get()
        if user_id:
            log_entry["user_id"] = user_id
        
        session_id = session_id_var.get()
        if session_id:
            log_entry["session_id"] = session_id
        
        # Add trace information if available
        if self.include_trace_info and OTEL_AVAILABLE:
            try:
                current_span = trace.get_current_span()
                if current_span.is_recording():
                    span_context = current_span.get_span_context()
                    log_entry["trace_id"] = format(span_context.trace_id, '032x')
                    log_entry["span_id"] = format(span_context.span_id, '016x')
            except Exception:
                pass
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add extra fields from LoggerAdapter or direct logging calls
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add custom attributes
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'exc_info', 'exc_text', 'stack_info',
                          'extra_fields'] and not key.startswith('_'):
                try:
                    # Ensure the value is JSON serializable
                    json.dumps(value)
                    log_entry[key] = value
                except (TypeError, ValueError):
                    log_entry[key] = str(value)
        
        return json.dumps(log_entry, ensure_ascii=False)


class CorrelationIdFilter(logging.Filter):
    """Filter to add correlation ID to all log records"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        correlation_id = correlation_id_var.get()
        if correlation_id:
            record.correlation_id = correlation_id
        return True


class StructuredLogger:
    """Enhanced logger with structured logging capabilities"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.name = name
    
    def _log_with_context(self, level: int, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log with additional context"""
        if extra:
            # Store extra fields in the record
            if hasattr(logging.getLoggerClass(), 'extra_fields'):
                self.logger.extra_fields = extra
            else:
                # Fallback: add to kwargs
                kwargs.update(extra)
        
        self.logger.log(level, message, **kwargs)
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log debug message with context"""
        self._log_with_context(logging.DEBUG, message, extra, **kwargs)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log info message with context"""
        self._log_with_context(logging.INFO, message, extra, **kwargs)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log warning message with context"""
        self._log_with_context(logging.WARNING, message, extra, **kwargs)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = False, **kwargs):
        """Log error message with context"""
        self._log_with_context(logging.ERROR, message, extra, exc_info=exc_info, **kwargs)
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None, exc_info: bool = False, **kwargs):
        """Log critical message with context"""
        self._log_with_context(logging.CRITICAL, message, extra, exc_info=exc_info, **kwargs)
    
    def exception(self, message: str, extra: Optional[Dict[str, Any]] = None, **kwargs):
        """Log exception with full traceback"""
        self._log_with_context(logging.ERROR, message, extra, exc_info=True, **kwargs)
    
    def log_business_event(self, event_type: str, details: Dict[str, Any], level: int = logging.INFO):
        """Log business events with structured data"""
        extra = {
            "event_type": "business_event",
            "business_event": event_type,
            "event_details": details,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self._log_with_context(level, f"Business event: {event_type}", extra)
    
    def log_performance(self, operation: str, duration: float, details: Optional[Dict[str, Any]] = None):
        """Log performance metrics"""
        extra = {
            "event_type": "performance",
            "operation": operation,
            "duration_seconds": duration,
            "performance_details": details or {}
        }
        self._log_with_context(logging.INFO, f"Performance: {operation} completed in {duration:.3f}s", extra)
    
    def log_security_event(self, event_type: str, details: Dict[str, Any], level: int = logging.WARNING):
        """Log security-related events"""
        extra = {
            "event_type": "security",
            "security_event": event_type,
            "security_details": details,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self._log_with_context(level, f"Security event: {event_type}", extra)
    
    def log_audit_trail(self, action: str, resource: str, user_id: str, details: Optional[Dict[str, Any]] = None):
        """Log audit trail events"""
        extra = {
            "event_type": "audit",
            "action": action,
            "resource": resource,
            "user_id": user_id,
            "audit_details": details or {},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self._log_with_context(logging.INFO, f"Audit: {user_id} performed {action} on {resource}", extra)


def init_structured_logging(
    level: str = "INFO",
    service_name: str = "scip-api",
    enable_json_format: bool = True,
    log_file: Optional[str] = None
) -> bool:
    """Initialize structured logging configuration"""
    
    try:
        # Configure logging level
        log_level = getattr(logging, level.upper(), logging.INFO)
        
        # Create formatters
        if enable_json_format:
            formatter = StructuredFormatter(service_name=service_name)
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Configure handlers
        handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        handlers.append(console_handler)
        
        # File handler (optional)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            handlers.append(file_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            handlers=handlers,
            format='%(message)s' if enable_json_format else None
        )
        
        # Add correlation ID filter to all handlers
        correlation_filter = CorrelationIdFilter()
        for handler in handlers:
            handler.addFilter(correlation_filter)
        
        # Set specific logger levels
        logging.getLogger('uvicorn').setLevel(logging.WARNING)
        logging.getLogger('fastapi').setLevel(logging.INFO)
        logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.WARNING)
        
        logger = get_logger(__name__)
        logger.info(f"Structured logging initialized", extra={
            "service": service_name,
            "level": level,
            "json_format": enable_json_format,
            "log_file": log_file
        })
        
        return True
        
    except Exception as e:
        print(f"Failed to initialize structured logging: {e}")
        return False


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance"""
    return StructuredLogger(name)


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Set correlation ID for request tracing"""
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    correlation_id_var.set(correlation_id)
    return correlation_id


def get_correlation_id() -> str:
    """Get current correlation ID"""
    return correlation_id_var.get()


def set_user_context(user_id: str, session_id: Optional[str] = None):
    """Set user context for logging"""
    user_id_var.set(user_id)
    if session_id:
        session_id_var.set(session_id)


def clear_context():
    """Clear all context variables"""
    correlation_id_var.set('')
    user_id_var.set('')
    session_id_var.set('')


def log_with_context(**context):
    """Decorator to add context to all logs within a function"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Set context
            original_context = {}
            for key, value in context.items():
                if key == 'correlation_id':
                    original_context['correlation_id'] = correlation_id_var.get()
                    correlation_id_var.set(value)
                elif key == 'user_id':
                    original_context['user_id'] = user_id_var.get()
                    user_id_var.set(value)
                elif key == 'session_id':
                    original_context['session_id'] = session_id_var.get()
                    session_id_var.set(value)
            
            try:
                return await func(*args, **kwargs)
            finally:
                # Restore original context
                for key, value in original_context.items():
                    if key == 'correlation_id':
                        correlation_id_var.set(value)
                    elif key == 'user_id':
                        user_id_var.set(value)
                    elif key == 'session_id':
                        session_id_var.set(value)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Set context
            original_context = {}
            for key, value in context.items():
                if key == 'correlation_id':
                    original_context['correlation_id'] = correlation_id_var.get()
                    correlation_id_var.set(value)
                elif key == 'user_id':
                    original_context['user_id'] = user_id_var.get()
                    user_id_var.set(value)
                elif key == 'session_id':
                    original_context['session_id'] = session_id_var.get()
                    session_id_var.set(value)
            
            try:
                return func(*args, **kwargs)
            finally:
                # Restore original context
                for key, value in original_context.items():
                    if key == 'correlation_id':
                        correlation_id_var.set(value)
                    elif key == 'user_id':
                        user_id_var.set(value)
                    elif key == 'session_id':
                        session_id_var.set(value)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


class LoggingMiddleware:
    """FastAPI middleware for request logging"""
    
    def __init__(self, app, logger_name: str = "request"):
        self.app = app
        self.logger = get_logger(logger_name)
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Generate correlation ID
        correlation_id = set_correlation_id()
        
        # Extract request info
        method = scope["method"]
        path = scope["path"]
        query_string = scope["query_string"].decode()
        client = scope.get("client", ("unknown", 0))
        
        start_time = time.time()
        
        # Log request start
        self.logger.info(
            f"Request started: {method} {path}",
            extra={
                "event_type": "request_start",
                "method": method,
                "path": path,
                "query_string": query_string,
                "client_ip": client[0],
                "client_port": client[1],
                "correlation_id": correlation_id
            }
        )
        
        # Process request
        try:
            await self.app(scope, receive, send)
            
        except Exception as e:
            # Log request error
            duration = time.time() - start_time
            self.logger.error(
                f"Request failed: {method} {path}",
                extra={
                    "event_type": "request_error",
                    "method": method,
                    "path": path,
                    "duration": duration,
                    "error": str(e),
                    "correlation_id": correlation_id
                },
                exc_info=True
            )
            raise
        
        finally:
            # Log request completion
            duration = time.time() - start_time
            self.logger.info(
                f"Request completed: {method} {path} in {duration:.3f}s",
                extra={
                    "event_type": "request_complete",
                    "method": method,
                    "path": path,
                    "duration": duration,
                    "correlation_id": correlation_id
                }
            )
            
            # Clear context
            clear_context()