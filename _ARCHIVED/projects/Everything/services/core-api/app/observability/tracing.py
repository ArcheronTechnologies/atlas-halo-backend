"""
OpenTelemetry Distributed Tracing

Production-ready distributed tracing with automatic instrumentation,
custom spans, and comprehensive context propagation.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager, asynccontextmanager
from functools import wraps
import time
import traceback
from datetime import datetime, timezone

try:
    from opentelemetry import trace, propagate
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.propagators.b3 import B3MultiFormat
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.trace import Status, StatusCode, SpanKind
    from opentelemetry.semconv.trace import SpanAttributes
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None
    
    # Fallback classes when OpenTelemetry is not available
    class SpanKind:
        INTERNAL = "internal"
        CLIENT = "client" 
        SERVER = "server"
        
    class Status:
        def __init__(self, status_code):
            self.status_code = status_code
            
    class StatusCode:
        OK = "ok"
        ERROR = "error"

from ..core.config import settings

logger = logging.getLogger(__name__)

# Global tracer instance
tracer = None


class TracingConfig:
    """OpenTelemetry tracing configuration"""
    def __init__(self):
        self.service_name = settings.otel_service_name
        self.endpoint = settings.otel_endpoint
        self.headers = self._parse_headers(settings.otel_headers)
        self.sample_rate = 1.0  # Sample 100% of traces in development
        self.enabled = bool(self.endpoint) and OTEL_AVAILABLE
    
    def _parse_headers(self, headers_str: str) -> Dict[str, str]:
        """Parse OTEL_HEADERS environment variable"""
        if not headers_str:
            return {}
        
        headers = {}
        for header in headers_str.split(','):
            if '=' in header:
                key, value = header.split('=', 1)
                headers[key.strip()] = value.strip()
        return headers


def init_tracing() -> bool:
    """Initialize OpenTelemetry tracing"""
    global tracer
    
    if not OTEL_AVAILABLE:
        logger.warning("OpenTelemetry not available. Install with: pip install opentelemetry-distro opentelemetry-instrumentation-auto")
        return False
    
    config = TracingConfig()
    
    if not config.enabled:
        logger.info("OpenTelemetry tracing disabled (no endpoint configured)")
        tracer = trace.NoOpTracer()
        return False
    
    try:
        # Create resource
        resource = Resource.create({
            SERVICE_NAME: config.service_name,
            SERVICE_VERSION: "1.0.0",
            "environment": "production"  # Could be configurable
        })
        
        # Set up tracer provider
        trace.set_tracer_provider(TracerProvider(resource=resource))
        tracer_provider = trace.get_tracer_provider()
        
        # Configure OTLP exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=config.endpoint,
            headers=config.headers,
            timeout=30
        )
        
        # Add batch span processor
        span_processor = BatchSpanProcessor(
            otlp_exporter,
            max_queue_size=2048,
            max_export_batch_size=512,
            export_timeout=30,
            schedule_delay=2
        )
        tracer_provider.add_span_processor(span_processor)
        
        # Set up propagators
        propagate.set_global_textmap(B3MultiFormat())
        
        # Get tracer
        tracer = trace.get_tracer(__name__)
        
        # Auto-instrument libraries
        FastAPIInstrumentor.instrument()
        RequestsInstrumentor.instrument()
        SQLAlchemyInstrumentor.instrument()
        RedisInstrumentor.instrument()
        
        logger.info(f"OpenTelemetry tracing initialized with endpoint: {config.endpoint}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry tracing: {e}")
        tracer = trace.NoOpTracer()
        return False


@contextmanager
def create_span(
    name: str,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None
):
    """Create a manual span with automatic error handling"""
    if not tracer:
        yield None
        return
    
    with tracer.start_as_current_span(name, kind=kind) as span:
        try:
            # Set attributes
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            
            yield span
            
        except Exception as e:
            # Record exception in span
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


@asynccontextmanager
async def create_async_span(
    name: str,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None
):
    """Create an async span with automatic error handling"""
    if not tracer:
        yield None
        return
    
    with tracer.start_as_current_span(name, kind=kind) as span:
        try:
            # Set attributes
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            
            yield span
            
        except Exception as e:
            # Record exception in span
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


def trace_function(
    name: Optional[str] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Optional[Dict[str, Any]] = None
):
    """Decorator to automatically trace function calls"""
    def decorator(func: Callable) -> Callable:
        span_name = name or f"{func.__module__}.{func.__qualname__}"
        
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with create_async_span(span_name, kind, attributes) as span:
                    if span:
                        # Add function parameters as attributes
                        span.set_attribute("function.name", func.__name__)
                        span.set_attribute("function.module", func.__module__)
                        
                        # Add argument info (be careful with sensitive data)
                        if args:
                            span.set_attribute("function.args_count", len(args))
                        if kwargs:
                            span.set_attribute("function.kwargs_count", len(kwargs))
                    
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with create_span(span_name, kind, attributes) as span:
                    if span:
                        # Add function parameters as attributes
                        span.set_attribute("function.name", func.__name__)
                        span.set_attribute("function.module", func.__module__)
                        
                        # Add argument info
                        if args:
                            span.set_attribute("function.args_count", len(args))
                        if kwargs:
                            span.set_attribute("function.kwargs_count", len(kwargs))
                    
                    return func(*args, **kwargs)
            return sync_wrapper
    return decorator


def trace_database_operation(operation: str, table: str):
    """Decorator specifically for database operations"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with create_span(
                f"db.{operation}",
                kind=SpanKind.CLIENT,
                attributes={
                    SpanAttributes.DB_OPERATION: operation,
                    SpanAttributes.DB_SQL_TABLE: table,
                    SpanAttributes.DB_SYSTEM: "postgresql"  # or from config
                }
            ) as span:
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    
                    if span:
                        span.set_attribute("db.rows_affected", getattr(result, 'rowcount', 0))
                        span.set_status(Status(StatusCode.OK))
                    
                    return result
                    
                except Exception as e:
                    if span:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
                finally:
                    if span:
                        span.set_attribute("db.duration", time.time() - start_time)
        return wrapper
    return decorator


def trace_http_request(method: str, url: str):
    """Decorator for HTTP requests"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with create_async_span(
                f"http.{method.lower()}",
                kind=SpanKind.CLIENT,
                attributes={
                    SpanAttributes.HTTP_METHOD: method,
                    SpanAttributes.HTTP_URL: url
                }
            ) as span:
                start_time = time.time()
                try:
                    response = await func(*args, **kwargs)
                    
                    if span and hasattr(response, 'status_code'):
                        span.set_attribute(SpanAttributes.HTTP_STATUS_CODE, response.status_code)
                        if 200 <= response.status_code < 400:
                            span.set_status(Status(StatusCode.OK))
                        else:
                            span.set_status(Status(StatusCode.ERROR))
                    
                    return response
                    
                except Exception as e:
                    if span:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
                finally:
                    if span:
                        span.set_attribute("http.duration", time.time() - start_time)
        return wrapper
    return decorator


def get_trace_context() -> Dict[str, str]:
    """Get current trace context for propagation"""
    if not tracer:
        return {}
    
    try:
        # Get current span context
        current_span = trace.get_current_span()
        if not current_span.is_recording():
            return {}
        
        # Create carrier dict
        carrier = {}
        propagate.inject(carrier)
        
        return carrier
    except Exception as e:
        logger.error(f"Failed to get trace context: {e}")
        return {}


def add_span_attributes(attributes: Dict[str, Any]):
    """Add attributes to current span"""
    if not tracer:
        return
    
    try:
        current_span = trace.get_current_span()
        if current_span.is_recording():
            for key, value in attributes.items():
                current_span.set_attribute(key, value)
    except Exception as e:
        logger.error(f"Failed to add span attributes: {e}")


def record_span_event(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Record an event in the current span"""
    if not tracer:
        return
    
    try:
        current_span = trace.get_current_span()
        if current_span.is_recording():
            current_span.add_event(name, attributes or {})
    except Exception as e:
        logger.error(f"Failed to record span event: {e}")


class TraceableContext:
    """Context manager for tracing complex operations"""
    
    def __init__(self, operation_name: str, component: str = "api"):
        self.operation_name = operation_name
        self.component = component
        self.span = None
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        if tracer:
            self.span = tracer.start_span(
                f"{self.component}.{self.operation_name}",
                attributes={
                    "component": self.component,
                    "operation": self.operation_name,
                    "start_time": datetime.now(timezone.utc).isoformat()
                }
            )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            duration = time.time() - self.start_time if self.start_time else 0
            self.span.set_attribute("duration", duration)
            
            if exc_type:
                self.span.record_exception(exc_val)
                self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
            else:
                self.span.set_status(Status(StatusCode.OK))
            
            self.span.end()
    
    def add_attribute(self, key: str, value: Any):
        """Add attribute to the current operation span"""
        if self.span:
            self.span.set_attribute(key, value)
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to the current operation span"""
        if self.span:
            self.span.add_event(name, attributes or {})