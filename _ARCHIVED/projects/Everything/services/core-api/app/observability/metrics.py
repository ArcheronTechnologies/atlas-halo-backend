"""
Prometheus Metrics Collection

Comprehensive metrics collection for supply chain API including
business metrics, performance metrics, and system health indicators.
"""

import time
import threading
import asyncio
from typing import Dict, List, Any, Optional, Union
from functools import wraps
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging

try:
    from prometheus_client import (
        Counter as PrometheusCounter,
        Histogram as PrometheusHistogram,
        Gauge as PrometheusGauge,
        Summary as PrometheusSummary,
        Info as PrometheusInfo,
        generate_latest,
        CollectorRegistry,
        start_http_server,
        multiprocess,
        CONTENT_TYPE_LATEST
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from ..core.config import settings

logger = logging.getLogger(__name__)

# Global metrics registry
registry = None
metrics_server = None


@dataclass
class MetricsConfig:
    """Metrics configuration"""
    enabled: bool = True
    port: int = settings.prometheus_port
    service_name: str = settings.otel_service_name
    namespace: str = "scip"
    collect_business_metrics: bool = True
    collect_performance_metrics: bool = True
    histogram_buckets: List[float] = field(default_factory=lambda: [
        .001, .005, .01, .025, .05, .075, .1, .25, .5, .75, 1.0, 2.5, 5.0, 7.5, 10.0, float("inf")
    ])


class MetricsCollector:
    """Centralized metrics collector for supply chain operations"""
    
    def __init__(self, config: MetricsConfig):
        self.config = config
        self.enabled = config.enabled and PROMETHEUS_AVAILABLE
        
        if not self.enabled:
            logger.warning("Prometheus metrics disabled or not available")
            return
        
        # Initialize registry
        global registry
        registry = CollectorRegistry()
        
        # HTTP Request Metrics
        self.http_requests_total = PrometheusCounter(
            name=f"{config.namespace}_http_requests_total",
            documentation="Total HTTP requests",
            labelnames=["method", "endpoint", "status_code", "service"],
            registry=registry
        )
        
        self.http_request_duration = PrometheusHistogram(
            name=f"{config.namespace}_http_request_duration_seconds",
            documentation="HTTP request duration in seconds",
            labelnames=["method", "endpoint", "service"],
            buckets=config.histogram_buckets,
            registry=registry
        )
        
        # Database Metrics
        self.db_queries_total = PrometheusCounter(
            name=f"{config.namespace}_database_queries_total",
            documentation="Total database queries",
            labelnames=["operation", "table", "status"],
            registry=registry
        )
        
        self.db_query_duration = PrometheusHistogram(
            name=f"{config.namespace}_database_query_duration_seconds",
            documentation="Database query duration in seconds",
            labelnames=["operation", "table"],
            buckets=config.histogram_buckets,
            registry=registry
        )
        
        self.db_connections = PrometheusGauge(
            name=f"{config.namespace}_database_connections",
            documentation="Current database connections",
            labelnames=["pool", "state"],
            registry=registry
        )
        
        # Cache Metrics
        self.cache_operations_total = PrometheusCounter(
            name=f"{config.namespace}_cache_operations_total",
            documentation="Total cache operations",
            labelnames=["operation", "backend", "status"],
            registry=registry
        )
        
        self.cache_hit_rate = PrometheusGauge(
            name=f"{config.namespace}_cache_hit_rate",
            documentation="Cache hit rate",
            labelnames=["backend"],
            registry=registry
        )
        
        # Business Metrics
        if config.collect_business_metrics:
            self._init_business_metrics()
        
        # Performance Metrics
        if config.collect_performance_metrics:
            self._init_performance_metrics()
        
        # System Health Metrics
        self._init_health_metrics()
    
    def _init_business_metrics(self):
        """Initialize business-specific metrics"""
        # Component Metrics
        self.components_created_total = PrometheusCounter(
            name=f"{self.config.namespace}_components_created_total",
            documentation="Total components created",
            labelnames=["category", "manufacturer"],
            registry=registry
        )
        
        self.component_searches_total = PrometheusCounter(
            name=f"{self.config.namespace}_component_searches_total",
            documentation="Total component searches",
            labelnames=["search_type", "results_found"],
            registry=registry
        )
        
        # RFQ Metrics
        self.rfqs_created_total = PrometheusCounter(
            name=f"{self.config.namespace}_rfqs_created_total",
            documentation="Total RFQs created",
            labelnames=["source", "customer_type"],
            registry=registry
        )
        
        self.rfq_processing_duration = PrometheusHistogram(
            name=f"{self.config.namespace}_rfq_processing_duration_seconds",
            documentation="RFQ processing duration in seconds",
            labelnames=["complexity", "item_count_range"],
            buckets=self.config.histogram_buckets,
            registry=registry
        )
        
        self.rfq_response_rate = PrometheusGauge(
            name=f"{self.config.namespace}_rfq_response_rate",
            documentation="RFQ response rate by suppliers",
            labelnames=["supplier_tier", "component_category"],
            registry=registry
        )
        
        # Supplier Metrics
        self.supplier_performance_score = PrometheusGauge(
            name=f"{self.config.namespace}_supplier_performance_score",
            documentation="Supplier performance score",
            labelnames=["supplier_id", "category"],
            registry=registry
        )
        
        self.price_changes_total = PrometheusCounter(
            name=f"{self.config.namespace}_price_changes_total",
            documentation="Total price changes detected",
            labelnames=["component_category", "change_type", "source"],
            registry=registry
        )
        
        # Market Intelligence Metrics
        self.market_alerts_total = PrometheusCounter(
            name=f"{self.config.namespace}_market_alerts_total",
            documentation="Total market alerts generated",
            labelnames=["alert_type", "severity", "source"],
            registry=registry
        )
        
        self.intelligence_accuracy = PrometheusGauge(
            name=f"{self.config.namespace}_intelligence_accuracy_score",
            documentation="Intelligence accuracy score",
            labelnames=["intelligence_type", "time_range"],
            registry=registry
        )
    
    def _init_performance_metrics(self):
        """Initialize performance metrics"""
        # Processing Time Metrics
        self.processing_time = PrometheusHistogram(
            name=f"{self.config.namespace}_processing_duration_seconds",
            documentation="Processing duration for various operations",
            labelnames=["operation_type", "complexity"],
            buckets=self.config.histogram_buckets,
            registry=registry
        )
        
        # Queue Metrics
        self.queue_size = PrometheusGauge(
            name=f"{self.config.namespace}_queue_size",
            documentation="Current queue size",
            labelnames=["queue_name", "priority"],
            registry=registry
        )
        
        self.queue_processing_rate = PrometheusGauge(
            name=f"{self.config.namespace}_queue_processing_rate",
            documentation="Queue processing rate (items per second)",
            labelnames=["queue_name"],
            registry=registry
        )
        
        # Resource Utilization
        self.memory_usage_bytes = PrometheusGauge(
            name=f"{self.config.namespace}_memory_usage_bytes",
            documentation="Memory usage in bytes",
            labelnames=["component"],
            registry=registry
        )
        
        self.cpu_usage_percent = PrometheusGauge(
            name=f"{self.config.namespace}_cpu_usage_percent",
            documentation="CPU usage percentage",
            labelnames=["component"],
            registry=registry
        )
    
    def _init_health_metrics(self):
        """Initialize health and availability metrics"""
        self.service_health = PrometheusGauge(
            name=f"{self.config.namespace}_service_health",
            documentation="Service health status (1=healthy, 0=unhealthy)",
            labelnames=["service_name", "component"],
            registry=registry
        )
        
        self.dependency_health = PrometheusGauge(
            name=f"{self.config.namespace}_dependency_health",
            documentation="External dependency health (1=healthy, 0=unhealthy)",
            labelnames=["dependency", "endpoint"],
            registry=registry
        )
        
        self.error_rate = PrometheusGauge(
            name=f"{self.config.namespace}_error_rate",
            documentation="Error rate percentage",
            labelnames=["component", "error_type"],
            registry=registry
        )
        
        # Info metrics
        self.build_info = PrometheusInfo(
            name=f"{self.config.namespace}_build_info",
            documentation="Build information",
            registry=registry
        )
        
        self.build_info.info({
            'version': '1.0.0',
            'service': self.config.service_name,
            'environment': 'production'
        })
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics"""
        if not self.enabled:
            return
        
        try:
            self.http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code=str(status_code),
                service=self.config.service_name
            ).inc()
            
            self.http_request_duration.labels(
                method=method,
                endpoint=endpoint,
                service=self.config.service_name
            ).observe(duration)
        except Exception as e:
            logger.error(f"Failed to record HTTP request metrics: {e}")
    
    def record_db_query(self, operation: str, table: str, duration: float, success: bool = True):
        """Record database query metrics"""
        if not self.enabled:
            return
        
        try:
            status = "success" if success else "error"
            self.db_queries_total.labels(
                operation=operation,
                table=table,
                status=status
            ).inc()
            
            self.db_query_duration.labels(
                operation=operation,
                table=table
            ).observe(duration)
        except Exception as e:
            logger.error(f"Failed to record DB query metrics: {e}")
    
    def record_cache_operation(self, operation: str, backend: str, success: bool = True):
        """Record cache operation metrics"""
        if not self.enabled:
            return
        
        try:
            status = "hit" if success and operation == "get" else "miss" if operation == "get" else "success" if success else "error"
            self.cache_operations_total.labels(
                operation=operation,
                backend=backend,
                status=status
            ).inc()
        except Exception as e:
            logger.error(f"Failed to record cache metrics: {e}")
    
    def update_cache_hit_rate(self, backend: str, hit_rate: float):
        """Update cache hit rate"""
        if not self.enabled:
            return
        
        try:
            self.cache_hit_rate.labels(backend=backend).set(hit_rate)
        except Exception as e:
            logger.error(f"Failed to update cache hit rate: {e}")
    
    def record_business_event(self, event_type: str, labels: Dict[str, str]):
        """Record business event metrics"""
        if not self.enabled or not self.config.collect_business_metrics:
            return
        
        try:
            if event_type == "component_created":
                self.components_created_total.labels(**labels).inc()
            elif event_type == "component_search":
                self.component_searches_total.labels(**labels).inc()
            elif event_type == "rfq_created":
                self.rfqs_created_total.labels(**labels).inc()
            elif event_type == "price_change":
                self.price_changes_total.labels(**labels).inc()
            elif event_type == "market_alert":
                self.market_alerts_total.labels(**labels).inc()
        except Exception as e:
            logger.error(f"Failed to record business event {event_type}: {e}")
    
    def record_processing_time(self, operation_type: str, duration: float, complexity: str = "normal"):
        """Record processing time"""
        if not self.enabled or not self.config.collect_performance_metrics:
            return
        
        try:
            self.processing_time.labels(
                operation_type=operation_type,
                complexity=complexity
            ).observe(duration)
        except Exception as e:
            logger.error(f"Failed to record processing time: {e}")
    
    def update_queue_metrics(self, queue_name: str, size: int, processing_rate: float, priority: str = "normal"):
        """Update queue metrics"""
        if not self.enabled or not self.config.collect_performance_metrics:
            return
        
        try:
            self.queue_size.labels(queue_name=queue_name, priority=priority).set(size)
            self.queue_processing_rate.labels(queue_name=queue_name).set(processing_rate)
        except Exception as e:
            logger.error(f"Failed to update queue metrics: {e}")
    
    def update_resource_usage(self, component: str, memory_bytes: int, cpu_percent: float):
        """Update resource usage metrics"""
        if not self.enabled or not self.config.collect_performance_metrics:
            return
        
        try:
            self.memory_usage_bytes.labels(component=component).set(memory_bytes)
            self.cpu_usage_percent.labels(component=component).set(cpu_percent)
        except Exception as e:
            logger.error(f"Failed to update resource usage: {e}")
    
    def update_service_health(self, service_name: str, component: str, healthy: bool):
        """Update service health status"""
        if not self.enabled:
            return
        
        try:
            health_value = 1.0 if healthy else 0.0
            self.service_health.labels(service_name=service_name, component=component).set(health_value)
        except Exception as e:
            logger.error(f"Failed to update service health: {e}")
    
    def update_dependency_health(self, dependency: str, endpoint: str, healthy: bool):
        """Update dependency health status"""
        if not self.enabled:
            return
        
        try:
            health_value = 1.0 if healthy else 0.0
            self.dependency_health.labels(dependency=dependency, endpoint=endpoint).set(health_value)
        except Exception as e:
            logger.error(f"Failed to update dependency health: {e}")
    
    def update_error_rate(self, component: str, error_type: str, rate: float):
        """Update error rate"""
        if not self.enabled:
            return
        
        try:
            self.error_rate.labels(component=component, error_type=error_type).set(rate)
        except Exception as e:
            logger.error(f"Failed to update error rate: {e}")


# Global metrics instance
metrics: Optional[MetricsCollector] = None


def init_metrics() -> bool:
    """Initialize Prometheus metrics collection"""
    global metrics, metrics_server
    
    if not PROMETHEUS_AVAILABLE:
        logger.warning("Prometheus client not available. Install with: pip install prometheus-client")
        return False
    
    try:
        config = MetricsConfig()
        metrics = MetricsCollector(config)
        
        if not metrics.enabled:
            return False
        
        # Start metrics HTTP server
        metrics_server = start_http_server(config.port, registry=registry)
        logger.info(f"Prometheus metrics server started on port {config.port}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize metrics: {e}")
        return False


def record_metric(metric_type: str, name: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None):
    """Generic function to record metrics"""
    if not metrics or not metrics.enabled:
        return
    
    labels = labels or {}
    
    try:
        if metric_type == "counter":
            # Find or create counter metric
            pass  # Would need dynamic metric creation
        elif metric_type == "histogram":
            pass  # Would need dynamic metric creation
        elif metric_type == "gauge":
            pass  # Would need dynamic metric creation
    except Exception as e:
        logger.error(f"Failed to record {metric_type} metric {name}: {e}")


def get_metrics_data() -> str:
    """Get Prometheus metrics data"""
    if not metrics or not metrics.enabled or not registry:
        return ""
    
    try:
        return generate_latest(registry)
    except Exception as e:
        logger.error(f"Failed to generate metrics data: {e}")
        return ""


# Decorators for automatic metrics collection

def time_function(operation_type: str = None, complexity: str = "normal"):
    """Decorator to automatically time function execution"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            operation = operation_type or f"{func.__module__}.{func.__name__}"
            
            try:
                result = await func(*args, **kwargs)
                if metrics:
                    metrics.record_processing_time(operation, time.time() - start_time, complexity)
                return result
            except Exception as e:
                if metrics:
                    metrics.record_processing_time(operation, time.time() - start_time, complexity)
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            operation = operation_type or f"{func.__module__}.{func.__name__}"
            
            try:
                result = func(*args, **kwargs)
                if metrics:
                    metrics.record_processing_time(operation, time.time() - start_time, complexity)
                return result
            except Exception as e:
                if metrics:
                    metrics.record_processing_time(operation, time.time() - start_time, complexity)
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def count_calls(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Decorator to count function calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if metrics:
                # Would increment a custom counter
                pass
            return func(*args, **kwargs)
        return wrapper
    return decorator