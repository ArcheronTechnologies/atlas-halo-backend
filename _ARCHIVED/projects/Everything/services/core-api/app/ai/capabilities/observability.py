"""
AI Capabilities Observability and Metrics

Comprehensive monitoring, logging, and metrics collection for AI capabilities
with integration for OpenTelemetry, Prometheus, and custom analytics.
"""

import logging
import time
import asyncio
import json
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from contextlib import asynccontextmanager
import statistics

# Optional OpenTelemetry imports
try:
    from opentelemetry import trace, metrics
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.trace import TracerProvider
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    # Fallback classes
    class trace:
        @staticmethod
        def get_tracer(name): return NoOpTracer()
    class metrics:
        @staticmethod 
        def get_meter(name): return NoOpMeter()

from .base import AICapability, CapabilityResult, CapabilityStatus

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics collected"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricPoint:
    """Individual metric data point"""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    unit: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Performance metrics for capability execution"""
    capability_name: str
    execution_count: int
    success_count: int
    failure_count: int
    avg_execution_time_ms: float
    min_execution_time_ms: float
    max_execution_time_ms: float
    p50_execution_time_ms: float
    p95_execution_time_ms: float
    p99_execution_time_ms: float
    cache_hit_rate: float
    error_rate: float
    throughput_per_second: float
    concurrent_executions_avg: float
    last_updated: datetime


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    description: str
    metric_name: str
    condition: str  # "gt", "lt", "eq", "gte", "lte"
    threshold: float
    duration_seconds: int
    severity: str  # "critical", "warning", "info"
    enabled: bool = True
    cooldown_seconds: int = 300


@dataclass
class Alert:
    """Active alert"""
    rule: AlertRule
    capability_name: str
    triggered_at: datetime
    current_value: float
    message: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class NoOpTracer:
    """No-op tracer when OpenTelemetry is not available"""
    def start_span(self, name, **kwargs):
        return NoOpSpan()

class NoOpSpan:
    """No-op span when OpenTelemetry is not available"""
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def set_status(self, *args): pass
    def set_attribute(self, *args): pass
    def add_event(self, *args): pass

class NoOpMeter:
    """No-op meter when OpenTelemetry is not available"""
    def create_counter(self, *args, **kwargs): return NoOpInstrument()
    def create_histogram(self, *args, **kwargs): return NoOpInstrument()
    def create_gauge(self, *args, **kwargs): return NoOpInstrument()

class NoOpInstrument:
    """No-op instrument when OpenTelemetry is not available"""
    def add(self, *args, **kwargs): pass
    def record(self, *args, **kwargs): pass
    def set(self, *args, **kwargs): pass


class MetricsCollector:
    """Centralized metrics collection and aggregation"""
    
    def __init__(self):
        self.metrics: Dict[str, List[MetricPoint]] = {}
        self.performance_cache: Dict[str, PerformanceMetrics] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.last_cleanup = datetime.now(timezone.utc)
        self.retention_hours = 24
        
        # Initialize OpenTelemetry if available
        if OPENTELEMETRY_AVAILABLE:
            self.tracer = trace.get_tracer(__name__)
            self.meter = metrics.get_meter(__name__)
            self._init_instruments()
        else:
            self.tracer = NoOpTracer()
            self.meter = NoOpMeter()
            logger.warning("OpenTelemetry not available, using no-op implementations")
    
    def _init_instruments(self):
        """Initialize OpenTelemetry instruments"""
        if not OPENTELEMETRY_AVAILABLE:
            return
            
        self.execution_counter = self.meter.create_counter(
            name="ai_capability_executions_total",
            description="Total number of capability executions",
            unit="1"
        )
        
        self.execution_duration = self.meter.create_histogram(
            name="ai_capability_execution_duration_ms",
            description="Capability execution duration in milliseconds",
            unit="ms"
        )
        
        self.cache_hit_counter = self.meter.create_counter(
            name="ai_capability_cache_hits_total",
            description="Total number of cache hits",
            unit="1"
        )
        
        self.error_counter = self.meter.create_counter(
            name="ai_capability_errors_total",
            description="Total number of capability errors",
            unit="1"
        )
        
        self.concurrent_executions = self.meter.create_gauge(
            name="ai_capability_concurrent_executions",
            description="Current number of concurrent executions",
            unit="1"
        )
    
    def record_metric(
        self, 
        name: str, 
        value: Union[int, float], 
        metric_type: MetricType,
        labels: Optional[Dict[str, str]] = None,
        unit: Optional[str] = None
    ):
        """Record a metric point"""
        metric_point = MetricPoint(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.now(timezone.utc),
            labels=labels or {},
            unit=unit
        )
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(metric_point)
        
        # Also record to OpenTelemetry if available
        self._record_otel_metric(metric_point)
        
        # Check for alerts
        self._check_alerts(metric_point)
        
        # Periodic cleanup
        self._maybe_cleanup()
    
    def _record_otel_metric(self, metric_point: MetricPoint):
        """Record metric to OpenTelemetry"""
        if not OPENTELEMETRY_AVAILABLE:
            return
            
        labels_dict = metric_point.labels
        
        try:
            if metric_point.metric_type == MetricType.COUNTER:
                if hasattr(self, 'execution_counter'):
                    self.execution_counter.add(metric_point.value, labels_dict)
            elif metric_point.metric_type == MetricType.HISTOGRAM:
                if hasattr(self, 'execution_duration'):
                    self.execution_duration.record(metric_point.value, labels_dict)
            elif metric_point.metric_type == MetricType.GAUGE:
                if hasattr(self, 'concurrent_executions'):
                    self.concurrent_executions.set(metric_point.value, labels_dict)
        except Exception as e:
            logger.warning(f"Failed to record OpenTelemetry metric: {e}")
    
    def record_capability_execution(
        self, 
        capability_name: str, 
        result: CapabilityResult,
        execution_time_ms: float,
        cache_hit: bool = False,
        concurrent_count: int = 0
    ):
        """Record capability execution metrics"""
        labels = {
            "capability_name": capability_name,
            "success": str(result.success).lower()
        }
        
        # Record execution count
        self.record_metric(
            "capability_executions_total",
            1,
            MetricType.COUNTER,
            labels
        )
        
        # Record execution time
        self.record_metric(
            "capability_execution_duration_ms",
            execution_time_ms,
            MetricType.HISTOGRAM,
            labels,
            "ms"
        )
        
        # Record cache hit
        if cache_hit:
            self.record_metric(
                "capability_cache_hits_total",
                1,
                MetricType.COUNTER,
                labels
            )
        
        # Record errors
        if not result.success:
            error_labels = {**labels, "error_type": "execution_failure"}
            self.record_metric(
                "capability_errors_total",
                1,
                MetricType.COUNTER,
                error_labels
            )
        
        # Record concurrent executions
        if concurrent_count > 0:
            self.record_metric(
                "capability_concurrent_executions",
                concurrent_count,
                MetricType.GAUGE,
                {"capability_name": capability_name}
            )
        
        # Update performance cache
        self._update_performance_metrics(capability_name, result, execution_time_ms, cache_hit)
    
    def _update_performance_metrics(
        self, 
        capability_name: str, 
        result: CapabilityResult,
        execution_time_ms: float,
        cache_hit: bool
    ):
        """Update aggregated performance metrics"""
        now = datetime.now(timezone.utc)
        
        if capability_name not in self.performance_cache:
            self.performance_cache[capability_name] = PerformanceMetrics(
                capability_name=capability_name,
                execution_count=0,
                success_count=0,
                failure_count=0,
                avg_execution_time_ms=0.0,
                min_execution_time_ms=float('inf'),
                max_execution_time_ms=0.0,
                p50_execution_time_ms=0.0,
                p95_execution_time_ms=0.0,
                p99_execution_time_ms=0.0,
                cache_hit_rate=0.0,
                error_rate=0.0,
                throughput_per_second=0.0,
                concurrent_executions_avg=0.0,
                last_updated=now
            )
        
        metrics = self.performance_cache[capability_name]
        
        # Update counters
        metrics.execution_count += 1
        if result.success:
            metrics.success_count += 1
        else:
            metrics.failure_count += 1
        
        # Update execution time stats
        metrics.avg_execution_time_ms = (
            (metrics.avg_execution_time_ms * (metrics.execution_count - 1) + execution_time_ms) 
            / metrics.execution_count
        )
        metrics.min_execution_time_ms = min(metrics.min_execution_time_ms, execution_time_ms)
        metrics.max_execution_time_ms = max(metrics.max_execution_time_ms, execution_time_ms)
        
        # Calculate percentiles from recent execution times
        recent_times = self._get_recent_execution_times(capability_name, hours=1)
        if recent_times:
            sorted_times = sorted(recent_times)
            n = len(sorted_times)
            metrics.p50_execution_time_ms = sorted_times[n // 2] if n > 0 else 0
            metrics.p95_execution_time_ms = sorted_times[int(n * 0.95)] if n > 0 else 0
            metrics.p99_execution_time_ms = sorted_times[int(n * 0.99)] if n > 0 else 0
        
        # Update rates
        metrics.error_rate = metrics.failure_count / metrics.execution_count
        
        # Calculate cache hit rate from recent cache hits
        recent_cache_hits = self._get_recent_cache_hits(capability_name, hours=1)
        recent_executions = len(recent_times) if recent_times else 1
        metrics.cache_hit_rate = len(recent_cache_hits) / recent_executions
        
        # Calculate throughput (executions per second over last hour)
        hour_ago = now - timedelta(hours=1)
        recent_execution_count = len([
            t for t in recent_times 
            if self._get_metric_timestamp(capability_name, "capability_executions_total") > hour_ago
        ])
        metrics.throughput_per_second = recent_execution_count / 3600.0
        
        metrics.last_updated = now
    
    def _get_recent_execution_times(self, capability_name: str, hours: int = 1) -> List[float]:
        """Get recent execution times for a capability"""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        execution_metrics = self.metrics.get("capability_execution_duration_ms", [])
        return [
            m.value for m in execution_metrics
            if m.labels.get("capability_name") == capability_name and m.timestamp > cutoff
        ]
    
    def _get_recent_cache_hits(self, capability_name: str, hours: int = 1) -> List[MetricPoint]:
        """Get recent cache hits for a capability"""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        cache_metrics = self.metrics.get("capability_cache_hits_total", [])
        return [
            m for m in cache_metrics
            if m.labels.get("capability_name") == capability_name and m.timestamp > cutoff
        ]
    
    def _get_metric_timestamp(self, capability_name: str, metric_name: str) -> datetime:
        """Get latest timestamp for a metric"""
        metrics_list = self.metrics.get(metric_name, [])
        relevant_metrics = [
            m for m in metrics_list 
            if m.labels.get("capability_name") == capability_name
        ]
        return max((m.timestamp for m in relevant_metrics), 
                  default=datetime.min.replace(tzinfo=timezone.utc))
    
    def get_performance_metrics(self, capability_name: Optional[str] = None) -> Dict[str, PerformanceMetrics]:
        """Get performance metrics for capabilities"""
        if capability_name:
            return {capability_name: self.performance_cache.get(capability_name)}
        return self.performance_cache.copy()
    
    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def _check_alerts(self, metric_point: MetricPoint):
        """Check if metric point triggers any alerts"""
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled or rule.metric_name != metric_point.name:
                continue
            
            capability_name = metric_point.labels.get("capability_name", "unknown")
            alert_key = f"{rule_name}_{capability_name}"
            
            # Check if alert is already active and in cooldown
            if alert_key in self.active_alerts:
                alert = self.active_alerts[alert_key]
                if not alert.resolved:
                    time_since_trigger = (datetime.now(timezone.utc) - alert.triggered_at).total_seconds()
                    if time_since_trigger < rule.cooldown_seconds:
                        continue
            
            # Evaluate condition
            triggered = self._evaluate_alert_condition(rule, metric_point.value)
            
            if triggered:
                alert = Alert(
                    rule=rule,
                    capability_name=capability_name,
                    triggered_at=datetime.now(timezone.utc),
                    current_value=metric_point.value,
                    message=f"{rule.description}: {metric_point.value} {rule.condition} {rule.threshold}"
                )
                
                self.active_alerts[alert_key] = alert
                logger.warning(f"Alert triggered: {alert.message}")
                
                # Could integrate with external alerting systems here
                self._send_alert_notification(alert)
    
    def _evaluate_alert_condition(self, rule: AlertRule, value: float) -> bool:
        """Evaluate if a value triggers an alert condition"""
        if rule.condition == "gt":
            return value > rule.threshold
        elif rule.condition == "gte":
            return value >= rule.threshold
        elif rule.condition == "lt":
            return value < rule.threshold
        elif rule.condition == "lte":
            return value <= rule.threshold
        elif rule.condition == "eq":
            return value == rule.threshold
        return False
    
    def _send_alert_notification(self, alert: Alert):
        """Send alert notification (placeholder for integration)"""
        # This could integrate with Slack, PagerDuty, email, etc.
        logger.critical(f"ALERT [{alert.rule.severity.upper()}]: {alert.message}")
    
    def resolve_alert(self, alert_key: str):
        """Manually resolve an alert"""
        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            alert.resolved = True
            alert.resolved_at = datetime.now(timezone.utc)
            logger.info(f"Alert resolved: {alert_key}")
    
    def get_active_alerts(self) -> Dict[str, Alert]:
        """Get all active alerts"""
        return {k: v for k, v in self.active_alerts.items() if not v.resolved}
    
    def _maybe_cleanup(self):
        """Periodic cleanup of old metrics"""
        now = datetime.now(timezone.utc)
        if (now - self.last_cleanup).total_seconds() < 3600:  # Cleanup every hour
            return
        
        cutoff = now - timedelta(hours=self.retention_hours)
        
        for metric_name in list(self.metrics.keys()):
            self.metrics[metric_name] = [
                m for m in self.metrics[metric_name] 
                if m.timestamp > cutoff
            ]
            
            # Remove empty metric lists
            if not self.metrics[metric_name]:
                del self.metrics[metric_name]
        
        # Cleanup resolved alerts older than 24 hours
        alert_cutoff = now - timedelta(hours=24)
        self.active_alerts = {
            k: v for k, v in self.active_alerts.items()
            if not v.resolved or v.resolved_at > alert_cutoff
        }
        
        self.last_cleanup = now
        logger.debug("Completed metrics cleanup")
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics for external systems"""
        return {
            "metrics": {
                name: [asdict(point) for point in points]
                for name, points in self.metrics.items()
            },
            "performance_metrics": {
                name: asdict(perf) for name, perf in self.performance_cache.items()
            },
            "active_alerts": {
                name: asdict(alert) for name, alert in self.get_active_alerts().items()
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class CapabilityObserver:
    """Observability wrapper for individual capabilities"""
    
    def __init__(self, capability: AICapability, metrics_collector: MetricsCollector):
        self.capability = capability
        self.metrics = metrics_collector
        self.tracer = metrics_collector.tracer
    
    @asynccontextmanager
    async def trace_execution(self, operation_name: str, **span_attributes):
        """Context manager for tracing capability operations"""
        with self.tracer.start_span(operation_name) as span:
            # Set span attributes
            span.set_attribute("capability.name", self.capability.name)
            span.set_attribute("capability.version", self.capability.version)
            
            for key, value in span_attributes.items():
                span.set_attribute(key, str(value))
            
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.add_event("exception", {"exception.message": str(e)})
                raise
    
    async def run_with_observability(self, payload: Dict[str, Any]) -> CapabilityResult:
        """Execute capability with full observability"""
        start_time = time.time()
        
        async with self.trace_execution("capability_execution", **payload) as span:
            # Record start metrics
            self.metrics.record_metric(
                "capability_starts_total",
                1,
                MetricType.COUNTER,
                {"capability_name": self.capability.name}
            )
            
            try:
                # Execute capability
                result = await self.capability.run(payload)
                
                # Calculate execution time
                execution_time_ms = (time.time() - start_time) * 1000
                
                # Record detailed metrics
                cache_hit = result.metadata.get('cache_hit', False) if result.metadata else False
                concurrent_count = result.metadata.get('concurrent_executions', 0) if result.metadata else 0
                
                self.metrics.record_capability_execution(
                    self.capability.name,
                    result,
                    execution_time_ms,
                    cache_hit,
                    concurrent_count
                )
                
                # Add span details
                span.set_attribute("execution.success", result.success)
                span.set_attribute("execution.time_ms", execution_time_ms)
                span.set_attribute("execution.cache_hit", cache_hit)
                
                if result.confidence is not None:
                    span.set_attribute("result.confidence", result.confidence)
                
                if result.error:
                    span.add_event("error", {"error.message": result.error})
                
                return result
                
            except Exception as e:
                execution_time_ms = (time.time() - start_time) * 1000
                
                # Record error metrics
                self.metrics.record_metric(
                    "capability_exceptions_total",
                    1,
                    MetricType.COUNTER,
                    {
                        "capability_name": self.capability.name,
                        "exception_type": type(e).__name__
                    }
                )
                
                raise


class ObservabilityManager:
    """Central observability manager for all AI capabilities"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.capability_observers: Dict[str, CapabilityObserver] = {}
        self._setup_default_alerts()
    
    def _setup_default_alerts(self):
        """Setup default alert rules"""
        # High error rate alert
        self.metrics_collector.add_alert_rule(AlertRule(
            name="high_error_rate",
            description="High error rate detected",
            metric_name="capability_executions_total",
            condition="gt",
            threshold=0.1,  # 10% error rate
            duration_seconds=300,
            severity="warning"
        ))
        
        # Slow execution alert
        self.metrics_collector.add_alert_rule(AlertRule(
            name="slow_execution",
            description="Slow capability execution detected",
            metric_name="capability_execution_duration_ms",
            condition="gt",
            threshold=30000,  # 30 seconds
            duration_seconds=60,
            severity="warning"
        ))
        
        # High concurrent executions
        self.metrics_collector.add_alert_rule(AlertRule(
            name="high_concurrency",
            description="High concurrent executions detected",
            metric_name="capability_concurrent_executions",
            condition="gt",
            threshold=50,
            duration_seconds=120,
            severity="critical"
        ))
    
    def register_capability(self, capability: AICapability) -> CapabilityObserver:
        """Register a capability for observability"""
        observer = CapabilityObserver(capability, self.metrics_collector)
        self.capability_observers[capability.name] = observer
        logger.info(f"Registered capability for observability: {capability.name}")
        return observer
    
    def get_capability_observer(self, capability_name: str) -> Optional[CapabilityObserver]:
        """Get observer for a capability"""
        return self.capability_observers.get(capability_name)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        performance_metrics = self.metrics_collector.get_performance_metrics()
        active_alerts = self.metrics_collector.get_active_alerts()
        
        # Calculate overall health score
        health_score = 1.0
        total_capabilities = len(performance_metrics)
        
        if total_capabilities > 0:
            total_error_rate = sum(m.error_rate for m in performance_metrics.values()) / total_capabilities
            health_score = max(0.0, 1.0 - total_error_rate)
        
        # Determine health status
        if health_score >= 0.95:
            health_status = "healthy"
        elif health_score >= 0.8:
            health_status = "degraded"
        else:
            health_status = "unhealthy"
        
        return {
            "health_status": health_status,
            "health_score": round(health_score, 3),
            "total_capabilities": total_capabilities,
            "active_alerts": len(active_alerts),
            "critical_alerts": len([a for a in active_alerts.values() if a.rule.severity == "critical"]),
            "capabilities_status": {
                name: {
                    "error_rate": metrics.error_rate,
                    "avg_execution_time_ms": metrics.avg_execution_time_ms,
                    "throughput_per_second": metrics.throughput_per_second,
                    "cache_hit_rate": metrics.cache_hit_rate
                }
                for name, metrics in performance_metrics.items()
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics for all capabilities"""
        return {
            "system_health": self.get_system_health(),
            "performance_metrics": {
                name: asdict(metrics) 
                for name, metrics in self.metrics_collector.get_performance_metrics().items()
            },
            "active_alerts": {
                name: asdict(alert) 
                for name, alert in self.metrics_collector.get_active_alerts().items()
            },
            "raw_metrics": self.metrics_collector.export_metrics()
        }


# Global observability manager instance
observability_manager = ObservabilityManager()