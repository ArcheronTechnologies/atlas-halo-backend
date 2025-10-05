"""
Atlas AI Comprehensive Metrics Collection System
Advanced monitoring and metrics for production deployment
"""

import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
from collections import defaultdict, deque
import psutil
import threading
import os

from ..config.production_settings import get_config
from ..database.postgis_database import get_database
from ..caching.redis_cache import get_cache


class MetricType(str, Enum):
    """Metric type enumeration."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    SET = "set"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    metric_pattern: str
    condition: str  # e.g., "> 100", "< 0.5"
    severity: AlertSeverity
    duration_seconds: int
    description: str
    enabled: bool = True


@dataclass
class Alert:
    """Active alert."""
    id: str
    rule_name: str
    message: str
    severity: AlertSeverity
    triggered_at: datetime
    value: float
    acknowledged: bool = False
    resolved: bool = False


class MetricsCollector:
    """Comprehensive metrics collection system."""
    
    def __init__(self, collection_interval: int = 30):
        self.collection_interval = collection_interval
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # System for tracking business metrics
        self.business_metrics = {
            'api_requests_total': 0,
            'api_requests_success': 0,
            'api_requests_error': 0,
            'location_predictions_total': 0,
            'area_analyses_total': 0,
            'batch_requests_total': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'model_predictions': 0,
            'threat_detections': 0,
            'high_risk_alerts': 0
        }
        
        # Alert system
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # Performance tracking
        self.response_times: deque = deque(maxlen=1000)
        self.error_rates: deque = deque(maxlen=100)
        
        # External dependencies
        self.db = None
        self.cache = None
        
        self._setup_default_alerts()
    
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        self.alert_rules = [
            AlertRule(
                name="high_response_time",
                metric_pattern="response_time_ms",
                condition="> 2000",
                severity=AlertSeverity.WARNING,
                duration_seconds=300,
                description="API response time is above 2 seconds"
            ),
            AlertRule(
                name="low_cache_hit_rate",
                metric_pattern="cache_hit_rate",
                condition="< 0.3",
                severity=AlertSeverity.WARNING,
                duration_seconds=600,
                description="Cache hit rate is below 30%"
            ),
            AlertRule(
                name="high_error_rate",
                metric_pattern="error_rate",
                condition="> 0.05",
                severity=AlertSeverity.ERROR,
                duration_seconds=120,
                description="Error rate is above 5%"
            ),
            AlertRule(
                name="high_memory_usage",
                metric_pattern="memory_usage_percent",
                condition="> 85",
                severity=AlertSeverity.WARNING,
                duration_seconds=300,
                description="Memory usage is above 85%"
            ),
            AlertRule(
                name="database_connection_failure",
                metric_pattern="database_available",
                condition="== 0",
                severity=AlertSeverity.CRITICAL,
                duration_seconds=30,
                description="Database connection is unavailable"
            )
        ]
    
    async def initialize(self):
        """Initialize metrics collector."""
        try:
            self.db = await get_database()
            self.cache = await get_cache()
            self.logger.info("✅ Metrics collector initialized")
            
            # Start collection tasks
            asyncio.create_task(self._collect_system_metrics())
            asyncio.create_task(self._collect_application_metrics())
            asyncio.create_task(self._check_alerts())
            
        except Exception as e:
            self.logger.error(f"Failed to initialize metrics collector: {e}")
    
    # Metric recording methods
    def increment_counter(self, name: str, value: float = 1.0, tags: Dict[str, str] = None):
        """Increment a counter metric."""
        self.counters[name] += value
        self.record_metric(name, self.counters[name], MetricType.COUNTER, tags)
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None):
        """Set a gauge metric."""
        self.gauges[name] = value
        self.record_metric(name, value, MetricType.GAUGE, tags)
    
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a histogram value."""
        self.histograms[name].append(value)
        # Keep only last 1000 values
        if len(self.histograms[name]) > 1000:
            self.histograms[name] = self.histograms[name][-1000:]
        
        self.record_metric(name, value, MetricType.HISTOGRAM, tags)
    
    def record_timer(self, name: str, duration_ms: float, tags: Dict[str, str] = None):
        """Record a timer metric."""
        self.timers[name].append(duration_ms)
        self.record_metric(name, duration_ms, MetricType.TIMER, tags)
    
    def record_metric(self, name: str, value: float, metric_type: MetricType, 
                     tags: Dict[str, str] = None):
        """Record a generic metric."""
        metric = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            tags=tags or {},
            metric_type=metric_type
        )
        
        self.metrics[name].append(metric)
    
    # Business metric tracking
    def track_api_request(self, endpoint: str, method: str, status_code: int, 
                         duration_ms: float):
        """Track API request metrics."""
        tags = {
            'endpoint': endpoint,
            'method': method,
            'status_code': str(status_code)
        }
        
        self.increment_counter('api_requests_total', tags=tags)
        
        if 200 <= status_code < 400:
            self.increment_counter('api_requests_success', tags=tags)
            self.business_metrics['api_requests_success'] += 1
        else:
            self.increment_counter('api_requests_error', tags=tags)
            self.business_metrics['api_requests_error'] += 1
        
        self.business_metrics['api_requests_total'] += 1
        self.record_timer('api_response_time', duration_ms, tags)
        self.response_times.append(duration_ms)
    
    def track_prediction_request(self, prediction_type: str, processing_time_ms: float,
                               cache_hit: bool = False):
        """Track prediction request metrics."""
        tags = {'type': prediction_type, 'cached': str(cache_hit)}
        
        self.increment_counter('predictions_total', tags=tags)
        self.record_timer('prediction_processing_time', processing_time_ms, tags)
        
        if prediction_type == 'location':
            self.business_metrics['location_predictions_total'] += 1
        elif prediction_type == 'area':
            self.business_metrics['area_analyses_total'] += 1
        elif prediction_type == 'batch':
            self.business_metrics['batch_requests_total'] += 1
        
        if cache_hit:
            self.business_metrics['cache_hits'] += 1
        else:
            self.business_metrics['cache_misses'] += 1
    
    def track_threat_detection(self, threat_level: str, confidence: float):
        """Track threat detection metrics."""
        tags = {'threat_level': threat_level}
        
        self.increment_counter('threat_detections_total', tags=tags)
        self.record_histogram('threat_confidence', confidence, tags)
        
        self.business_metrics['threat_detections'] += 1
        
        if threat_level in ['high', 'critical']:
            self.increment_counter('high_risk_alerts_total', tags=tags)
            self.business_metrics['high_risk_alerts'] += 1
    
    # System metrics collection
    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        while True:
            try:
                # CPU and Memory
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                self.set_gauge('cpu_usage_percent', cpu_percent)
                self.set_gauge('memory_usage_percent', memory.percent)
                self.set_gauge('memory_usage_mb', memory.used / 1024 / 1024)
                self.set_gauge('disk_usage_percent', disk.percent)
                
                # Network I/O
                network = psutil.net_io_counters()
                self.set_gauge('network_bytes_sent', network.bytes_sent)
                self.set_gauge('network_bytes_recv', network.bytes_recv)
                
                # Process-specific metrics
                process = psutil.Process()
                self.set_gauge('process_memory_mb', process.memory_info().rss / 1024 / 1024)
                self.set_gauge('process_cpu_percent', process.cpu_percent())
                self.set_gauge('process_threads', process.num_threads())
                
                # Database availability
                if self.db:
                    try:
                        await self.db.execute_query("SELECT 1")
                        self.set_gauge('database_available', 1)
                        
                        # Database connection pool metrics if available
                        if hasattr(self.db, 'pool'):
                            pool = self.db.pool
                            self.set_gauge('database_pool_size', pool.get_size())
                            self.set_gauge('database_pool_available', pool.get_idle_size())
                    except Exception:
                        self.set_gauge('database_available', 0)
                
                # Cache availability and stats
                if self.cache:
                    try:
                        await self.cache.ping()
                        self.set_gauge('cache_available', 1)
                        
                        # Get cache statistics if available
                        try:
                            stats = await self.cache.get_cache_stats()
                            if isinstance(stats, dict):
                                for key, value in stats.items():
                                    if isinstance(value, (int, float)):
                                        self.set_gauge(f'cache_{key}', value)
                        except Exception:
                            pass  # Cache stats not available
                    except Exception:
                        self.set_gauge('cache_available', 0)
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"System metrics collection error: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_application_metrics(self):
        """Collect application-specific metrics."""
        while True:
            try:
                # Calculate derived metrics
                total_requests = self.business_metrics['api_requests_total']
                error_requests = self.business_metrics['api_requests_error']
                
                if total_requests > 0:
                    error_rate = error_requests / total_requests
                    self.set_gauge('error_rate', error_rate)
                    self.error_rates.append(error_rate)
                
                # Cache hit rate
                total_cache_ops = (self.business_metrics['cache_hits'] + 
                                 self.business_metrics['cache_misses'])
                if total_cache_ops > 0:
                    hit_rate = self.business_metrics['cache_hits'] / total_cache_ops
                    self.set_gauge('cache_hit_rate', hit_rate)
                
                # Response time percentiles
                if self.response_times:
                    times = sorted(list(self.response_times))
                    n = len(times)
                    
                    self.set_gauge('response_time_p50', times[int(n * 0.5)])
                    self.set_gauge('response_time_p95', times[int(n * 0.95)])
                    self.set_gauge('response_time_p99', times[int(n * 0.99)])
                    self.set_gauge('response_time_avg', sum(times) / n)
                
                # Business metrics
                for metric_name, value in self.business_metrics.items():
                    self.set_gauge(f'business_{metric_name}', value)
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Application metrics collection error: {e}")
                await asyncio.sleep(60)
    
    # Alert system
    async def _check_alerts(self):
        """Check alert conditions."""
        while True:
            try:
                for rule in self.alert_rules:
                    if not rule.enabled:
                        continue
                    
                    await self._evaluate_alert_rule(rule)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Alert checking error: {e}")
                await asyncio.sleep(30)
    
    async def _evaluate_alert_rule(self, rule: AlertRule):
        """Evaluate a single alert rule."""
        try:
            # Get recent metric values
            if rule.metric_pattern not in self.metrics:
                return
            
            recent_metrics = [m for m in self.metrics[rule.metric_pattern] 
                            if m.timestamp > datetime.utcnow() - timedelta(seconds=rule.duration_seconds)]
            
            if not recent_metrics:
                return
            
            # Evaluate condition
            latest_value = recent_metrics[-1].value
            condition_met = self._evaluate_condition(latest_value, rule.condition)
            
            alert_id = f"{rule.name}_{rule.metric_pattern}"
            
            if condition_met and alert_id not in self.active_alerts:
                # Trigger alert
                alert = Alert(
                    id=alert_id,
                    rule_name=rule.name,
                    message=f"{rule.description}. Current value: {latest_value}",
                    severity=rule.severity,
                    triggered_at=datetime.utcnow(),
                    value=latest_value
                )
                
                self.active_alerts[alert_id] = alert
                self.alert_history.append(alert)
                
                self.logger.warning(f"🚨 Alert triggered: {alert.message}")
                
            elif not condition_met and alert_id in self.active_alerts:
                # Resolve alert
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                del self.active_alerts[alert_id]
                
                self.logger.info(f"✅ Alert resolved: {rule.name}")
        
        except Exception as e:
            self.logger.error(f"Alert rule evaluation error: {e}")
    
    def _evaluate_condition(self, value: float, condition: str) -> bool:
        """Evaluate alert condition."""
        try:
            # Parse condition (e.g., "> 100", "< 0.5", "== 0")
            condition = condition.strip()
            
            if condition.startswith(">="):
                threshold = float(condition[2:].strip())
                return value >= threshold
            elif condition.startswith("<="):
                threshold = float(condition[2:].strip())
                return value <= threshold
            elif condition.startswith(">"):
                threshold = float(condition[1:].strip())
                return value > threshold
            elif condition.startswith("<"):
                threshold = float(condition[1:].strip())
                return value < threshold
            elif condition.startswith("=="):
                threshold = float(condition[2:].strip())
                return value == threshold
            elif condition.startswith("!="):
                threshold = float(condition[2:].strip())
                return value != threshold
            
            return False
            
        except Exception:
            return False
    
    # Data export and reporting
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics summary for the last N hours."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        summary = {
            "time_range_hours": hours,
            "system_metrics": {},
            "business_metrics": dict(self.business_metrics),
            "active_alerts": [asdict(alert) for alert in self.active_alerts.values()],
            "alert_count": len(self.active_alerts),
            "performance_summary": {}
        }
        
        # System metrics summary
        for metric_name in ['cpu_usage_percent', 'memory_usage_percent', 'error_rate', 'cache_hit_rate']:
            if metric_name in self.metrics:
                recent_values = [m.value for m in self.metrics[metric_name] 
                               if m.timestamp > cutoff]
                if recent_values:
                    summary["system_metrics"][metric_name] = {
                        "current": recent_values[-1],
                        "average": sum(recent_values) / len(recent_values),
                        "min": min(recent_values),
                        "max": max(recent_values)
                    }
        
        # Performance summary
        if self.response_times:
            times = list(self.response_times)
            summary["performance_summary"] = {
                "avg_response_time_ms": sum(times) / len(times),
                "total_requests": len(times),
                "requests_per_hour": len(times) * (1 / hours)
            }
        
        return summary
    
    def export_metrics_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        for metric_name, metric_queue in self.metrics.items():
            if not metric_queue:
                continue
            
            latest_metric = metric_queue[-1]
            
            # Prometheus metric line
            metric_line = f"atlas_ai_{metric_name}"
            
            # Add tags
            if latest_metric.tags:
                tag_str = ",".join([f'{k}="{v}"' for k, v in latest_metric.tags.items()])
                metric_line += f"{{{tag_str}}}"
            
            metric_line += f" {latest_metric.value}"
            lines.append(metric_line)
        
        return "\n".join(lines)


# Global metrics collector instance
_global_metrics: Optional[MetricsCollector] = None


async def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _global_metrics
    
    if _global_metrics is None:
        config = get_config()
        collection_interval = getattr(config.monitoring, 'collection_interval', 30)
        
        _global_metrics = MetricsCollector(collection_interval)
        await _global_metrics.initialize()
    
    return _global_metrics


# Convenience functions
async def track_api_request(endpoint: str, method: str, status_code: int, duration_ms: float):
    """Track API request metrics."""
    collector = await get_metrics_collector()
    collector.track_api_request(endpoint, method, status_code, duration_ms)


async def track_prediction(prediction_type: str, processing_time_ms: float, cache_hit: bool = False):
    """Track prediction metrics."""
    collector = await get_metrics_collector()
    collector.track_prediction_request(prediction_type, processing_time_ms, cache_hit)


async def track_threat_detection(threat_level: str, confidence: float):
    """Track threat detection metrics."""
    collector = await get_metrics_collector()
    collector.track_threat_detection(threat_level, confidence)


async def get_system_metrics() -> Dict[str, Any]:
    """Get current system metrics."""
    collector = await get_metrics_collector()
    return collector.get_metrics_summary(hours=1)


# Decorator for automatic performance tracking
def track_performance(endpoint_name: str):
    """Decorator for automatic performance tracking."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                
                # Determine status code from result
                status_code = 200
                if hasattr(result, 'status_code'):
                    status_code = result.status_code
                
                await track_api_request(endpoint_name, "API", status_code, duration_ms)
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                await track_api_request(endpoint_name, "API", 500, duration_ms)
                raise
        
        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    async def test_metrics_system():
        """Test metrics collection system."""
        collector = MetricsCollector()
        await collector.initialize()
        
        # Simulate some metrics
        collector.track_api_request("/api/predict", "POST", 200, 150.5)
        collector.track_prediction_request("location", 145.2, cache_hit=False)
        collector.track_threat_detection("medium", 0.75)
        
        # Wait a bit for collection
        await asyncio.sleep(2)
        
        # Get summary
        summary = collector.get_metrics_summary(hours=1)
        print("Metrics Summary:")
        print(json.dumps(summary, indent=2, default=str))
        
        # Export Prometheus format
        prometheus_data = collector.export_metrics_prometheus()
        print("\nPrometheus Export:")
        print(prometheus_data)
    
    asyncio.run(test_metrics_system())