"""
Atlas AI Production Monitoring and Metrics System
Comprehensive metrics collection for performance monitoring and alerting
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import os
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

# Mock Prometheus client (production would use actual prometheus_client)
class MockPrometheusMetrics:
    """Mock Prometheus metrics for environments without prometheus_client"""
    
    def __init__(self):
        self.counters = defaultdict(float)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        self.summaries = defaultdict(list)
    
    def counter(self, name: str, description: str = "", labelnames: list = None):
        return MockCounter(name, self.counters, labelnames or [])
    
    def gauge(self, name: str, description: str = "", labelnames: list = None):
        return MockGauge(name, self.gauges, labelnames or [])
    
    def histogram(self, name: str, description: str = "", labelnames: list = None, buckets: list = None):
        return MockHistogram(name, self.histograms, labelnames or [])
    
    def summary(self, name: str, description: str = "", labelnames: list = None):
        return MockSummary(name, self.summaries, labelnames or [])

class MockCounter:
    def __init__(self, name: str, storage: dict, labelnames: list):
        self.name = name
        self.storage = storage
        self.labelnames = labelnames
    
    def labels(self, **kwargs):
        key = f"{self.name}_{hash(frozenset(kwargs.items()))}"
        return MockCounterLabeled(key, self.storage)
    
    def inc(self, amount: float = 1):
        self.storage[self.name] += amount

class MockCounterLabeled:
    def __init__(self, key: str, storage: dict):
        self.key = key
        self.storage = storage
    
    def inc(self, amount: float = 1):
        self.storage[self.key] += amount

class MockGauge:
    def __init__(self, name: str, storage: dict, labelnames: list):
        self.name = name
        self.storage = storage
        self.labelnames = labelnames
    
    def set(self, value: float):
        self.storage[self.name] = value
    
    def inc(self, amount: float = 1):
        self.storage[self.name] += amount
    
    def dec(self, amount: float = 1):
        self.storage[self.name] -= amount
    
    def labels(self, **kwargs):
        key = f"{self.name}_{hash(frozenset(kwargs.items()))}"
        return MockGaugeLabeled(key, self.storage)

class MockGaugeLabeled:
    def __init__(self, key: str, storage: dict):
        self.key = key
        self.storage = storage
    
    def set(self, value: float):
        self.storage[self.key] = value

class MockHistogram:
    def __init__(self, name: str, storage: dict, labelnames: list):
        self.name = name
        self.storage = storage
        self.labelnames = labelnames
    
    def observe(self, value: float):
        self.storage[self.name].append(value)
    
    def time(self):
        return MockHistogramTimer(self)
    
    def labels(self, **kwargs):
        key = f"{self.name}_{hash(frozenset(kwargs.items()))}"
        return MockHistogramLabeled(key, self.storage)

class MockHistogramLabeled:
    def __init__(self, key: str, storage: dict):
        self.key = key
        self.storage = storage
    
    def observe(self, value: float):
        if self.key not in self.storage:
            self.storage[self.key] = []
        self.storage[self.key].append(value)
    
    def time(self):
        return MockHistogramTimer(self)

class MockHistogramTimer:
    def __init__(self, histogram):
        self.histogram = histogram
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.histogram.observe(duration)

class MockSummary:
    def __init__(self, name: str, storage: dict, labelnames: list):
        self.name = name
        self.storage = storage
        self.labelnames = labelnames
    
    def observe(self, value: float):
        self.storage[self.name].append(value)

# Try to import prometheus_client, fall back to mock
try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
    registry = CollectorRegistry()
    
    def create_counter(*args, **kwargs):
        return Counter(*args, registry=registry, **kwargs)
    
    def create_gauge(*args, **kwargs):
        return Gauge(*args, registry=registry, **kwargs)
    
    def create_histogram(*args, **kwargs):
        return Histogram(*args, registry=registry, **kwargs)
    
    def create_summary(*args, **kwargs):
        return Summary(*args, registry=registry, **kwargs)
        
except ImportError:
    logger.warning("prometheus_client not available, using mock metrics")
    PROMETHEUS_AVAILABLE = False
    mock_metrics = MockPrometheusMetrics()
    
    def create_counter(*args, **kwargs):
        return mock_metrics.counter(*args, **kwargs)
    
    def create_gauge(*args, **kwargs):
        return mock_metrics.gauge(*args, **kwargs)
    
    def create_histogram(*args, **kwargs):
        return mock_metrics.histogram(*args, **kwargs)
    
    def create_summary(*args, **kwargs):
        return mock_metrics.summary(*args, **kwargs)

# Application Metrics
# HTTP Request Metrics
http_requests_total = create_counter(
    'atlas_ai_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

http_request_duration_seconds = create_histogram(
    'atlas_ai_http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=[.005, .01, .025, .05, .075, .1, .25, .5, .75, 1.0, 2.5, 5.0, 7.5, 10.0]
)

# Database Metrics
database_connections_active = create_gauge(
    'atlas_ai_database_connections_active',
    'Active database connections'
)

database_query_duration_seconds = create_histogram(
    'atlas_ai_database_query_duration_seconds',
    'Database query duration in seconds',
    ['query_type'],
    buckets=[.001, .005, .01, .025, .05, .1, .25, .5, 1.0, 2.5, 5.0]
)

database_queries_total = create_counter(
    'atlas_ai_database_queries_total',
    'Total database queries',
    ['query_type', 'status']
)

# Cache Metrics
cache_operations_total = create_counter(
    'atlas_ai_cache_operations_total',
    'Total cache operations',
    ['operation', 'result']
)

cache_hit_rate = create_gauge(
    'atlas_ai_cache_hit_rate',
    'Cache hit rate percentage'
)

# Business Metrics
incidents_total = create_counter(
    'atlas_ai_incidents_total',
    'Total incidents created',
    ['incident_type', 'severity']
)

detections_total = create_counter(
    'atlas_ai_detections_total',
    'Total AI detections',
    ['detection_type', 'sensor_type']
)

detection_confidence = create_histogram(
    'atlas_ai_detection_confidence',
    'AI detection confidence scores',
    ['detection_type'],
    buckets=[.1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99]
)

# ML Model Metrics
model_predictions_total = create_counter(
    'atlas_ai_model_predictions_total',
    'Total ML model predictions',
    ['model_name', 'prediction_type']
)

model_accuracy = create_gauge(
    'atlas_ai_model_accuracy',
    'ML model accuracy percentage',
    ['model_name']
)

model_training_duration_seconds = create_histogram(
    'atlas_ai_model_training_duration_seconds',
    'Model training duration in seconds',
    ['model_name']
)

# Stream Processing Metrics
stream_messages_processed_total = create_counter(
    'atlas_ai_stream_messages_processed_total',
    'Total stream messages processed',
    ['stream_type', 'result']
)

stream_processing_lag_seconds = create_gauge(
    'atlas_ai_stream_processing_lag_seconds',
    'Stream processing lag in seconds',
    ['stream_type']
)

# User Metrics
active_users = create_gauge(
    'atlas_ai_active_users',
    'Currently active users',
    ['user_role']
)

user_sessions_total = create_counter(
    'atlas_ai_user_sessions_total',
    'Total user sessions',
    ['user_role', 'session_type']
)

# System Resource Metrics
memory_usage_bytes = create_gauge(
    'atlas_ai_memory_usage_bytes',
    'Memory usage in bytes',
    ['component']
)

cpu_usage_percent = create_gauge(
    'atlas_ai_cpu_usage_percent',
    'CPU usage percentage',
    ['component']
)

disk_usage_percent = create_gauge(
    'atlas_ai_disk_usage_percent',
    'Disk usage percentage',
    ['mount_point']
)

# Error Metrics
errors_total = create_counter(
    'atlas_ai_errors_total',
    'Total errors',
    ['component', 'error_type', 'severity']
)

# Alert Manager Integration
@dataclass
class Alert:
    name: str
    severity: str
    message: str
    timestamp: datetime
    labels: Dict[str, str]
    annotations: Dict[str, str]

class AlertManager:
    """Production alert management system"""
    
    def __init__(self):
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.notification_channels = []
        self.alert_rules = {}
        
    def add_notification_channel(self, channel_type: str, config: Dict[str, Any]):
        """Add notification channel (email, slack, webhook, etc.)"""
        self.notification_channels.append({
            'type': channel_type,
            'config': config,
            'enabled': True
        })
        
    def create_alert_rule(self, name: str, condition: str, severity: str, 
                         message: str, labels: Dict[str, str] = None):
        """Create alert rule for monitoring conditions"""
        self.alert_rules[name] = {
            'condition': condition,
            'severity': severity,
            'message': message,
            'labels': labels or {},
            'enabled': True,
            'last_triggered': None
        }
        
    def trigger_alert(self, name: str, message: str, severity: str = "warning",
                     labels: Dict[str, str] = None, annotations: Dict[str, str] = None):
        """Trigger an alert"""
        alert = Alert(
            name=name,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            labels=labels or {},
            annotations=annotations or {}
        )
        
        # Store active alert
        self.active_alerts[name] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        asyncio.create_task(self._send_notifications(alert))
        
        # Update metrics
        errors_total.labels(
            component="alert_manager",
            error_type=name,
            severity=severity
        ).inc()
        
        logger.warning(f"Alert triggered: {name} - {message}")
        
    def resolve_alert(self, name: str):
        """Resolve an active alert"""
        if name in self.active_alerts:
            alert = self.active_alerts.pop(name)
            logger.info(f"Alert resolved: {name}")
            
    async def _send_notifications(self, alert: Alert):
        """Send alert notifications through configured channels"""
        for channel in self.notification_channels:
            if not channel['enabled']:
                continue
                
            try:
                await self._send_notification(channel, alert)
            except Exception as e:
                logger.error(f"Failed to send notification via {channel['type']}: {e}")
                
    async def _send_notification(self, channel: Dict, alert: Alert):
        """Send notification through specific channel"""
        channel_type = channel['type']
        
        if channel_type == 'webhook':
            await self._send_webhook_notification(channel['config'], alert)
        elif channel_type == 'email':
            await self._send_email_notification(channel['config'], alert)
        elif channel_type == 'slack':
            await self._send_slack_notification(channel['config'], alert)
        elif channel_type == 'log':
            await self._send_log_notification(alert)
            
    async def _send_webhook_notification(self, config: Dict, alert: Alert):
        """Send webhook notification"""
        import aiohttp
        
        payload = {
            'alert_name': alert.name,
            'severity': alert.severity,
            'message': alert.message,
            'timestamp': alert.timestamp.isoformat(),
            'labels': alert.labels,
            'annotations': alert.annotations
        }
        
        async with aiohttp.ClientSession() as session:
            await session.post(
                config['url'],
                json=payload,
                headers=config.get('headers', {}),
                timeout=aiohttp.ClientTimeout(total=10)
            )
            
    async def _send_email_notification(self, config: Dict, alert: Alert):
        """Send email notification"""
        # Mock email sending - in production would use actual email service
        logger.info(f"Email notification sent for alert: {alert.name}")
        
    async def _send_slack_notification(self, config: Dict, alert: Alert):
        """Send Slack notification"""
        # Mock Slack notification - in production would use Slack API
        logger.info(f"Slack notification sent for alert: {alert.name}")
        
    async def _send_log_notification(self, alert: Alert):
        """Log alert notification"""
        logger.warning(f"ALERT: {alert.name} - {alert.message} (severity: {alert.severity})")

# Global alert manager instance
alert_manager = AlertManager()

# Health Check System
class HealthChecker:
    """Comprehensive health checking system"""
    
    def __init__(self):
        self.health_checks = {}
        self.check_intervals = {}
        self.last_check_results = {}
        self.running = False
        
    def register_health_check(self, name: str, check_func, interval: int = 60):
        """Register a health check function"""
        self.health_checks[name] = check_func
        self.check_intervals[name] = interval
        self.last_check_results[name] = None
        
    async def start_monitoring(self):
        """Start health monitoring background tasks"""
        self.running = True
        
        # Start health check tasks
        for name in self.health_checks:
            asyncio.create_task(self._run_health_check_loop(name))
            
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.running = False
        
    async def _run_health_check_loop(self, check_name: str):
        """Run health check in a loop"""
        while self.running:
            try:
                result = await self._run_health_check(check_name)
                self.last_check_results[check_name] = result
                
                if not result['healthy']:
                    alert_manager.trigger_alert(
                        name=f"health_check_{check_name}",
                        message=f"Health check failed: {check_name} - {result.get('error', 'Unknown error')}",
                        severity="critical",
                        labels={'component': check_name}
                    )
                else:
                    alert_manager.resolve_alert(f"health_check_{check_name}")
                    
            except Exception as e:
                logger.error(f"Health check {check_name} failed with exception: {e}")
                alert_manager.trigger_alert(
                    name=f"health_check_{check_name}",
                    message=f"Health check exception: {check_name} - {str(e)}",
                    severity="critical"
                )
                
            await asyncio.sleep(self.check_intervals[check_name])
            
    async def _run_health_check(self, check_name: str) -> Dict[str, Any]:
        """Run a single health check"""
        check_func = self.health_checks[check_name]
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = check_func()
                
            duration = time.time() - start_time
            
            return {
                'healthy': result.get('healthy', True),
                'duration': duration,
                'timestamp': datetime.now().isoformat(),
                'details': result
            }
        except Exception as e:
            duration = time.time() - start_time
            return {
                'healthy': False,
                'duration': duration,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

# Global health checker
health_checker = HealthChecker()

# Metrics Middleware
async def metrics_middleware(request, call_next):
    """FastAPI middleware for collecting HTTP metrics"""
    start_time = time.time()
    method = request.method
    path = request.url.path
    
    # Process request
    response = await call_next(request)
    
    # Collect metrics
    duration = time.time() - start_time
    status_code = str(response.status_code)
    
    # Update metrics
    http_requests_total.labels(
        method=method,
        endpoint=path,
        status_code=status_code
    ).inc()
    
    http_request_duration_seconds.labels(
        method=method,
        endpoint=path
    ).observe(duration)
    
    return response

# Setup Functions
def setup_metrics():
    """Initialize metrics and monitoring system"""
    logger.info("Setting up Atlas AI monitoring system")
    
    # Setup alert channels
    alert_manager.add_notification_channel('log', {})
    
    # Setup alert rules
    alert_manager.create_alert_rule(
        name="high_error_rate",
        condition="error_rate > 0.05",
        severity="warning",
        message="High error rate detected"
    )
    
    alert_manager.create_alert_rule(
        name="database_connection_failure",
        condition="database_connections_active == 0",
        severity="critical",
        message="Database connection failure"
    )
    
    # Register health checks
    health_checker.register_health_check("database", check_database_health, 30)
    health_checker.register_health_check("cache", check_cache_health, 30)
    health_checker.register_health_check("disk_space", check_disk_space, 60)
    
    # Start monitoring
    asyncio.create_task(health_checker.start_monitoring())
    
    logger.info("Monitoring system setup completed")

def get_metrics():
    """Get current metrics for Prometheus endpoint"""
    if PROMETHEUS_AVAILABLE:
        return generate_latest(registry).decode('utf-8')
    else:
        # Return mock metrics in Prometheus format
        metrics_output = []
        
        # Add mock counters
        for name, value in mock_metrics.counters.items():
            metrics_output.append(f"# TYPE {name} counter")
            metrics_output.append(f"{name} {value}")
        
        # Add mock gauges
        for name, value in mock_metrics.gauges.items():
            metrics_output.append(f"# TYPE {name} gauge")
            metrics_output.append(f"{name} {value}")
            
        return "\n".join(metrics_output)

# Health Check Functions
async def check_database_health():
    """Check database connectivity and performance"""
    try:
        from ..database.postgis_database import get_database
        
        db = await get_database()
        start_time = time.time()
        result = await db.execute_query("SELECT 1 as health_check")
        duration = time.time() - start_time
        
        database_query_duration_seconds.labels(query_type="health_check").observe(duration)
        
        if result and len(result) > 0:
            database_queries_total.labels(query_type="health_check", status="success").inc()
            return {'healthy': True, 'response_time': duration}
        else:
            database_queries_total.labels(query_type="health_check", status="failure").inc()
            return {'healthy': False, 'error': 'No result returned'}
            
    except Exception as e:
        database_queries_total.labels(query_type="health_check", status="error").inc()
        return {'healthy': False, 'error': str(e)}

async def check_cache_health():
    """Check cache connectivity and performance"""
    try:
        from ..caching.redis_cache import get_cache
        
        cache = await get_cache()
        start_time = time.time()
        
        # Test cache operations
        test_key = "health_check_test"
        await cache.set(test_key, "ok", ttl=10)
        result = await cache.get(test_key)
        
        duration = time.time() - start_time
        
        if result == "ok":
            cache_operations_total.labels(operation="health_check", result="hit").inc()
            return {'healthy': True, 'response_time': duration}
        else:
            cache_operations_total.labels(operation="health_check", result="miss").inc()
            return {'healthy': False, 'error': 'Cache test failed'}
            
    except Exception as e:
        cache_operations_total.labels(operation="health_check", result="error").inc()
        return {'healthy': False, 'error': str(e)}

def check_disk_space():
    """Check disk space usage"""
    try:
        import shutil
        
        # Check root filesystem
        total, used, free = shutil.disk_usage("/")
        usage_percent = (used / total) * 100
        
        disk_usage_percent.labels(mount_point="/").set(usage_percent)
        
        if usage_percent > 90:
            return {'healthy': False, 'error': f'Disk usage is {usage_percent:.1f}%'}
        elif usage_percent > 80:
            return {'healthy': True, 'warning': f'Disk usage is {usage_percent:.1f}%'}
        else:
            return {'healthy': True, 'disk_usage_percent': usage_percent}
            
    except Exception as e:
        return {'healthy': False, 'error': str(e)}

# Performance Monitoring
class PerformanceMonitor:
    """System performance monitoring and analysis"""
    
    def __init__(self):
        self.metrics_buffer = deque(maxlen=1000)
        self.analysis_results = {}
        
    async def collect_system_metrics(self):
        """Collect system performance metrics"""
        try:
            import psutil
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_usage_percent.labels(component="system").set(cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_usage_bytes.labels(component="system").set(memory.used)
            
            # Process-specific metrics
            process = psutil.Process()
            cpu_usage_percent.labels(component="atlas_ai").set(process.cpu_percent())
            memory_usage_bytes.labels(component="atlas_ai").set(process.memory_info().rss)
            
            # Store for analysis
            metrics_data = {
                'timestamp': datetime.now(),
                'cpu_system': cpu_percent,
                'memory_used': memory.used,
                'memory_percent': memory.percent,
                'process_cpu': process.cpu_percent(),
                'process_memory': process.memory_info().rss
            }
            
            self.metrics_buffer.append(metrics_data)
            
        except ImportError:
            # Fallback when psutil is not available
            logger.warning("psutil not available, using mock system metrics")
            cpu_usage_percent.labels(component="system").set(25.0)
            memory_usage_bytes.labels(component="system").set(1024*1024*1024)  # 1GB
            
    async def analyze_performance_trends(self):
        """Analyze performance trends and detect anomalies"""
        if len(self.metrics_buffer) < 10:
            return
            
        # Calculate averages and trends
        cpu_values = [m['cpu_system'] for m in list(self.metrics_buffer)[-60:]]  # Last 60 samples
        memory_values = [m['memory_percent'] for m in list(self.metrics_buffer)[-60:]]
        
        avg_cpu = sum(cpu_values) / len(cpu_values)
        avg_memory = sum(memory_values) / len(memory_values)
        
        # Detect high resource usage
        if avg_cpu > 80:
            alert_manager.trigger_alert(
                name="high_cpu_usage",
                message=f"High CPU usage detected: {avg_cpu:.1f}%",
                severity="warning",
                labels={'component': 'system', 'metric': 'cpu'}
            )
            
        if avg_memory > 85:
            alert_manager.trigger_alert(
                name="high_memory_usage", 
                message=f"High memory usage detected: {avg_memory:.1f}%",
                severity="warning",
                labels={'component': 'system', 'metric': 'memory'}
            )

# Global performance monitor
performance_monitor = PerformanceMonitor()

# Start monitoring tasks
async def start_monitoring_tasks():
    """Start all monitoring background tasks"""
    # System metrics collection
    async def metrics_collection_loop():
        while True:
            await performance_monitor.collect_system_metrics()
            await asyncio.sleep(60)  # Collect every minute
    
    # Performance analysis
    async def performance_analysis_loop():
        while True:
            await performance_monitor.analyze_performance_trends()
            await asyncio.sleep(300)  # Analyze every 5 minutes
    
    asyncio.create_task(metrics_collection_loop())
    asyncio.create_task(performance_analysis_loop())

# Export key components
__all__ = [
    'setup_metrics',
    'get_metrics', 
    'metrics_middleware',
    'alert_manager',
    'health_checker',
    'performance_monitor',
    'start_monitoring_tasks',
    # Metrics
    'http_requests_total',
    'http_request_duration_seconds',
    'database_connections_active',
    'database_query_duration_seconds',
    'cache_operations_total',
    'incidents_total',
    'detections_total',
    'model_predictions_total',
    'errors_total'
]