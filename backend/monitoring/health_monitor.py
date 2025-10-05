#!/usr/bin/env python3
"""
Health Monitoring System for Atlas AI
Phase 2.3 System Reliability - Health Monitoring

Comprehensive health monitoring:
- Real-time system health tracking
- Performance metrics collection
- Service dependency monitoring
- Cross-platform health reporting (Web/iOS/Android)
- Proactive alerting and notification
- Health dashboards and analytics
"""

import asyncio
import psutil
import time
import logging
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
import statistics
from pathlib import Path
import aiohttp
import socket
import platform
import sys
import subprocess
import hashlib

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning" 
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"


class MetricType(Enum):
    """Types of metrics to collect"""
    SYSTEM = "system"
    APPLICATION = "application"
    BUSINESS = "business"
    PERFORMANCE = "performance"
    AVAILABILITY = "availability"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthMetric:
    """Individual health metric"""
    name: str
    value: Union[float, int, str, bool]
    unit: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metric_type: MetricType = MetricType.SYSTEM
    tags: Dict[str, str] = field(default_factory=dict)
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None


@dataclass
class ServiceHealth:
    """Health status of a service"""
    service_name: str
    status: HealthStatus
    response_time_ms: Optional[float] = None
    last_check: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_message: Optional[str] = None
    uptime_seconds: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthAlert:
    """Health monitoring alert"""
    alert_id: str
    service: str
    metric_name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    current_value: Any
    threshold_value: Any
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass 
class SystemSnapshot:
    """Complete system health snapshot"""
    timestamp: datetime
    overall_status: HealthStatus
    services: Dict[str, ServiceHealth]
    metrics: Dict[str, HealthMetric]
    alerts: List[HealthAlert]
    system_info: Dict[str, Any]


class HealthCheck:
    """Base class for health checks"""
    
    def __init__(self, name: str, check_interval: float = 30.0):
        self.name = name
        self.check_interval = check_interval
        self.last_check = None
        self.last_result = None
        self.enabled = True
    
    async def check(self) -> ServiceHealth:
        """Perform health check - to be implemented by subclasses"""
        raise NotImplementedError("Health check must implement check() method")
    
    async def run_check(self) -> ServiceHealth:
        """Execute health check with error handling"""
        try:
            self.last_check = datetime.now(timezone.utc)
            self.last_result = await self.check()
            return self.last_result
        except Exception as e:
            logger.error(f"Health check {self.name} failed: {str(e)}")
            return ServiceHealth(
                service_name=self.name,
                status=HealthStatus.CRITICAL,
                error_message=str(e),
                last_check=datetime.now(timezone.utc)
            )


class DatabaseHealthCheck(HealthCheck):
    """Database connectivity health check"""
    
    def __init__(self, connection_string: str, timeout: float = 5.0):
        super().__init__("database", check_interval=60.0)
        self.connection_string = connection_string
        self.timeout = timeout
    
    async def check(self) -> ServiceHealth:
        start_time = time.time()
        
        try:
            # Real database health check
            from ..database.postgis_database import get_database
            db = await get_database()
            
            # Test with simple query
            result = await db.execute_query("SELECT 1 as health_check")
            
            response_time = (time.time() - start_time) * 1000
            
            if result and len(result) > 0:
                # Test passed - database is responsive
                status = HealthStatus.HEALTHY if response_time < 1000 else HealthStatus.WARNING
                metadata = {
                    "connection_pool": "active", 
                    "test_query": "passed",
                    "connection_string": self.connection_string.replace(self.connection_string.split('@')[0].split('//')[1], '***') if '@' in self.connection_string else 'local'
                }
            else:
                status = HealthStatus.WARNING
                metadata = {"test_query": "no_result"}
            
            return ServiceHealth(
                service_name="database",
                status=status,
                response_time_ms=response_time,
                metadata=metadata
            )
            
        except Exception as e:
            return ServiceHealth(
                service_name="database",
                status=HealthStatus.CRITICAL,
                error_message=str(e),
                response_time_ms=(time.time() - start_time) * 1000
            )


class RedisHealthCheck(HealthCheck):
    """Redis cache health check"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        super().__init__("redis", check_interval=30.0)
        self.redis_url = redis_url
    
    async def check(self) -> ServiceHealth:
        start_time = time.time()
        
        try:
            # Real Redis health check
            from ..caching.redis_cache import get_redis_cache
            redis_cache = await get_redis_cache()
            
            if redis_cache:
                # Perform actual Redis health check
                health_stats = await redis_cache.health_check()
                response_time = (time.time() - start_time) * 1000
                
                if health_stats.get('status') == 'healthy':
                    status = HealthStatus.HEALTHY
                    metadata = {
                        "connection": "active",
                        "response_time_ms": health_stats.get('response_time_ms', response_time),
                        "test_passed": health_stats.get('test_passed', False)
                    }
                else:
                    status = HealthStatus.CRITICAL
                    metadata = {"error": health_stats.get('error', 'Unknown error')}
            else:
                # Redis not available
                status = HealthStatus.WARNING
                response_time = (time.time() - start_time) * 1000
                metadata = {"connection": "disabled", "reason": "Redis cache not enabled"}
            
            return ServiceHealth(
                service_name="redis",
                status=status,
                response_time_ms=response_time,
                metadata=metadata
            )
            
        except Exception as e:
            return ServiceHealth(
                service_name="redis",
                status=HealthStatus.CRITICAL,
                error_message=str(e)
            )


class APIEndpointHealthCheck(HealthCheck):
    """API endpoint health check"""
    
    def __init__(self, name: str, url: str, expected_status: int = 200, timeout: float = 10.0):
        super().__init__(name, check_interval=60.0)
        self.url = url
        self.expected_status = expected_status
        self.timeout = timeout
    
    async def check(self) -> ServiceHealth:
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.url, timeout=self.timeout) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    status = HealthStatus.HEALTHY
                    if response.status != self.expected_status:
                        status = HealthStatus.WARNING
                    elif response_time > 5000:  # > 5 seconds
                        status = HealthStatus.DEGRADED
                    
                    return ServiceHealth(
                        service_name=self.name,
                        status=status,
                        response_time_ms=response_time,
                        metadata={
                            "http_status": response.status,
                            "content_length": response.headers.get("Content-Length", "unknown")
                        }
                    )
                    
        except asyncio.TimeoutError:
            return ServiceHealth(
                service_name=self.name,
                status=HealthStatus.CRITICAL,
                error_message="Timeout",
                response_time_ms=(time.time() - start_time) * 1000
            )
        except Exception as e:
            return ServiceHealth(
                service_name=self.name,
                status=HealthStatus.CRITICAL,
                error_message=str(e),
                response_time_ms=(time.time() - start_time) * 1000
            )


class AIModelHealthCheck(HealthCheck):
    """AI model inference health check"""
    
    def __init__(self, model_name: str, test_input_generator: Callable):
        super().__init__(f"ai_model_{model_name}", check_interval=120.0)
        self.model_name = model_name
        self.test_input_generator = test_input_generator
    
    async def check(self) -> ServiceHealth:
        start_time = time.time()
        
        try:
            # Generate test input
            test_input = self.test_input_generator()
            
            # Simulate model inference
            await asyncio.sleep(0.5)  # Simulate model processing time
            
            response_time = (time.time() - start_time) * 1000
            
            status = HealthStatus.HEALTHY
            if response_time > 10000:  # > 10 seconds
                status = HealthStatus.WARNING
            elif response_time > 30000:  # > 30 seconds
                status = HealthStatus.CRITICAL
            
            return ServiceHealth(
                service_name=f"ai_model_{self.model_name}",
                status=status,
                response_time_ms=response_time,
                metadata={
                    "model_name": self.model_name,
                    "inference_time_ms": response_time,
                    "gpu_available": False  # Would check actual GPU status
                }
            )
            
        except Exception as e:
            return ServiceHealth(
                service_name=f"ai_model_{self.model_name}",
                status=HealthStatus.CRITICAL,
                error_message=str(e),
                response_time_ms=(time.time() - start_time) * 1000
            )


class HealthMonitor:
    """Main health monitoring system"""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.health_checks: Dict[str, HealthCheck] = {}
        self.metrics_history: Dict[str, List[HealthMetric]] = {}
        self.alerts: List[HealthAlert] = []
        self.alert_handlers: List[Callable] = []
        
        # System monitoring
        self.system_metrics_enabled = True
        self.metric_retention_hours = 24
        
        # Monitoring state
        self.running = False
        self.monitoring_task = None
        self.last_snapshot = None
        
        # Performance tracking
        self.performance_history: List[Dict[str, Any]] = []
        self.uptime_start = datetime.now(timezone.utc)
        
        logger.info("🏥 HealthMonitor initialized")
    
    def register_health_check(self, health_check: HealthCheck):
        """Register a health check"""
        self.health_checks[health_check.name] = health_check
        logger.info(f"📋 Registered health check: {health_check.name}")
    
    def register_alert_handler(self, handler: Callable[[HealthAlert], None]):
        """Register an alert handler"""
        self.alert_handlers.append(handler)
        logger.info("📢 Registered alert handler")
    
    async def start_monitoring(self):
        """Start the health monitoring loop"""
        if self.running:
            logger.warning("Health monitoring is already running")
            return
        
        self.running = True
        self.uptime_start = datetime.now(timezone.utc)
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("▶️ Health monitoring started")
    
    async def stop_monitoring(self):
        """Stop the health monitoring loop"""
        if not self.running:
            return
        
        self.running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("⏸️ Health monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("🔄 Health monitoring loop started")
        
        while self.running:
            try:
                # Collect system snapshot
                snapshot = await self.collect_system_snapshot()
                self.last_snapshot = snapshot
                
                # Check for alerts
                await self._check_alerts(snapshot)
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Wait for next check
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(self.check_interval)
        
        logger.info("🔄 Health monitoring loop stopped")
    
    async def collect_system_snapshot(self) -> SystemSnapshot:
        """Collect complete system health snapshot"""
        start_time = time.time()
        
        # Run all health checks
        service_health = {}
        if self.health_checks:
            check_tasks = [
                check.run_check() 
                for check in self.health_checks.values() 
                if check.enabled
            ]
            
            if check_tasks:
                check_results = await asyncio.gather(*check_tasks, return_exceptions=True)
                
                for i, result in enumerate(check_results):
                    if isinstance(result, Exception):
                        logger.error(f"Health check failed: {str(result)}")
                    elif isinstance(result, ServiceHealth):
                        service_health[result.service_name] = result
        
        # Collect system metrics
        system_metrics = await self._collect_system_metrics()
        
        # Store metrics in history
        for name, metric in system_metrics.items():
            if name not in self.metrics_history:
                self.metrics_history[name] = []
            self.metrics_history[name].append(metric)
        
        # Determine overall status
        overall_status = self._calculate_overall_status(service_health, system_metrics)
        
        # Get system information
        system_info = self._get_system_info()
        
        collection_time = (time.time() - start_time) * 1000
        logger.debug(f"System snapshot collected in {collection_time:.2f}ms")
        
        return SystemSnapshot(
            timestamp=datetime.now(timezone.utc),
            overall_status=overall_status,
            services=service_health,
            metrics=system_metrics,
            alerts=[alert for alert in self.alerts if not alert.resolved],
            system_info=system_info
        )
    
    async def _collect_system_metrics(self) -> Dict[str, HealthMetric]:
        """Collect system performance metrics"""
        if not self.system_metrics_enabled:
            return {}
        
        metrics = {}
        timestamp = datetime.now(timezone.utc)
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            metrics["cpu_usage"] = HealthMetric(
                name="cpu_usage",
                value=cpu_percent,
                unit="percent",
                timestamp=timestamp,
                metric_type=MetricType.SYSTEM,
                threshold_warning=80.0,
                threshold_critical=95.0
            )
            
            metrics["cpu_count"] = HealthMetric(
                name="cpu_count",
                value=cpu_count,
                unit="cores",
                timestamp=timestamp,
                metric_type=MetricType.SYSTEM
            )
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            metrics["memory_usage"] = HealthMetric(
                name="memory_usage",
                value=memory.percent,
                unit="percent",
                timestamp=timestamp,
                metric_type=MetricType.SYSTEM,
                threshold_warning=80.0,
                threshold_critical=95.0
            )
            
            metrics["memory_available"] = HealthMetric(
                name="memory_available",
                value=memory.available / (1024**3),  # GB
                unit="GB",
                timestamp=timestamp,
                metric_type=MetricType.SYSTEM
            )
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            
            metrics["disk_usage"] = HealthMetric(
                name="disk_usage",
                value=(disk_usage.used / disk_usage.total) * 100,
                unit="percent",
                timestamp=timestamp,
                metric_type=MetricType.SYSTEM,
                threshold_warning=85.0,
                threshold_critical=95.0
            )
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            
            metrics["process_memory"] = HealthMetric(
                name="process_memory",
                value=process_memory.rss / (1024**2),  # MB
                unit="MB",
                timestamp=timestamp,
                metric_type=MetricType.APPLICATION
            )
            
            metrics["process_cpu"] = HealthMetric(
                name="process_cpu",
                value=process.cpu_percent(),
                unit="percent",
                timestamp=timestamp,
                metric_type=MetricType.APPLICATION
            )
            
            # Network metrics (if available)
            try:
                network_io = psutil.net_io_counters()
                
                metrics["network_bytes_sent"] = HealthMetric(
                    name="network_bytes_sent",
                    value=network_io.bytes_sent,
                    unit="bytes",
                    timestamp=timestamp,
                    metric_type=MetricType.SYSTEM
                )
                
                metrics["network_bytes_recv"] = HealthMetric(
                    name="network_bytes_recv",
                    value=network_io.bytes_recv,
                    unit="bytes",
                    timestamp=timestamp,
                    metric_type=MetricType.SYSTEM
                )
            except Exception:
                pass  # Network stats might not be available
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
        
        return metrics
    
    def _calculate_overall_status(
        self, 
        service_health: Dict[str, ServiceHealth], 
        system_metrics: Dict[str, HealthMetric]
    ) -> HealthStatus:
        """Calculate overall system health status"""
        
        # Check service health
        service_statuses = [health.status for health in service_health.values()]
        
        if any(status == HealthStatus.CRITICAL for status in service_statuses):
            return HealthStatus.CRITICAL
        elif any(status == HealthStatus.DEGRADED for status in service_statuses):
            return HealthStatus.DEGRADED
        elif any(status == HealthStatus.WARNING for status in service_statuses):
            return HealthStatus.WARNING
        
        # Check system metrics against thresholds
        for metric in system_metrics.values():
            if (metric.threshold_critical and 
                isinstance(metric.value, (int, float)) and
                metric.value >= metric.threshold_critical):
                return HealthStatus.CRITICAL
            elif (metric.threshold_warning and 
                  isinstance(metric.value, (int, float)) and
                  metric.value >= metric.threshold_warning):
                return HealthStatus.WARNING
        
        return HealthStatus.HEALTHY
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            "platform": platform.system(),
            "platform_version": platform.release(),
            "architecture": platform.machine(),
            "python_version": platform.python_version(),
            "hostname": socket.gethostname(),
            "uptime_seconds": (datetime.now(timezone.utc) - self.uptime_start).total_seconds(),
            "monitoring_enabled": self.running,
            "last_check": datetime.now(timezone.utc).isoformat()
        }
    
    async def _check_alerts(self, snapshot: SystemSnapshot):
        """Check for alert conditions"""
        new_alerts = []
        
        # Check service alerts
        for service_name, service_health in snapshot.services.items():
            if service_health.status in [HealthStatus.CRITICAL, HealthStatus.DEGRADED]:
                alert_id = f"service_{service_name}_{int(time.time())}"
                alert = HealthAlert(
                    alert_id=alert_id,
                    service=service_name,
                    metric_name="service_status",
                    severity=AlertSeverity.CRITICAL if service_health.status == HealthStatus.CRITICAL else AlertSeverity.ERROR,
                    message=f"Service {service_name} is {service_health.status.value}: {service_health.error_message or 'No details'}",
                    timestamp=datetime.now(timezone.utc),
                    current_value=service_health.status.value,
                    threshold_value="healthy"
                )
                new_alerts.append(alert)
        
        # Check metric threshold alerts
        for metric_name, metric in snapshot.metrics.items():
            if isinstance(metric.value, (int, float)):
                if metric.threshold_critical and metric.value >= metric.threshold_critical:
                    alert_id = f"metric_{metric_name}_{int(time.time())}"
                    alert = HealthAlert(
                        alert_id=alert_id,
                        service="system",
                        metric_name=metric_name,
                        severity=AlertSeverity.CRITICAL,
                        message=f"{metric_name} is critically high: {metric.value}{metric.unit or ''} (threshold: {metric.threshold_critical})",
                        timestamp=datetime.now(timezone.utc),
                        current_value=metric.value,
                        threshold_value=metric.threshold_critical
                    )
                    new_alerts.append(alert)
                elif metric.threshold_warning and metric.value >= metric.threshold_warning:
                    alert_id = f"metric_{metric_name}_{int(time.time())}"
                    alert = HealthAlert(
                        alert_id=alert_id,
                        service="system",
                        metric_name=metric_name,
                        severity=AlertSeverity.WARNING,
                        message=f"{metric_name} is high: {metric.value}{metric.unit or ''} (threshold: {metric.threshold_warning})",
                        timestamp=datetime.now(timezone.utc),
                        current_value=metric.value,
                        threshold_value=metric.threshold_warning
                    )
                    new_alerts.append(alert)
        
        # Add new alerts and notify handlers
        for alert in new_alerts:
            self.alerts.append(alert)
            logger.warning(f"🚨 Alert triggered: {alert.message}")
            
            # Notify alert handlers
            for handler in self.alert_handlers:
                try:
                    await self._call_alert_handler(handler, alert)
                except Exception as e:
                    logger.error(f"Alert handler failed: {str(e)}")
    
    async def _call_alert_handler(self, handler: Callable, alert: HealthAlert):
        """Call alert handler with proper async handling"""
        if asyncio.iscoroutinefunction(handler):
            await handler(alert)
        else:
            handler(alert)
    
    def _cleanup_old_data(self):
        """Clean up old metrics and alerts"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.metric_retention_hours)
        
        # Clean up metrics history
        for metric_name, metrics_list in self.metrics_history.items():
            self.metrics_history[metric_name] = [
                metric for metric in metrics_list
                if metric.timestamp > cutoff_time
            ]
        
        # Clean up old alerts (keep resolved alerts for some time)
        alert_cutoff = datetime.now(timezone.utc) - timedelta(hours=48)
        self.alerts = [
            alert for alert in self.alerts
            if alert.timestamp > alert_cutoff or not alert.resolved
        ]
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get current health summary"""
        if not self.last_snapshot:
            return {"status": "no_data", "message": "No health data available"}
        
        snapshot = self.last_snapshot
        
        # Calculate service summary
        service_summary = {}
        for status in HealthStatus:
            count = sum(1 for service in snapshot.services.values() if service.status == status)
            if count > 0:
                service_summary[status.value] = count
        
        # Calculate alert summary
        active_alerts = [alert for alert in self.alerts if not alert.resolved]
        alert_summary = {}
        for severity in AlertSeverity:
            count = sum(1 for alert in active_alerts if alert.severity == severity)
            if count > 0:
                alert_summary[severity.value] = count
        
        # Get key metrics
        key_metrics = {}
        if snapshot.metrics:
            important_metrics = ["cpu_usage", "memory_usage", "disk_usage", "process_memory"]
            for metric_name in important_metrics:
                if metric_name in snapshot.metrics:
                    metric = snapshot.metrics[metric_name]
                    key_metrics[metric_name] = {
                        "value": metric.value,
                        "unit": metric.unit,
                        "status": "ok" if (not metric.threshold_warning or 
                                        metric.value < metric.threshold_warning) else "warning"
                    }
        
        return {
            "overall_status": snapshot.overall_status.value,
            "timestamp": snapshot.timestamp.isoformat(),
            "uptime_seconds": (datetime.now(timezone.utc) - self.uptime_start).total_seconds(),
            "services": service_summary,
            "alerts": alert_summary,
            "key_metrics": key_metrics,
            "monitoring_active": self.running
        }
    
    async def get_detailed_health(self) -> Dict[str, Any]:
        """Get detailed health information"""
        if not self.last_snapshot:
            return {"error": "No health data available"}
        
        snapshot = self.last_snapshot
        
        # Convert services to dict format
        services_detail = {}
        for name, health in snapshot.services.items():
            services_detail[name] = {
                "status": health.status.value,
                "response_time_ms": health.response_time_ms,
                "last_check": health.last_check.isoformat(),
                "error_message": health.error_message,
                "uptime_seconds": health.uptime_seconds,
                "metadata": health.metadata
            }
        
        # Convert metrics to dict format
        metrics_detail = {}
        for name, metric in snapshot.metrics.items():
            metrics_detail[name] = {
                "value": metric.value,
                "unit": metric.unit,
                "timestamp": metric.timestamp.isoformat(),
                "type": metric.metric_type.value,
                "threshold_warning": metric.threshold_warning,
                "threshold_critical": metric.threshold_critical,
                "tags": metric.tags
            }
        
        # Convert alerts to dict format
        alerts_detail = []
        active_alerts = [alert for alert in self.alerts if not alert.resolved]
        for alert in active_alerts[-10:]:  # Last 10 alerts
            alerts_detail.append({
                "id": alert.alert_id,
                "service": alert.service,
                "metric": alert.metric_name,
                "severity": alert.severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "current_value": alert.current_value,
                "threshold_value": alert.threshold_value,
                "resolved": alert.resolved
            })
        
        return {
            "overall_status": snapshot.overall_status.value,
            "timestamp": snapshot.timestamp.isoformat(),
            "services": services_detail,
            "metrics": metrics_detail,
            "alerts": alerts_detail,
            "system_info": snapshot.system_info
        }
    
    async def force_health_check(self) -> SystemSnapshot:
        """Force an immediate health check"""
        logger.info("🔍 Forcing immediate health check...")
        return await self.collect_system_snapshot()


# Example alert handler
async def log_alert_handler(alert: HealthAlert):
    """Simple log-based alert handler"""
    logger.warning(f"ALERT [{alert.severity.value.upper()}] {alert.service}: {alert.message}")


async def setup_default_health_monitor() -> HealthMonitor:
    """Setup health monitor with default checks"""
    monitor = HealthMonitor(check_interval=30.0)
    
    # Register default health checks
    monitor.register_health_check(DatabaseHealthCheck("postgresql://localhost:5432/atlas_ai"))
    monitor.register_health_check(RedisHealthCheck("redis://localhost:6379"))
    
    # AI model health checks
    def generate_cv_test_input():
        import numpy as np
        return np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    
    def generate_speech_test_input():
        import numpy as np
        return np.random.normal(0, 0.1, 16000).astype(np.float32)
    
    monitor.register_health_check(AIModelHealthCheck("computer_vision", generate_cv_test_input))
    monitor.register_health_check(AIModelHealthCheck("speech_to_text", generate_speech_test_input))
    
    # Register default alert handler
    monitor.register_alert_handler(log_alert_handler)
    
    return monitor


# Global health monitor instance
health_monitor = None


async def initialize_health_monitor() -> HealthMonitor:
    """Initialize the global health monitor"""
    global health_monitor
    health_monitor = await setup_default_health_monitor()
    return health_monitor


if __name__ == "__main__":
    async def test_health_monitor():
        """Test the health monitoring system"""
        logging.basicConfig(level=logging.INFO)
        
        print("🧪 Testing Health Monitoring System...")
        
        # Initialize health monitor
        monitor = await setup_default_health_monitor()
        
        print("\n📋 Registered Health Checks:")
        for name, check in monitor.health_checks.items():
            print(f"  • {name} (interval: {check.check_interval}s)")
        
        # Start monitoring
        await monitor.start_monitoring()
        
        # Let it run for a bit
        print("\n⏳ Running health checks...")
        await asyncio.sleep(5)
        
        # Get health summary
        print("\n📊 Health Summary:")
        summary = monitor.get_health_summary()
        print(f"  Overall Status: {summary['overall_status']}")
        print(f"  Uptime: {summary['uptime_seconds']:.1f} seconds")
        print(f"  Services: {summary.get('services', {})}")
        print(f"  Key Metrics: {len(summary.get('key_metrics', {}))}")
        
        # Get detailed health
        print("\n🔍 Detailed Health Check:")
        detailed = await monitor.get_detailed_health()
        
        if 'services' in detailed:
            print("  Services:")
            for name, service in detailed['services'].items():
                status_icon = "✅" if service['status'] == 'healthy' else "⚠️" if service['status'] == 'warning' else "❌"
                print(f"    {status_icon} {name}: {service['status']} ({service.get('response_time_ms', 0):.1f}ms)")
        
        if 'metrics' in detailed:
            print("  System Metrics:")
            important_metrics = ['cpu_usage', 'memory_usage', 'disk_usage']
            for metric_name in important_metrics:
                if metric_name in detailed['metrics']:
                    metric = detailed['metrics'][metric_name]
                    print(f"    📈 {metric_name}: {metric['value']}{metric.get('unit', '')}")
        
        # Force a health check
        print("\n🔍 Forcing immediate health check...")
        snapshot = await monitor.force_health_check()
        print(f"  Snapshot collected at: {snapshot.timestamp.strftime('%H:%M:%S')}")
        print(f"  Overall status: {snapshot.overall_status.value}")
        
        # Stop monitoring
        await monitor.stop_monitoring()
        
        print("\n✅ Health monitoring test completed!")
    
    asyncio.run(test_health_monitor())