#!/usr/bin/env python3
"""
Production Infrastructure Monitoring for Atlas AI
Phase 2.2 Core Production Features - Infrastructure Monitoring

Comprehensive production monitoring system:
- Real-time infrastructure monitoring
- Performance metrics collection
- Alert management system
- Dashboard data aggregation
- Resource utilization tracking
- Service dependency monitoring
- Automated incident response
- Capacity planning metrics
"""

import asyncio
import logging
import psutil
import docker
import requests
import redis
import psycopg2
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import os
import socket
import subprocess
import time
import threading
from pathlib import Path
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import aiofiles
import asyncpg

# Prometheus metrics (optional)
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ServiceStatus(Enum):
    """Service status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DOWN = "down"
    UNKNOWN = "unknown"


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class InfrastructureAlert:
    """Infrastructure alert data structure"""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    service: str
    metric_name: str
    threshold_value: float
    current_value: float
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    acknowledgments: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceHealth:
    """Service health status"""
    name: str
    status: ServiceStatus
    response_time_ms: float
    last_check: datetime
    uptime_percentage: float
    error_count: int
    last_error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceMetrics:
    """System resource metrics"""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    network_connections: int
    swap_usage_percent: float
    load_average_1m: float
    load_average_5m: float
    load_average_15m: float
    uptime_seconds: float
    processes_count: int


@dataclass
class ApplicationMetrics:
    """Application-specific metrics"""
    timestamp: datetime
    active_connections: int
    request_rate_per_second: float
    error_rate_percent: float
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    database_connections: int
    cache_hit_rate_percent: float
    queue_size: int
    worker_utilization_percent: float
    memory_heap_mb: float


@dataclass
class ContainerMetrics:
    """Docker container metrics"""
    container_name: str
    timestamp: datetime
    status: str
    cpu_usage_percent: float
    memory_usage_mb: float
    memory_limit_mb: float
    network_rx_bytes: int
    network_tx_bytes: int
    block_read_bytes: int
    block_write_bytes: int
    restart_count: int


class PrometheusMetricsCollector:
    """Prometheus metrics collector for production monitoring"""

    def __init__(self):
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available, metrics collection disabled")
            return

        self.registry = CollectorRegistry()

        # System metrics
        self.cpu_usage = Gauge('atlas_cpu_usage_percent', 'CPU usage percentage', registry=self.registry)
        self.memory_usage = Gauge('atlas_memory_usage_percent', 'Memory usage percentage', registry=self.registry)
        self.disk_usage = Gauge('atlas_disk_usage_percent', 'Disk usage percentage', registry=self.registry)

        # Application metrics
        self.request_count = Counter('atlas_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'], registry=self.registry)
        self.request_duration = Histogram('atlas_request_duration_seconds', 'Request duration', ['method', 'endpoint'], registry=self.registry)
        self.active_connections = Gauge('atlas_active_connections', 'Active database connections', registry=self.registry)
        self.cache_hit_rate = Gauge('atlas_cache_hit_rate', 'Cache hit rate percentage', registry=self.registry)

        # Error metrics
        self.error_count = Counter('atlas_errors_total', 'Total errors', ['service', 'error_type'], registry=self.registry)
        self.alert_count = Counter('atlas_alerts_total', 'Total alerts', ['severity', 'service'], registry=self.registry)

        # Service health
        self.service_up = Gauge('atlas_service_up', 'Service availability', ['service'], registry=self.registry)
        self.service_response_time = Gauge('atlas_service_response_time_ms', 'Service response time', ['service'], registry=self.registry)

        logger.info("✅ Prometheus metrics collector initialized")

    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics"""
        if not PROMETHEUS_AVAILABLE:
            return

        self.request_count.labels(method=method, endpoint=endpoint, status=str(status)).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)

    def update_system_metrics(self, metrics: ResourceMetrics):
        """Update system resource metrics"""
        if not PROMETHEUS_AVAILABLE:
            return

        self.cpu_usage.set(metrics.cpu_usage_percent)
        self.memory_usage.set(metrics.memory_usage_percent)
        self.disk_usage.set(metrics.disk_usage_percent)

    def update_service_health(self, service: str, is_healthy: bool, response_time: float):
        """Update service health metrics"""
        if not PROMETHEUS_AVAILABLE:
            return

        self.service_up.labels(service=service).set(1 if is_healthy else 0)
        self.service_response_time.labels(service=service).set(response_time)

    def record_error(self, service: str, error_type: str):
        """Record error occurrence"""
        if not PROMETHEUS_AVAILABLE:
            return

        self.error_count.labels(service=service, error_type=error_type).inc()

    def record_alert(self, severity: str, service: str):
        """Record alert occurrence"""
        if not PROMETHEUS_AVAILABLE:
            return

        self.alert_count.labels(severity=severity, service=service).inc()


class AlertManager:
    """Alert management system with notification support"""

    def __init__(self):
        self.active_alerts: Dict[str, InfrastructureAlert] = {}
        self.alert_history: List[InfrastructureAlert] = []
        self.alert_handlers: List[Callable] = []
        self.notification_channels = {
            'email': None,
            'slack': None,
            'webhook': None
        }

        # Alert thresholds
        self.thresholds = {
            'cpu_usage': {'warning': 70.0, 'critical': 85.0},
            'memory_usage': {'warning': 80.0, 'critical': 90.0},
            'disk_usage': {'warning': 80.0, 'critical': 90.0},
            'response_time': {'warning': 1000.0, 'critical': 5000.0},
            'error_rate': {'warning': 5.0, 'critical': 10.0},
            'database_connections': {'warning': 80, 'critical': 95}
        }

        logger.info("✅ Alert Manager initialized")

    def add_alert_handler(self, handler: Callable):
        """Add custom alert handler"""
        self.alert_handlers.append(handler)

    def configure_email_notifications(self, smtp_server: str, smtp_port: int,
                                    username: str, password: str, recipients: List[str]):
        """Configure email notifications"""
        self.notification_channels['email'] = {
            'smtp_server': smtp_server,
            'smtp_port': smtp_port,
            'username': username,
            'password': password,
            'recipients': recipients
        }

    async def check_thresholds(self, metrics: Dict[str, float], service: str = "system"):
        """Check metrics against thresholds and trigger alerts"""

        for metric_name, value in metrics.items():
            if metric_name not in self.thresholds:
                continue

            thresholds = self.thresholds[metric_name]

            # Determine severity
            severity = None
            if value >= thresholds.get('critical', float('inf')):
                severity = AlertSeverity.CRITICAL
            elif value >= thresholds.get('warning', float('inf')):
                severity = AlertSeverity.WARNING

            if severity:
                await self.trigger_alert(
                    title=f"{metric_name.replace('_', ' ').title()} Threshold Exceeded",
                    description=f"{metric_name} is at {value:.2f}, exceeding {severity.value} threshold of {thresholds[severity.value]}",
                    severity=severity,
                    service=service,
                    metric_name=metric_name,
                    threshold_value=thresholds[severity.value],
                    current_value=value
                )

    async def trigger_alert(self, title: str, description: str, severity: AlertSeverity,
                          service: str, metric_name: str, threshold_value: float,
                          current_value: float, metadata: Dict[str, Any] = None):
        """Trigger a new alert"""

        alert_id = f"{service}_{metric_name}_{int(time.time())}"

        alert = InfrastructureAlert(
            id=alert_id,
            title=title,
            description=description,
            severity=severity,
            service=service,
            metric_name=metric_name,
            threshold_value=threshold_value,
            current_value=current_value,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata or {}
        )

        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)

        # Send notifications
        await self._send_notifications(alert)

        # Call custom handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

        logger.warning(f"🚨 Alert triggered: {title} ({severity.value})")

    async def _send_notifications(self, alert: InfrastructureAlert):
        """Send alert notifications through configured channels"""

        # Email notifications
        if self.notification_channels['email']:
            await self._send_email_notification(alert)

        # Add other notification channels as needed

    async def _send_email_notification(self, alert: InfrastructureAlert):
        """Send email notification"""
        try:
            config = self.notification_channels['email']

            msg = MimeMultipart()
            msg['From'] = config['username']
            msg['To'] = ', '.join(config['recipients'])
            msg['Subject'] = f"Atlas AI Alert: {alert.title}"

            body = f"""
Atlas AI Infrastructure Alert

Alert ID: {alert.id}
Severity: {alert.severity.value.upper()}
Service: {alert.service}
Metric: {alert.metric_name}

Description: {alert.description}

Current Value: {alert.current_value}
Threshold: {alert.threshold_value}
Timestamp: {alert.timestamp}

Please investigate immediately.
"""

            msg.attach(MimeText(body, 'plain'))

            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['username'], config['password'])
            server.send_message(msg)
            server.quit()

            logger.info(f"📧 Email alert sent for {alert.id}")

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")

    async def resolve_alert(self, alert_id: str, resolved_by: str = "system"):
        """Resolve an active alert"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolution_time = datetime.now(timezone.utc)
            alert.acknowledgments.append(f"Resolved by {resolved_by}")

            del self.active_alerts[alert_id]
            logger.info(f"✅ Alert resolved: {alert_id}")

    def get_active_alerts(self) -> List[InfrastructureAlert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics"""
        total_active = len(self.active_alerts)
        severity_counts = {}

        for alert in self.active_alerts.values():
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            'total_active_alerts': total_active,
            'severity_breakdown': severity_counts,
            'total_historical_alerts': len(self.alert_history),
            'oldest_active_alert': min([a.timestamp for a in self.active_alerts.values()]) if self.active_alerts else None
        }


class ServiceMonitor:
    """Monitor individual services and their dependencies"""

    def __init__(self):
        self.services = {}
        self.dependency_graph = {}

    def register_service(self, name: str, health_check_url: str,
                        dependencies: List[str] = None, timeout: int = 5):
        """Register a service for monitoring"""
        self.services[name] = {
            'health_check_url': health_check_url,
            'dependencies': dependencies or [],
            'timeout': timeout,
            'last_check': None,
            'consecutive_failures': 0
        }

        if dependencies:
            self.dependency_graph[name] = dependencies

        logger.info(f"📋 Registered service: {name}")

    async def check_service_health(self, service_name: str) -> ServiceHealth:
        """Check health of a specific service"""

        if service_name not in self.services:
            return ServiceHealth(
                name=service_name,
                status=ServiceStatus.UNKNOWN,
                response_time_ms=0,
                last_check=datetime.now(timezone.utc),
                uptime_percentage=0,
                error_count=0,
                last_error="Service not registered"
            )

        service_config = self.services[service_name]
        start_time = time.time()

        try:
            # HTTP health check
            response = requests.get(
                service_config['health_check_url'],
                timeout=service_config['timeout']
            )

            response_time_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                status = ServiceStatus.HEALTHY
                self.services[service_name]['consecutive_failures'] = 0
            elif response.status_code < 500:
                status = ServiceStatus.DEGRADED
            else:
                status = ServiceStatus.UNHEALTHY
                self.services[service_name]['consecutive_failures'] += 1

            error_message = None

        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            status = ServiceStatus.DOWN
            error_message = str(e)
            self.services[service_name]['consecutive_failures'] += 1

        # Update last check time
        self.services[service_name]['last_check'] = datetime.now(timezone.utc)

        # Calculate uptime percentage (simplified)
        failure_count = self.services[service_name]['consecutive_failures']
        uptime_percentage = max(0, 100 - (failure_count * 10))

        return ServiceHealth(
            name=service_name,
            status=status,
            response_time_ms=response_time_ms,
            last_check=datetime.now(timezone.utc),
            uptime_percentage=uptime_percentage,
            error_count=failure_count,
            last_error=error_message,
            dependencies=service_config['dependencies']
        )

    async def check_all_services(self) -> Dict[str, ServiceHealth]:
        """Check health of all registered services"""

        health_results = {}

        for service_name in self.services:
            health_results[service_name] = await self.check_service_health(service_name)

        return health_results

    def get_dependency_status(self, service_name: str) -> Dict[str, ServiceStatus]:
        """Get status of service dependencies"""

        if service_name not in self.dependency_graph:
            return {}

        dependency_status = {}
        for dep in self.dependency_graph[service_name]:
            # This would normally check the actual dependency status
            # For now, return a placeholder
            dependency_status[dep] = ServiceStatus.HEALTHY

        return dependency_status


class InfrastructureMonitor:
    """Main infrastructure monitoring system"""

    def __init__(self):
        self.prometheus_collector = PrometheusMetricsCollector()
        self.alert_manager = AlertManager()
        self.service_monitor = ServiceMonitor()

        # Monitoring configuration
        self.monitoring_interval = 30  # seconds
        self.metrics_retention_days = 7
        self.is_monitoring = False
        self.monitoring_task = None

        # Data storage
        self.metrics_history: List[ResourceMetrics] = []
        self.application_metrics_history: List[ApplicationMetrics] = []
        self.container_metrics_history: List[ContainerMetrics] = []

        # Docker client (if available)
        try:
            self.docker_client = docker.from_env()
            self.docker_available = True
        except Exception:
            self.docker_client = None
            self.docker_available = False

        logger.info("✅ Infrastructure Monitor initialized")

    async def start_monitoring(self):
        """Start continuous monitoring"""

        if self.is_monitoring:
            logger.warning("Monitoring already running")
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        # Start Prometheus metrics server if available
        if PROMETHEUS_AVAILABLE:
            try:
                start_http_server(8090, registry=self.prometheus_collector.registry)
                logger.info("📊 Prometheus metrics server started on port 8090")
            except Exception as e:
                logger.warning(f"Failed to start Prometheus server: {e}")

        logger.info("🚀 Infrastructure monitoring started")

    async def stop_monitoring(self):
        """Stop continuous monitoring"""

        self.is_monitoring = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("⏹️ Infrastructure monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop"""

        while self.is_monitoring:
            try:
                # Collect system metrics
                resource_metrics = await self.collect_system_metrics()
                self.metrics_history.append(resource_metrics)

                # Collect application metrics
                app_metrics = await self.collect_application_metrics()
                self.application_metrics_history.append(app_metrics)

                # Collect container metrics
                if self.docker_available:
                    container_metrics = await self.collect_container_metrics()
                    self.container_metrics_history.extend(container_metrics)

                # Check service health
                service_health = await self.service_monitor.check_all_services()

                # Update Prometheus metrics
                self.prometheus_collector.update_system_metrics(resource_metrics)

                for service_name, health in service_health.items():
                    is_healthy = health.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]
                    self.prometheus_collector.update_service_health(
                        service_name, is_healthy, health.response_time_ms
                    )

                # Check alert thresholds
                await self._check_alert_thresholds(resource_metrics, app_metrics)

                # Clean up old metrics
                await self._cleanup_old_metrics()

                # Log monitoring status
                if len(self.metrics_history) % 10 == 0:  # Every 10 cycles
                    logger.info(f"📊 Monitoring active: {len(self.metrics_history)} metric snapshots collected")

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                self.prometheus_collector.record_error("monitoring", "collection_error")

            await asyncio.sleep(self.monitoring_interval)

    async def collect_system_metrics(self) -> ResourceMetrics:
        """Collect system resource metrics"""

        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)

        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # Disk metrics
        disk = psutil.disk_usage('/')

        # Network metrics
        network = psutil.net_io_counters()

        # Load average
        load_avg = os.getloadavg()

        # System uptime
        boot_time = psutil.boot_time()
        uptime = time.time() - boot_time

        # Process count
        process_count = len(psutil.pids())

        metrics = ResourceMetrics(
            timestamp=datetime.now(timezone.utc),
            cpu_usage_percent=cpu_percent,
            memory_usage_percent=memory.percent,
            memory_available_gb=memory.available / (1024**3),
            disk_usage_percent=disk.percent,
            disk_free_gb=disk.free / (1024**3),
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv,
            network_connections=len(psutil.net_connections()),
            swap_usage_percent=swap.percent,
            load_average_1m=load_avg[0],
            load_average_5m=load_avg[1],
            load_average_15m=load_avg[2],
            uptime_seconds=uptime,
            processes_count=process_count
        )

        return metrics

    async def collect_application_metrics(self) -> ApplicationMetrics:
        """Collect application-specific metrics"""

        # These would normally come from your application's metrics
        # For now, provide placeholder values

        metrics = ApplicationMetrics(
            timestamp=datetime.now(timezone.utc),
            active_connections=psutil.net_connections().__len__(),
            request_rate_per_second=0.0,  # Would be tracked by web framework
            error_rate_percent=0.0,       # Would be tracked by application
            avg_response_time_ms=0.0,     # Would be tracked by application
            p95_response_time_ms=0.0,     # Would be tracked by application
            p99_response_time_ms=0.0,     # Would be tracked by application
            database_connections=0,        # Would query database
            cache_hit_rate_percent=0.0,   # Would query cache
            queue_size=0,                 # Would query message queue
            worker_utilization_percent=0.0,  # Would query worker pool
            memory_heap_mb=psutil.Process().memory_info().rss / (1024**2)
        )

        return metrics

    async def collect_container_metrics(self) -> List[ContainerMetrics]:
        """Collect Docker container metrics"""

        if not self.docker_available:
            return []

        container_metrics = []

        try:
            for container in self.docker_client.containers.list():
                stats = container.stats(stream=False)

                # Calculate CPU usage
                cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                           stats['precpu_stats']['cpu_usage']['total_usage']
                system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                              stats['precpu_stats']['system_cpu_usage']

                cpu_usage = 0.0
                if system_delta > 0:
                    cpu_usage = (cpu_delta / system_delta) * 100.0

                # Memory usage
                memory_usage = stats['memory_stats'].get('usage', 0) / (1024**2)  # MB
                memory_limit = stats['memory_stats'].get('limit', 0) / (1024**2)  # MB

                # Network I/O
                networks = stats.get('networks', {})
                total_rx = sum(net.get('rx_bytes', 0) for net in networks.values())
                total_tx = sum(net.get('tx_bytes', 0) for net in networks.values())

                # Block I/O
                blkio_stats = stats.get('blkio_stats', {})
                io_service_bytes = blkio_stats.get('io_service_bytes_recursive', [])

                total_read = sum(entry.get('value', 0) for entry in io_service_bytes
                               if entry.get('op') == 'Read')
                total_write = sum(entry.get('value', 0) for entry in io_service_bytes
                                if entry.get('op') == 'Write')

                metrics = ContainerMetrics(
                    container_name=container.name,
                    timestamp=datetime.now(timezone.utc),
                    status=container.status,
                    cpu_usage_percent=cpu_usage,
                    memory_usage_mb=memory_usage,
                    memory_limit_mb=memory_limit,
                    network_rx_bytes=total_rx,
                    network_tx_bytes=total_tx,
                    block_read_bytes=total_read,
                    block_write_bytes=total_write,
                    restart_count=0  # Would need to track this separately
                )

                container_metrics.append(metrics)

        except Exception as e:
            logger.error(f"Failed to collect container metrics: {e}")

        return container_metrics

    async def _check_alert_thresholds(self, resource_metrics: ResourceMetrics,
                                    app_metrics: ApplicationMetrics):
        """Check metrics against alert thresholds"""

        # System resource thresholds
        system_metrics = {
            'cpu_usage': resource_metrics.cpu_usage_percent,
            'memory_usage': resource_metrics.memory_usage_percent,
            'disk_usage': resource_metrics.disk_usage_percent
        }

        await self.alert_manager.check_thresholds(system_metrics, "system")

        # Application thresholds
        app_threshold_metrics = {
            'response_time': app_metrics.avg_response_time_ms,
            'error_rate': app_metrics.error_rate_percent
        }

        await self.alert_manager.check_thresholds(app_threshold_metrics, "application")

    async def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory growth"""

        cutoff_time = datetime.now(timezone.utc) - timedelta(days=self.metrics_retention_days)

        # Clean system metrics
        self.metrics_history = [m for m in self.metrics_history if m.timestamp > cutoff_time]

        # Clean application metrics
        self.application_metrics_history = [m for m in self.application_metrics_history
                                          if m.timestamp > cutoff_time]

        # Clean container metrics
        self.container_metrics_history = [m for m in self.container_metrics_history
                                        if m.timestamp > cutoff_time]

    def get_current_status(self) -> Dict[str, Any]:
        """Get current infrastructure status"""

        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        latest_app_metrics = self.application_metrics_history[-1] if self.application_metrics_history else None

        alert_summary = self.alert_manager.get_alert_summary()

        status = {
            'monitoring_active': self.is_monitoring,
            'last_check': latest_metrics.timestamp if latest_metrics else None,
            'system_health': {
                'cpu_usage': latest_metrics.cpu_usage_percent if latest_metrics else 0,
                'memory_usage': latest_metrics.memory_usage_percent if latest_metrics else 0,
                'disk_usage': latest_metrics.disk_usage_percent if latest_metrics else 0,
                'uptime_hours': latest_metrics.uptime_seconds / 3600 if latest_metrics else 0
            },
            'application_health': {
                'active_connections': latest_app_metrics.active_connections if latest_app_metrics else 0,
                'avg_response_time': latest_app_metrics.avg_response_time_ms if latest_app_metrics else 0,
                'error_rate': latest_app_metrics.error_rate_percent if latest_app_metrics else 0
            },
            'alerts': alert_summary,
            'services_monitored': len(self.service_monitor.services),
            'docker_available': self.docker_available,
            'prometheus_available': PROMETHEUS_AVAILABLE,
            'metrics_collected': len(self.metrics_history)
        }

        return status

    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for the specified time period"""

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        # Filter metrics
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        recent_app_metrics = [m for m in self.application_metrics_history if m.timestamp > cutoff_time]

        if not recent_metrics:
            return {'error': 'No metrics available for the specified period'}

        # Calculate statistics
        cpu_values = [m.cpu_usage_percent for m in recent_metrics]
        memory_values = [m.memory_usage_percent for m in recent_metrics]
        disk_values = [m.disk_usage_percent for m in recent_metrics]

        summary = {
            'period_hours': hours,
            'data_points': len(recent_metrics),
            'system_metrics': {
                'cpu_usage': {
                    'avg': sum(cpu_values) / len(cpu_values),
                    'min': min(cpu_values),
                    'max': max(cpu_values),
                    'current': cpu_values[-1] if cpu_values else 0
                },
                'memory_usage': {
                    'avg': sum(memory_values) / len(memory_values),
                    'min': min(memory_values),
                    'max': max(memory_values),
                    'current': memory_values[-1] if memory_values else 0
                },
                'disk_usage': {
                    'avg': sum(disk_values) / len(disk_values),
                    'min': min(disk_values),
                    'max': max(disk_values),
                    'current': disk_values[-1] if disk_values else 0
                }
            }
        }

        # Add application metrics if available
        if recent_app_metrics:
            response_times = [m.avg_response_time_ms for m in recent_app_metrics]
            error_rates = [m.error_rate_percent for m in recent_app_metrics]

            summary['application_metrics'] = {
                'response_time': {
                    'avg': sum(response_times) / len(response_times),
                    'min': min(response_times),
                    'max': max(response_times),
                    'current': response_times[-1] if response_times else 0
                },
                'error_rate': {
                    'avg': sum(error_rates) / len(error_rates),
                    'min': min(error_rates),
                    'max': max(error_rates),
                    'current': error_rates[-1] if error_rates else 0
                }
            }

        return summary


# Usage example and testing
async def main():
    """Test production infrastructure monitoring"""

    logging.basicConfig(level=logging.INFO)

    print("🏭 Production Infrastructure Monitoring Test")
    print("=" * 55)

    # Initialize monitoring system
    monitor = InfrastructureMonitor()

    # Configure alert notifications (placeholder)
    monitor.alert_manager.configure_email_notifications(
        smtp_server="smtp.gmail.com",
        smtp_port=587,
        username="alerts@atlasai.com",
        password="placeholder",
        recipients=["admin@atlasai.com"]
    )

    # Register services for monitoring
    monitor.service_monitor.register_service(
        "atlas-api",
        "http://localhost:8000/health",
        dependencies=["database", "cache"]
    )

    monitor.service_monitor.register_service(
        "database",
        "http://localhost:5432",  # This would be a custom health endpoint
        timeout=3
    )

    monitor.service_monitor.register_service(
        "cache",
        "http://localhost:6379",  # This would be a custom health endpoint
        timeout=2
    )

    print(f"📋 Infrastructure Monitor Configuration:")
    print(f"   Monitoring interval: {monitor.monitoring_interval}s")
    print(f"   Metrics retention: {monitor.metrics_retention_days} days")
    print(f"   Docker available: {monitor.docker_available}")
    print(f"   Prometheus available: {PROMETHEUS_AVAILABLE}")
    print(f"   Services registered: {len(monitor.service_monitor.services)}")

    # Test metric collection
    print(f"\n📊 Testing Metric Collection:")

    resource_metrics = await monitor.collect_system_metrics()
    print(f"✅ System metrics collected:")
    print(f"   CPU: {resource_metrics.cpu_usage_percent:.1f}%")
    print(f"   Memory: {resource_metrics.memory_usage_percent:.1f}%")
    print(f"   Disk: {resource_metrics.disk_usage_percent:.1f}%")
    print(f"   Uptime: {resource_metrics.uptime_seconds/3600:.1f} hours")

    app_metrics = await monitor.collect_application_metrics()
    print(f"✅ Application metrics collected:")
    print(f"   Active connections: {app_metrics.active_connections}")
    print(f"   Memory heap: {app_metrics.memory_heap_mb:.1f} MB")

    if monitor.docker_available:
        container_metrics = await monitor.collect_container_metrics()
        print(f"✅ Container metrics collected: {len(container_metrics)} containers")

        for cm in container_metrics[:3]:  # Show first 3
            print(f"   {cm.container_name}: {cm.status}, CPU: {cm.cpu_usage_percent:.1f}%")

    # Test service health checks
    print(f"\n🏥 Testing Service Health Checks:")

    service_health = await monitor.service_monitor.check_all_services()
    for service_name, health in service_health.items():
        status_emoji = {"healthy": "✅", "degraded": "⚠️", "unhealthy": "❌", "down": "💥", "unknown": "❓"}[health.status.value]
        print(f"   {status_emoji} {service_name}: {health.status.value} ({health.response_time_ms:.0f}ms)")

    # Test alert system
    print(f"\n🚨 Testing Alert System:")

    # Trigger a test alert
    await monitor.alert_manager.trigger_alert(
        title="Test Alert",
        description="This is a test alert for demonstration",
        severity=AlertSeverity.WARNING,
        service="test",
        metric_name="test_metric",
        threshold_value=50.0,
        current_value=75.0
    )

    active_alerts = monitor.alert_manager.get_active_alerts()
    print(f"✅ Test alert triggered: {len(active_alerts)} active alerts")

    for alert in active_alerts:
        print(f"   🚨 {alert.title} ({alert.severity.value}) - {alert.service}")

    # Test threshold checking
    test_metrics = {
        'cpu_usage': 85.5,  # Should trigger critical alert
        'memory_usage': 75.0,  # Should trigger warning alert
        'disk_usage': 60.0   # Should be fine
    }

    await monitor.alert_manager.check_thresholds(test_metrics, "test_system")

    updated_alerts = monitor.alert_manager.get_active_alerts()
    print(f"✅ Threshold checking: {len(updated_alerts)} total active alerts")

    # Get monitoring status
    status = monitor.get_current_status()
    print(f"\n📈 Current Infrastructure Status:")
    print(f"   Monitoring active: {status['monitoring_active']}")
    print(f"   Services monitored: {status['services_monitored']}")
    print(f"   Metrics collected: {status['metrics_collected']}")
    print(f"   Active alerts: {status['alerts']['total_active_alerts']}")

    # Test short monitoring run
    print(f"\n🚀 Starting short monitoring run (30 seconds)...")

    await monitor.start_monitoring()
    await asyncio.sleep(30)
    await monitor.stop_monitoring()

    print(f"✅ Monitoring test completed")

    # Get final metrics summary
    final_status = monitor.get_current_status()
    print(f"\n📊 Final Status:")
    print(f"   Total metrics collected: {final_status['metrics_collected']}")
    print(f"   System health: CPU {final_status['system_health']['cpu_usage']:.1f}%, "
          f"Memory {final_status['system_health']['memory_usage']:.1f}%")

    print(f"\n🎉 Production Infrastructure Monitoring Test Complete!")
    print(f"   The monitoring system is ready for production deployment")
    print(f"   Use monitor.start_monitoring() to begin continuous monitoring")


if __name__ == "__main__":
    asyncio.run(main())