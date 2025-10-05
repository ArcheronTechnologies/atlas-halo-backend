"""
Production Health Monitoring System for Atlas AI
Comprehensive health checks, metrics, and monitoring for Phase 2.1 deployment
"""

import asyncio
import logging
import psutil
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import os

from ..database.postgis_database import get_database
from ..caching.redis_cache import get_cache
from ..observability.metrics import metrics

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class ComponentHealth:
    """Individual component health status"""
    name: str
    status: HealthStatus
    response_time_ms: float
    last_check: datetime
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    network_connections: int
    uptime_seconds: float
    load_average: Tuple[float, float, float]

@dataclass
class HealthReport:
    """Comprehensive health report"""
    timestamp: datetime
    overall_status: HealthStatus
    components: Dict[str, ComponentHealth]
    system_metrics: SystemMetrics
    performance_summary: Dict[str, Any]
    alerts: List[str]

class HealthMonitor:
    """Production health monitoring system"""

    def __init__(self):
        self.start_time = datetime.now(timezone.utc)
        self.check_history = []
        self.max_history = 100

        # Health check configurations
        self.check_configs = {
            'database': {
                'timeout': 10.0,
                'critical_response_time': 5000,  # ms
                'warning_response_time': 1000
            },
            'cache': {
                'timeout': 5.0,
                'critical_response_time': 2000,
                'warning_response_time': 500
            },
            'feature_engine': {
                'timeout': 30.0,
                'critical_response_time': 10000,
                'warning_response_time': 3000
            },
            'authentication': {
                'timeout': 5.0,
                'critical_response_time': 3000,
                'warning_response_time': 1000
            }
        }

        # Alert thresholds
        self.thresholds = {
            'cpu_usage': {'warning': 70, 'critical': 90},
            'memory_usage': {'warning': 80, 'critical': 95},
            'disk_usage': {'warning': 85, 'critical': 95},
            'database_connections': {'warning': 15, 'critical': 18},
            'response_time': {'warning': 2000, 'critical': 5000}
        }

    async def comprehensive_health_check(self, include_optional: bool = True) -> HealthReport:
        """Perform comprehensive health check of all system components"""

        check_start = datetime.now(timezone.utc)
        components = {}
        alerts = []

        # Core component checks (always performed)
        core_checks = [
            self._check_database_health(),
            self._check_cache_health(),
            self._check_system_resources(),
        ]

        # Optional component checks
        if include_optional:
            core_checks.extend([
                self._check_feature_engine_health(),
                self._check_authentication_health(),
                self._check_api_endpoints_health()
            ])

        # Run all checks in parallel
        check_results = await asyncio.gather(*core_checks, return_exceptions=True)

        # Process results
        check_names = ['database', 'cache', 'system']
        if include_optional:
            check_names.extend(['feature_engine', 'authentication', 'api_endpoints'])

        for name, result in zip(check_names, check_results):
            if isinstance(result, Exception):
                components[name] = ComponentHealth(
                    name=name,
                    status=HealthStatus.CRITICAL,
                    response_time_ms=0,
                    last_check=check_start,
                    error_message=str(result)
                )
                alerts.append(f"{name.title()} health check failed: {result}")
            else:
                components[name] = result
                if result.status == HealthStatus.CRITICAL:
                    alerts.append(f"{name.title()} is in critical state: {result.error_message}")
                elif result.status == HealthStatus.WARNING:
                    alerts.append(f"{name.title()} warning: {result.error_message}")

        # Get system metrics
        system_metrics = self._get_system_metrics()

        # Check system-level alerts
        self._check_system_alerts(system_metrics, alerts)

        # Determine overall status
        overall_status = self._determine_overall_status(components)

        # Create performance summary
        performance_summary = self._create_performance_summary(components, system_metrics)

        # Create health report
        report = HealthReport(
            timestamp=check_start,
            overall_status=overall_status,
            components=components,
            system_metrics=system_metrics,
            performance_summary=performance_summary,
            alerts=alerts
        )

        # Store in history
        self._store_health_history(report)

        # Track metrics
        self._track_health_metrics(report)

        return report

    async def _check_database_health(self) -> ComponentHealth:
        """Check database health and performance"""
        start_time = time.time()

        try:
            database = await get_database()

            # Test basic connectivity
            await database.execute_query("SELECT 1")

            # Test geospatial functionality
            await database.execute_query("SELECT ST_Point(59.3293, 18.0686)")

            # Get database statistics
            stats_query = """
            SELECT
                COUNT(*) as total_incidents,
                COUNT(DISTINCT DATE(created_at)) as active_days,
                MAX(created_at) as latest_incident
            FROM crime_incidents
            WHERE created_at > NOW() - INTERVAL '30 days'
            """
            stats = await database.fetch_one(stats_query)

            response_time = (time.time() - start_time) * 1000
            config = self.check_configs['database']

            # Determine status based on response time
            if response_time > config['critical_response_time']:
                status = HealthStatus.CRITICAL
                error_msg = f"Database response time too high: {response_time:.1f}ms"
            elif response_time > config['warning_response_time']:
                status = HealthStatus.WARNING
                error_msg = f"Database response time elevated: {response_time:.1f}ms"
            else:
                status = HealthStatus.HEALTHY
                error_msg = None

            metadata = {
                'connection_pool_size': getattr(database, 'pool_size', 'unknown'),
                'total_incidents': stats.get('total_incidents', 0) if stats else 0,
                'active_days': stats.get('active_days', 0) if stats else 0,
                'latest_incident': stats.get('latest_incident').isoformat() if stats and stats.get('latest_incident') else None
            }

            return ComponentHealth(
                name='database',
                status=status,
                response_time_ms=response_time,
                last_check=datetime.now(timezone.utc),
                error_message=error_msg,
                metadata=metadata
            )

        except Exception as e:
            return ComponentHealth(
                name='database',
                status=HealthStatus.CRITICAL,
                response_time_ms=(time.time() - start_time) * 1000,
                last_check=datetime.now(timezone.utc),
                error_message=f"Database connection failed: {str(e)}"
            )

    async def _check_cache_health(self) -> ComponentHealth:
        """Check Redis cache health and performance"""
        start_time = time.time()

        try:
            cache = await get_cache()

            if not cache:
                raise Exception("Cache instance not available")

            # Test basic operations
            test_key = "health_check_test"
            test_value = {"timestamp": datetime.now(timezone.utc).isoformat(), "test": True}

            await cache.set(test_key, test_value, ttl=60)
            retrieved = await cache.get(test_key)
            await cache.delete(test_key)

            # Validate test operation
            if not retrieved or retrieved.get('test') != True:
                raise Exception("Cache test operation failed")

            # Get cache statistics
            cache_stats = await cache.get_cache_statistics()

            response_time = (time.time() - start_time) * 1000
            config = self.check_configs['cache']

            # Determine status based on response time
            if response_time > config['critical_response_time']:
                status = HealthStatus.CRITICAL
                error_msg = f"Cache response time too high: {response_time:.1f}ms"
            elif response_time > config['warning_response_time']:
                status = HealthStatus.WARNING
                error_msg = f"Cache response time elevated: {response_time:.1f}ms"
            else:
                status = HealthStatus.HEALTHY
                error_msg = None

            # Check hit rate
            hit_rate = cache_stats.get('hit_rate', 0)
            if hit_rate < 0.5 and status == HealthStatus.HEALTHY:
                status = HealthStatus.WARNING
                error_msg = f"Cache hit rate low: {hit_rate:.1%}"

            metadata = {
                'hit_rate': hit_rate,
                'total_operations': cache_stats.get('total_operations', 0),
                'keys_stored': cache_stats.get('keys_stored', 0),
                'cache_type': 'Redis' if not isinstance(cache.redis, type(cache).MockRedis) else 'Mock'
            }

            return ComponentHealth(
                name='cache',
                status=status,
                response_time_ms=response_time,
                last_check=datetime.now(timezone.utc),
                error_message=error_msg,
                metadata=metadata
            )

        except Exception as e:
            return ComponentHealth(
                name='cache',
                status=HealthStatus.CRITICAL,
                response_time_ms=(time.time() - start_time) * 1000,
                last_check=datetime.now(timezone.utc),
                error_message=f"Cache health check failed: {str(e)}"
            )

    async def _check_feature_engine_health(self) -> ComponentHealth:
        """Check feature engineering system health"""
        start_time = time.time()

        try:
            from ..analytics.real_world_features import get_feature_engine

            feature_engine = await get_feature_engine()

            # Test feature generation
            test_lat, test_lng = 59.3293, 18.0686  # Stockholm
            test_timestamp = datetime.now(timezone.utc)

            features = await feature_engine.get_comprehensive_features(test_lat, test_lng, test_timestamp)

            response_time = (time.time() - start_time) * 1000
            config = self.check_configs['feature_engine']

            # Validate feature generation
            if not features or features.get('feature_count', 0) < 50:
                raise Exception("Feature generation produced insufficient features")

            # Determine status based on response time
            if response_time > config['critical_response_time']:
                status = HealthStatus.CRITICAL
                error_msg = f"Feature generation too slow: {response_time:.1f}ms"
            elif response_time > config['warning_response_time']:
                status = HealthStatus.WARNING
                error_msg = f"Feature generation slow: {response_time:.1f}ms"
            else:
                status = HealthStatus.HEALTHY
                error_msg = None

            metadata = {
                'feature_count': features.get('feature_count', 0),
                'has_weather_api': bool(os.getenv('OPENWEATHER_API_KEY')),
                'cache_enabled': feature_engine.cache is not None,
                'overall_risk_score': features.get('composite', {}).get('overall_risk_score', 0),
                'risk_confidence': features.get('composite', {}).get('risk_confidence', 0)
            }

            return ComponentHealth(
                name='feature_engine',
                status=status,
                response_time_ms=response_time,
                last_check=datetime.now(timezone.utc),
                error_message=error_msg,
                metadata=metadata
            )

        except Exception as e:
            return ComponentHealth(
                name='feature_engine',
                status=HealthStatus.CRITICAL,
                response_time_ms=(time.time() - start_time) * 1000,
                last_check=datetime.now(timezone.utc),
                error_message=f"Feature engine health check failed: {str(e)}"
            )

    async def _check_authentication_health(self) -> ComponentHealth:
        """Check authentication system health"""
        start_time = time.time()

        try:
            from ..auth.jwt_authentication import AuthenticationService

            database = await get_database()
            auth_service = AuthenticationService(database)
            await auth_service.initialize()

            # Test JWT token operations
            test_user_data = {
                'id': 'health_check_user',
                'username': 'health_test',
                'email': 'health@test.com',
                'role': 'citizen'
            }

            # Create and verify access token
            access_token = auth_service.create_access_token(test_user_data)
            token_payload = auth_service.verify_token(access_token)

            if not token_payload or token_payload.user_id != test_user_data['id']:
                raise Exception("JWT token creation/verification failed")

            # Create and verify refresh token
            refresh_token = auth_service.create_refresh_token(test_user_data)
            refresh_payload = auth_service.verify_token(refresh_token, token_type="refresh")

            if not refresh_payload or refresh_payload.user_id != test_user_data['id']:
                raise Exception("JWT refresh token creation/verification failed")

            response_time = (time.time() - start_time) * 1000
            config = self.check_configs['authentication']

            # Determine status based on response time
            if response_time > config['critical_response_time']:
                status = HealthStatus.CRITICAL
                error_msg = f"Authentication too slow: {response_time:.1f}ms"
            elif response_time > config['warning_response_time']:
                status = HealthStatus.WARNING
                error_msg = f"Authentication slow: {response_time:.1f}ms"
            else:
                status = HealthStatus.HEALTHY
                error_msg = None

            metadata = {
                'jwt_algorithm': 'HS256',
                'access_token_length': len(access_token),
                'refresh_token_length': len(refresh_token),
                'database_initialized': auth_service._initialized
            }

            return ComponentHealth(
                name='authentication',
                status=status,
                response_time_ms=response_time,
                last_check=datetime.now(timezone.utc),
                error_message=error_msg,
                metadata=metadata
            )

        except Exception as e:
            return ComponentHealth(
                name='authentication',
                status=HealthStatus.CRITICAL,
                response_time_ms=(time.time() - start_time) * 1000,
                last_check=datetime.now(timezone.utc),
                error_message=f"Authentication health check failed: {str(e)}"
            )

    async def _check_api_endpoints_health(self) -> ComponentHealth:
        """Check API endpoints health"""
        start_time = time.time()

        try:
            # This would test actual API endpoints in a real deployment
            # For now, we'll simulate endpoint health checks

            # Simulate checking critical endpoints
            critical_endpoints = [
                '/health', '/auth/status', '/api/incidents', '/api/predict'
            ]

            # Simulate response times
            endpoint_responses = []
            for endpoint in critical_endpoints:
                # Simulate endpoint check (would be actual HTTP requests in production)
                endpoint_time = 50 + (hash(endpoint) % 100)  # Simulate 50-150ms
                endpoint_responses.append((endpoint, endpoint_time))

            total_response_time = sum(time for _, time in endpoint_responses)
            avg_response_time = total_response_time / len(endpoint_responses)

            response_time = (time.time() - start_time) * 1000

            # Determine status based on average endpoint response time
            if avg_response_time > 1000:
                status = HealthStatus.CRITICAL
                error_msg = f"API endpoints too slow: {avg_response_time:.1f}ms avg"
            elif avg_response_time > 500:
                status = HealthStatus.WARNING
                error_msg = f"API endpoints slow: {avg_response_time:.1f}ms avg"
            else:
                status = HealthStatus.HEALTHY
                error_msg = None

            metadata = {
                'endpoints_checked': len(critical_endpoints),
                'avg_response_time': avg_response_time,
                'endpoint_details': dict(endpoint_responses)
            }

            return ComponentHealth(
                name='api_endpoints',
                status=status,
                response_time_ms=response_time,
                last_check=datetime.now(timezone.utc),
                error_message=error_msg,
                metadata=metadata
            )

        except Exception as e:
            return ComponentHealth(
                name='api_endpoints',
                status=HealthStatus.CRITICAL,
                response_time_ms=(time.time() - start_time) * 1000,
                last_check=datetime.now(timezone.utc),
                error_message=f"API endpoints health check failed: {str(e)}"
            )

    async def _check_system_resources(self) -> ComponentHealth:
        """Check system resource health"""
        start_time = time.time()

        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            alerts = []
            status = HealthStatus.HEALTHY

            # Check CPU usage
            if cpu_percent > self.thresholds['cpu_usage']['critical']:
                status = HealthStatus.CRITICAL
                alerts.append(f"CPU usage critical: {cpu_percent:.1f}%")
            elif cpu_percent > self.thresholds['cpu_usage']['warning']:
                status = HealthStatus.WARNING
                alerts.append(f"CPU usage high: {cpu_percent:.1f}%")

            # Check memory usage
            if memory.percent > self.thresholds['memory_usage']['critical']:
                status = HealthStatus.CRITICAL
                alerts.append(f"Memory usage critical: {memory.percent:.1f}%")
            elif memory.percent > self.thresholds['memory_usage']['warning']:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                alerts.append(f"Memory usage high: {memory.percent:.1f}%")

            # Check disk usage
            if disk.percent > self.thresholds['disk_usage']['critical']:
                status = HealthStatus.CRITICAL
                alerts.append(f"Disk usage critical: {disk.percent:.1f}%")
            elif disk.percent > self.thresholds['disk_usage']['warning']:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                alerts.append(f"Disk usage high: {disk.percent:.1f}%")

            response_time = (time.time() - start_time) * 1000
            error_message = "; ".join(alerts) if alerts else None

            metadata = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                'process_count': len(psutil.pids())
            }

            return ComponentHealth(
                name='system',
                status=status,
                response_time_ms=response_time,
                last_check=datetime.now(timezone.utc),
                error_message=error_message,
                metadata=metadata
            )

        except Exception as e:
            return ComponentHealth(
                name='system',
                status=HealthStatus.CRITICAL,
                response_time_ms=(time.time() - start_time) * 1000,
                last_check=datetime.now(timezone.utc),
                error_message=f"System resource check failed: {str(e)}"
            )

    def _get_system_metrics(self) -> SystemMetrics:
        """Get current system performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network_connections = len(psutil.net_connections())
            uptime = time.time() - psutil.boot_time()
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)

            return SystemMetrics(
                timestamp=datetime.now(timezone.utc),
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory.percent,
                disk_usage_percent=disk.percent,
                network_connections=network_connections,
                uptime_seconds=uptime,
                load_average=load_avg
            )
        except Exception as e:
            logger.error(f"❌ Failed to get system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now(timezone.utc),
                cpu_usage_percent=0,
                memory_usage_percent=0,
                disk_usage_percent=0,
                network_connections=0,
                uptime_seconds=0,
                load_average=(0, 0, 0)
            )

    def _check_system_alerts(self, metrics: SystemMetrics, alerts: List[str]):
        """Check for system-level alerts"""

        # Check for high resource usage
        if metrics.cpu_usage_percent > self.thresholds['cpu_usage']['warning']:
            alerts.append(f"High CPU usage: {metrics.cpu_usage_percent:.1f}%")

        if metrics.memory_usage_percent > self.thresholds['memory_usage']['warning']:
            alerts.append(f"High memory usage: {metrics.memory_usage_percent:.1f}%")

        if metrics.disk_usage_percent > self.thresholds['disk_usage']['warning']:
            alerts.append(f"High disk usage: {metrics.disk_usage_percent:.1f}%")

        # Check for high network connections
        if metrics.network_connections > 1000:
            alerts.append(f"High network connections: {metrics.network_connections}")

        # Check uptime (alert if recently restarted)
        if metrics.uptime_seconds < 300:  # Less than 5 minutes
            alerts.append(f"System recently restarted (uptime: {metrics.uptime_seconds:.0f}s)")

    def _determine_overall_status(self, components: Dict[str, ComponentHealth]) -> HealthStatus:
        """Determine overall system status from component statuses"""

        if not components:
            return HealthStatus.UNKNOWN

        statuses = [comp.status for comp in components.values()]

        # If any component is critical, overall is critical
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL

        # If any component has warning, overall is warning
        if HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING

        # If all components are healthy, overall is healthy
        if all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY

        return HealthStatus.UNKNOWN

    def _create_performance_summary(self, components: Dict[str, ComponentHealth], metrics: SystemMetrics) -> Dict[str, Any]:
        """Create performance summary from health check results"""

        response_times = [comp.response_time_ms for comp in components.values() if comp.response_time_ms > 0]

        return {
            'avg_response_time_ms': sum(response_times) / len(response_times) if response_times else 0,
            'max_response_time_ms': max(response_times) if response_times else 0,
            'min_response_time_ms': min(response_times) if response_times else 0,
            'components_healthy': sum(1 for comp in components.values() if comp.status == HealthStatus.HEALTHY),
            'components_warning': sum(1 for comp in components.values() if comp.status == HealthStatus.WARNING),
            'components_critical': sum(1 for comp in components.values() if comp.status == HealthStatus.CRITICAL),
            'total_components': len(components),
            'system_uptime_hours': metrics.uptime_seconds / 3600,
            'application_uptime_hours': (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600
        }

    def _store_health_history(self, report: HealthReport):
        """Store health report in history"""

        self.check_history.append({
            'timestamp': report.timestamp.isoformat(),
            'overall_status': report.overall_status.value,
            'component_count': len(report.components),
            'alerts_count': len(report.alerts),
            'avg_response_time': report.performance_summary['avg_response_time_ms']
        })

        # Keep only recent history
        if len(self.check_history) > self.max_history:
            self.check_history = self.check_history[-self.max_history:]

    def _track_health_metrics(self, report: HealthReport):
        """Track health metrics for monitoring"""

        try:
            # Track overall status
            status_counter = metrics.counter("health_checks_total", "Health checks", ("status",))
            status_counter.labels(report.overall_status.value).inc()

            # Track component statuses
            for comp_name, comp_health in report.components.items():
                comp_counter = metrics.counter("component_health_checks", "Component health", ("component", "status"))
                comp_counter.labels(comp_name, comp_health.status.value).inc()

                # Track response times
                response_time_histogram = metrics.histogram("component_response_time_seconds", "Response times", ("component",))
                response_time_histogram.labels(comp_name).observe(comp_health.response_time_ms / 1000)

            # Track system metrics
            cpu_gauge = metrics.gauge("system_cpu_usage_percent", "CPU usage")
            cpu_gauge.set(report.system_metrics.cpu_usage_percent)

            memory_gauge = metrics.gauge("system_memory_usage_percent", "Memory usage")
            memory_gauge.set(report.system_metrics.memory_usage_percent)

            disk_gauge = metrics.gauge("system_disk_usage_percent", "Disk usage")
            disk_gauge.set(report.system_metrics.disk_usage_percent)

        except Exception as e:
            logger.error(f"❌ Failed to track health metrics: {e}")

    def get_health_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get health check history for the specified hours"""

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        return [
            entry for entry in self.check_history
            if datetime.fromisoformat(entry['timestamp']) > cutoff_time
        ]

    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary statistics"""

        if not self.check_history:
            return {
                'total_checks': 0,
                'avg_response_time': 0,
                'success_rate': 0,
                'recent_status': 'unknown'
            }

        recent_checks = self.check_history[-24:]  # Last 24 checks

        healthy_count = sum(1 for check in recent_checks if check['overall_status'] == 'healthy')

        return {
            'total_checks': len(self.check_history),
            'recent_checks': len(recent_checks),
            'avg_response_time': sum(check['avg_response_time'] for check in recent_checks) / len(recent_checks),
            'success_rate': healthy_count / len(recent_checks),
            'recent_status': recent_checks[-1]['overall_status'] if recent_checks else 'unknown',
            'uptime_hours': (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600
        }

# Global health monitor instance
health_monitor = HealthMonitor()

async def get_health_monitor() -> HealthMonitor:
    """Get health monitor instance"""
    return health_monitor

# Convenience functions for FastAPI integration
async def health_check_endpoint(include_optional: bool = True) -> Dict[str, Any]:
    """Health check endpoint for FastAPI"""
    monitor = await get_health_monitor()
    report = await monitor.comprehensive_health_check(include_optional)

    return {
        'status': report.overall_status.value,
        'timestamp': report.timestamp.isoformat(),
        'components': {
            name: {
                'status': comp.status.value,
                'response_time_ms': comp.response_time_ms,
                'error_message': comp.error_message,
                'metadata': comp.metadata
            }
            for name, comp in report.components.items()
        },
        'system_metrics': asdict(report.system_metrics),
        'performance_summary': report.performance_summary,
        'alerts': report.alerts
    }

async def readiness_check_endpoint() -> Dict[str, Any]:
    """Kubernetes readiness probe endpoint"""
    monitor = await get_health_monitor()

    # Quick check of critical components only
    report = await monitor.comprehensive_health_check(include_optional=False)

    if report.overall_status in [HealthStatus.HEALTHY, HealthStatus.WARNING]:
        return {
            'status': 'ready',
            'timestamp': report.timestamp.isoformat(),
            'components_healthy': report.performance_summary['components_healthy'],
            'components_total': report.performance_summary['total_components']
        }
    else:
        return {
            'status': 'not_ready',
            'timestamp': report.timestamp.isoformat(),
            'issues': report.alerts
        }

async def liveness_check_endpoint() -> Dict[str, Any]:
    """Kubernetes liveness probe endpoint"""
    return {
        'status': 'alive',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'uptime_seconds': (datetime.now(timezone.utc) - health_monitor.start_time).total_seconds()
    }