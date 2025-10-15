"""
Health Monitoring and Checks

Comprehensive health monitoring for all system components including
databases, external services, and internal subsystems.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Callable, NamedTuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from ..core.config import settings
from .metrics import metrics
from .logging import get_logger

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    name: str
    status: HealthStatus
    message: str
    duration: float
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class HealthCheck:
    """Individual health check implementation"""
    
    def __init__(
        self, 
        name: str, 
        check_func: Callable,
        timeout: float = 30.0,
        interval: float = 60.0,
        critical: bool = True
    ):
        self.name = name
        self.check_func = check_func
        self.timeout = timeout
        self.interval = interval
        self.critical = critical
        self.last_result: Optional[HealthCheckResult] = None
        self.consecutive_failures = 0
    
    async def execute(self) -> HealthCheckResult:
        """Execute the health check"""
        start_time = time.time()
        timestamp = datetime.now(timezone.utc)
        
        try:
            if asyncio.iscoroutinefunction(self.check_func):
                result = await asyncio.wait_for(self.check_func(), timeout=self.timeout)
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self.check_func
                )
            
            duration = time.time() - start_time
            
            # Parse result
            if isinstance(result, dict):
                status = HealthStatus(result.get('status', 'healthy'))
                message = result.get('message', 'OK')
                details = result.get('details', {})
            elif isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                message = 'OK' if result else 'Check failed'
                details = {}
            else:
                status = HealthStatus.HEALTHY
                message = str(result) or 'OK'
                details = {}
            
            self.consecutive_failures = 0
            
            health_result = HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                duration=duration,
                timestamp=timestamp,
                details=details
            )
            
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            self.consecutive_failures += 1
            
            health_result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {self.timeout}s",
                duration=duration,
                timestamp=timestamp,
                error="timeout"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.consecutive_failures += 1
            
            health_result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                duration=duration,
                timestamp=timestamp,
                error=str(e)
            )
        
        self.last_result = health_result
        
        # Record metrics
        if metrics:
            healthy = health_result.status == HealthStatus.HEALTHY
            metrics.update_service_health(self.name, "health_check", healthy)
        
        return health_result


class HealthMonitor:
    """Central health monitoring system"""
    
    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        self.enabled = True
        self.overall_health_cache: Optional[Dict[str, Any]] = None
        self.cache_expiry: Optional[datetime] = None
        self.cache_duration = 30  # seconds
    
    def add_check(self, health_check: HealthCheck):
        """Add a health check"""
        self.checks[health_check.name] = health_check
        logger.info(f"Added health check: {health_check.name}")
    
    def remove_check(self, name: str):
        """Remove a health check"""
        if name in self.checks:
            del self.checks[name]
            logger.info(f"Removed health check: {name}")
    
    async def run_check(self, name: str) -> Optional[HealthCheckResult]:
        """Run a specific health check"""
        if name not in self.checks:
            return None
        
        check = self.checks[name]
        result = await check.execute()
        
        # Log result
        if result.status == HealthStatus.HEALTHY:
            logger.debug(f"Health check {name}: {result.message}")
        else:
            logger.warning(
                f"Health check {name} failed: {result.message}",
                extra={
                    "health_check": name,
                    "status": result.status.value,
                    "duration": result.duration,
                    "consecutive_failures": check.consecutive_failures
                }
            )
        
        return result
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks concurrently"""
        if not self.checks:
            return {}
        
        # Run all checks concurrently
        tasks = [self.run_check(name) for name in self.checks.keys()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        check_results = {}
        for name, result in zip(self.checks.keys(), results):
            if isinstance(result, Exception):
                check_results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check error: {str(result)}",
                    duration=0.0,
                    timestamp=datetime.now(timezone.utc),
                    error=str(result)
                )
            else:
                check_results[name] = result
        
        return check_results
    
    async def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        
        # Check cache
        now = datetime.now(timezone.utc)
        if (self.overall_health_cache and self.cache_expiry and 
            now < self.cache_expiry):
            return self.overall_health_cache
        
        # Run health checks
        check_results = await self.run_all_checks()
        
        # Calculate overall status
        overall_status = HealthStatus.HEALTHY
        critical_failures = 0
        total_failures = 0
        
        for check_name, result in check_results.items():
            check = self.checks[check_name]
            
            if result.status != HealthStatus.HEALTHY:
                total_failures += 1
                if check.critical:
                    critical_failures += 1
        
        # Determine overall status
        if critical_failures > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif total_failures > 0:
            overall_status = HealthStatus.DEGRADED
        
        # Get system info if available
        system_info = {}
        if PSUTIL_AVAILABLE:
            try:
                system_info = {
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_usage": {
                        path: psutil.disk_usage(path).percent 
                        for path in ['/'] if psutil.disk_usage(path)
                    },
                    "load_average": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None,
                    "uptime": time.time() - psutil.boot_time()
                }
            except Exception as e:
                logger.warning(f"Failed to get system info: {e}")
        
        health_data = {
            "status": overall_status.value,
            "timestamp": now.isoformat(),
            "service": settings.otel_service_name,
            "version": "1.0.0",
            "uptime": time.time(),  # Could be more accurate
            "checks": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "duration": result.duration,
                    "timestamp": result.timestamp.isoformat(),
                    "critical": self.checks[name].critical,
                    "consecutive_failures": self.checks[name].consecutive_failures
                }
                for name, result in check_results.items()
            },
            "summary": {
                "total_checks": len(check_results),
                "healthy_checks": sum(1 for r in check_results.values() if r.status == HealthStatus.HEALTHY),
                "critical_failures": critical_failures,
                "total_failures": total_failures
            },
            "system": system_info
        }
        
        # Cache result
        self.overall_health_cache = health_data
        self.cache_expiry = now + timedelta(seconds=self.cache_duration)
        
        return health_data
    
    async def start_monitoring(self, interval: float = 60.0):
        """Start background health monitoring"""
        if self.monitoring_task and not self.monitoring_task.done():
            logger.warning("Health monitoring already running")
            return
        
        self.monitoring_task = asyncio.create_task(self._monitoring_loop(interval))
        logger.info(f"Health monitoring started with {interval}s interval")
    
    async def stop_monitoring(self):
        """Stop background health monitoring"""
        if self.monitoring_task and not self.monitoring_task.done():
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Health monitoring stopped")
    
    async def _monitoring_loop(self, interval: float):
        """Background monitoring loop"""
        while self.enabled:
            try:
                await self.run_all_checks()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring loop error: {e}")
                await asyncio.sleep(min(interval, 60))


# Global health monitor instance
health_monitor = HealthMonitor()


# Standard health check functions

async def database_health_check() -> Dict[str, Any]:
    """Check database connectivity"""
    try:
        from ..db.session import get_session
        
        with next(get_session()) as session:
            # Simple query to test connection
            result = session.execute("SELECT 1").fetchone()
            
        return {
            "status": "healthy",
            "message": "Database connection OK",
            "details": {"result": result[0] if result else None}
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "message": f"Database connection failed: {str(e)}"
        }


async def redis_health_check() -> Dict[str, Any]:
    """Check Redis connectivity"""
    try:
        from ..cache.redis_cache import get_redis_client
        
        redis_client = await get_redis_client()
        if redis_client:
            await redis_client.ping()
            info = await redis_client.info()
            
            return {
                "status": "healthy",
                "message": "Redis connection OK",
                "details": {
                    "version": info.get("redis_version"),
                    "uptime": info.get("uptime_in_seconds"),
                    "connected_clients": info.get("connected_clients")
                }
            }
        else:
            return {
                "status": "degraded",
                "message": "Redis not configured"
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Redis connection failed: {str(e)}"
        }


async def neo4j_health_check() -> Dict[str, Any]:
    """Check Neo4j connectivity"""
    try:
        from ..graph import neo4j_client
        
        health = neo4j_client.get_health()
        
        if health.is_healthy:
            return {
                "status": "healthy",
                "message": "Neo4j connection OK",
                "details": {
                    "response_time": health.response_time,
                    "server_info": health.server_info
                }
            }
        else:
            return {
                "status": "unhealthy",
                "message": f"Neo4j unhealthy: {health.error_message}"
            }
    except Exception as e:
        return {
            "status": "degraded",
            "message": f"Neo4j check failed: {str(e)}"
        }


async def kafka_health_check() -> Dict[str, Any]:
    """Check Kafka connectivity"""
    try:
        from ..events import kafka_client
        
        if kafka_client.is_healthy():
            health = kafka_client.get_health()
            return {
                "status": "healthy", 
                "message": "Kafka connection OK",
                "details": {
                    "brokers": len(health.brokers) if health.brokers else 0,
                    "response_time": health.response_time
                }
            }
        else:
            return {
                "status": "unhealthy",
                "message": "Kafka connection unhealthy"
            }
    except Exception as e:
        return {
            "status": "degraded",
            "message": f"Kafka check failed: {str(e)}"
        }


def init_health_checks():
    """Initialize standard health checks"""
    
    # Database check
    health_monitor.add_check(HealthCheck(
        name="database",
        check_func=database_health_check,
        timeout=10.0,
        critical=True
    ))
    
    # Redis check
    health_monitor.add_check(HealthCheck(
        name="redis",
        check_func=redis_health_check,
        timeout=5.0,
        critical=False
    ))
    
    # Neo4j check
    health_monitor.add_check(HealthCheck(
        name="neo4j",
        check_func=neo4j_health_check,
        timeout=10.0,
        critical=False
    ))
    
    # Kafka check
    health_monitor.add_check(HealthCheck(
        name="kafka",
        check_func=kafka_health_check,
        timeout=10.0,
        critical=False
    ))
    
    # Octopart dependency check (only if API key configured)
    import os
    if os.getenv("OCTOPART_API_KEY"):
        async def octopart_health_check():
            try:
                import httpx
                endpoint = os.getenv("OCTOPART_ENDPOINT", "https://api.octopart.com/v4/graphql")
                payload = {"query": "query{__typename}"}
                headers = {"Content-Type": "application/json", "token": os.getenv("OCTOPART_API_KEY")}
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.post(endpoint, json=payload, headers=headers)
                if resp.status_code == 200:
                    return {"status": "healthy", "message": "Octopart GraphQL reachable"}
                return {"status": "degraded", "message": f"HTTP {resp.status_code}"}
            except Exception as e:
                return {"status": "degraded", "message": f"Octopart check failed: {e}"}

        health_monitor.add_check(HealthCheck(
            name="octopart",
            check_func=octopart_health_check,
            timeout=7.0,
            critical=False
        ))
    
    logger.info("Standard health checks initialized")
