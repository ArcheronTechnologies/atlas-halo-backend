"""
Atlas AI Comprehensive Health Check System
Production-ready health monitoring with detailed diagnostics and alerting
"""

import asyncio
import logging
import time
import psutil
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import json

from ..database.postgis_database import get_database
from ..caching.redis_cache import get_cache
from ..config.production_settings import get_config


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class HealthCheckType(str, Enum):
    CRITICAL = "critical"     # Service fails if this fails
    IMPORTANT = "important"   # Service degraded if this fails
    OPTIONAL = "optional"     # Nice to have, doesn't affect service


@dataclass
class HealthCheckResult:
    """Individual health check result."""
    name: str
    status: HealthStatus
    check_type: HealthCheckType
    response_time_ms: float
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    error: Optional[str] = None


@dataclass
class SystemHealth:
    """Overall system health status."""
    status: HealthStatus
    timestamp: datetime
    checks: List[HealthCheckResult]
    system_info: Dict[str, Any]
    summary: Dict[str, int]
    uptime_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "checks": [asdict(check) for check in self.checks],
            "system_info": self.system_info,
            "summary": self.summary,
            "uptime_seconds": self.uptime_seconds
        }


class HealthChecker:
    """Individual health check implementation."""
    
    def __init__(self, name: str, check_type: HealthCheckType, timeout: float = 5.0):
        self.name = name
        self.check_type = check_type
        self.timeout = timeout
        self.logger = logging.getLogger(f"health.{name}")
    
    async def check(self) -> HealthCheckResult:
        """Execute health check with timeout."""
        start_time = time.time()
        
        try:
            # Run check with timeout
            result = await asyncio.wait_for(
                self._execute_check(),
                timeout=self.timeout
            )
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=result.get("status", HealthStatus.UNKNOWN),
                check_type=self.check_type,
                response_time_ms=response_time,
                message=result.get("message", "Check completed"),
                details=result.get("details", {}),
                timestamp=datetime.now()
            )
            
        except asyncio.TimeoutError:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                check_type=self.check_type,
                response_time_ms=response_time,
                message=f"Check timed out after {self.timeout}s",
                details={},
                timestamp=datetime.now(),
                error="timeout"
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                check_type=self.check_type,
                response_time_ms=response_time,
                message=f"Check failed: {str(e)}",
                details={},
                timestamp=datetime.now(),
                error=str(e)
            )
    
    async def _execute_check(self) -> Dict[str, Any]:
        """Override this method in subclasses."""
        raise NotImplementedError


class DatabaseHealthChecker(HealthChecker):
    """Database connectivity and performance health check."""
    
    def __init__(self):
        super().__init__("database", HealthCheckType.CRITICAL, timeout=10.0)
    
    async def _execute_check(self) -> Dict[str, Any]:
        try:
            db = await get_database()
            
            # Test basic connectivity
            start_time = time.time()
            result = await db.execute_query("SELECT 1 as test, NOW() as server_time")
            query_time = (time.time() - start_time) * 1000
            
            if not result:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "message": "Database query returned no results",
                    "details": {"query_time_ms": query_time}
                }
            
            server_time = result[0]["server_time"]
            
            # Test PostGIS extension
            postgis_result = await db.execute_query("SELECT PostGIS_Version() as version")
            postgis_version = postgis_result[0]["version"] if postgis_result else "unknown"
            
            # Check connection pool status
            pool_info = {
                "size": db.pool.get_size(),
                "checked_in": db.pool.get_idle_size(),
                "checked_out": db.pool.get_size() - db.pool.get_idle_size(),
                "max_size": db.pool.get_max_size()
            }
            
            # Performance check - slow query warning
            status = HealthStatus.HEALTHY
            message = "Database is healthy"
            
            if query_time > 1000:  # 1 second
                status = HealthStatus.DEGRADED
                message = f"Database responding slowly ({query_time:.1f}ms)"
            elif query_time > 5000:  # 5 seconds
                status = HealthStatus.UNHEALTHY
                message = f"Database very slow ({query_time:.1f}ms)"
            
            return {
                "status": status,
                "message": message,
                "details": {
                    "query_time_ms": query_time,
                    "server_time": server_time.isoformat(),
                    "postgis_version": postgis_version,
                    "pool": pool_info
                }
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Database connection failed: {str(e)}",
                "details": {"error": str(e)}
            }


class CacheHealthChecker(HealthChecker):
    """Redis cache health check."""
    
    def __init__(self):
        super().__init__("cache", HealthCheckType.IMPORTANT, timeout=5.0)
    
    async def _execute_check(self) -> Dict[str, Any]:
        try:
            cache = await get_cache()
            
            # Test set/get operations
            test_key = "health_check_test"
            test_value = f"test_{int(time.time())}"
            
            start_time = time.time()
            
            # Set operation
            await cache.set(test_key, test_value, ttl=10)
            
            # Get operation
            retrieved_value = await cache.get(test_key)
            
            # Delete operation
            await cache.delete(test_key)
            
            operation_time = (time.time() - start_time) * 1000
            
            if retrieved_value != test_value:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "message": "Cache set/get operation failed",
                    "details": {
                        "expected": test_value,
                        "actual": retrieved_value,
                        "operation_time_ms": operation_time
                    }
                }
            
            # Check memory usage if available
            try:
                info = await cache.info()
                memory_usage = info.get('used_memory_human', 'unknown')
                memory_peak = info.get('used_memory_peak_human', 'unknown')
            except:
                memory_usage = memory_peak = 'unknown'
            
            status = HealthStatus.HEALTHY
            message = "Cache is healthy"
            
            if operation_time > 100:  # 100ms
                status = HealthStatus.DEGRADED
                message = f"Cache responding slowly ({operation_time:.1f}ms)"
            
            return {
                "status": status,
                "message": message,
                "details": {
                    "operation_time_ms": operation_time,
                    "memory_usage": memory_usage,
                    "memory_peak": memory_peak
                }
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Cache operation failed: {str(e)}",
                "details": {"error": str(e)}
            }


class SystemResourcesHealthChecker(HealthChecker):
    """System resources (CPU, memory, disk) health check."""
    
    def __init__(self):
        super().__init__("system_resources", HealthCheckType.IMPORTANT, timeout=3.0)
    
    async def _execute_check(self) -> Dict[str, Any]:
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_mb = memory.available / 1024 / 1024
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / 1024 / 1024 / 1024
            
            # Network I/O
            net_io = psutil.net_io_counters()
            
            # Process information
            process = psutil.Process(os.getpid())
            process_memory_mb = process.memory_info().rss / 1024 / 1024
            process_cpu_percent = process.cpu_percent()
            
            # Determine status based on thresholds
            status = HealthStatus.HEALTHY
            warnings = []
            
            if cpu_percent > 90:
                status = HealthStatus.UNHEALTHY
                warnings.append(f"High CPU usage: {cpu_percent:.1f}%")
            elif cpu_percent > 70:
                status = HealthStatus.DEGRADED
                warnings.append(f"Elevated CPU usage: {cpu_percent:.1f}%")
            
            if memory_percent > 95:
                status = HealthStatus.UNHEALTHY
                warnings.append(f"Critical memory usage: {memory_percent:.1f}%")
            elif memory_percent > 80:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                warnings.append(f"High memory usage: {memory_percent:.1f}%")
            
            if disk_percent > 95:
                status = HealthStatus.UNHEALTHY
                warnings.append(f"Critical disk usage: {disk_percent:.1f}%")
            elif disk_percent > 85:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.DEGRADED
                warnings.append(f"High disk usage: {disk_percent:.1f}%")
            
            message = "System resources are healthy"
            if warnings:
                message = "; ".join(warnings)
            
            return {
                "status": status,
                "message": message,
                "details": {
                    "cpu": {
                        "usage_percent": cpu_percent,
                        "core_count": cpu_count,
                        "process_usage_percent": process_cpu_percent
                    },
                    "memory": {
                        "usage_percent": memory_percent,
                        "available_mb": memory_available_mb,
                        "process_usage_mb": process_memory_mb
                    },
                    "disk": {
                        "usage_percent": disk_percent,
                        "free_gb": disk_free_gb
                    },
                    "network": {
                        "bytes_sent": net_io.bytes_sent,
                        "bytes_recv": net_io.bytes_recv
                    }
                }
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"System resources check failed: {str(e)}",
                "details": {"error": str(e)}
            }


class ExternalServicesHealthChecker(HealthChecker):
    """External services connectivity health check."""
    
    def __init__(self):
        super().__init__("external_services", HealthCheckType.OPTIONAL, timeout=10.0)
    
    async def _execute_check(self) -> Dict[str, Any]:
        try:
            import aiohttp
            
            services_to_check = [
                {"name": "Google DNS", "url": "https://8.8.8.8", "timeout": 3},
                {"name": "OpenStreetMap", "url": "https://www.openstreetmap.org", "timeout": 5},
            ]
            
            results = {}
            all_healthy = True
            
            async with aiohttp.ClientSession() as session:
                for service in services_to_check:
                    try:
                        start_time = time.time()
                        async with session.get(
                            service["url"], 
                            timeout=aiohttp.ClientTimeout(total=service["timeout"])
                        ) as response:
                            response_time = (time.time() - start_time) * 1000
                            
                            results[service["name"]] = {
                                "status": "healthy" if response.status < 400 else "unhealthy",
                                "response_time_ms": response_time,
                                "status_code": response.status
                            }
                            
                            if response.status >= 400:
                                all_healthy = False
                                
                    except Exception as e:
                        results[service["name"]] = {
                            "status": "unhealthy",
                            "error": str(e)
                        }
                        all_healthy = False
            
            status = HealthStatus.HEALTHY if all_healthy else HealthStatus.DEGRADED
            message = "External services accessible" if all_healthy else "Some external services unavailable"
            
            return {
                "status": status,
                "message": message,
                "details": {"services": results}
            }
            
        except ImportError:
            return {
                "status": HealthStatus.UNKNOWN,
                "message": "aiohttp not available for external service checks",
                "details": {}
            }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"External services check failed: {str(e)}",
                "details": {"error": str(e)}
            }


class HealthMonitor:
    """Main health monitoring system."""
    
    def __init__(self):
        self.checkers: List[HealthChecker] = [
            DatabaseHealthChecker(),
            CacheHealthChecker(),
            SystemResourcesHealthChecker(),
            ExternalServicesHealthChecker()
        ]
        self.logger = logging.getLogger(__name__)
        self.start_time = time.time()
    
    async def check_health(self, include_optional: bool = True) -> SystemHealth:
        """Perform comprehensive health check."""
        
        # Filter checkers based on include_optional
        checkers_to_run = self.checkers
        if not include_optional:
            checkers_to_run = [
                checker for checker in self.checkers 
                if checker.check_type != HealthCheckType.OPTIONAL
            ]
        
        # Run all health checks concurrently
        check_results = await asyncio.gather(
            *[checker.check() for checker in checkers_to_run],
            return_exceptions=True
        )
        
        # Convert exceptions to failed health checks
        results = []
        for i, result in enumerate(check_results):
            if isinstance(result, Exception):
                checker = checkers_to_run[i]
                results.append(HealthCheckResult(
                    name=checker.name,
                    status=HealthStatus.UNHEALTHY,
                    check_type=checker.check_type,
                    response_time_ms=0,
                    message=f"Health check exception: {str(result)}",
                    details={},
                    timestamp=datetime.now(),
                    error=str(result)
                ))
            else:
                results.append(result)
        
        # Determine overall status
        overall_status = self._determine_overall_status(results)
        
        # Generate summary
        summary = {
            "total": len(results),
            "healthy": len([r for r in results if r.status == HealthStatus.HEALTHY]),
            "degraded": len([r for r in results if r.status == HealthStatus.DEGRADED]),
            "unhealthy": len([r for r in results if r.status == HealthStatus.UNHEALTHY]),
            "unknown": len([r for r in results if r.status == HealthStatus.UNKNOWN])
        }
        
        # System information
        system_info = {
            "python_version": os.sys.version.split()[0],
            "platform": os.uname().sysname if hasattr(os, 'uname') else 'unknown',
            "process_id": os.getpid(),
            "environment": get_config().environment.value,
            "debug_mode": get_config().debug
        }
        
        uptime = time.time() - self.start_time
        
        return SystemHealth(
            status=overall_status,
            timestamp=datetime.now(),
            checks=results,
            system_info=system_info,
            summary=summary,
            uptime_seconds=uptime
        )
    
    def _determine_overall_status(self, results: List[HealthCheckResult]) -> HealthStatus:
        """Determine overall system status from individual check results."""
        
        # Check critical services first
        critical_results = [r for r in results if r.check_type == HealthCheckType.CRITICAL]
        for result in critical_results:
            if result.status == HealthStatus.UNHEALTHY:
                return HealthStatus.UNHEALTHY
        
        # Check important services
        important_results = [r for r in results if r.check_type == HealthCheckType.IMPORTANT]
        degraded_important = [r for r in important_results if r.status in [HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]]
        
        if len(degraded_important) >= len(important_results) // 2:  # Half or more important services degraded
            return HealthStatus.DEGRADED
        
        # Check if any critical services are degraded
        degraded_critical = [r for r in critical_results if r.status == HealthStatus.DEGRADED]
        if degraded_critical:
            return HealthStatus.DEGRADED
        
        # Check if any services have issues
        all_issues = [r for r in results if r.status in [HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]]
        if all_issues:
            return HealthStatus.DEGRADED
        
        return HealthStatus.HEALTHY
    
    async def quick_health_check(self) -> Dict[str, Any]:
        """Quick health check for readiness probes."""
        
        # Only check critical services for quick check
        critical_checkers = [
            checker for checker in self.checkers 
            if checker.check_type == HealthCheckType.CRITICAL
        ]
        
        results = await asyncio.gather(
            *[checker.check() for checker in critical_checkers],
            return_exceptions=True
        )
        
        # Check if any critical service failed
        for result in results:
            if isinstance(result, Exception) or result.status == HealthStatus.UNHEALTHY:
                return {
                    "status": "unhealthy",
                    "timestamp": datetime.now().isoformat()
                }
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat()
        }


# Global health monitor instance
health_monitor = HealthMonitor()


def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance."""
    return health_monitor


# Health check endpoint implementations
async def health_check_endpoint(include_optional: bool = True) -> Dict[str, Any]:
    """Main health check endpoint."""
    try:
        health = await health_monitor.check_health(include_optional)
        return health.to_dict()
    except Exception as e:
        logging.getLogger(__name__).error(f"Health check failed: {e}")
        return {
            "status": HealthStatus.UNHEALTHY.value,
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }


async def readiness_check_endpoint() -> Dict[str, Any]:
    """Kubernetes readiness probe endpoint."""
    return await health_monitor.quick_health_check()


async def liveness_check_endpoint() -> Dict[str, Any]:
    """Kubernetes liveness probe endpoint."""
    return {
        "status": "alive",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": time.time() - health_monitor.start_time
    }


# Example usage
if __name__ == "__main__":
    async def demo():
        print("Atlas AI Health Check System Demo")
        
        health = await health_monitor.check_health()
        print(f"Overall Status: {health.status.value}")
        print(f"Checks: {len(health.checks)}")
        
        for check in health.checks:
            print(f"  {check.name}: {check.status.value} ({check.response_time_ms:.1f}ms)")
    
    asyncio.run(demo())