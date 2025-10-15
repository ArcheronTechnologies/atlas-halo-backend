"""
Health and Metrics Router

Endpoints for health checks, metrics, and system status monitoring.
"""

from fastapi import APIRouter, Response, HTTPException
from fastapi.responses import PlainTextResponse
import time
from typing import Dict, Any

from ..observability import health_monitor, metrics, get_logger
from ..observability.metrics import get_metrics_data

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Get overall system health status"""
    try:
        health_data = await health_monitor.get_overall_health()
        
        # Set appropriate HTTP status based on health
        status_code = 200
        if health_data["status"] == "unhealthy":
            status_code = 503
        elif health_data["status"] == "degraded":
            status_code = 200  # Still accepting traffic but degraded
        
        return Response(
            content=health_data,
            status_code=status_code,
            media_type="application/json"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Health check system failure"
        )


@router.get("/health/live", response_model=Dict[str, str])
async def liveness_probe():
    """Kubernetes liveness probe - basic service availability"""
    return {
        "status": "alive",
        "timestamp": str(time.time())
    }


@router.get("/health/ready", response_model=Dict[str, Any])
async def readiness_probe():
    """Kubernetes readiness probe - service ready to accept traffic"""
    try:
        health_data = await health_monitor.get_overall_health()
        
        # Check if critical services are healthy
        critical_healthy = all(
            check["status"] == "healthy" 
            for name, check in health_data.get("checks", {}).items()
            if health_monitor.checks.get(name, {}).critical
        )
        
        if critical_healthy:
            return {
                "status": "ready",
                "timestamp": str(time.time()),
                "critical_services": "healthy"
            }
        else:
            raise HTTPException(
                status_code=503,
                detail="Critical services not ready"
            )
            
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Service not ready"
        )


@router.get("/health/startup", response_model=Dict[str, str])  
async def startup_probe():
    """Kubernetes startup probe - service has started successfully"""
    # Simple startup check - could be enhanced with more comprehensive checks
    return {
        "status": "started",
        "timestamp": str(time.time())
    }


@router.get("/health/{service_name}")
async def service_health_check(service_name: str):
    """Get health status for a specific service"""
    try:
        result = await health_monitor.run_check(service_name)
        
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Health check '{service_name}' not found"
            )
        
        status_code = 200 if result.status.value == "healthy" else 503
        
        return Response(
            content={
                "name": result.name,
                "status": result.status.value,
                "message": result.message,
                "duration": result.duration,
                "timestamp": result.timestamp.isoformat(),
                "details": result.details
            },
            status_code=status_code,
            media_type="application/json"
        )
        
    except Exception as e:
        logger.error(f"Service health check failed for {service_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )


@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    try:
        metrics_data = get_metrics_data()
        
        if not metrics_data:
            raise HTTPException(
                status_code=503,
                detail="Metrics collection not available"
            )
        
        return Response(
            content=metrics_data,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
        
    except Exception as e:
        logger.error(f"Metrics endpoint failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Metrics collection failed"
        )


@router.get("/status", response_model=Dict[str, Any])
async def system_status():
    """Comprehensive system status including health, metrics, and configuration"""
    try:
        # Get health data
        health_data = await health_monitor.get_overall_health()
        
        # Get basic metrics if available
        metrics_summary = {}
        if metrics and metrics.enabled:
            try:
                metrics_summary = {
                    "enabled": True,
                    "port": metrics.config.port,
                    "namespace": metrics.config.namespace
                }
            except Exception:
                metrics_summary = {"enabled": False, "error": "Failed to get metrics info"}
        else:
            metrics_summary = {"enabled": False}
        
        # System status
        status_data = {
            "service": {
                "name": health_data.get("service"),
                "version": health_data.get("version"),
                "status": health_data.get("status"),
                "uptime": health_data.get("uptime")
            },
            "health": {
                "overall_status": health_data.get("status"),
                "total_checks": health_data.get("summary", {}).get("total_checks", 0),
                "healthy_checks": health_data.get("summary", {}).get("healthy_checks", 0),
                "failed_checks": health_data.get("summary", {}).get("total_failures", 0)
            },
            "metrics": metrics_summary,
            "system": health_data.get("system", {}),
            "timestamp": health_data.get("timestamp")
        }
        
        return status_data
        
    except Exception as e:
        logger.error(f"System status failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="System status check failed"
        )