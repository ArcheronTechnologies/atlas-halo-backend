"""
Admin Monitoring Dashboard API
Provides real-time system metrics and status via REST API
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import psutil
import asyncio

from ..database.postgis_database import get_database

router = APIRouter(prefix="/admin/monitoring", tags=["admin", "monitoring"])


# =============================================================================
# SYSTEM STATUS
# =============================================================================

@router.get("/status")
async def get_system_status():
    """
    Get overall system status and health
    """
    db = await get_database()

    # Database status
    try:
        db_test = await db.execute_query("SELECT 1")
        database_healthy = True
        database_latency_ms = 0  # Would measure actual latency
    except Exception as e:
        database_healthy = False
        database_latency_ms = None

    # Redis status (for WebSocket)
    redis_healthy = True  # Would check actual Redis connection

    # System resources
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    return {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "healthy" if (database_healthy and redis_healthy) else "degraded",
        "components": {
            "database": {
                "healthy": database_healthy,
                "latency_ms": database_latency_ms
            },
            "redis": {
                "healthy": redis_healthy
            },
            "websocket": {
                "healthy": redis_healthy  # WebSocket depends on Redis
            }
        },
        "resources": {
            "cpu": {
                "usage_percent": cpu_percent,
                "status": "healthy" if cpu_percent < 80 else "warning"
            },
            "memory": {
                "usage_percent": memory.percent,
                "used_gb": memory.used / (1024**3),
                "total_gb": memory.total / (1024**3),
                "status": "healthy" if memory.percent < 85 else "warning"
            },
            "disk": {
                "usage_percent": disk.percent,
                "used_gb": disk.used / (1024**3),
                "total_gb": disk.total / (1024**3),
                "status": "healthy" if disk.percent < 80 else "warning"
            }
        }
    }


# =============================================================================
# DATA INGESTION METRICS
# =============================================================================

@router.get("/ingestion/stats")
async def get_ingestion_stats(hours: int = 24):
    """
    Get data ingestion statistics for the last N hours
    """
    db = await get_database()

    cutoff_time = datetime.now() - timedelta(hours=hours)

    # Total incidents collected
    query_total = """
    SELECT COUNT(*) as total
    FROM crime_incidents
    WHERE created_at >= $1
    """
    result_total = await db.execute_query_single(query_total, cutoff_time)

    # Incidents by source
    query_by_source = """
    SELECT source, COUNT(*) as count
    FROM crime_incidents
    WHERE created_at >= $1
    GROUP BY source
    ORDER BY count DESC
    """
    result_by_source = await db.execute_query(query_by_source, cutoff_time)

    # Incidents by city
    query_by_city = """
    SELECT metadata->>'city' as city, COUNT(*) as count
    FROM crime_incidents
    WHERE created_at >= $1
    AND metadata->>'city' IS NOT NULL
    GROUP BY metadata->>'city'
    ORDER BY count DESC
    LIMIT 10
    """
    result_by_city = await db.execute_query(query_by_city, cutoff_time)

    # Average quality score
    query_quality = """
    SELECT AVG(confidence_score) as avg_quality
    FROM crime_incidents
    WHERE created_at >= $1
    AND confidence_score IS NOT NULL
    """
    result_quality = await db.execute_query_single(query_quality, cutoff_time)

    return {
        "time_range_hours": hours,
        "total_incidents": result_total['total'] if result_total else 0,
        "by_source": [{"source": r['source'], "count": r['count']} for r in result_by_source],
        "by_city": [{"city": r['city'], "count": r['count']} for r in result_by_city],
        "average_quality_score": float(result_quality['avg_quality']) if result_quality and result_quality['avg_quality'] else None
    }


# =============================================================================
# DATABASE METRICS
# =============================================================================

@router.get("/database/stats")
async def get_database_stats():
    """
    Get database statistics and performance metrics
    """
    db = await get_database()

    # Table sizes
    query_sizes = """
    SELECT
        schemaname,
        tablename,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
        pg_total_relation_size(schemaname||'.'||tablename) AS size_bytes
    FROM pg_tables
    WHERE schemaname = 'public'
    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
    LIMIT 10
    """
    result_sizes = await db.execute_query(query_sizes)

    # Total database size
    query_db_size = """
    SELECT pg_size_pretty(pg_database_size(current_database())) as size
    """
    result_db_size = await db.execute_query_single(query_db_size)

    # Connection stats
    query_connections = """
    SELECT
        count(*) FILTER (WHERE state = 'active') as active,
        count(*) FILTER (WHERE state = 'idle') as idle,
        count(*) as total
    FROM pg_stat_activity
    WHERE datname = current_database()
    """
    result_connections = await db.execute_query_single(query_connections)

    # Index hit ratio
    query_cache = """
    SELECT
        sum(heap_blks_read) as heap_read,
        sum(heap_blks_hit) as heap_hit,
        sum(heap_blks_hit) / NULLIF(sum(heap_blks_hit) + sum(heap_blks_read), 0) * 100 AS cache_hit_ratio
    FROM pg_statio_user_tables
    """
    result_cache = await db.execute_query_single(query_cache)

    return {
        "database_size": result_db_size['size'] if result_db_size else "Unknown",
        "tables": [
            {
                "name": r['tablename'],
                "size": r['size'],
                "size_bytes": r['size_bytes']
            }
            for r in result_sizes
        ],
        "connections": {
            "active": result_connections['active'] if result_connections else 0,
            "idle": result_connections['idle'] if result_connections else 0,
            "total": result_connections['total'] if result_connections else 0,
            "max": 20  # From pool configuration
        },
        "cache_hit_ratio": float(result_cache['cache_hit_ratio']) if result_cache and result_cache['cache_hit_ratio'] else None
    }


# =============================================================================
# RECENT ACTIVITY
# =============================================================================

@router.get("/activity/recent")
async def get_recent_activity(limit: int = 50):
    """
    Get recent incidents and system activity
    """
    db = await get_database()

    query = """
    SELECT
        id,
        incident_type,
        latitude,
        longitude,
        severity,
        source,
        confidence_score,
        occurred_at,
        created_at,
        metadata->>'city' as city
    FROM crime_incidents
    ORDER BY created_at DESC
    LIMIT $1
    """

    results = await db.execute_query(query, limit)

    return {
        "count": len(results),
        "incidents": [
            {
                "id": str(r['id']),
                "type": r['incident_type'],
                "location": {
                    "latitude": float(r['latitude']),
                    "longitude": float(r['longitude']),
                    "city": r['city']
                },
                "severity": r['severity'],
                "source": r['source'],
                "quality_score": float(r['confidence_score']) if r['confidence_score'] else None,
                "occurred_at": r['occurred_at'].isoformat() if r['occurred_at'] else None,
                "created_at": r['created_at'].isoformat()
            }
            for r in results
        ]
    }


# =============================================================================
# QUALITY METRICS
# =============================================================================

@router.get("/quality/summary")
async def get_quality_summary(hours: int = 24):
    """
    Get data quality summary for the last N hours
    """
    db = await get_database()

    cutoff_time = datetime.now() - timedelta(hours=hours)

    # Quality score distribution
    query_distribution = """
    SELECT
        CASE
            WHEN confidence_score >= 0.9 THEN 'excellent'
            WHEN confidence_score >= 0.7 THEN 'good'
            WHEN confidence_score >= 0.5 THEN 'fair'
            ELSE 'poor'
        END as quality_tier,
        COUNT(*) as count
    FROM crime_incidents
    WHERE created_at >= $1
    AND confidence_score IS NOT NULL
    GROUP BY quality_tier
    ORDER BY quality_tier
    """
    result_distribution = await db.execute_query(query_distribution, cutoff_time)

    # Average quality by source
    query_by_source = """
    SELECT
        source,
        AVG(confidence_score) as avg_quality,
        COUNT(*) as count
    FROM crime_incidents
    WHERE created_at >= $1
    AND confidence_score IS NOT NULL
    GROUP BY source
    ORDER BY avg_quality DESC
    """
    result_by_source = await db.execute_query(query_by_source, cutoff_time)

    return {
        "time_range_hours": hours,
        "quality_distribution": [
            {"tier": r['quality_tier'], "count": r['count']}
            for r in result_distribution
        ],
        "by_source": [
            {
                "source": r['source'],
                "average_quality": float(r['avg_quality']),
                "count": r['count']
            }
            for r in result_by_source
        ]
    }


# =============================================================================
# ALERTS
# =============================================================================

@router.post("/alerts/webhook")
async def receive_alert_webhook(alert_data: Dict):
    """
    Receive alerts from AlertManager and store/process them
    """
    # Store alert in database for historical tracking
    # Send notifications to admin users
    # Could trigger automated responses

    return {
        "status": "received",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/alerts/active")
async def get_active_alerts():
    """
    Get currently active system alerts
    """
    # This would query AlertManager API or local alert storage
    # For now, return empty list

    return {
        "active_alerts": []
    }


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

@router.get("/performance/summary")
async def get_performance_summary():
    """
    Get system performance summary
    """
    # This would aggregate metrics from Prometheus
    # For now, return basic system metrics

    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()

    # Disk I/O
    disk_io = psutil.disk_io_counters()

    # Network I/O
    net_io = psutil.net_io_counters()

    return {
        "timestamp": datetime.now().isoformat(),
        "cpu": {
            "usage_percent": cpu_percent,
            "count": psutil.cpu_count()
        },
        "memory": {
            "usage_percent": memory.percent,
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "total_gb": memory.total / (1024**3)
        },
        "disk_io": {
            "read_mb": disk_io.read_bytes / (1024**2),
            "write_mb": disk_io.write_bytes / (1024**2)
        },
        "network_io": {
            "sent_mb": net_io.bytes_sent / (1024**2),
            "recv_mb": net_io.bytes_recv / (1024**2)
        }
    }