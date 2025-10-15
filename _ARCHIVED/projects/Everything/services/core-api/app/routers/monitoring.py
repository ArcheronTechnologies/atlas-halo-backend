"""
Monitoring and Scalability API Endpoints

This module provides comprehensive monitoring, performance analysis, and 
auto-scaling endpoints for the SCIP platform.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import logging

from ..monitoring.performance import performance_monitor, get_performance_report
from ..monitoring.distributed import distributed_monitor, get_cluster_status, get_cluster_health
from ..scaling.auto_scaler import auto_scaler, run_scaling_evaluation, execute_auto_scaling
from ..cache.redis_cache import cache
from ..db.optimization import db_optimizer
from ..db.session import get_session
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)
router = APIRouter()


# Performance Monitoring Endpoints

@router.get("/health")
async def get_system_health():
    """Get current system health status"""
    try:
        return performance_monitor.get_system_health()
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve system health")


@router.get("/performance/report")
async def get_comprehensive_performance_report():
    """Get comprehensive performance report"""
    try:
        return get_performance_report()
    except Exception as e:
        logger.error(f"Error generating performance report: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate performance report")


@router.get("/performance/trends")
async def get_performance_trends(hours: int = Query(24, ge=1, le=168)):
    """Get performance trends over specified time period"""
    try:
        return performance_monitor.get_performance_trends(hours=hours)
    except Exception as e:
        logger.error(f"Error getting performance trends: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve performance trends")


@router.get("/performance/endpoints")
async def get_endpoint_statistics(hours: int = Query(24, ge=1, le=168)):
    """Get statistics for all API endpoints"""
    try:
        stats = performance_monitor.get_endpoint_statistics(hours=hours)
        return [stat.__dict__ for stat in stats]
    except Exception as e:
        logger.error(f"Error getting endpoint statistics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve endpoint statistics")


@router.get("/alerts")
async def get_alerts(
    severity: Optional[str] = Query(None, regex="^(warning|error|critical)$"),
    limit: int = Query(50, ge=1, le=1000)
):
    """Get recent alerts filtered by severity"""
    try:
        return performance_monitor.get_alerts(severity=severity, limit=limit)
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve alerts")


# Distributed Monitoring Endpoints

@router.get("/cluster/overview")
async def get_cluster_overview():
    """Get comprehensive cluster overview"""
    try:
        return await get_cluster_status()
    except Exception as e:
        logger.error(f"Error getting cluster overview: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cluster overview")


@router.get("/cluster/health")
async def get_cluster_health_status():
    """Get cluster health summary"""
    try:
        return await get_cluster_health()
    except Exception as e:
        logger.error(f"Error getting cluster health: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cluster health")


@router.get("/cluster/nodes")
async def get_cluster_nodes():
    """Get information about all cluster nodes"""
    try:
        nodes = await distributed_monitor.discovery.discover_nodes()
        return [node.__dict__ for node in nodes]
    except Exception as e:
        logger.error(f"Error getting cluster nodes: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve cluster nodes")


@router.get("/cluster/nodes/{node_id}")
async def get_node_details(node_id: str):
    """Get detailed information about a specific node"""
    try:
        node_details = await distributed_monitor.get_node_details(node_id)
        if not node_details:
            raise HTTPException(status_code=404, detail="Node not found")
        return node_details
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting node details: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve node details")


# Auto-Scaling Endpoints

@router.get("/scaling/evaluation")
async def evaluate_scaling_needs():
    """Evaluate current scaling needs and get recommendations"""
    try:
        return await run_scaling_evaluation()
    except Exception as e:
        logger.error(f"Error evaluating scaling needs: {e}")
        raise HTTPException(status_code=500, detail="Failed to evaluate scaling needs")


@router.get("/scaling/recommendations")
async def get_scaling_recommendations():
    """Get detailed scaling recommendations with predictive analysis"""
    try:
        return await auto_scaler.get_scaling_recommendations()
    except Exception as e:
        logger.error(f"Error getting scaling recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to get scaling recommendations")


@router.post("/scaling/execute")
async def execute_scaling_decisions(background_tasks: BackgroundTasks):
    """Execute automatic scaling based on current metrics"""
    try:
        # Run scaling in background
        background_tasks.add_task(_background_scaling_task)
        
        return {
            "status": "initiated",
            "message": "Auto-scaling evaluation and execution started",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error initiating auto-scaling: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate auto-scaling")


async def _background_scaling_task():
    """Background task for scaling execution"""
    try:
        result = await execute_auto_scaling()
        logger.info(f"Auto-scaling completed: {result}")
    except Exception as e:
        logger.error(f"Error in background scaling task: {e}")


@router.get("/scaling/history")
async def get_scaling_history(limit: int = Query(50, ge=1, le=500)):
    """Get history of scaling decisions and actions"""
    try:
        # Get recent scaling history from auto-scaler
        history = auto_scaler.scaling_history[-limit:] if hasattr(auto_scaler, 'scaling_history') else []
        
        return {
            "scaling_history": [
                {
                    "timestamp": item["timestamp"].isoformat(),
                    "decision": item["decision"].__dict__
                }
                for item in history
            ],
            "total_count": len(history)
        }
    except Exception as e:
        logger.error(f"Error getting scaling history: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve scaling history")


# Database Optimization Endpoints

@router.get("/database/performance")
async def get_database_performance(session: Session = Depends(get_session)):
    """Get database performance report"""
    try:
        report = await db_optimizer.get_performance_report(session)
        return report
    except Exception as e:
        logger.error(f"Error getting database performance: {e}")
        raise HTTPException(status_code=500, detail="Failed to get database performance report")


@router.get("/database/optimization/suggestions")
async def get_database_optimization_suggestions(session: Session = Depends(get_session)):
    """Get database optimization suggestions"""
    try:
        suggestions = await db_optimizer.suggest_indexes(session)
        optimization_needs = await db_optimizer.optimize_queries(session)
        
        return {
            "index_suggestions": suggestions,
            "optimization_needs": optimization_needs,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting optimization suggestions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get optimization suggestions")


@router.post("/database/maintenance")
async def run_database_maintenance(
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session)
):
    """Run database maintenance tasks"""
    try:
        background_tasks.add_task(_run_db_maintenance, session)
        
        return {
            "status": "initiated",
            "message": "Database maintenance tasks started",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error initiating database maintenance: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate database maintenance")


async def _run_db_maintenance(session: Session):
    """Background task for database maintenance"""
    try:
        result = await db_optimizer.run_maintenance_tasks(session)
        logger.info(f"Database maintenance completed: {result}")
    except Exception as e:
        logger.error(f"Error in database maintenance task: {e}")


@router.get("/database/slow-queries")
async def get_slow_queries(
    limit: int = Query(20, ge=1, le=100),
    session: Session = Depends(get_session)
):
    """Get recent slow queries for analysis"""
    try:
        from ..db.optimization import analyze_slow_queries
        slow_queries = await analyze_slow_queries(session, limit=limit)
        return {
            "slow_queries": slow_queries,
            "count": len(slow_queries),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting slow queries: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve slow queries")


# Cache Management Endpoints

@router.get("/cache/stats")
async def get_cache_statistics():
    """Get cache performance statistics"""
    try:
        stats = await cache.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cache statistics")


@router.post("/cache/invalidate/{category}")
async def invalidate_cache_category(category: str):
    """Invalidate all keys in a cache category"""
    try:
        deleted_count = await cache.invalidate_category(category)
        return {
            "status": "completed",
            "category": category,
            "keys_deleted": deleted_count,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error invalidating cache category {category}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to invalidate cache category {category}")


@router.post("/cache/clear")
async def clear_all_cache():
    """Clear all cache data (use with caution)"""
    try:
        # This would clear all cache categories
        categories = ['components', 'suppliers', 'rfqs', 'pricing', 'intelligence', 'api_responses']
        total_deleted = 0
        
        for category in categories:
            deleted = await cache.invalidate_category(category)
            total_deleted += deleted
        
        return {
            "status": "completed", 
            "total_keys_deleted": total_deleted,
            "categories_cleared": categories,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")


# System Optimization Endpoints

@router.post("/optimize/all")
async def run_comprehensive_optimization(
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_session)
):
    """Run comprehensive system optimization (database, cache, scaling)"""
    try:
        background_tasks.add_task(_comprehensive_optimization_task, session)
        
        return {
            "status": "initiated",
            "message": "Comprehensive system optimization started",
            "components": ["database", "cache", "scaling"],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error initiating comprehensive optimization: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate system optimization")


async def _comprehensive_optimization_task(session: Session):
    """Background task for comprehensive optimization"""
    try:
        results = {
            "database_maintenance": None,
            "cache_optimization": None,
            "scaling_evaluation": None,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Run database maintenance
        try:
            db_result = await db_optimizer.run_maintenance_tasks(session)
            results["database_maintenance"] = db_result
            logger.info("Database maintenance completed in optimization task")
        except Exception as e:
            logger.error(f"Database maintenance failed in optimization: {e}")
            results["database_maintenance"] = {"error": str(e)}
        
        # Cache optimization (clear expired keys, optimize memory)
        try:
            cache_stats = await cache.get_stats()
            results["cache_optimization"] = {
                "status": "analyzed",
                "stats": cache_stats
            }
            logger.info("Cache optimization completed")
        except Exception as e:
            logger.error(f"Cache optimization failed: {e}")
            results["cache_optimization"] = {"error": str(e)}
        
        # Scaling evaluation
        try:
            scaling_result = await run_scaling_evaluation()
            results["scaling_evaluation"] = scaling_result
            logger.info("Scaling evaluation completed")
        except Exception as e:
            logger.error(f"Scaling evaluation failed: {e}")
            results["scaling_evaluation"] = {"error": str(e)}
        
        logger.info(f"Comprehensive optimization completed: {results}")
        
    except Exception as e:
        logger.error(f"Error in comprehensive optimization task: {e}")


# Load Balancer and Service Discovery Endpoints

@router.get("/load-balancer/stats")
async def get_load_balancer_stats():
    """Get load balancer statistics for all services"""
    try:
        from ..scaling.load_balancer import service_registry
        return await service_registry.get_all_service_stats()
    except Exception as e:
        logger.error(f"Error getting load balancer stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get load balancer statistics")


@router.post("/load-balancer/register")
async def register_service_instance(
    service_name: str,
    host: str,
    port: int,
    weight: int = 100,
    region: str = "default"
):
    """Register a new service instance"""
    try:
        from ..scaling.load_balancer import service_registry, ServiceInstance
        
        instance = ServiceInstance(
            id=f"{service_name}-{host}-{port}",
            host=host,
            port=port,
            weight=weight,
            region=region
        )
        
        await service_registry.register_service_instance(service_name, instance)
        
        return {
            "status": "registered",
            "service": service_name,
            "instance_id": instance.id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error registering service instance: {e}")
        raise HTTPException(status_code=500, detail="Failed to register service instance")


@router.get("/cache/advanced/stats")
async def get_advanced_cache_stats():
    """Get advanced multi-tier cache statistics"""
    try:
        from ..cache.advanced_cache import cache_coordinator
        return await cache_coordinator.get_all_stats()
    except Exception as e:
        logger.error(f"Error getting advanced cache stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get advanced cache statistics")


@router.post("/cache/warm")
async def warm_cache(background_tasks: BackgroundTasks):
    """Trigger cache warming process"""
    try:
        from ..cache.advanced_cache import advanced_cache
        background_tasks.add_task(advanced_cache.warm_cache)
        
        return {
            "status": "initiated",
            "message": "Cache warming started",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error initiating cache warming: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate cache warming")


# Load Testing Endpoints

@router.post("/load-test/smoke")
async def run_smoke_test(background_tasks: BackgroundTasks, base_url: str = "http://localhost:8000"):
    """Run smoke test"""
    try:
        from ..testing.load_test import load_tester, create_smoke_test_config
        
        config = create_smoke_test_config(base_url)
        background_tasks.add_task(_run_load_test_task, config)
        
        return {
            "status": "initiated",
            "test_type": "smoke_test",
            "duration": f"{config.duration_seconds}s",
            "concurrent_users": config.concurrent_users,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error initiating smoke test: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate smoke test")


@router.post("/load-test/performance")
async def run_performance_test(background_tasks: BackgroundTasks, base_url: str = "http://localhost:8000"):
    """Run performance load test"""
    try:
        from ..testing.load_test import load_tester, create_load_test_config
        
        config = create_load_test_config(base_url)
        background_tasks.add_task(_run_load_test_task, config)
        
        return {
            "status": "initiated",
            "test_type": "load_test",
            "duration": f"{config.duration_seconds}s",
            "concurrent_users": config.concurrent_users,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error initiating load test: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate load test")


@router.get("/benchmark/database")
async def benchmark_database():
    """Run database performance benchmark"""
    try:
        from ..testing.load_test import PerformanceBenchmark
        results = await PerformanceBenchmark.benchmark_database_operations()
        return results
    except Exception as e:
        logger.error(f"Error running database benchmark: {e}")
        raise HTTPException(status_code=500, detail="Failed to run database benchmark")


@router.get("/benchmark/cache")
async def benchmark_cache():
    """Run cache performance benchmark"""
    try:
        from ..testing.load_test import PerformanceBenchmark
        results = await PerformanceBenchmark.benchmark_cache_operations()
        return results
    except Exception as e:
        logger.error(f"Error running cache benchmark: {e}")
        raise HTTPException(status_code=500, detail="Failed to run cache benchmark")


async def _run_load_test_task(config):
    """Background task for running load tests"""
    try:
        from ..testing.load_test import load_tester
        results = await load_tester.run_load_test(config)
        logger.info(f"Load test completed: {config.name} - Success rate: {results.success_rate:.1f}%")
    except Exception as e:
        logger.error(f"Load test task failed: {e}")


# Real-time Monitoring Endpoints

@router.get("/realtime/metrics")
async def get_realtime_metrics():
    """Get real-time system metrics"""
    try:
        health = performance_monitor.get_system_health()
        cluster_health = await get_cluster_health()
        cache_stats = await cache.get_stats()
        
        return {
            "system": health,
            "cluster": cluster_health,
            "cache": cache_stats,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting real-time metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get real-time metrics")


@router.get("/dashboard")
async def get_monitoring_dashboard():
    """Get comprehensive monitoring dashboard data"""
    try:
        # Gather all dashboard data in parallel
        import asyncio
        
        dashboard_tasks = [
            get_cluster_health(),
            performance_monitor.get_system_health(),
            cache.get_stats(),
            auto_scaler.get_scaling_recommendations()
        ]
        
        cluster_health, system_health, cache_stats, scaling_recs = await asyncio.gather(
            *dashboard_tasks, return_exceptions=True
        )
        
        # Handle any exceptions
        dashboard_data = {
            "cluster_health": cluster_health if not isinstance(cluster_health, Exception) else {"error": str(cluster_health)},
            "system_health": system_health if not isinstance(system_health, Exception) else {"error": str(system_health)},
            "cache_stats": cache_stats if not isinstance(cache_stats, Exception) else {"error": str(cache_stats)},
            "scaling_recommendations": scaling_recs if not isinstance(scaling_recs, Exception) else {"error": str(scaling_recs)},
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Error getting monitoring dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to get monitoring dashboard data")