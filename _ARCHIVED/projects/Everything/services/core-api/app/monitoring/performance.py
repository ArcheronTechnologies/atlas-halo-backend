"""
Production Performance Monitoring and Optimization

This module provides comprehensive performance monitoring, metrics collection,
and automatic optimization for the SCIP platform.
"""

import asyncio
import time
import logging
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import json
from functools import wraps
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """Individual request metrics"""
    path: str
    method: str
    status_code: int
    duration: float
    timestamp: datetime
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_size: int = 0
    response_size: int = 0


@dataclass
class SystemMetrics:
    """System-level metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    disk_usage_percent: float
    active_connections: int
    timestamp: datetime


@dataclass
class APIEndpointStats:
    """Statistics for API endpoints"""
    path: str
    method: str
    total_requests: int
    success_rate: float
    avg_duration: float
    p95_duration: float
    p99_duration: float
    error_count: int
    last_24h_requests: int


class PerformanceMonitor:
    """Comprehensive performance monitoring system"""
    
    def __init__(self):
        self.metrics_buffer = deque(maxlen=10000)  # Last 10k requests
        self.system_metrics = deque(maxlen=1440)    # 24 hours of minute-level data
        self.endpoint_stats = defaultdict(lambda: {
            'requests': deque(maxlen=1000),
            'durations': deque(maxlen=1000),
            'errors': deque(maxlen=100)
        })
        
        # Performance thresholds
        self.thresholds = {
            'slow_request': 2.0,        # 2 seconds
            'very_slow_request': 5.0,   # 5 seconds
            'high_memory': 80.0,        # 80% memory usage
            'high_cpu': 80.0,           # 80% CPU usage
            'low_success_rate': 95.0,   # Below 95% success rate
        }
        
        # Alerts and notifications
        self.alerts = deque(maxlen=1000)
        self.alert_handlers = []
        
        # Start background tasks
        self._monitoring_active = True
        self._start_background_monitoring()
    
    def _start_background_monitoring(self):
        """Start background system monitoring"""
        def monitor_system():
            while self._monitoring_active:
                try:
                    self._collect_system_metrics()
                    self._analyze_performance()
                    time.sleep(60)  # Collect every minute
                except Exception as e:
                    logger.error(f"Error in system monitoring: {e}")
                    time.sleep(60)
        
        # Start in background thread
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
    
    def _collect_system_metrics(self):
        """Collect system-level performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network connections (approximate active connections)
            connections = len(psutil.net_connections(kind='tcp'))
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_gb=memory.used / (1024**3),
                disk_usage_percent=disk.percent,
                active_connections=connections,
                timestamp=datetime.now(timezone.utc)
            )
            
            self.system_metrics.append(metrics)
            
            # Check for system-level alerts
            self._check_system_alerts(metrics)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _check_system_alerts(self, metrics: SystemMetrics):
        """Check system metrics against thresholds and generate alerts"""
        alerts = []
        
        if metrics.cpu_percent > self.thresholds['high_cpu']:
            alerts.append({
                'type': 'high_cpu',
                'severity': 'warning',
                'message': f'High CPU usage: {metrics.cpu_percent:.1f}%',
                'value': metrics.cpu_percent,
                'threshold': self.thresholds['high_cpu']
            })
        
        if metrics.memory_percent > self.thresholds['high_memory']:
            alerts.append({
                'type': 'high_memory',
                'severity': 'warning',
                'message': f'High memory usage: {metrics.memory_percent:.1f}%',
                'value': metrics.memory_percent,
                'threshold': self.thresholds['high_memory']
            })
        
        for alert in alerts:
            alert['timestamp'] = datetime.now(timezone.utc)
            self.alerts.append(alert)
            self._notify_alert(alert)
    
    def _analyze_performance(self):
        """Analyze performance trends and generate insights"""
        try:
            # Analyze endpoint performance
            for endpoint_key, stats in self.endpoint_stats.items():
                if len(stats['requests']) < 10:  # Need minimum requests for analysis
                    continue
                
                recent_requests = list(stats['requests'])[-100:]  # Last 100 requests
                recent_durations = [r.duration for r in recent_requests]
                
                if recent_durations:
                    avg_duration = sum(recent_durations) / len(recent_durations)
                    error_count = sum(1 for r in recent_requests if r.status_code >= 400)
                    success_rate = (len(recent_requests) - error_count) / len(recent_requests) * 100
                    
                    # Check for performance degradation
                    if avg_duration > self.thresholds['slow_request']:
                        alert = {
                            'type': 'slow_endpoint',
                            'severity': 'warning',
                            'message': f'Slow endpoint {endpoint_key}: {avg_duration:.2f}s average',
                            'endpoint': endpoint_key,
                            'value': avg_duration,
                            'threshold': self.thresholds['slow_request'],
                            'timestamp': datetime.now(timezone.utc)
                        }
                        self.alerts.append(alert)
                        self._notify_alert(alert)
                    
                    if success_rate < self.thresholds['low_success_rate']:
                        alert = {
                            'type': 'low_success_rate',
                            'severity': 'error',
                            'message': f'Low success rate {endpoint_key}: {success_rate:.1f}%',
                            'endpoint': endpoint_key,
                            'value': success_rate,
                            'threshold': self.thresholds['low_success_rate'],
                            'timestamp': datetime.now(timezone.utc)
                        }
                        self.alerts.append(alert)
                        self._notify_alert(alert)
            
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")
    
    def _notify_alert(self, alert: Dict[str, Any]):
        """Notify registered alert handlers"""
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
    
    def add_alert_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """Add alert notification handler"""
        self.alert_handlers.append(handler)
    
    def record_request(self, metrics: RequestMetrics):
        """Record request metrics"""
        self.metrics_buffer.append(metrics)
        
        # Update endpoint-specific stats
        endpoint_key = f"{metrics.method} {metrics.path}"
        stats = self.endpoint_stats[endpoint_key]
        stats['requests'].append(metrics)
        stats['durations'].append(metrics.duration)
        
        if metrics.status_code >= 400:
            stats['errors'].append(metrics)
    
    def get_endpoint_statistics(self, hours: int = 24) -> List[APIEndpointStats]:
        """Get statistics for all endpoints"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        endpoint_stats = []
        
        for endpoint_key, stats in self.endpoint_stats.items():
            if not stats['requests']:
                continue
            
            method, path = endpoint_key.split(' ', 1)
            
            # Filter requests within time window
            recent_requests = [r for r in stats['requests'] if r.timestamp >= cutoff_time]
            if not recent_requests:
                continue
            
            durations = [r.duration for r in recent_requests]
            errors = [r for r in recent_requests if r.status_code >= 400]
            
            # Calculate percentiles
            durations.sort()
            p95_idx = int(len(durations) * 0.95)
            p99_idx = int(len(durations) * 0.99)
            
            endpoint_stat = APIEndpointStats(
                path=path,
                method=method,
                total_requests=len(recent_requests),
                success_rate=(len(recent_requests) - len(errors)) / len(recent_requests) * 100,
                avg_duration=sum(durations) / len(durations),
                p95_duration=durations[p95_idx] if durations else 0.0,
                p99_duration=durations[p99_idx] if durations else 0.0,
                error_count=len(errors),
                last_24h_requests=len(recent_requests)
            )
            
            endpoint_stats.append(endpoint_stat)
        
        return sorted(endpoint_stats, key=lambda x: x.total_requests, reverse=True)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status"""
        if not self.system_metrics:
            return {'status': 'unknown', 'message': 'No metrics available'}
        
        latest_metrics = self.system_metrics[-1]
        health_status = 'healthy'
        issues = []
        
        if latest_metrics.cpu_percent > self.thresholds['high_cpu']:
            health_status = 'degraded'
            issues.append(f'High CPU usage: {latest_metrics.cpu_percent:.1f}%')
        
        if latest_metrics.memory_percent > self.thresholds['high_memory']:
            health_status = 'degraded'
            issues.append(f'High memory usage: {latest_metrics.memory_percent:.1f}%')
        
        # Check recent alerts
        recent_alerts = [a for a in self.alerts if 
                        a['timestamp'] > datetime.now(timezone.utc) - timedelta(minutes=5)]
        
        if any(a['severity'] == 'error' for a in recent_alerts):
            health_status = 'unhealthy'
        
        return {
            'status': health_status,
            'cpu_percent': latest_metrics.cpu_percent,
            'memory_percent': latest_metrics.memory_percent,
            'memory_used_gb': latest_metrics.memory_used_gb,
            'disk_usage_percent': latest_metrics.disk_usage_percent,
            'active_connections': latest_metrics.active_connections,
            'issues': issues,
            'recent_alerts': len(recent_alerts),
            'timestamp': latest_metrics.timestamp.isoformat()
        }
    
    def get_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance trends over time"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        # System trends
        recent_system_metrics = [m for m in self.system_metrics if m.timestamp >= cutoff_time]
        
        system_trends = {}
        if recent_system_metrics:
            system_trends = {
                'cpu_trend': {
                    'current': recent_system_metrics[-1].cpu_percent,
                    'average': sum(m.cpu_percent for m in recent_system_metrics) / len(recent_system_metrics),
                    'peak': max(m.cpu_percent for m in recent_system_metrics)
                },
                'memory_trend': {
                    'current': recent_system_metrics[-1].memory_percent,
                    'average': sum(m.memory_percent for m in recent_system_metrics) / len(recent_system_metrics),
                    'peak': max(m.memory_percent for m in recent_system_metrics)
                }
            }
        
        # Request trends
        recent_requests = [r for r in self.metrics_buffer if r.timestamp >= cutoff_time]
        
        request_trends = {}
        if recent_requests:
            durations = [r.duration for r in recent_requests]
            error_count = sum(1 for r in recent_requests if r.status_code >= 400)
            
            request_trends = {
                'total_requests': len(recent_requests),
                'requests_per_hour': len(recent_requests) / hours,
                'average_duration': sum(durations) / len(durations),
                'error_rate': error_count / len(recent_requests) * 100,
                'success_rate': (len(recent_requests) - error_count) / len(recent_requests) * 100
            }
        
        return {
            'time_window_hours': hours,
            'system_trends': system_trends,
            'request_trends': request_trends,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def get_alerts(self, severity: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        alerts = list(self.alerts)
        
        if severity:
            alerts = [a for a in alerts if a.get('severity') == severity]
        
        # Sort by timestamp (newest first) and limit
        alerts.sort(key=lambda x: x['timestamp'], reverse=True)
        return alerts[:limit]
    
    def shutdown(self):
        """Shutdown monitoring"""
        self._monitoring_active = False


class PerformanceMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for performance monitoring"""
    
    def __init__(self, app, monitor: PerformanceMonitor):
        super().__init__(app)
        self.monitor = monitor
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Get request size
        request_size = 0
        if hasattr(request, 'body'):
            try:
                body = await request.body()
                request_size = len(body)
            except:
                pass
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Get response size
        response_size = 0
        if hasattr(response, 'body'):
            try:
                response_size = len(response.body)
            except:
                pass
        
        # Extract user info
        user_id = None
        if hasattr(request.state, 'user'):
            user_id = getattr(request.state.user, 'id', None)
        
        # Record metrics
        metrics = RequestMetrics(
            path=request.url.path,
            method=request.method,
            status_code=response.status_code,
            duration=duration,
            timestamp=datetime.now(timezone.utc),
            user_id=user_id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get('user-agent'),
            request_size=request_size,
            response_size=response_size
        )
        
        self.monitor.record_request(metrics)
        
        # Add performance headers
        response.headers['X-Response-Time'] = str(round(duration * 1000, 2))
        response.headers['X-Request-ID'] = request.headers.get('X-Request-ID', 'unknown')
        
        return response


def performance_timer(category: str = "general"):
    """Decorator to time function execution"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"Performance [{category}] {func.__name__}: {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Performance [{category}] {func.__name__} FAILED: {duration:.3f}s - {e}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"Performance [{category}] {func.__name__}: {duration:.3f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Performance [{category}] {func.__name__} FAILED: {duration:.3f}s - {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


class PerformanceOptimizer:
    """Automatic performance optimization"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.optimization_history = []
    
    async def analyze_and_optimize(self) -> Dict[str, Any]:
        """Analyze performance and suggest optimizations"""
        optimizations = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'suggestions': [],
            'automatic_actions': [],
            'performance_score': 0.0
        }
        
        # Analyze endpoint performance
        endpoint_stats = self.monitor.get_endpoint_statistics(hours=1)
        
        for stat in endpoint_stats:
            if stat.avg_duration > 2.0:  # Slow endpoint
                optimizations['suggestions'].append({
                    'type': 'slow_endpoint',
                    'endpoint': f"{stat.method} {stat.path}",
                    'issue': f'Average response time: {stat.avg_duration:.2f}s',
                    'recommendation': 'Add caching, optimize database queries, or implement pagination'
                })
            
            if stat.success_rate < 95.0:  # Low success rate
                optimizations['suggestions'].append({
                    'type': 'reliability',
                    'endpoint': f"{stat.method} {stat.path}",
                    'issue': f'Success rate: {stat.success_rate:.1f}%',
                    'recommendation': 'Add error handling, input validation, and retry logic'
                })
        
        # Analyze system performance
        health = self.monitor.get_system_health()
        
        if health['cpu_percent'] > 80:
            optimizations['suggestions'].append({
                'type': 'system_resource',
                'issue': f'High CPU usage: {health["cpu_percent"]:.1f}%',
                'recommendation': 'Scale horizontally or optimize CPU-intensive operations'
            })
        
        if health['memory_percent'] > 80:
            optimizations['suggestions'].append({
                'type': 'system_resource',
                'issue': f'High memory usage: {health["memory_percent"]:.1f}%',
                'recommendation': 'Optimize memory usage, implement object pooling, or scale vertically'
            })
        
        # Calculate performance score
        avg_response_time = sum(stat.avg_duration for stat in endpoint_stats) / len(endpoint_stats) if endpoint_stats else 0
        avg_success_rate = sum(stat.success_rate for stat in endpoint_stats) / len(endpoint_stats) if endpoint_stats else 100
        
        performance_score = min(100, 
            (100 - min(avg_response_time * 10, 50)) *  # Response time component
            (avg_success_rate / 100) *                # Success rate component  
            (1 - health['cpu_percent'] / 200) *       # CPU component
            (1 - health['memory_percent'] / 200)      # Memory component
        )
        
        optimizations['performance_score'] = max(0, performance_score)
        
        return optimizations


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


# Alert handlers
def log_alert_handler(alert: Dict[str, Any]):
    """Log alerts to application logs"""
    level = logging.ERROR if alert['severity'] == 'error' else logging.WARNING
    logger.log(level, f"Performance Alert: {alert['message']}")


def slack_alert_handler(alert: Dict[str, Any]):
    """Send alerts to Slack (implementation would depend on webhook setup)"""
    if alert['severity'] == 'error':
        # Would send to Slack webhook
        logger.info(f"Would send Slack alert: {alert['message']}")


# Register default alert handlers
performance_monitor.add_alert_handler(log_alert_handler)
# performance_monitor.add_alert_handler(slack_alert_handler)  # Enable when configured


def get_performance_report() -> Dict[str, Any]:
    """Get comprehensive performance report"""
    return {
        'system_health': performance_monitor.get_system_health(),
        'endpoint_stats': [asdict(stat) for stat in performance_monitor.get_endpoint_statistics()],
        'trends': performance_monitor.get_performance_trends(),
        'recent_alerts': performance_monitor.get_alerts(limit=20),
        'generated_at': datetime.now(timezone.utc).isoformat()
    }


# Cleanup function
def shutdown_performance_monitoring():
    """Shutdown performance monitoring"""
    performance_monitor.shutdown()