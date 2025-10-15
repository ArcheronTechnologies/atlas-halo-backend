"""
Distributed Monitoring and Cluster Management

This module provides distributed monitoring capabilities across multiple instances,
load balancers, and data centers for comprehensive system visibility.
"""

import asyncio
import json
import logging
import socket
import uuid
from typing import Dict, List, Any, Optional, Set
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib
import time

from ..cache.redis_cache import cache
from ..monitoring.performance import performance_monitor, SystemMetrics

logger = logging.getLogger(__name__)


@dataclass
class NodeInfo:
    """Information about a cluster node"""
    node_id: str
    hostname: str
    ip_address: str
    port: int
    node_type: str  # api, worker, database, cache
    region: str
    availability_zone: str
    instance_size: str
    started_at: datetime
    last_heartbeat: datetime
    status: str  # healthy, degraded, unhealthy, offline
    version: str
    
    
@dataclass  
class ClusterMetrics:
    """Cluster-wide aggregated metrics"""
    total_nodes: int
    healthy_nodes: int
    degraded_nodes: int
    unhealthy_nodes: int
    total_cpu_percent: float
    total_memory_percent: float
    total_request_rate: float
    average_response_time: float
    cluster_error_rate: float
    data_centers: List[str]
    regions: List[str]
    timestamp: datetime


class NodeDiscovery:
    """Service discovery and node registration"""
    
    def __init__(self, redis_cache=None):
        self.cache = redis_cache or cache
        self.node_id = self._generate_node_id()
        self.node_info = self._get_node_info()
        self.discovery_key = "cluster:nodes"
        self.heartbeat_interval = 30  # seconds
        self._heartbeat_task = None
        
    def _generate_node_id(self) -> str:
        """Generate unique node ID"""
        hostname = socket.gethostname()
        timestamp = int(time.time())
        unique_id = uuid.uuid4().hex[:8]
        return f"{hostname}-{timestamp}-{unique_id}"
    
    def _get_node_info(self) -> NodeInfo:
        """Get current node information"""
        hostname = socket.gethostname()
        
        # Get local IP address
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip_address = s.getsockname()[0]
            s.close()
        except:
            ip_address = "127.0.0.1"
        
        return NodeInfo(
            node_id=self.node_id,
            hostname=hostname,
            ip_address=ip_address,
            port=8000,  # Default FastAPI port
            node_type="api",  # Could be detected from environment
            region="us-east-1",  # Could be detected from cloud metadata
            availability_zone="us-east-1a",  # Could be detected
            instance_size="t3.medium",  # Could be detected
            started_at=datetime.now(timezone.utc),
            last_heartbeat=datetime.now(timezone.utc),
            status="healthy",
            version="1.0.0"
        )
    
    async def register_node(self) -> bool:
        """Register this node with the cluster"""
        try:
            node_data = asdict(self.node_info)
            # Convert datetime objects to ISO strings
            for key, value in node_data.items():
                if isinstance(value, datetime):
                    node_data[key] = value.isoformat()
            
            await self.cache.set(
                'cluster_nodes',
                self.node_id,
                node_data,
                ttl=120  # 2 minutes TTL
            )
            
            logger.info(f"Registered node {self.node_id} with cluster")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register node: {e}")
            return False
    
    async def start_heartbeat(self):
        """Start sending heartbeats to indicate node health"""
        if self._heartbeat_task and not self._heartbeat_task.done():
            return  # Already running
        
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
    
    async def _heartbeat_loop(self):
        """Continuous heartbeat loop"""
        while True:
            try:
                await self._send_heartbeat()
                await asyncio.sleep(self.heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _send_heartbeat(self):
        """Send heartbeat with current health status"""
        try:
            # Update node status based on current health
            health = performance_monitor.get_system_health()
            
            if health['status'] == 'healthy':
                self.node_info.status = 'healthy'
            elif health['status'] == 'degraded':
                self.node_info.status = 'degraded'
            else:
                self.node_info.status = 'unhealthy'
            
            self.node_info.last_heartbeat = datetime.now(timezone.utc)
            
            # Include current metrics in heartbeat
            heartbeat_data = {
                'node_id': self.node_id,
                'status': self.node_info.status,
                'timestamp': self.node_info.last_heartbeat.isoformat(),
                'metrics': health
            }
            
            await self.cache.set(
                'cluster_heartbeats',
                self.node_id,
                heartbeat_data,
                ttl=90  # 90 seconds TTL
            )
            
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")
    
    async def discover_nodes(self) -> List[NodeInfo]:
        """Discover all nodes in the cluster"""
        try:
            # Get all registered nodes
            node_keys = await self.cache.get('cluster_nodes', '*')
            if not node_keys:
                return []
            
            nodes = []
            for node_id, node_data in node_keys.items():
                try:
                    # Parse datetime strings back to datetime objects
                    if isinstance(node_data.get('started_at'), str):
                        node_data['started_at'] = datetime.fromisoformat(node_data['started_at'])
                    if isinstance(node_data.get('last_heartbeat'), str):
                        node_data['last_heartbeat'] = datetime.fromisoformat(node_data['last_heartbeat'])
                    
                    node_info = NodeInfo(**node_data)
                    
                    # Check if node is still alive based on heartbeat
                    cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=2)
                    if node_info.last_heartbeat < cutoff_time:
                        node_info.status = 'offline'
                    
                    nodes.append(node_info)
                    
                except Exception as e:
                    logger.error(f"Error parsing node data for {node_id}: {e}")
                    continue
            
            return nodes
            
        except Exception as e:
            logger.error(f"Error discovering nodes: {e}")
            return []
    
    async def stop_heartbeat(self):
        """Stop heartbeat task"""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass


class DistributedMonitor:
    """Distributed monitoring coordinator"""
    
    def __init__(self):
        self.discovery = NodeDiscovery()
        self.cluster_metrics_history = []
        self.alert_thresholds = {
            'min_healthy_nodes': 2,
            'max_cluster_error_rate': 5.0,
            'max_average_response_time': 3.0,
            'max_cluster_cpu': 80.0,
            'max_cluster_memory': 85.0
        }
    
    async def start_monitoring(self):
        """Start distributed monitoring"""
        try:
            # Register this node
            await self.discovery.register_node()
            
            # Start heartbeat
            await self.discovery.start_heartbeat()
            
            logger.info("Distributed monitoring started")
            
        except Exception as e:
            logger.error(f"Error starting distributed monitoring: {e}")
    
    async def get_cluster_overview(self) -> Dict[str, Any]:
        """Get comprehensive cluster overview"""
        try:
            nodes = await self.discovery.discover_nodes()
            cluster_metrics = await self._calculate_cluster_metrics(nodes)
            
            # Categorize nodes by type and status
            nodes_by_type = defaultdict(list)
            nodes_by_status = defaultdict(list)
            nodes_by_region = defaultdict(list)
            
            for node in nodes:
                nodes_by_type[node.node_type].append(node)
                nodes_by_status[node.status].append(node)
                nodes_by_region[node.region].append(node)
            
            # Get load balancer status (would be implemented based on LB type)
            load_balancer_status = await self._get_load_balancer_status()
            
            return {
                'cluster_metrics': asdict(cluster_metrics),
                'total_nodes': len(nodes),
                'nodes_by_type': {k: len(v) for k, v in nodes_by_type.items()},
                'nodes_by_status': {k: len(v) for k, v in nodes_by_status.items()},
                'nodes_by_region': {k: len(v) for k, v in nodes_by_region.items()},
                'load_balancer': load_balancer_status,
                'alerts': await self._check_cluster_alerts(cluster_metrics),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting cluster overview: {e}")
            return {'error': str(e)}
    
    async def _calculate_cluster_metrics(self, nodes: List[NodeInfo]) -> ClusterMetrics:
        """Calculate aggregated cluster metrics"""
        if not nodes:
            return ClusterMetrics(
                total_nodes=0,
                healthy_nodes=0,
                degraded_nodes=0,
                unhealthy_nodes=0,
                total_cpu_percent=0.0,
                total_memory_percent=0.0,
                total_request_rate=0.0,
                average_response_time=0.0,
                cluster_error_rate=0.0,
                data_centers=[],
                regions=[],
                timestamp=datetime.now(timezone.utc)
            )
        
        # Count nodes by status
        healthy_count = sum(1 for n in nodes if n.status == 'healthy')
        degraded_count = sum(1 for n in nodes if n.status == 'degraded')
        unhealthy_count = sum(1 for n in nodes if n.status == 'unhealthy')
        
        # Collect unique regions and AZs
        regions = list(set(n.region for n in nodes))
        data_centers = list(set(n.availability_zone for n in nodes))
        
        # Get aggregated performance metrics
        total_cpu = 0.0
        total_memory = 0.0
        total_requests = 0.0
        total_response_time = 0.0
        total_errors = 0.0
        active_nodes = 0
        
        for node in nodes:
            if node.status in ['healthy', 'degraded']:
                try:
                    # Get node-specific metrics from heartbeat
                    heartbeat = await cache.get('cluster_heartbeats', node.node_id)
                    if heartbeat and 'metrics' in heartbeat:
                        metrics = heartbeat['metrics']
                        total_cpu += metrics.get('cpu_percent', 0.0)
                        total_memory += metrics.get('memory_percent', 0.0)
                        active_nodes += 1
                        
                        # Get endpoint stats for this node (would be more sophisticated)
                        # For now, use global stats
                        endpoint_stats = performance_monitor.get_endpoint_statistics(hours=1)
                        if endpoint_stats:
                            total_requests += sum(stat.total_requests for stat in endpoint_stats)
                            total_response_time += sum(stat.avg_duration for stat in endpoint_stats)
                            total_errors += sum(stat.error_count for stat in endpoint_stats)
                        
                except Exception as e:
                    logger.error(f"Error getting metrics for node {node.node_id}: {e}")
        
        # Calculate averages
        avg_cpu = total_cpu / max(active_nodes, 1)
        avg_memory = total_memory / max(active_nodes, 1)
        avg_response_time = total_response_time / max(len(performance_monitor.get_endpoint_statistics(hours=1)) * active_nodes, 1)
        cluster_error_rate = (total_errors / max(total_requests, 1)) * 100
        
        return ClusterMetrics(
            total_nodes=len(nodes),
            healthy_nodes=healthy_count,
            degraded_nodes=degraded_count,
            unhealthy_nodes=unhealthy_count,
            total_cpu_percent=avg_cpu,
            total_memory_percent=avg_memory,
            total_request_rate=total_requests / 3600,  # Requests per second
            average_response_time=avg_response_time,
            cluster_error_rate=cluster_error_rate,
            data_centers=data_centers,
            regions=regions,
            timestamp=datetime.now(timezone.utc)
        )
    
    async def _get_load_balancer_status(self) -> Dict[str, Any]:
        """Get load balancer health status"""
        # This would integrate with actual load balancer APIs (AWS ALB, nginx, etc.)
        return {
            'status': 'healthy',
            'active_targets': 3,
            'healthy_targets': 3,
            'unhealthy_targets': 0,
            'requests_per_second': 45.2,
            'average_target_response_time': 0.235
        }
    
    async def _check_cluster_alerts(self, metrics: ClusterMetrics) -> List[Dict[str, Any]]:
        """Check cluster-wide alert conditions"""
        alerts = []
        
        if metrics.healthy_nodes < self.alert_thresholds['min_healthy_nodes']:
            alerts.append({
                'type': 'cluster_availability',
                'severity': 'critical',
                'message': f'Only {metrics.healthy_nodes} healthy nodes available (minimum: {self.alert_thresholds["min_healthy_nodes"]})',
                'value': metrics.healthy_nodes,
                'threshold': self.alert_thresholds['min_healthy_nodes']
            })
        
        if metrics.cluster_error_rate > self.alert_thresholds['max_cluster_error_rate']:
            alerts.append({
                'type': 'cluster_error_rate',
                'severity': 'error',
                'message': f'High cluster error rate: {metrics.cluster_error_rate:.1f}%',
                'value': metrics.cluster_error_rate,
                'threshold': self.alert_thresholds['max_cluster_error_rate']
            })
        
        if metrics.average_response_time > self.alert_thresholds['max_average_response_time']:
            alerts.append({
                'type': 'cluster_performance',
                'severity': 'warning',
                'message': f'High cluster average response time: {metrics.average_response_time:.2f}s',
                'value': metrics.average_response_time,
                'threshold': self.alert_thresholds['max_average_response_time']
            })
        
        if metrics.total_cpu_percent > self.alert_thresholds['max_cluster_cpu']:
            alerts.append({
                'type': 'cluster_cpu',
                'severity': 'warning',
                'message': f'High cluster CPU usage: {metrics.total_cpu_percent:.1f}%',
                'value': metrics.total_cpu_percent,
                'threshold': self.alert_thresholds['max_cluster_cpu']
            })
        
        # Add timestamp to all alerts
        for alert in alerts:
            alert['timestamp'] = datetime.now(timezone.utc).isoformat()
        
        return alerts
    
    async def get_node_details(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific node"""
        try:
            nodes = await self.discovery.discover_nodes()
            target_node = next((n for n in nodes if n.node_id == node_id), None)
            
            if not target_node:
                return None
            
            # Get latest heartbeat data
            heartbeat = await cache.get('cluster_heartbeats', node_id)
            
            return {
                'node_info': asdict(target_node),
                'latest_metrics': heartbeat.get('metrics', {}) if heartbeat else {},
                'heartbeat_timestamp': heartbeat.get('timestamp') if heartbeat else None,
                'uptime_seconds': (datetime.now(timezone.utc) - target_node.started_at).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Error getting node details for {node_id}: {e}")
            return None
    
    async def get_cluster_health_summary(self) -> Dict[str, Any]:
        """Get high-level cluster health summary"""
        try:
            overview = await self.get_cluster_overview()
            
            # Calculate health score
            total_nodes = overview['total_nodes']
            healthy_nodes = overview['nodes_by_status'].get('healthy', 0)
            
            if total_nodes == 0:
                health_score = 0.0
            else:
                # Base score on node health
                node_health_score = (healthy_nodes / total_nodes) * 100
                
                # Adjust based on performance metrics
                metrics = overview.get('cluster_metrics', {})
                cpu_penalty = max(0, metrics.get('total_cpu_percent', 0) - 70) * 0.5
                memory_penalty = max(0, metrics.get('total_memory_percent', 0) - 80) * 0.3
                response_time_penalty = max(0, (metrics.get('average_response_time', 0) - 1.0) * 20)
                
                health_score = max(0, node_health_score - cpu_penalty - memory_penalty - response_time_penalty)
            
            status = 'healthy'
            if health_score < 50:
                status = 'critical'
            elif health_score < 80:
                status = 'degraded'
            elif len(overview.get('alerts', [])) > 0:
                status = 'warning'
            
            return {
                'status': status,
                'health_score': round(health_score, 1),
                'total_nodes': total_nodes,
                'healthy_nodes': healthy_nodes,
                'active_alerts': len(overview.get('alerts', [])),
                'regions': len(overview.get('cluster_metrics', {}).get('regions', [])),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting cluster health summary: {e}")
            return {
                'status': 'unknown',
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def stop_monitoring(self):
        """Stop distributed monitoring"""
        await self.discovery.stop_heartbeat()
        logger.info("Distributed monitoring stopped")


# Global distributed monitor instance
distributed_monitor = DistributedMonitor()


async def get_cluster_status() -> Dict[str, Any]:
    """Get complete cluster status"""
    return await distributed_monitor.get_cluster_overview()


async def get_cluster_health() -> Dict[str, Any]:
    """Get cluster health summary"""
    return await distributed_monitor.get_cluster_health_summary()


async def start_distributed_monitoring():
    """Initialize and start distributed monitoring"""
    await distributed_monitor.start_monitoring()


async def stop_distributed_monitoring():
    """Stop distributed monitoring"""
    await distributed_monitor.stop_monitoring()