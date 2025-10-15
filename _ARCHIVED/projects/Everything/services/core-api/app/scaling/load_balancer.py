"""
Intelligent Load Balancing and Service Discovery

This module provides advanced load balancing capabilities with health-aware routing,
circuit breakers, and automatic failover for horizontal scaling.
"""

import asyncio
import logging
import random
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import aiohttp
from collections import defaultdict, deque

from ..cache.redis_cache import cache

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_RESPONSE_TIME = "weighted_response_time"
    CONSISTENT_HASH = "consistent_hash"
    GEOGRAPHIC = "geographic"


@dataclass
class ServiceInstance:
    """Represents a service instance in the load balancer"""
    id: str
    host: str
    port: int
    weight: int = 100
    health_status: HealthStatus = HealthStatus.HEALTHY
    current_connections: int = 0
    avg_response_time: float = 0.0
    error_count: int = 0
    last_health_check: Optional[datetime] = None
    region: str = "default"
    availability_zone: str = "default"
    version: str = "1.0.0"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    instance_id: str
    status: HealthStatus
    response_time: float
    error: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def can_execute(self) -> bool:
        """Check if request can be executed"""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open"
                return True
            return False
        elif self.state == "half-open":
            return True
        return False
    
    def record_success(self):
        """Record successful execution"""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


class LoadBalancer:
    """Intelligent load balancer with multiple strategies"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.instances: Dict[str, ServiceInstance] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.strategy = LoadBalancingStrategy.WEIGHTED_RESPONSE_TIME
        self.round_robin_counter = 0
        
        # Health checking
        self.health_check_interval = 30  # seconds
        self.health_check_timeout = 5    # seconds
        self._health_check_task = None
        
        # Metrics
        self.request_history = deque(maxlen=10000)
        self.error_history = deque(maxlen=1000)
        
        # Configuration
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        
    async def register_instance(self, instance: ServiceInstance):
        """Register a new service instance"""
        self.instances[instance.id] = instance
        self.circuit_breakers[instance.id] = CircuitBreaker()
        
        # Store in cache for service discovery
        await cache.set(
            'service_instances',
            f"{self.service_name}:{instance.id}",
            asdict(instance),
            ttl=300  # 5 minutes
        )
        
        logger.info(f"Registered instance {instance.id} for service {self.service_name}")
    
    async def deregister_instance(self, instance_id: str):
        """Deregister a service instance"""
        if instance_id in self.instances:
            del self.instances[instance_id]
            del self.circuit_breakers[instance_id]
            
            # Remove from cache
            await cache.delete('service_instances', f"{self.service_name}:{instance_id}")
            
            logger.info(f"Deregistered instance {instance_id} from service {self.service_name}")
    
    async def get_healthy_instances(self) -> List[ServiceInstance]:
        """Get all healthy instances"""
        healthy = []
        for instance in self.instances.values():
            if (instance.health_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED] and
                self.circuit_breakers[instance.id].can_execute()):
                healthy.append(instance)
        return healthy
    
    async def select_instance(self, request_context: Dict[str, Any] = None) -> Optional[ServiceInstance]:
        """Select the best instance based on the current strategy"""
        healthy_instances = await self.get_healthy_instances()
        
        if not healthy_instances:
            logger.warning(f"No healthy instances available for service {self.service_name}")
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_RESPONSE_TIME:
            return self._weighted_response_time_select(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            return self._consistent_hash_select(healthy_instances, request_context)
        elif self.strategy == LoadBalancingStrategy.GEOGRAPHIC:
            return self._geographic_select(healthy_instances, request_context)
        
        # Fallback to round robin
        return self._round_robin_select(healthy_instances)
    
    def _round_robin_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Round robin selection"""
        if not instances:
            return None
        
        selected = instances[self.round_robin_counter % len(instances)]
        self.round_robin_counter += 1
        return selected
    
    def _least_connections_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance with least connections"""
        return min(instances, key=lambda x: x.current_connections)
    
    def _weighted_response_time_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance based on weighted response time and health"""
        # Calculate weights based on inverse response time and health
        weights = []
        for instance in instances:
            # Base weight from instance configuration
            base_weight = instance.weight
            
            # Adjust for response time (lower is better)
            response_time_factor = 1.0 / max(instance.avg_response_time, 0.001)
            
            # Adjust for health status
            health_factor = 1.0
            if instance.health_status == HealthStatus.DEGRADED:
                health_factor = 0.5
            
            # Adjust for current load
            connection_factor = 1.0 / max(instance.current_connections + 1, 1)
            
            final_weight = base_weight * response_time_factor * health_factor * connection_factor
            weights.append(final_weight)
        
        # Weighted random selection
        total_weight = sum(weights)
        if total_weight == 0:
            return instances[0]
        
        r = random.uniform(0, total_weight)
        cumulative = 0
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return instances[i]
        
        return instances[-1]  # Fallback
    
    def _consistent_hash_select(self, instances: List[ServiceInstance], context: Dict[str, Any]) -> ServiceInstance:
        """Consistent hashing for session affinity"""
        if not context:
            return self._round_robin_select(instances)
        
        # Create hash based on user ID, session ID, or IP
        hash_key = context.get('user_id') or context.get('session_id') or context.get('client_ip', 'default')
        hash_value = int(hashlib.md5(str(hash_key).encode()).hexdigest(), 16)
        
        # Map hash to instance
        return instances[hash_value % len(instances)]
    
    def _geographic_select(self, instances: List[ServiceInstance], context: Dict[str, Any]) -> ServiceInstance:
        """Geographic routing based on client location"""
        if not context or 'region' not in context:
            return self._weighted_response_time_select(instances)
        
        client_region = context['region']
        
        # Prefer instances in the same region
        same_region_instances = [i for i in instances if i.region == client_region]
        if same_region_instances:
            return self._weighted_response_time_select(same_region_instances)
        
        # Fallback to all instances
        return self._weighted_response_time_select(instances)
    
    async def execute_request(self, 
                            method: str, 
                            path: str, 
                            context: Dict[str, Any] = None,
                            **kwargs) -> Tuple[Any, ServiceInstance]:
        """Execute request with load balancing and retries"""
        last_error = None
        
        for attempt in range(self.max_retries):
            instance = await self.select_instance(context)
            if not instance:
                raise Exception(f"No healthy instances available for service {self.service_name}")
            
            circuit_breaker = self.circuit_breakers[instance.id]
            if not circuit_breaker.can_execute():
                continue
            
            try:
                # Track connection
                instance.current_connections += 1
                start_time = time.time()
                
                # Make HTTP request
                url = f"http://{instance.host}:{instance.port}{path}"
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                    async with session.request(method, url, **kwargs) as response:
                        response_time = time.time() - start_time
                        
                        # Update instance metrics
                        instance.avg_response_time = (instance.avg_response_time * 0.9 + response_time * 0.1)
                        instance.current_connections -= 1
                        
                        # Record success
                        circuit_breaker.record_success()
                        
                        # Log request
                        self.request_history.append({
                            'timestamp': datetime.now(timezone.utc),
                            'instance_id': instance.id,
                            'method': method,
                            'path': path,
                            'status_code': response.status,
                            'response_time': response_time
                        })
                        
                        if response.status >= 200 and response.status < 400:
                            return await response.json(), instance
                        else:
                            raise aiohttp.ClientError(f"HTTP {response.status}")
            
            except Exception as e:
                # Track error
                instance.current_connections = max(0, instance.current_connections - 1)
                instance.error_count += 1
                circuit_breaker.record_failure()
                
                # Log error
                self.error_history.append({
                    'timestamp': datetime.now(timezone.utc),
                    'instance_id': instance.id,
                    'error': str(e),
                    'attempt': attempt + 1
                })
                
                last_error = e
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
        
        raise Exception(f"All retry attempts failed. Last error: {last_error}")
    
    async def start_health_checks(self):
        """Start background health checking"""
        if self._health_check_task and not self._health_check_task.done():
            return  # Already running
        
        self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def _health_check_loop(self):
        """Background health check loop"""
        while True:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _perform_health_checks(self):
        """Perform health checks on all instances"""
        tasks = []
        for instance in self.instances.values():
            task = asyncio.create_task(self._check_instance_health(instance))
            tasks.append(task)
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, HealthCheckResult):
                    await self._update_instance_health(result)
    
    async def _check_instance_health(self, instance: ServiceInstance) -> HealthCheckResult:
        """Check health of a single instance"""
        start_time = time.time()
        
        try:
            url = f"http://{instance.host}:{instance.port}/health"
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.health_check_timeout)) as session:
                async with session.get(url) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        status = HealthStatus.HEALTHY
                    elif response.status in [503, 429]:
                        status = HealthStatus.DEGRADED
                    else:
                        status = HealthStatus.UNHEALTHY
                    
                    return HealthCheckResult(
                        instance_id=instance.id,
                        status=status,
                        response_time=response_time
                    )
        
        except asyncio.TimeoutError:
            return HealthCheckResult(
                instance_id=instance.id,
                status=HealthStatus.UNHEALTHY,
                response_time=self.health_check_timeout,
                error="Health check timeout"
            )
        except Exception as e:
            return HealthCheckResult(
                instance_id=instance.id,
                status=HealthStatus.OFFLINE,
                response_time=time.time() - start_time,
                error=str(e)
            )
    
    async def _update_instance_health(self, result: HealthCheckResult):
        """Update instance health based on check result"""
        if result.instance_id not in self.instances:
            return
        
        instance = self.instances[result.instance_id]
        instance.health_status = result.status
        instance.last_health_check = result.timestamp
        
        # Update response time average
        if result.response_time > 0:
            instance.avg_response_time = (instance.avg_response_time * 0.8 + result.response_time * 0.2)
        
        # Log health status changes
        logger.info(f"Instance {result.instance_id} health: {result.status.value} "
                   f"(response time: {result.response_time:.3f}s)")
    
    async def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        healthy_count = len(await self.get_healthy_instances())
        total_count = len(self.instances)
        
        # Calculate request statistics
        recent_requests = [r for r in self.request_history 
                          if r['timestamp'] > datetime.now(timezone.utc) - timedelta(minutes=5)]
        
        avg_response_time = 0.0
        if recent_requests:
            avg_response_time = sum(r['response_time'] for r in recent_requests) / len(recent_requests)
        
        # Error rate
        recent_errors = [e for e in self.error_history 
                        if e['timestamp'] > datetime.now(timezone.utc) - timedelta(minutes=5)]
        
        error_rate = 0.0
        if recent_requests:
            error_rate = len(recent_errors) / (len(recent_requests) + len(recent_errors)) * 100
        
        return {
            'service_name': self.service_name,
            'strategy': self.strategy.value,
            'total_instances': total_count,
            'healthy_instances': healthy_count,
            'unhealthy_instances': total_count - healthy_count,
            'requests_last_5min': len(recent_requests),
            'errors_last_5min': len(recent_errors),
            'avg_response_time': avg_response_time,
            'error_rate_percent': error_rate,
            'instances': {
                instance.id: {
                    'health': instance.health_status.value,
                    'connections': instance.current_connections,
                    'avg_response_time': instance.avg_response_time,
                    'error_count': instance.error_count,
                    'region': instance.region
                }
                for instance in self.instances.values()
            }
        }
    
    async def stop_health_checks(self):
        """Stop health checking"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass


class ServiceRegistry:
    """Global service registry for multiple services"""
    
    def __init__(self):
        self.load_balancers: Dict[str, LoadBalancer] = {}
    
    def get_load_balancer(self, service_name: str) -> LoadBalancer:
        """Get or create load balancer for a service"""
        if service_name not in self.load_balancers:
            self.load_balancers[service_name] = LoadBalancer(service_name)
        return self.load_balancers[service_name]
    
    async def register_service_instance(self, service_name: str, instance: ServiceInstance):
        """Register a service instance"""
        lb = self.get_load_balancer(service_name)
        await lb.register_instance(instance)
    
    async def discover_service(self, service_name: str, context: Dict[str, Any] = None) -> Optional[ServiceInstance]:
        """Discover and select a service instance"""
        if service_name not in self.load_balancers:
            return None
        
        lb = self.load_balancers[service_name]
        return await lb.select_instance(context)
    
    async def execute_service_request(self, 
                                    service_name: str,
                                    method: str,
                                    path: str,
                                    context: Dict[str, Any] = None,
                                    **kwargs) -> Tuple[Any, ServiceInstance]:
        """Execute request against a service"""
        if service_name not in self.load_balancers:
            raise Exception(f"Service {service_name} not registered")
        
        lb = self.load_balancers[service_name]
        return await lb.execute_request(method, path, context, **kwargs)
    
    async def get_all_service_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all services"""
        stats = {}
        for service_name, lb in self.load_balancers.items():
            stats[service_name] = await lb.get_load_balancer_stats()
        return stats
    
    async def start_all_health_checks(self):
        """Start health checks for all services"""
        for lb in self.load_balancers.values():
            await lb.start_health_checks()
    
    async def stop_all_health_checks(self):
        """Stop health checks for all services"""
        for lb in self.load_balancers.values():
            await lb.stop_health_checks()


# Global service registry
service_registry = ServiceRegistry()


# Convenience functions
async def register_api_instance(host: str, port: int, weight: int = 100, region: str = "default"):
    """Register an API server instance"""
    instance = ServiceInstance(
        id=f"api-{host}-{port}",
        host=host,
        port=port,
        weight=weight,
        region=region,
        metadata={"type": "api_server"}
    )
    await service_registry.register_service_instance("api", instance)


async def discover_api_instance(context: Dict[str, Any] = None) -> Optional[ServiceInstance]:
    """Discover an API instance"""
    return await service_registry.discover_service("api", context)


async def get_service_stats() -> Dict[str, Any]:
    """Get all service statistics"""
    return await service_registry.get_all_service_stats()