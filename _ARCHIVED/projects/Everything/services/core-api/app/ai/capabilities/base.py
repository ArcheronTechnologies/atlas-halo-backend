"""
Base AI Capability Interface

Defines the core interface that all AI capabilities must implement,
providing standardized execution, error handling, and observability.
"""

import abc
import time
import asyncio
import logging
import hashlib
import json
from typing import Dict, Any, Optional, List, Union, Type, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple token bucket rate limiter"""
    
    def __init__(self, rate_per_second: float):
        self.rate = rate_per_second
        self.tokens = rate_per_second
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Acquire a token. Returns True if successful, False if rate limited."""
        async with self.lock:
            now = time.time()
            time_passed = now - self.last_update
            self.tokens = min(self.rate, self.tokens + time_passed * self.rate)
            self.last_update = now
            
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False


class CapabilityStatus(Enum):
    """Status of a capability"""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNAVAILABLE = "unavailable"
    LOADING = "loading"
    ERROR = "error"


@dataclass
class CapabilityConfig:
    """Configuration for an AI capability"""
    name: str
    version: str
    enabled: bool = True
    timeout_seconds: float = 30.0
    retry_count: int = 2
    fallback_capability: Optional[str] = None
    model_config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    observability_enabled: bool = True
    # Caching configuration
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes default
    cache_max_size: int = 1000
    # Concurrency configuration
    max_concurrent_executions: int = 10
    rate_limit_per_second: Optional[float] = None


@dataclass
class CapabilityResult:
    """Result from capability execution"""
    success: bool
    data: Any = None
    confidence: Optional[float] = None
    execution_time_ms: Optional[float] = None
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    capability_name: Optional[str] = None
    capability_version: Optional[str] = None
    fallback_used: bool = False


class AICapability(abc.ABC):
    """
    Base class for all AI capabilities.
    
    Provides standardized interface for AI components with built-in
    error handling, timeouts, observability, and fallback mechanisms.
    """
    
    def __init__(self, config: CapabilityConfig):
        self.config = config
        self._status = CapabilityStatus.LOADING
        self._last_health_check = None
        self._metrics = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'avg_execution_time': 0.0,
            'last_execution_time': None,
            'cache_hits': 0,
            'cache_misses': 0,
            'concurrent_executions': 0
        }
        
        # Caching
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_access_times: Dict[str, datetime] = {}
        
        # Concurrency control
        self._execution_semaphore = asyncio.Semaphore(config.max_concurrent_executions)
        self._rate_limiter = None
        if config.rate_limit_per_second:
            self._rate_limiter = RateLimiter(config.rate_limit_per_second)
        
    @property
    def name(self) -> str:
        return self.config.name
    
    @property
    def version(self) -> str:
        return self.config.version
    
    @property
    def status(self) -> CapabilityStatus:
        return self._status
    
    @property
    def metrics(self) -> Dict[str, Any]:
        return self._metrics.copy()
    
    @abc.abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the capability. Load models, establish connections, etc.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abc.abstractmethod
    async def execute(self, payload: Dict[str, Any]) -> CapabilityResult:
        """
        Execute the capability's core functionality.
        
        Args:
            payload: Input data for the capability
            
        Returns:
            CapabilityResult with execution results
        """
        pass
    
    @abc.abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the capability is healthy and ready to serve requests.
        
        Returns:
            True if healthy, False otherwise
        """
        pass
    
    async def cleanup(self) -> None:
        """
        Cleanup resources. Override if needed.
        """
        self._cache.clear()
        self._cache_access_times.clear()
    
    def _generate_cache_key(self, payload: Dict[str, Any]) -> str:
        """Generate cache key from payload"""
        # Create a stable hash of the payload
        payload_str = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.md5(f"{self.name}:{self.version}:{payload_str}".encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[CapabilityResult]:
        """Get result from cache if valid"""
        if not self.config.cache_enabled or cache_key not in self._cache:
            return None
        
        # Check TTL
        cache_time = self._cache_access_times.get(cache_key)
        if cache_time:
            age = datetime.now(timezone.utc) - cache_time
            if age.total_seconds() > self.config.cache_ttl_seconds:
                # Expired, remove from cache
                self._cache.pop(cache_key, None)
                self._cache_access_times.pop(cache_key, None)
                return None
        
        cached_data = self._cache.get(cache_key)
        if cached_data:
            # Update access time
            self._cache_access_times[cache_key] = datetime.now(timezone.utc)
            
            # Reconstruct CapabilityResult
            result = CapabilityResult(**cached_data)
            result.metadata['cache_hit'] = True
            
            self._metrics['cache_hits'] += 1
            return result
        
        return None
    
    def _store_in_cache(self, cache_key: str, result: CapabilityResult) -> None:
        """Store result in cache"""
        if not self.config.cache_enabled or not result.success:
            return
        
        # Clean up cache if at max size
        if len(self._cache) >= self.config.cache_max_size:
            self._cleanup_cache()
        
        # Store serializable version
        cache_data = {
            'success': result.success,
            'data': result.data,
            'confidence': result.confidence,
            'execution_time_ms': result.execution_time_ms,
            'error': result.error,
            'warnings': result.warnings,
            'metadata': result.metadata,
            'capability_name': result.capability_name,
            'capability_version': result.capability_version,
            'fallback_used': result.fallback_used
        }
        
        self._cache[cache_key] = cache_data
        self._cache_access_times[cache_key] = datetime.now(timezone.utc)
        self._metrics['cache_misses'] += 1
    
    def _cleanup_cache(self) -> None:
        """Remove oldest cache entries"""
        if not self._cache_access_times:
            return
        
        # Remove oldest 25% of entries
        sorted_keys = sorted(
            self._cache_access_times.keys(),
            key=lambda k: self._cache_access_times[k]
        )
        
        remove_count = max(1, len(sorted_keys) // 4)
        for key in sorted_keys[:remove_count]:
            self._cache.pop(key, None)
            self._cache_access_times.pop(key, None)
    
    async def run(self, payload: Dict[str, Any]) -> CapabilityResult:
        """
        Execute capability with full observability, caching, and error handling.
        
        Args:
            payload: Input data for the capability
            
        Returns:
            CapabilityResult with execution results and metadata
        """
        start_time = time.time()
        
        # Check if capability is enabled
        if not self.config.enabled:
            return CapabilityResult(
                success=False,
                error="Capability is disabled",
                capability_name=self.name,
                capability_version=self.version
            )
        
        # Check health status
        if self._status == CapabilityStatus.UNAVAILABLE:
            return CapabilityResult(
                success=False,
                error="Capability is unavailable",
                capability_name=self.name,
                capability_version=self.version
            )
        
        # Generate cache key
        cache_key = self._generate_cache_key(payload)
        
        # Check cache first
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            return cached_result
        
        # Apply rate limiting
        if self._rate_limiter:
            if not await self._rate_limiter.acquire():
                return CapabilityResult(
                    success=False,
                    error="Rate limit exceeded",
                    capability_name=self.name,
                    capability_version=self.version
                )
        
        # Acquire concurrency semaphore
        async with self._execution_semaphore:
            self._metrics['concurrent_executions'] += 1
            
            try:
                # Execute with timeout
                try:
                    result = await asyncio.wait_for(
                        self.execute(payload),
                        timeout=self.config.timeout_seconds
                    )
                except asyncio.TimeoutError:
                    result = CapabilityResult(
                        success=False,
                        error=f"Execution timed out after {self.config.timeout_seconds}s",
                        capability_name=self.name,
                        capability_version=self.version
                    )
                
                # Update metrics
                execution_time = (time.time() - start_time) * 1000
                self._update_metrics(result.success if result else False, execution_time)
                
                # Enhance result with metadata
                if result:
                    result.execution_time_ms = execution_time
                    result.capability_name = self.name
                    result.capability_version = self.version
                    result.metadata = result.metadata or {}
                    result.metadata['cache_hit'] = False
                    result.metadata['concurrent_executions'] = self._metrics['concurrent_executions']
                    
                    # Store in cache
                    self._store_in_cache(cache_key, result)
                else:
                    result = CapabilityResult(
                        success=False,
                        error="No result returned from capability",
                        capability_name=self.name,
                        capability_version=self.version,
                        execution_time_ms=execution_time
                    )
                
                return result
                
            except Exception as e:
                # Update metrics
                execution_time = (time.time() - start_time) * 1000
                self._update_metrics(False, execution_time)
                
                logger.error(f"Error executing capability {self.name}: {e}", exc_info=True)
                
                return CapabilityResult(
                    success=False,
                    error=str(e),
                    execution_time_ms=execution_time,
                    capability_name=self.name,
                    capability_version=self.version
                )
            
            finally:
                self._metrics['concurrent_executions'] -= 1
    
    async def run_with_retry(self, payload: Dict[str, Any]) -> CapabilityResult:
        """
        Execute capability with retry logic.
        
        Args:
            payload: Input data for the capability
            
        Returns:
            CapabilityResult with execution results
        """
        last_result = None
        
        for attempt in range(self.config.retry_count + 1):
            result = await self.run(payload)
            
            if result.success:
                return result
            
            last_result = result
            
            if attempt < self.config.retry_count:
                # Exponential backoff
                wait_time = 2 ** attempt
                logger.warning(
                    f"Capability {self.name} failed (attempt {attempt + 1}), "
                    f"retrying in {wait_time}s: {result.error}"
                )
                await asyncio.sleep(wait_time)
        
        return last_result
    
    async def update_status(self, status: CapabilityStatus) -> None:
        """Update capability status"""
        old_status = self._status
        self._status = status
        
        if old_status != status:
            logger.info(f"Capability {self.name} status changed: {old_status} -> {status}")
    
    def _update_metrics(self, success: bool, execution_time: float) -> None:
        """Update capability metrics"""
        self._metrics['total_executions'] += 1
        self._metrics['last_execution_time'] = datetime.now(timezone.utc).isoformat()
        
        if success:
            self._metrics['successful_executions'] += 1
        else:
            self._metrics['failed_executions'] += 1
        
        # Update rolling average execution time
        total = self._metrics['total_executions']
        current_avg = self._metrics['avg_execution_time']
        self._metrics['avg_execution_time'] = (
            (current_avg * (total - 1) + execution_time) / total
        )


class NoOpCapability(AICapability):
    """
    No-operation capability for testing and fallbacks.
    """
    
    def __init__(self, name: str = "noop"):
        config = CapabilityConfig(
            name=name,
            version="1.0.0",
            enabled=True
        )
        super().__init__(config)
    
    async def initialize(self) -> bool:
        await self.update_status(CapabilityStatus.HEALTHY)
        return True
    
    async def execute(self, payload: Dict[str, Any]) -> CapabilityResult:
        return CapabilityResult(
            success=True,
            data={"message": "No-op capability executed successfully"},
            confidence=1.0
        )
    
    async def health_check(self) -> bool:
        return True


class FallbackCapability(AICapability):
    """
    Fallback capability that provides basic functionality when primary capabilities fail.
    """
    
    def __init__(self, primary_capability_name: str):
        config = CapabilityConfig(
            name=f"{primary_capability_name}_fallback",
            version="1.0.0",
            enabled=True
        )
        super().__init__(config)
        self.primary_name = primary_capability_name
    
    async def initialize(self) -> bool:
        await self.update_status(CapabilityStatus.HEALTHY)
        return True
    
    async def execute(self, payload: Dict[str, Any]) -> CapabilityResult:
        return CapabilityResult(
            success=True,
            data={
                "message": f"Fallback for {self.primary_name}",
                "fallback_reason": "Primary capability unavailable",
                "original_payload": payload
            },
            confidence=0.1,  # Low confidence for fallback
            warnings=[f"Using fallback for {self.primary_name}"]
        )
    
    async def health_check(self) -> bool:
        return True