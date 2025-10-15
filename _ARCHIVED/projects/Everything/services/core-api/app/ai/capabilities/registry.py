"""
Capability Registry

Central registry for AI capabilities with discovery, dependency management,
and lifecycle control.
"""

import logging
from typing import Dict, List, Optional, Set, Any, Type
from dataclasses import dataclass
import asyncio

from .base import AICapability, CapabilityStatus, CapabilityConfig, FallbackCapability

logger = logging.getLogger(__name__)


@dataclass
class CapabilityRegistration:
    """Registration information for a capability"""
    capability: AICapability
    dependencies: Set[str]
    dependents: Set[str]
    priority: int
    tags: Set[str]
    auto_start: bool = True


class CapabilityRegistry:
    """
    Central registry for AI capabilities.
    
    Manages capability lifecycle, dependencies, discovery, and provides
    a unified interface for capability orchestration.
    """
    
    def __init__(self):
        self._capabilities: Dict[str, CapabilityRegistration] = {}
        self._capability_classes: Dict[str, Type[AICapability]] = {}
        self._initialization_order: List[str] = []
        self._started = False
        
    def register_capability_class(
        self, 
        capability_class: Type[AICapability],
        tags: Optional[Set[str]] = None
    ) -> None:
        """
        Register a capability class for lazy instantiation.
        
        Args:
            capability_class: The capability class to register
            tags: Optional tags for categorization
        """
        # This would normally extract name from class metadata
        name = getattr(capability_class, '_capability_name', capability_class.__name__)
        self._capability_classes[name] = capability_class
        
        logger.info(f"Registered capability class: {name}")
    
    def register(
        self,
        capability: AICapability,
        dependencies: Optional[List[str]] = None,
        priority: int = 100,
        tags: Optional[Set[str]] = None,
        auto_start: bool = True
    ) -> None:
        """
        Register a capability instance.
        
        Args:
            capability: The capability instance to register
            dependencies: List of capability names this depends on
            priority: Priority for initialization order (lower = higher priority)
            tags: Tags for categorization and discovery
            auto_start: Whether to auto-start during registry initialization
        """
        name = capability.name
        
        if name in self._capabilities:
            logger.warning(f"Capability {name} already registered, replacing")
        
        # Validate dependencies
        deps = set(dependencies or [])
        for dep in deps:
            if dep not in self._capabilities and dep not in self._capability_classes:
                logger.warning(f"Dependency {dep} for {name} not yet registered")
        
        registration = CapabilityRegistration(
            capability=capability,
            dependencies=deps,
            dependents=set(),
            priority=priority,
            tags=tags or set(),
            auto_start=auto_start
        )
        
        self._capabilities[name] = registration
        
        # Update dependents for existing capabilities
        for dep_name in deps:
            if dep_name in self._capabilities:
                self._capabilities[dep_name].dependents.add(name)
        
        # Update dependents for this capability
        for reg_name, reg in self._capabilities.items():
            if name in reg.dependencies:
                registration.dependents.add(reg_name)
        
        logger.info(f"Registered capability: {name} (deps: {deps})")
        
        # Recalculate initialization order
        self._calculate_initialization_order()
    
    def get(self, name: str) -> Optional[AICapability]:
        """Get a capability by name"""
        registration = self._capabilities.get(name)
        return registration.capability if registration else None
    
    def get_capability(self, name: str) -> Optional[AICapability]:
        """Get a capability by name (alias for get method)"""
        return self.get(name)
    
    def list_capabilities(self, tags: Optional[Set[str]] = None) -> List[str]:
        """
        List registered capabilities, optionally filtered by tags.
        
        Args:
            tags: Optional tags to filter by
            
        Returns:
            List of capability names
        """
        if not tags:
            return list(self._capabilities.keys())
        
        matching = []
        for name, registration in self._capabilities.items():
            if tags.intersection(registration.tags):
                matching.append(name)
        
        return matching
    
    def get_status(self, name: str) -> Optional[CapabilityStatus]:
        """Get capability status"""
        capability = self.get(name)
        return capability.status if capability else None
    
    def get_dependencies(self, name: str) -> Set[str]:
        """Get capability dependencies"""
        registration = self._capabilities.get(name)
        return registration.dependencies.copy() if registration else set()
    
    def get_dependents(self, name: str) -> Set[str]:
        """Get capabilities that depend on this one"""
        registration = self._capabilities.get(name)
        return registration.dependents.copy() if registration else set()
    
    def find_by_tags(self, tags: Set[str]) -> List[AICapability]:
        """Find capabilities that have any of the specified tags"""
        matching = []
        for registration in self._capabilities.values():
            if tags.intersection(registration.tags):
                matching.append(registration.capability)
        return matching
    
    async def initialize_all(self) -> Dict[str, bool]:
        """
        Initialize all registered capabilities in dependency order.
        
        Returns:
            Dict mapping capability names to initialization success
        """
        results = {}
        
        logger.info("Initializing capabilities in dependency order...")
        
        for name in self._initialization_order:
            registration = self._capabilities.get(name)
            if not registration or not registration.auto_start:
                continue
                
            capability = registration.capability
            
            try:
                logger.info(f"Initializing capability: {name}")
                success = await capability.initialize()
                results[name] = success
                
                if success:
                    await capability.update_status(CapabilityStatus.HEALTHY)
                    logger.info(f"✓ Capability {name} initialized successfully")
                else:
                    await capability.update_status(CapabilityStatus.ERROR)
                    logger.error(f"✗ Capability {name} initialization failed")
                    
                    # Try to create fallback
                    await self._create_fallback(name)
                    
            except Exception as e:
                logger.error(f"✗ Error initializing capability {name}: {e}", exc_info=True)
                results[name] = False
                await capability.update_status(CapabilityStatus.ERROR)
                await self._create_fallback(name)
        
        self._started = True
        
        # Log summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        logger.info(f"Capability initialization complete: {successful}/{total} successful")
        
        return results
    
    async def health_check_all(self) -> Dict[str, bool]:
        """
        Perform health checks on all capabilities.
        
        Returns:
            Dict mapping capability names to health status
        """
        results = {}
        
        for name, registration in self._capabilities.items():
            try:
                healthy = await registration.capability.health_check()
                results[name] = healthy
                
                # Update status based on health check
                if healthy:
                    if registration.capability.status == CapabilityStatus.ERROR:
                        await registration.capability.update_status(CapabilityStatus.HEALTHY)
                else:
                    await registration.capability.update_status(CapabilityStatus.DEGRADED)
                    
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                results[name] = False
                await registration.capability.update_status(CapabilityStatus.ERROR)
        
        return results
    
    async def shutdown_all(self) -> None:
        """Shutdown all capabilities in reverse dependency order"""
        logger.info("Shutting down capabilities...")
        
        # Shutdown in reverse order
        for name in reversed(self._initialization_order):
            registration = self._capabilities.get(name)
            if not registration:
                continue
                
            try:
                await registration.capability.cleanup()
                await registration.capability.update_status(CapabilityStatus.UNAVAILABLE)
                logger.info(f"✓ Capability {name} shut down")
            except Exception as e:
                logger.error(f"Error shutting down capability {name}: {e}")
        
        self._started = False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for all capabilities"""
        metrics = {
            'total_capabilities': len(self._capabilities),
            'healthy_capabilities': 0,
            'degraded_capabilities': 0,
            'error_capabilities': 0,
            'capabilities': {}
        }
        
        for name, registration in self._capabilities.items():
            capability = registration.capability
            cap_metrics = capability.metrics
            cap_metrics['status'] = capability.status.value
            
            metrics['capabilities'][name] = cap_metrics
            
            # Count by status
            if capability.status == CapabilityStatus.HEALTHY:
                metrics['healthy_capabilities'] += 1
            elif capability.status == CapabilityStatus.DEGRADED:
                metrics['degraded_capabilities'] += 1
            elif capability.status == CapabilityStatus.ERROR:
                metrics['error_capabilities'] += 1
        
        return metrics
    
    def _calculate_initialization_order(self) -> None:
        """Calculate capability initialization order using topological sort"""
        # Simple topological sort
        in_degree = {}
        for name in self._capabilities:
            in_degree[name] = len(self._capabilities[name].dependencies)
        
        # Sort by priority first, then by in-degree
        available = sorted(
            [name for name, degree in in_degree.items() if degree == 0],
            key=lambda n: self._capabilities[n].priority
        )
        
        order = []
        
        while available:
            # Pick highest priority capability with no dependencies
            current = available.pop(0)
            order.append(current)
            
            # Update dependents
            for dependent in self._capabilities[current].dependents:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    # Insert in priority order
                    priority = self._capabilities[dependent].priority
                    inserted = False
                    for i, name in enumerate(available):
                        if self._capabilities[name].priority > priority:
                            available.insert(i, dependent)
                            inserted = True
                            break
                    if not inserted:
                        available.append(dependent)
        
        # Check for cycles
        if len(order) != len(self._capabilities):
            remaining = set(self._capabilities.keys()) - set(order)
            logger.error(f"Circular dependency detected among capabilities: {remaining}")
            # Add remaining capabilities anyway
            order.extend(remaining)
        
        self._initialization_order = order
        logger.debug(f"Capability initialization order: {order}")
    
    async def _create_fallback(self, failed_capability_name: str) -> None:
        """Create a fallback capability for a failed one"""
        fallback_name = f"{failed_capability_name}_fallback"
        
        if fallback_name not in self._capabilities:
            fallback = FallbackCapability(failed_capability_name)
            await fallback.initialize()
            
            self.register(
                fallback,
                priority=1000,  # Low priority
                tags={'fallback'},
                auto_start=False
            )
            
            logger.info(f"Created fallback capability: {fallback_name}")


# Global capability registry instance
capability_registry = CapabilityRegistry()