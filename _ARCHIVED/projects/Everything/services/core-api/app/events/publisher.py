"""
Event Publisher

High-level interface for publishing domain events throughout the application.
Provides convenient methods for different event types and handles fallbacks
when Kafka is unavailable.
"""

from typing import Optional, Dict, List, Any, Union, Callable
from datetime import datetime
import logging
import asyncio
from dataclasses import asdict

from .kafka_client import kafka_client, PublishResult
from .schemas import (
    BaseEvent, EventFactory, EventType,
    UserCreatedPayload, UserUpdatedPayload, UserLoginPayload,
    ComponentCreatedPayload, ComponentUpdatedPayload, ComponentPriceChangedPayload,
    RFQCreatedPayload, RFQSubmittedPayload, RFQRespondedPayload,
    POCreatedPayload, POApprovedPayload,
    MarketAlertPayload, GeopoliticalAlertPayload, SupplierAlertPayload,
    SystemHealthCheckPayload, AuditLogPayload
)

logger = logging.getLogger(__name__)


class PublisherError(Exception):
    """Base exception for publisher errors"""
    pass


class EventPublisher:
    """High-level event publisher with fallback handling"""
    
    def __init__(self):
        self._fallback_handlers: List[Callable[[BaseEvent], None]] = []
        self._metrics = {
            'events_published': 0,
            'events_failed': 0,
            'fallback_invocations': 0
        }
    
    def add_fallback_handler(self, handler: Callable[[BaseEvent], None]):
        """Add a fallback handler for when Kafka is unavailable"""
        self._fallback_handlers.append(handler)
    
    async def publish_event(
        self,
        event: BaseEvent,
        topic_suffix: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> PublishResult:
        """
        Publish a single event
        
        Args:
            event: The event to publish
            topic_suffix: Optional topic suffix override
            timeout: Optional timeout for publishing
            
        Returns:
            PublishResult indicating success/failure
        """
        try:
            if not kafka_client.is_healthy():
                logger.warning("Kafka client not healthy, using fallback handlers")
                await self._handle_fallback(event)
                return PublishResult(
                    success=False,
                    error="Kafka unavailable, used fallback",
                    topic=topic_suffix or "fallback"
                )
            
            result = await kafka_client.publish_event(
                event, 
                topic_suffix=topic_suffix,
                timeout=timeout
            )
            
            if result.success:
                self._metrics['events_published'] += 1
                logger.debug(f"Published event {event.header.event_id} to {result.topic}")
            else:
                self._metrics['events_failed'] += 1
                logger.error(f"Failed to publish event {event.header.event_id}: {result.error}")
                await self._handle_fallback(event)
            
            return result
            
        except Exception as e:
            self._metrics['events_failed'] += 1
            logger.error(f"Error publishing event {event.header.event_id}: {e}")
            await self._handle_fallback(event)
            return PublishResult(success=False, error=str(e))
    
    async def publish_batch(
        self,
        events: List[BaseEvent],
        topic_suffix: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> List[PublishResult]:
        """
        Publish multiple events as a batch
        
        Args:
            events: List of events to publish
            topic_suffix: Optional topic suffix override
            timeout: Optional timeout for publishing
            
        Returns:
            List of PublishResults for each event
        """
        if not events:
            return []
        
        try:
            if not kafka_client.is_healthy():
                logger.warning("Kafka client not healthy, using fallback handlers")
                for event in events:
                    await self._handle_fallback(event)
                return [
                    PublishResult(
                        success=False,
                        error="Kafka unavailable, used fallback",
                        topic=topic_suffix or "fallback"
                    )
                    for _ in events
                ]
            
            results = await kafka_client.publish_batch(
                events,
                topic_suffix=topic_suffix,
                timeout=timeout
            )
            
            # Update metrics
            for result in results:
                if result.success:
                    self._metrics['events_published'] += 1
                else:
                    self._metrics['events_failed'] += 1
            
            # Handle fallbacks for failed events
            failed_events = [
                events[i] for i, result in enumerate(results)
                if not result.success
            ]
            for event in failed_events:
                await self._handle_fallback(event)
            
            return results
            
        except Exception as e:
            self._metrics['events_failed'] += len(events)
            logger.error(f"Error publishing event batch: {e}")
            
            # Handle fallback for all events
            for event in events:
                await self._handle_fallback(event)
            
            return [
                PublishResult(success=False, error=str(e))
                for _ in events
            ]
    
    async def _handle_fallback(self, event: BaseEvent):
        """Handle event through fallback handlers"""
        if not self._fallback_handlers:
            logger.warning(f"No fallback handlers configured for event {event.header.event_id}")
            return
        
        self._metrics['fallback_invocations'] += 1
        
        for handler in self._fallback_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Fallback handler failed for event {event.header.event_id}: {e}")
    
    def get_metrics(self) -> Dict[str, int]:
        """Get publisher metrics"""
        return self._metrics.copy()
    
    # Convenience methods for different event types
    
    async def publish_user_created(
        self,
        user_id: str,
        email: str,
        name: str,
        roles: List[str],
        correlation_id: Optional[str] = None,
        **kwargs
    ) -> PublishResult:
        """Publish user created event"""
        event = EventFactory.create_user_created_event(
            user_id=user_id,
            email=email,
            name=name,
            roles=roles,
            correlation_id=correlation_id,
            **kwargs
        )
        return await self.publish_event(event, topic_suffix="users")
    
    async def publish_user_login(
        self,
        user_id: str,
        email: str,
        login_method: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs
    ) -> PublishResult:
        """Publish user login event"""
        payload = UserLoginPayload(
            user_id=user_id,
            email=email,
            login_method=login_method,
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            login_at=datetime.now()
        )
        event = EventFactory.create_event(EventType.USER_LOGIN, payload, **kwargs)
        return await self.publish_event(event, topic_suffix="users")
    
    async def publish_component_created(
        self,
        component_id: str,
        manufacturer_part_number: str,
        manufacturer_id: str,
        category: str,
        description: str,
        specifications: Dict[str, Any],
        created_by: str,
        **kwargs
    ) -> PublishResult:
        """Publish component created event"""
        event = EventFactory.create_component_created_event(
            component_id=component_id,
            manufacturer_part_number=manufacturer_part_number,
            manufacturer_id=manufacturer_id,
            category=category,
            description=description,
            specifications=specifications,
            created_by=created_by,
            **kwargs
        )
        return await self.publish_event(event, topic_suffix="components")
    
    async def publish_component_price_changed(
        self,
        component_id: str,
        supplier_id: str,
        previous_price: float,
        new_price: float,
        currency: str,
        quantity_break: int,
        effective_date: datetime,
        source: str = "manual",
        **kwargs
    ) -> PublishResult:
        """Publish component price changed event"""
        price_change_percent = ((new_price - previous_price) / previous_price) * 100
        
        payload = ComponentPriceChangedPayload(
            component_id=component_id,
            supplier_id=supplier_id,
            previous_price=previous_price,
            new_price=new_price,
            currency=currency,
            quantity_break=quantity_break,
            price_change_percent=price_change_percent,
            effective_date=effective_date,
            source=source
        )
        event = EventFactory.create_event(EventType.COMPONENT_PRICE_CHANGED, payload, **kwargs)
        return await self.publish_event(event, topic_suffix="components")
    
    async def publish_rfq_created(
        self,
        rfq_id: str,
        rfq_number: str,
        customer_id: str,
        items: List[Dict[str, Any]],
        created_by: str,
        due_date: Optional[datetime] = None,
        source: str = "manual",
        **kwargs
    ) -> PublishResult:
        """Publish RFQ created event"""
        event = EventFactory.create_rfq_created_event(
            rfq_id=rfq_id,
            rfq_number=rfq_number,
            customer_id=customer_id,
            items=items,
            created_by=created_by,
            due_date=due_date,
            source=source,
            **kwargs
        )
        return await self.publish_event(event, topic_suffix="rfqs")
    
    async def publish_rfq_submitted(
        self,
        rfq_id: str,
        rfq_number: str,
        customer_id: str,
        submitted_to: List[str],
        items: List[Dict[str, Any]],
        submitted_by: str,
        **kwargs
    ) -> PublishResult:
        """Publish RFQ submitted event"""
        payload = RFQSubmittedPayload(
            rfq_id=rfq_id,
            rfq_number=rfq_number,
            customer_id=customer_id,
            submitted_to=submitted_to,
            items=items,
            submitted_by=submitted_by,
            submitted_at=datetime.now()
        )
        event = EventFactory.create_event(EventType.RFQ_SUBMITTED, payload, **kwargs)
        return await self.publish_event(event, topic_suffix="rfqs")
    
    async def publish_market_alert(
        self,
        alert_type: str,
        component_ids: List[str],
        regions: List[str],
        severity: str,
        confidence: float,
        description: str,
        data_sources: Optional[List[str]] = None,
        estimated_impact: Optional[Dict[str, Any]] = None,
        recommendation: str = "",
        **kwargs
    ) -> PublishResult:
        """Publish market alert event"""
        event = EventFactory.create_market_alert_event(
            alert_type=alert_type,
            component_ids=component_ids,
            regions=regions,
            severity=severity,
            confidence=confidence,
            description=description,
            data_sources=data_sources or [],
            estimated_impact=estimated_impact or {},
            recommendation=recommendation,
            **kwargs
        )
        return await self.publish_event(event, topic_suffix="intelligence")
    
    async def publish_geopolitical_alert(
        self,
        event_type: str,
        affected_countries: List[str],
        affected_regions: List[str],
        risk_level: str,
        description: str,
        potential_impact: Dict[str, Any],
        recommended_actions: List[str],
        source_urls: List[str],
        **kwargs
    ) -> PublishResult:
        """Publish geopolitical alert event"""
        payload = GeopoliticalAlertPayload(
            alert_id=f"geo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            event_type=event_type,
            affected_countries=affected_countries,
            affected_regions=affected_regions,
            risk_level=risk_level,
            description=description,
            potential_impact=potential_impact,
            recommended_actions=recommended_actions,
            source_urls=source_urls,
            created_at=datetime.now()
        )
        event = EventFactory.create_event(EventType.GEOPOLITICAL_ALERT, payload, **kwargs)
        return await self.publish_event(event, topic_suffix="intelligence")
    
    async def publish_supplier_alert(
        self,
        supplier_id: str,
        alert_type: str,
        severity: str,
        description: str,
        affected_components: List[str],
        risk_score: float,
        recommended_actions: List[str],
        data_sources: List[str],
        **kwargs
    ) -> PublishResult:
        """Publish supplier alert event"""
        payload = SupplierAlertPayload(
            alert_id=f"supplier_{supplier_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            supplier_id=supplier_id,
            alert_type=alert_type,
            severity=severity,
            description=description,
            affected_components=affected_components,
            risk_score=risk_score,
            recommended_actions=recommended_actions,
            data_sources=data_sources,
            created_at=datetime.now()
        )
        event = EventFactory.create_event(EventType.SUPPLIER_ALERT, payload, **kwargs)
        return await self.publish_event(event, topic_suffix="intelligence")
    
    async def publish_system_health_check(
        self,
        service_name: str,
        status: str,
        response_time: float,
        memory_usage: float,
        cpu_usage: float,
        active_connections: int,
        error_rate: float,
        **kwargs
    ) -> PublishResult:
        """Publish system health check event"""
        payload = SystemHealthCheckPayload(
            service_name=service_name,
            status=status,
            response_time=response_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            active_connections=active_connections,
            error_rate=error_rate,
            timestamp=datetime.now()
        )
        event = EventFactory.create_event(EventType.SYSTEM_HEALTH_CHECK, payload, **kwargs)
        return await self.publish_event(event, topic_suffix="system")
    
    async def publish_audit_log(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        user_id: str,
        success: bool,
        ip_address: Optional[str] = None,
        before_state: Optional[Dict[str, Any]] = None,
        after_state: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        **kwargs
    ) -> PublishResult:
        """Publish audit log event"""
        event = EventFactory.create_audit_log_event(
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=user_id,
            success=success,
            ip_address=ip_address,
            before_state=before_state,
            after_state=after_state,
            error_message=error_message,
            **kwargs
        )
        return await self.publish_event(event, topic_suffix="audit")


# Global publisher instance
event_publisher = EventPublisher()


# Default fallback handler that logs events
def log_event_fallback(event: BaseEvent):
    """Default fallback handler that logs events when Kafka is unavailable"""
    logger.info(f"FALLBACK: {event.header.event_type} - {event.header.event_id}")
    logger.debug(f"Event payload: {event.payload}")


# Add default fallback handler
event_publisher.add_fallback_handler(log_event_fallback)


# Context manager for batch publishing
class BatchPublisher:
    """Context manager for batch event publishing"""
    
    def __init__(self, publisher: EventPublisher, topic_suffix: Optional[str] = None):
        self.publisher = publisher
        self.topic_suffix = topic_suffix
        self.events: List[BaseEvent] = []
    
    def add_event(self, event: BaseEvent):
        """Add event to batch"""
        self.events.append(event)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.events and exc_type is None:
            results = await self.publisher.publish_batch(
                self.events,
                topic_suffix=self.topic_suffix
            )
            failed_count = sum(1 for r in results if not r.success)
            if failed_count > 0:
                logger.warning(f"Failed to publish {failed_count} out of {len(results)} events")
        
        # Clear events regardless of outcome
        self.events.clear()


def create_batch_publisher(topic_suffix: Optional[str] = None) -> BatchPublisher:
    """Create a batch publisher context manager"""
    return BatchPublisher(event_publisher, topic_suffix)


# Simple emit function for backward compatibility
async def emit(event_type: str, data: Dict[str, Any], correlation_id: Optional[str] = None):
    """
    Simple event emission function for backward compatibility
    
    Args:
        event_type: The type of event (e.g., "rfq.status_changed")
        data: Event data dictionary
        correlation_id: Optional correlation ID
    """
    try:
        from .schemas import EventFactory, EventType
        
        # Map simple event types to our schema types
        event_type_map = {
            "rfq.status_changed": EventType.RFQ_UPDATED,
            "rfq.winner_selected": EventType.RFQ_RESPONDED, 
            "po.created": EventType.PO_CREATED,
            "po.status_changed": EventType.PO_APPROVED,
            "po.item_added": EventType.PO_CREATED,
            "po.item_deleted": EventType.PO_APPROVED
        }
        
        schema_event_type = event_type_map.get(event_type, EventType.AUDIT_LOG)
        
        # Create a generic event with the data as payload
        event = EventFactory.create_event(
            schema_event_type,
            data,
            correlation_id=correlation_id
        )
        
        # Publish the event
        result = await event_publisher.publish_event(event)
        
        if not result.success:
            logger.warning(f"Failed to emit event {event_type}: {result.error}")
            
    except Exception as e:
        logger.error(f"Error emitting event {event_type}: {e}")


# Synchronous wrapper for emit
def emit_sync(event_type: str, data: Dict[str, Any], correlation_id: Optional[str] = None):
    """Synchronous wrapper for emit function"""
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, schedule the coroutine
            asyncio.create_task(emit(event_type, data, correlation_id))
        else:
            # If not in async context, run it
            loop.run_until_complete(emit(event_type, data, correlation_id))
    except Exception as e:
        logger.error(f"Error in emit_sync for {event_type}: {e}")


# Alias for backward compatibility
emit = emit_sync

