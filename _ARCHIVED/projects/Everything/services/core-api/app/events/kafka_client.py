import os
import asyncio
import json
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class PublishResult:
    """Result of publishing an event"""
    success: bool
    error: Optional[str] = None
    topic: Optional[str] = None
    partition: Optional[int] = None
    offset: Optional[int] = None


class KafkaClient:
    """Kafka client for event publishing"""
    
    def __init__(self):
        self._producer = None
        self._healthy = False
    
    async def start(self):
        """Initialize Kafka producer"""
        brokers = os.getenv("KAFKA_BROKERS")
        if not brokers:
            logger.warning("KAFKA_BROKERS not configured, Kafka client disabled")
            return
        
        try:
            from aiokafka import AIOKafkaProducer  # type: ignore
            self._producer = AIOKafkaProducer(
                bootstrap_servers=brokers.split(","),
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            await self._producer.start()
            self._healthy = True
            logger.info("Kafka producer started successfully")
        except Exception as e:
            logger.error(f"Failed to start Kafka producer: {e}")
            self._healthy = False
    
    async def stop(self):
        """Stop Kafka producer"""
        if self._producer is not None:
            try:
                await self._producer.stop()
                logger.info("Kafka producer stopped")
            except Exception as e:
                logger.error(f"Error stopping Kafka producer: {e}")
            finally:
                self._producer = None
                self._healthy = False
    
    def is_healthy(self) -> bool:
        """Check if Kafka client is healthy"""
        return self._healthy and self._producer is not None
    
    async def send(self, topic: str, value: bytes) -> PublishResult:
        """Send raw bytes to topic"""
        if not self.is_healthy():
            return PublishResult(
                success=False,
                error="Kafka client not healthy",
                topic=topic
            )
        
        try:
            record_metadata = await self._producer.send_and_wait(topic, value)
            return PublishResult(
                success=True,
                topic=record_metadata.topic,
                partition=record_metadata.partition,
                offset=record_metadata.offset
            )
        except Exception as e:
            logger.error(f"Failed to send message to {topic}: {e}")
            return PublishResult(
                success=False,
                error=str(e),
                topic=topic
            )
    
    async def publish_event(
        self, 
        event, 
        topic_suffix: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> PublishResult:
        """Publish a structured event"""
        if not self.is_healthy():
            return PublishResult(
                success=False,
                error="Kafka client not healthy"
            )
        
        try:
            # Convert event to dict for serialization
            event_dict = {
                "header": {
                    "event_id": event.header.event_id,
                    "event_type": event.header.event_type,
                    "timestamp": event.header.timestamp.isoformat(),
                    "correlation_id": event.header.correlation_id,
                    "causation_id": event.header.causation_id,
                    "source": event.header.source,
                    "version": event.header.version
                },
                "payload": event.payload.__dict__ if hasattr(event.payload, '__dict__') else event.payload
            }
            
            # Determine topic
            base_topic = "scip.events"
            topic = f"{base_topic}.{topic_suffix}" if topic_suffix else base_topic
            
            # Send event
            record_metadata = await self._producer.send_and_wait(
                topic, 
                event_dict
            )
            
            return PublishResult(
                success=True,
                topic=record_metadata.topic,
                partition=record_metadata.partition,
                offset=record_metadata.offset
            )
            
        except Exception as e:
            logger.error(f"Failed to publish event {event.header.event_id}: {e}")
            return PublishResult(
                success=False,
                error=str(e)
            )
    
    async def publish_batch(
        self,
        events: List,
        topic_suffix: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> List[PublishResult]:
        """Publish multiple events as a batch"""
        if not events:
            return []
        
        if not self.is_healthy():
            return [
                PublishResult(
                    success=False,
                    error="Kafka client not healthy"
                )
                for _ in events
            ]
        
        results = []
        for event in events:
            result = await self.publish_event(event, topic_suffix, timeout)
            results.append(result)
        
        return results


# Global client instance
kafka_client = KafkaClient()

# Legacy functions for backward compatibility
async def start():
    await kafka_client.start()

async def stop():
    await kafka_client.stop()

async def send(topic: str, value: bytes):
    result = await kafka_client.send(topic, value)
    if not result.success:
        logger.warning(f"Failed to send to {topic}: {result.error}")

