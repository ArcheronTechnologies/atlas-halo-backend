"""
Redis Pub/Sub Manager for Real-Time Communication
Production-ready Redis integration for WebSocket notifications
Supports both single instance and cluster mode
"""

import json
import asyncio
import logging
import redis.asyncio as redis
from redis.asyncio.cluster import RedisCluster
from typing import Dict, Any, Set, Optional, Callable, Union
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class RedisPubSubManager:
    """Production Redis pub/sub manager for real-time notifications"""

    def __init__(self, redis_url: str = None, use_cluster: bool = None):
        """
        Initialize Redis connection

        Args:
            redis_url: Redis connection URL (single or cluster)
            use_cluster: True for cluster mode, False for single instance
                        If None, auto-detect from REDIS_CLUSTER_ENABLED env var
        """
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.use_cluster = use_cluster if use_cluster is not None else \
                          os.getenv('REDIS_CLUSTER_ENABLED', 'false').lower() == 'true'
        self.cluster_nodes = os.getenv('REDIS_CLUSTER_NODES',
                                      '172.20.0.11:7000,172.20.0.12:7001,172.20.0.13:7002').split(',')

        self.redis_client: Optional[Union[redis.Redis, RedisCluster]] = None
        self.pubsub: Optional[redis.client.PubSub] = None
        self.channels: Set[str] = set()
        self.subscribers: Dict[str, Set[Callable]] = {}
        self._listening = False

    async def connect(self) -> None:
        """Connect to Redis server (single or cluster)"""
        try:
            if self.use_cluster:
                # Connect to Redis Cluster
                startup_nodes = [{"host": node.split(':')[0], "port": int(node.split(':')[1])}
                               for node in self.cluster_nodes]

                self.redis_client = RedisCluster(
                    startup_nodes=startup_nodes,
                    decode_responses=False,
                    skip_full_coverage_check=True
                )
                await self.redis_client.ping()
                logger.info(f"✅ Redis Cluster connected successfully ({len(self.cluster_nodes)} nodes)")
            else:
                # Connect to single Redis instance
                self.redis_client = redis.from_url(self.redis_url)
                await self.redis_client.ping()
                logger.info("✅ Redis pub/sub connected successfully (single instance)")

            self.pubsub = self.redis_client.pubsub()

        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Redis"""
        if self.pubsub:
            await self.pubsub.unsubscribe()
            await self.pubsub.close()
        if self.redis_client:
            await self.redis_client.close()
        self._listening = False
        logger.info("🔌 Redis pub/sub disconnected")

    async def publish(self, channel: str, message: Dict[str, Any]) -> int:
        """Publish message to Redis channel"""
        if not self.redis_client:
            await self.connect()

        try:
            message_data = {
                'timestamp': datetime.now().isoformat(),
                'channel': channel,
                'data': message
            }

            result = await self.redis_client.publish(
                channel,
                json.dumps(message_data, default=str)
            )

            logger.debug(f"📡 Published to {channel}: {result} subscribers")
            return result

        except Exception as e:
            logger.error(f"❌ Error publishing to {channel}: {e}")
            raise

    async def subscribe(self, channel: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to Redis channel with callback"""
        if not self.redis_client:
            await self.connect()

        try:
            if channel not in self.channels:
                await self.pubsub.subscribe(channel)
                self.channels.add(channel)
                logger.info(f"📻 Subscribed to channel: {channel}")

            if channel not in self.subscribers:
                self.subscribers[channel] = set()
            self.subscribers[channel].add(callback)

            # Start listening if not already
            if not self._listening:
                asyncio.create_task(self._listen())

        except Exception as e:
            logger.error(f"❌ Error subscribing to {channel}: {e}")
            raise

    async def unsubscribe(self, channel: str, callback: Callable = None) -> None:
        """Unsubscribe from Redis channel"""
        if channel in self.subscribers:
            if callback:
                self.subscribers[channel].discard(callback)
                if not self.subscribers[channel]:
                    del self.subscribers[channel]
            else:
                del self.subscribers[channel]

        if channel not in self.subscribers and channel in self.channels:
            await self.pubsub.unsubscribe(channel)
            self.channels.remove(channel)
            logger.info(f"📻 Unsubscribed from channel: {channel}")

    async def _listen(self) -> None:
        """Listen for Redis messages and dispatch to callbacks"""
        if self._listening:
            return

        self._listening = True
        logger.info("👂 Starting Redis message listener")

        try:
            async for message in self.pubsub.listen():
                if message['type'] == 'message':
                    try:
                        channel = message['channel'].decode('utf-8')
                        data = json.loads(message['data'].decode('utf-8'))

                        # Dispatch to all subscribers for this channel
                        if channel in self.subscribers:
                            for callback in self.subscribers[channel]:
                                try:
                                    if asyncio.iscoroutinefunction(callback):
                                        asyncio.create_task(callback(data))
                                    else:
                                        callback(data)
                                except Exception as e:
                                    logger.error(f"❌ Error in message callback: {e}")

                    except Exception as e:
                        logger.error(f"❌ Error processing Redis message: {e}")

        except Exception as e:
            logger.error(f"❌ Redis listener error: {e}")
        finally:
            self._listening = False

    async def publish_incident_alert(self, incident_data: Dict[str, Any]) -> None:
        """Publish real-time incident alert"""
        alert = {
            'type': 'incident_alert',
            'incident_id': incident_data.get('id'),
            'incident_type': incident_data.get('incident_type'),
            'severity_level': incident_data.get('severity_level'),
            'latitude': incident_data.get('latitude'),
            'longitude': incident_data.get('longitude'),
            'location_address': incident_data.get('location_address'),
            'description': incident_data.get('description')[:100] + '...' if len(incident_data.get('description', '')) > 100 else incident_data.get('description'),
            'reported_time': incident_data.get('reported_time')
        }

        # Publish to global incident channel
        await self.publish('incidents:alerts', alert)

        # Publish to severity-specific channel
        severity = incident_data.get('severity_level', 'moderate')
        await self.publish(f'incidents:severity:{severity}', alert)

        # Publish to location-specific channel (rounded coordinates for privacy)
        lat = round(float(incident_data.get('latitude', 0)), 2)
        lon = round(float(incident_data.get('longitude', 0)), 2)
        await self.publish(f'incidents:location:{lat}:{lon}', alert)

    async def publish_safety_zone_update(self, zone_data: Dict[str, Any]) -> None:
        """Publish safety zone risk level changes"""
        update = {
            'type': 'safety_zone_update',
            'zone_id': zone_data.get('id'),
            'zone_name': zone_data.get('zone_name'),
            'risk_level': zone_data.get('current_risk_level'),
            'risk_score': zone_data.get('risk_score'),
            'center_latitude': zone_data.get('center_latitude'),
            'center_longitude': zone_data.get('center_longitude'),
            'incident_count_24h': zone_data.get('incident_count_24h')
        }

        await self.publish('safety_zones:updates', update)
        await self.publish(f'safety_zones:risk:{update["risk_level"]}', update)

    async def publish_user_notification(self, user_id: str, notification: Dict[str, Any]) -> None:
        """Publish user-specific notification"""
        message = {
            'type': 'user_notification',
            'user_id': user_id,
            'notification': notification
        }

        await self.publish(f'users:{user_id}:notifications', message)


# Global Redis pub/sub manager instance
redis_pubsub = RedisPubSubManager()


async def get_redis_pubsub() -> RedisPubSubManager:
    """Dependency to get Redis pub/sub manager"""
    if not redis_pubsub.redis_client:
        await redis_pubsub.connect()
    return redis_pubsub