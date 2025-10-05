"""
Smart Data Collection Scheduler
Intelligent scheduling with adaptive windows and deduplication
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from .polisen_data_collector import PolisenDataCollector
from ..database.postgis_database import get_database
from ..websockets.incident_broadcaster import broadcast_new_incident

logger = logging.getLogger(__name__)


@dataclass
class CollectionStats:
    """Statistics for collection operations"""
    city: str
    start_time: datetime
    end_time: datetime
    incidents_fetched: int
    incidents_stored: int
    incidents_updated: int
    incidents_skipped: int
    errors: List[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()

    @property
    def success_rate(self) -> float:
        if self.incidents_fetched == 0:
            return 0.0
        return (self.incidents_stored / self.incidents_fetched) * 100


class SmartDataScheduler:
    """
    Intelligent scheduler that:
    - Adapts collection windows based on last successful run
    - Prevents duplicate data collection
    - Tracks metrics per city
    - Implements circuit breaker for failing sources
    """

    def __init__(self, collection_interval_minutes: int = 60):
        self.db = None
        self.collector = None
        self.collection_interval = collection_interval_minutes
        self.last_collection_times: Dict[str, datetime] = {}
        self.circuit_breaker_failures: Dict[str, int] = {}
        self.is_running = False

        # Major Swedish cities
        self.cities = [
            "Stockholm",
            "Göteborg",
            "Malmö",
            "Uppsala",
            "Västerås",
            "Örebro",
            "Linköping",
            "Helsingborg"
        ]

    async def initialize(self):
        """Initialize database and collector"""
        self.db = await get_database()
        self.collector = PolisenDataCollector()
        logger.info("✅ Smart Data Scheduler initialized")

    async def get_optimal_collection_window(self, city: str) -> Tuple[datetime, datetime]:
        """
        Determine optimal time window for collection based on last collected data.
        Returns (start_date, end_date)
        """

        # Check database for the most recent incident from this city
        query = """
        SELECT MAX(occurred_at) as last_incident
        FROM crime_incidents
        WHERE source = 'polisen.se'
        AND (metadata->>'city' = $1 OR metadata->>'location_name' = $1)
        """

        result = await self.db.execute_query_single(query, city)

        if result and result.get('last_incident'):
            # Start from last known incident with 2-hour buffer for overlap
            start_date = result['last_incident'] - timedelta(hours=2)
        else:
            # First collection - go back 24 hours
            start_date = datetime.now() - timedelta(hours=24)

        # Ensure we don't go back more than 7 days (API limitation)
        seven_days_ago = datetime.now() - timedelta(days=7)
        if start_date < seven_days_ago:
            start_date = seven_days_ago

        end_date = datetime.now()

        logger.debug(f"📅 {city}: Collection window {start_date} to {end_date}")
        return start_date, end_date

    async def collect_city_data(self, city: str) -> CollectionStats:
        """Collect data for a single city with intelligent windowing"""

        start_time = datetime.now()
        stats = CollectionStats(
            city=city,
            start_time=start_time,
            end_time=start_time,  # Will update later
            incidents_fetched=0,
            incidents_stored=0,
            incidents_updated=0,
            incidents_skipped=0
        )

        try:
            # Check circuit breaker
            if self.circuit_breaker_failures.get(city, 0) >= 5:
                logger.warning(f"⚠️ Circuit breaker open for {city}, skipping collection")
                stats.errors.append("Circuit breaker open")
                return stats

            # Get optimal time window
            start_date, end_date = await self.get_optimal_collection_window(city)

            # Calculate days_back from start_date
            days_back = max(1, (end_date - start_date).days + 1)

            # Collect events
            logger.info(f"🔍 Collecting {city} incidents (last {days_back} days)")
            incidents = await self.collector.collect_events(
                location_name=city,
                days_back=days_back,
                max_events=2000
            )

            stats.incidents_fetched = len(incidents)

            # Filter by date range
            relevant_incidents = []
            for inc in incidents:
                try:
                    # Parse datetime, handling Polisen's format: "2025-10-04 7:08:39 +02:00"
                    dt_str = inc.datetime.replace(' +', 'T+').replace(' ', 'T', 1)
                    incident_dt = datetime.fromisoformat(dt_str)

                    # Make timezone-aware comparison
                    if incident_dt.tzinfo is None:
                        from datetime import timezone
                        incident_dt = incident_dt.replace(tzinfo=timezone.utc)

                    start_aware = start_date if start_date.tzinfo else start_date.replace(tzinfo=timezone.utc)
                    end_aware = end_date if end_date.tzinfo else end_date.replace(tzinfo=timezone.utc)

                    if start_aware <= incident_dt <= end_aware:
                        relevant_incidents.append(inc)
                except Exception as parse_error:
                    logger.debug(f"Failed to parse datetime '{inc.datetime}': {parse_error}")
                    continue

            logger.info(f"📊 {city}: {len(relevant_incidents)} incidents in time window")

            # Store with duplicate detection
            stored_count = 0
            updated_count = 0
            skipped_count = 0

            for incident in relevant_incidents:
                try:
                    # Parse datetime with format fix
                    dt_str = incident.datetime.replace(' +', 'T+').replace(' ', 'T', 1)
                    occurred_at = datetime.fromisoformat(dt_str)
                except Exception:
                    # Fallback to current time if parsing fails
                    occurred_at = datetime.now()

                incident_data = {
                    'incident_type': incident.type or 'other',
                    'severity': 3,  # Default severity
                    'description': incident.summary or '',
                    'latitude': incident.location.get('latitude', 0),
                    'longitude': incident.location.get('longitude', 0),
                    'occurred_at': occurred_at,
                    'source': 'polisen.se',
                    'source_id': str(incident.id),
                    'confidence_score': 0.9,
                    'metadata': {
                        'city': city,
                        'location_name': incident.location.get('name', ''),
                        'original_url': incident.url
                    }
                }

                try:
                    result_id = await self.db.store_incident(incident_data)
                    if result_id:
                        # Check if it was an update or new insert (simplified - could enhance)
                        stored_count += 1

                        # Broadcast to WebSocket clients in real-time
                        try:
                            incident_data['id'] = result_id
                            await broadcast_new_incident(
                                incident_data,
                                metadata={'city': city, 'municipality': city}
                            )
                            logger.debug(f"📡 Broadcast new incident: {result_id[:8]}")
                        except Exception as broadcast_error:
                            logger.warning(f"Failed to broadcast incident: {broadcast_error}")
                            # Don't fail the whole operation if broadcast fails

                except Exception as e:
                    logger.warning(f"Failed to store incident {incident.id}: {e}")
                    skipped_count += 1

            stats.incidents_stored = stored_count
            stats.incidents_skipped = skipped_count

            # Reset circuit breaker on success
            self.circuit_breaker_failures[city] = 0
            self.last_collection_times[city] = datetime.now()

            logger.info(
                f"✅ {city}: Stored {stored_count}, Skipped {skipped_count} "
                f"({stats.success_rate:.1f}% success rate)"
            )

        except Exception as e:
            logger.error(f"❌ Error collecting {city} data: {e}", exc_info=True)
            stats.errors.append(str(e))

            # Increment circuit breaker
            self.circuit_breaker_failures[city] = \
                self.circuit_breaker_failures.get(city, 0) + 1

        finally:
            stats.end_time = datetime.now()

        return stats

    async def run_collection_cycle(self) -> List[CollectionStats]:
        """Run one complete collection cycle across all cities"""

        logger.info(f"🚀 Starting collection cycle for {len(self.cities)} cities")
        cycle_stats = []

        for city in self.cities:
            try:
                stats = await self.collect_city_data(city)
                cycle_stats.append(stats)

                # Small delay between cities to avoid overwhelming the API
                await asyncio.sleep(5)

            except Exception as e:
                logger.error(f"Failed to collect {city}: {e}")

        # Log summary
        total_stored = sum(s.incidents_stored for s in cycle_stats)
        total_fetched = sum(s.incidents_fetched for s in cycle_stats)

        logger.info(
            f"📊 Cycle complete: {total_stored} incidents stored "
            f"from {total_fetched} fetched"
        )

        return cycle_stats

    async def run_scheduled_collection(self):
        """
        Main scheduler loop - runs collection cycles at regular intervals
        """

        if not self.db or not self.collector:
            await self.initialize()

        self.is_running = True
        logger.info(
            f"🕐 Smart scheduler started: Collection every "
            f"{self.collection_interval} minutes"
        )

        while self.is_running:
            try:
                # Run collection cycle
                stats = await self.run_collection_cycle()

                # Store collection metrics (could send to monitoring system)
                await self._log_cycle_metrics(stats)

                # Wait until next cycle
                logger.info(
                    f"⏳ Waiting {self.collection_interval} minutes "
                    f"until next cycle..."
                )
                await asyncio.sleep(self.collection_interval * 60)

            except Exception as e:
                logger.error(f"Scheduler error: {e}", exc_info=True)
                # Wait 5 minutes on error before retry
                await asyncio.sleep(300)

    async def _log_cycle_metrics(self, stats: List[CollectionStats]):
        """Log cycle metrics to database or monitoring system"""

        # Store in metrics table (would need to create this table)
        for stat in stats:
            logger.info(
                f"📈 Metrics [{stat.city}]: "
                f"Duration={stat.duration_seconds:.1f}s, "
                f"Fetched={stat.incidents_fetched}, "
                f"Stored={stat.incidents_stored}, "
                f"Success={stat.success_rate:.1f}%"
            )

    def stop(self):
        """Stop the scheduler gracefully"""
        logger.info("🛑 Stopping smart scheduler...")
        self.is_running = False


# Singleton instance
_scheduler_instance: Optional[SmartDataScheduler] = None


async def get_smart_scheduler() -> SmartDataScheduler:
    """Get or create scheduler instance"""
    global _scheduler_instance

    if _scheduler_instance is None:
        _scheduler_instance = SmartDataScheduler()
        await _scheduler_instance.initialize()

    return _scheduler_instance


async def start_scheduled_collection():
    """Start the smart scheduler (can be called from main.py)"""
    scheduler = await get_smart_scheduler()
    await scheduler.run_scheduled_collection()


if __name__ == "__main__":
    # Test the scheduler
    async def test_scheduler():
        scheduler = SmartDataScheduler(collection_interval_minutes=5)
        await scheduler.initialize()

        # Run one cycle
        stats = await scheduler.run_collection_cycle()

        print("\n📊 Collection Results:")
        for stat in stats:
            print(f"\n{stat.city}:")
            print(f"  Duration: {stat.duration_seconds:.1f}s")
            print(f"  Fetched: {stat.incidents_fetched}")
            print(f"  Stored: {stat.incidents_stored}")
            print(f"  Success Rate: {stat.success_rate:.1f}%")
            if stat.errors:
                print(f"  Errors: {', '.join(stat.errors)}")

    asyncio.run(test_scheduler())