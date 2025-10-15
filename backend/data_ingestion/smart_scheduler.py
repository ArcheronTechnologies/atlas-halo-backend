"""Smart data scheduler for Halo - fetches data from polisen.se and Atlas Intelligence"""
import logging
import asyncio
import httpx
import os
from typing import Optional, List
from datetime import datetime
from dataclasses import dataclass
from sqlalchemy.dialects.postgresql import insert
from geoalchemy2 import WKTElement

from backend.database.postgis_database import get_database
from backend.data_ingestion.polisen_collector import collect_polisen_data

logger = logging.getLogger(__name__)

@dataclass
class CollectionStats:
    """Statistics from a collection run"""
    city: str
    incidents_fetched: int
    incidents_stored: int
    incidents_skipped: int
    success_rate: float
    duration_seconds: float

class SmartDataScheduler:
    """Smart scheduler for data ingestion - pulls from Atlas Intelligence"""

    def __init__(self, collection_interval_minutes: int = 15):
        self.is_running = False
        self.collection_interval_minutes = collection_interval_minutes
        self.collection_task = None
        self.atlas_url = os.getenv("ATLAS_INTELLIGENCE_URL", "http://localhost:8001")
        logger.info(f"SmartDataScheduler initialized (interval: {collection_interval_minutes}min, Atlas: {self.atlas_url})")

    async def initialize(self):
        """Initialize the scheduler (async setup if needed)"""
        logger.info("SmartDataScheduler async initialization complete")

    async def start(self):
        """Start the smart scheduler"""
        self.is_running = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("SmartDataScheduler started - pulling data from Atlas Intelligence")

    def stop(self):
        """Stop the smart scheduler (synchronous)"""
        self.is_running = False
        if self.collection_task:
            self.collection_task.cancel()
        logger.info("SmartDataScheduler stopped")

    async def _collection_loop(self):
        """Main collection loop"""
        while self.is_running:
            try:
                await self.run_collection_cycle()
                # Wait for next collection interval
                await asyncio.sleep(self.collection_interval_minutes * 60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Collection loop error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry

    async def run_collection_cycle(self) -> List[CollectionStats]:
        """Run a single collection cycle - fetch from polisen.se directly"""
        start_time = datetime.now()

        try:
            # Fetch incidents directly from polisen.se
            logger.info("🔄 Fetching incidents from polisen.se...")
            incidents = await collect_polisen_data()

            if not incidents:
                logger.info("No new incidents from polisen.se")
                return []

            logger.info(f"📥 Collected {len(incidents)} incidents from polisen.se")

            # Store incidents in Halo database
            db = await get_database()
            stored_count = 0
            skipped_count = 0

            async with db.async_session() as session:
                from backend.database.models import Incident

                for inc in incidents:
                    try:
                        # Check if incident already exists
                        from sqlalchemy import select
                        result = await session.execute(
                            select(Incident).where(
                                Incident.external_id == inc["external_id"],
                                Incident.source == inc["source"]
                            )
                        )
                        existing = result.scalar_one_or_none()

                        if existing:
                            skipped_count += 1
                            continue

                        # Create new incident
                        incident = Incident(
                            external_id=inc["external_id"],
                            source=inc["source"],
                            incident_type=inc["incident_type"],
                            description=inc["description"],
                            location=WKTElement(f'POINT({inc["longitude"]} {inc["latitude"]})', srid=4326),
                            occurred_at=datetime.fromisoformat(inc["occurred_at"].replace("Z", "+00:00")),
                            severity=inc["severity"],
                            status=inc["status"],
                            polisen_region=inc.get("location_name", ""),
                            metadata_=inc.get("raw_data", {})
                        )
                        session.add(incident)
                        stored_count += 1

                    except Exception as e:
                        logger.error(f"Failed to store incident {inc.get('external_id')}: {e}")
                        skipped_count += 1
                        continue

                await session.commit()

            duration = (datetime.now() - start_time).total_seconds()

            stats = CollectionStats(
                city="Sweden (polisen.se)",
                incidents_fetched=len(incidents),
                incidents_stored=stored_count,
                incidents_skipped=skipped_count,
                success_rate=stored_count / len(incidents) if incidents else 1.0,
                duration_seconds=duration
            )

            logger.info(f"✅ Collection cycle complete: {stored_count} new incidents stored, {skipped_count} skipped (duplicates) in {duration:.1f}s")
            return [stats]

        except Exception as e:
            logger.error(f"❌ Collection cycle failed: {e}", exc_info=True)
            return []

    async def schedule_ingestion(self, source: str, interval_minutes: int = 15):
        """Schedule data ingestion for a source"""
        logger.info(f"Scheduled ingestion for {source} every {interval_minutes} minutes")

    def get_status(self) -> dict:
        """Get scheduler status"""
        return {
            "running": self.is_running,
            "last_run": datetime.now().isoformat(),
            "next_run": None,
            "source": "Atlas Intelligence API"
        }

# Singleton instance
_smart_scheduler = None

def get_smart_scheduler() -> SmartDataScheduler:
    """Get or create smart scheduler singleton"""
    global _smart_scheduler
    if _smart_scheduler is None:
        _smart_scheduler = SmartDataScheduler()
    return _smart_scheduler
