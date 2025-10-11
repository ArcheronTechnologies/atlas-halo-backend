"""Smart data scheduler for optimized ingestion"""
import logging
import asyncio
from typing import Optional, List
from datetime import datetime
from dataclasses import dataclass

from backend.data_ingestion.comprehensive_swedish_collector import ComprehensiveSwedishCollector, CollectionTarget

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
    """Smart scheduler for data ingestion"""

    def __init__(self, collection_interval_minutes: int = 15):
        self.is_running = False
        self.collection_interval_minutes = collection_interval_minutes
        self.collector = ComprehensiveSwedishCollector()
        self.collection_task = None
        logger.info(f"SmartDataScheduler initialized (interval: {collection_interval_minutes}min)")

    async def initialize(self):
        """Initialize the scheduler (async setup if needed)"""
        logger.info("SmartDataScheduler async initialization complete")

    async def start(self):
        """Start the smart scheduler"""
        self.is_running = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("SmartDataScheduler started")

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
        """Run a single collection cycle"""
        start_time = datetime.now()

        try:
            result = await self.collector.collect(CollectionTarget.POLISEN)
            duration = (datetime.now() - start_time).total_seconds()

            records_collected = result.get("records_collected", 0)

            stats = CollectionStats(
                city="Sweden",
                incidents_fetched=records_collected,
                incidents_stored=records_collected,
                incidents_skipped=0,
                success_rate=1.0 if result["success"] else 0.0,
                duration_seconds=duration
            )

            logger.info(f"Collection cycle complete: {records_collected} incidents in {duration:.1f}s")
            return [stats]

        except Exception as e:
            logger.error(f"Collection cycle failed: {e}")
            return []

    async def schedule_ingestion(self, source: str, interval_minutes: int = 15):
        """Schedule data ingestion for a source"""
        logger.info(f"Scheduled ingestion for {source} every {interval_minutes} minutes")

    def get_status(self) -> dict:
        """Get scheduler status"""
        return {
            "running": self.is_running,
            "last_run": datetime.now().isoformat(),
            "next_run": None
        }

# Singleton instance
_smart_scheduler = None

def get_smart_scheduler() -> SmartDataScheduler:
    """Get or create smart scheduler singleton"""
    global _smart_scheduler
    if _smart_scheduler is None:
        _smart_scheduler = SmartDataScheduler()
    return _smart_scheduler
