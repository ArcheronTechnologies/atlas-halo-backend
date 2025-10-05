"""
Background Data Ingestion Service
Runs continuous data collection from Swedish police and other sources
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

from ..data_ingestion.smart_scheduler import SmartDataScheduler, get_smart_scheduler
from ..database.postgis_database import get_database

logger = logging.getLogger(__name__)


@dataclass
class IngestionConfig:
    """Configuration for data ingestion service"""
    collection_interval_minutes: int = 15  # Collect every 15 minutes
    enabled: bool = True
    cities: list[str] = None

    def __post_init__(self):
        if self.cities is None:
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


class DataIngestionService:
    """
    Background service that continuously collects crime data from Swedish sources

    Features:
    - Runs SmartDataScheduler in background
    - Automatic deduplication and data quality checks
    - Graceful shutdown handling
    - Health monitoring and metrics
    """

    def __init__(self, config: Optional[IngestionConfig] = None):
        self.config = config or IngestionConfig()
        self.scheduler: Optional[SmartDataScheduler] = None
        self.is_running = False
        self.task: Optional[asyncio.Task] = None
        self.last_collection_time: Optional[datetime] = None
        self.total_collections = 0
        self.total_incidents_collected = 0

    async def start(self):
        """Start the background data ingestion service"""
        if self.is_running:
            logger.warning("Data ingestion service already running")
            return

        if not self.config.enabled:
            logger.info("Data ingestion service disabled in config")
            return

        logger.info(f"🚀 Starting Data Ingestion Service (interval: {self.config.collection_interval_minutes}min)")

        # Initialize scheduler
        self.scheduler = SmartDataScheduler(
            collection_interval_minutes=self.config.collection_interval_minutes
        )
        await self.scheduler.initialize()

        # Start background task
        self.is_running = True
        self.task = asyncio.create_task(self._run_collection_loop())

        logger.info("✅ Data Ingestion Service started successfully")

    async def stop(self):
        """Stop the background data ingestion service"""
        if not self.is_running:
            return

        logger.info("🛑 Stopping Data Ingestion Service...")

        self.is_running = False

        if self.scheduler:
            self.scheduler.stop()

        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

        logger.info("✅ Data Ingestion Service stopped")

    async def _run_collection_loop(self):
        """Main collection loop - runs continuously"""
        try:
            while self.is_running:
                try:
                    logger.info(f"🔄 Starting collection cycle #{self.total_collections + 1}")

                    # Run collection cycle
                    start_time = datetime.now()
                    stats = await self.scheduler.run_collection_cycle()

                    # Update metrics
                    self.last_collection_time = datetime.now()
                    self.total_collections += 1

                    # Calculate total incidents from this cycle
                    cycle_incidents = sum(s.incidents_stored for s in stats)
                    self.total_incidents_collected += cycle_incidents

                    duration = (datetime.now() - start_time).total_seconds()

                    logger.info(
                        f"✅ Collection cycle #{self.total_collections} complete: "
                        f"{cycle_incidents} incidents in {duration:.1f}s"
                    )

                    # Wait until next cycle
                    logger.info(
                        f"⏳ Next collection in {self.config.collection_interval_minutes} minutes..."
                    )
                    await asyncio.sleep(self.config.collection_interval_minutes * 60)

                except Exception as e:
                    logger.error(f"❌ Collection cycle error: {e}", exc_info=True)
                    # Wait 5 minutes before retry on error
                    await asyncio.sleep(300)

        except asyncio.CancelledError:
            logger.info("Collection loop cancelled")
            raise

    async def get_status(self) -> dict:
        """Get current status of the ingestion service"""
        return {
            "enabled": self.config.enabled,
            "running": self.is_running,
            "collection_interval_minutes": self.config.collection_interval_minutes,
            "total_collections": self.total_collections,
            "total_incidents_collected": self.total_incidents_collected,
            "last_collection_time": self.last_collection_time.isoformat() if self.last_collection_time else None,
            "cities": self.config.cities
        }

    async def trigger_immediate_collection(self) -> dict:
        """Manually trigger an immediate collection cycle"""
        if not self.scheduler:
            raise RuntimeError("Scheduler not initialized")

        logger.info("🔄 Manual collection triggered")
        stats = await self.scheduler.run_collection_cycle()

        return {
            "success": True,
            "cities_collected": len(stats),
            "total_incidents": sum(s.incidents_stored for s in stats),
            "stats": [
                {
                    "city": s.city,
                    "fetched": s.incidents_fetched,
                    "stored": s.incidents_stored,
                    "skipped": s.incidents_skipped,
                    "success_rate": s.success_rate,
                    "duration_seconds": s.duration_seconds
                }
                for s in stats
            ]
        }


# Singleton instance
_service_instance: Optional[DataIngestionService] = None


async def get_ingestion_service(config: Optional[IngestionConfig] = None) -> DataIngestionService:
    """Get or create the data ingestion service singleton"""
    global _service_instance

    if _service_instance is None:
        _service_instance = DataIngestionService(config)

    return _service_instance


async def start_ingestion_service(config: Optional[IngestionConfig] = None):
    """Start the data ingestion service (called from main.py)"""
    service = await get_ingestion_service(config)
    await service.start()


async def stop_ingestion_service():
    """Stop the data ingestion service (called from main.py)"""
    if _service_instance:
        await _service_instance.stop()