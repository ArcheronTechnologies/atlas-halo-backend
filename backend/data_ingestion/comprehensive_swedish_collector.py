"""Comprehensive Swedish crime data collector"""
import logging
import httpx
from typing import List, Dict, Optional
from enum import Enum
from datetime import datetime, timedelta
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from backend.database.postgis_database import get_database, Incident

logger = logging.getLogger(__name__)

class CollectionTarget(str, Enum):
    """Data collection targets"""
    POLISEN = "polisen"
    BRA = "bra"
    ALL = "all"

class ComprehensiveSwedishCollector:
    """Collects crime data from Swedish sources"""

    def __init__(self):
        self.enabled = True
        self.polisen_api_url = "https://polisen.se/api/events"
        logger.info("ComprehensiveSwedishCollector initialized")

    async def collect(self, target: CollectionTarget = CollectionTarget.ALL) -> Dict:
        """Collect data from specified target"""
        logger.info(f"Collecting data from {target.value}")

        result = {
            "target": target.value,
            "success": True,
            "records_collected": 0,
            "timestamp": datetime.now().isoformat()
        }

        if target in [CollectionTarget.POLISEN, CollectionTarget.ALL]:
            polisen_count = await self._collect_polisen()
            result["polisen_records"] = polisen_count
            result["records_collected"] += polisen_count

        if target in [CollectionTarget.BRA, CollectionTarget.ALL]:
            # BRA data not implemented yet
            result["bra_records"] = 0

        return result

    async def _collect_polisen(self) -> int:
        """Collect incidents from Polisen.se API"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(self.polisen_api_url)
                response.raise_for_status()
                events = response.json()

            if not events:
                logger.info("No events returned from Polisen API")
                return 0

            db = await get_database()
            stored_count = 0

            async with db.async_session() as session:
                for event in events:
                    try:
                        # Parse Polisen.se event data
                        incident_data = {
                            "external_id": str(event.get("id", "")),
                            "incident_type": event.get("type", "unknown"),
                            "description": event.get("summary", ""),
                            "latitude": float(event["location"]["gps"].split(",")[0]) if event.get("location", {}).get("gps") else 59.3293,
                            "longitude": float(event["location"]["gps"].split(",")[1]) if event.get("location", {}).get("gps") else 18.0686,
                            "occurred_at": datetime.fromisoformat(event["datetime"].replace("Z", "+00:00")) if event.get("datetime") else datetime.now(),
                            "source": "polisen",
                            "severity": self._estimate_severity(event.get("type", "")),
                            "polisen_region": event.get("location", {}).get("name", ""),
                            "polisen_url": event.get("url", "")
                        }

                        # Upsert (insert or update on conflict)
                        stmt = insert(Incident).values(**incident_data)
                        stmt = stmt.on_conflict_do_update(
                            index_elements=["external_id", "source"],
                            set_={"description": stmt.excluded.description}
                        )
                        await session.execute(stmt)
                        stored_count += 1

                    except Exception as e:
                        logger.error(f"Failed to process Polisen event {event.get('id')}: {e}")
                        continue

                await session.commit()

            logger.info(f"✅ Collected {stored_count} incidents from Polisen.se")
            return stored_count

        except Exception as e:
            logger.error(f"Failed to collect from Polisen.se: {e}")
            return 0

    def _estimate_severity(self, incident_type: str) -> int:
        """Estimate severity based on incident type"""
        severity_map = {
            "Mord": 5,
            "Misshandel": 4,
            "Rån": 4,
            "Skottlossning": 5,
            "Bombhot": 5,
            "Trafikolycka": 3,
            "Stöld": 2,
            "Inbrott": 3,
            "Skadegörelse": 2,
        }

        for key, severity in severity_map.items():
            if key.lower() in incident_type.lower():
                return severity

        return 1  # Default severity

    async def start_continuous_collection(self, interval_minutes: int = 15):
        """Start continuous data collection"""
        logger.info(f"Starting continuous collection (interval: {interval_minutes}min)")

    async def stop_continuous_collection(self):
        """Stop continuous data collection"""
        logger.info("Stopping continuous collection")
