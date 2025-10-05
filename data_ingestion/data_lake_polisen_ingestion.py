"""
Data Lake Polisen.se Ingestion
Centralized data collection for all Atlas AI platforms
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta, timezone
import random
from typing import Dict, List, Optional, Any, Tuple
import json
import time
from pathlib import Path
import hashlib
from uuid import uuid4

from ..database.postgis_database import PostGISDatabase, get_database
from sqlalchemy import text

logger = logging.getLogger(__name__)


class DataLakePolisenIngestion:
    """Centralized Polisen.se data ingestion for the data lake"""

    def __init__(self, database: PostGISDatabase):
        self.db = database
        self.base_url = "https://polisen.se/api/events"
        self.user_agents = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        ]

    async def fetch_polisen_events(
        self,
        limit: int = 500,
        offset: int = 0,
        location_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch events from Polisen API with rate limiting"""

        params = {
            'limit': limit,
            'offset': offset
        }

        if location_type:
            params['locationtype'] = location_type

        headers = {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'application/json',
            'Accept-Language': 'sv-SE,sv;q=0.9,en;q=0.8'
        }

        try:
            async with aiohttp.ClientSession() as session:
                await asyncio.sleep(random.uniform(1.0, 3.0))  # Rate limiting

                async with session.get(
                    self.base_url,
                    params=params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:

                    if response.status == 200:
                        data = await response.json()
                        events = data if isinstance(data, list) else []
                        logger.info(f"✅ Fetched {len(events)} events from Polisen API")
                        return events
                    else:
                        logger.error(f"❌ API request failed with status {response.status}")
                        return []

        except Exception as e:
            logger.error(f"❌ Error fetching Polisen events: {e}")
            return []

    def calculate_data_quality_score(self, event: Dict[str, Any]) -> float:
        """Calculate data quality score based on completeness and reliability"""

        score = 0.0
        max_score = 10.0

        # Basic required fields
        if event.get('id'):
            score += 1.0
        if event.get('datetime'):
            score += 1.0
        if event.get('name'):
            score += 1.0
        if event.get('type'):
            score += 1.0

        # Location quality
        location = event.get('location', {})
        if location.get('name'):
            score += 1.0

        if location.get('gps'):
            gps = location['gps'].split(',')
            if len(gps) == 2:
                try:
                    lat, lon = float(gps[0]), float(gps[1])
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        score += 2.0  # GPS coordinates are very valuable
                except (ValueError, TypeError):
                    pass

        # Content quality
        summary = event.get('summary', '')
        if len(summary) > 20:
            score += 1.0
        if len(summary) > 100:
            score += 1.0

        # URL presence (indicates more details available)
        if event.get('url'):
            score += 1.0

        return min(score / max_score, 1.0)

    def normalize_incident_type(self, polisen_type: str) -> str:
        """Normalize Polisen crime types to standard categories"""

        type_lower = polisen_type.lower()

        # Direct mappings
        type_mappings = {
            'stöld': 'theft',
            'rån': 'robbery',
            'misshandel': 'assault',
            'våldtäkt': 'violence',
            'skadegörelse': 'vandalism',
            'narkotika': 'drug_activity',
            'narkotikabrott': 'drug_activity',
            'trafikolycka': 'traffic_accident',
            'brand': 'fire',
            'mord': 'violence',
            'dråp': 'violence',
            'hot': 'threat',
            'bedrägeri': 'fraud',
            'inbrott': 'theft',
            'våld': 'violence'
        }

        # Check for exact matches first
        for swedish_term, english_type in type_mappings.items():
            if swedish_term in type_lower:
                return english_type

        # Fallback patterns
        if any(word in type_lower for word in ['stöld', 'tjuvnad', 'snatteri']):
            return 'theft'
        elif any(word in type_lower for word in ['våld', 'misshandel', 'slagsmål']):
            return 'assault'
        elif any(word in type_lower for word in ['trafik', 'olycka', 'krock']):
            return 'traffic_accident'
        elif any(word in type_lower for word in ['narkotika', 'droger', 'amfetamin']):
            return 'drug_activity'
        elif any(word in type_lower for word in ['brand', 'eld']):
            return 'fire'

        return 'other'

    def calculate_severity_level(self, event: Dict[str, Any], incident_type: str) -> str:
        """Calculate severity level based on incident type and content"""

        summary = event.get('summary', '').lower()
        event_type = event.get('type', '').lower()

        # Critical incidents
        critical_keywords = ['mord', 'dråp', 'skjuten', 'död', 'explosion', 'dödsfall', 'terrorist']
        if any(keyword in summary or keyword in event_type for keyword in critical_keywords):
            return 'critical'

        # High severity incidents
        high_keywords = ['rån', 'våldtäkt', 'misshandel', 'grov', 'kniv', 'vapen', 'sjukhus']
        if any(keyword in summary or keyword in event_type for keyword in high_keywords):
            return 'high'

        # Type-based severity
        if incident_type in ['violence', 'robbery', 'assault']:
            return 'high'
        elif incident_type in ['theft', 'vandalism', 'drug_activity']:
            return 'moderate'
        elif incident_type in ['traffic_accident']:
            # Check if serious
            if any(word in summary for word in ['död', 'svårt', 'allvarlig', 'sjukhus']):
                return 'high'
            return 'moderate'

        return 'low'

    async def store_raw_incident(self, event: Dict[str, Any]) -> Optional[str]:
        """Store raw incident data in the data lake"""

        try:
            # Extract basic information
            polisen_id = str(event.get('id', ''))
            if not polisen_id:
                return None

            # Check if already exists
            existing_query = """
                SELECT id FROM data_lake.raw_incidents
                WHERE source = 'polisen' AND source_id = $1
            """
            existing = await self.db.execute_query(existing_query, polisen_id)
            if existing:
                logger.debug(f"Event {polisen_id} already exists in data lake")
                return None

            # Parse location
            location = event.get('location', {})
            latitude, longitude = None, None
            gps_coords = location.get('gps', '')

            if gps_coords:
                try:
                    lat_str, lon_str = gps_coords.split(',')
                    latitude = float(lat_str.strip())
                    longitude = float(lon_str.strip())
                except (ValueError, TypeError, AttributeError):
                    logger.warning(f"Invalid GPS coordinates for event {polisen_id}: {gps_coords}")

            # Parse datetime
            incident_time = None
            try:
                datetime_str = event.get('datetime', '')
                if datetime_str:
                    # Parse Polisen datetime format
                    incident_time = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            except (ValueError, TypeError):
                logger.warning(f"Invalid datetime for event {polisen_id}: {event.get('datetime')}")

            # Calculate metrics
            data_quality_score = self.calculate_data_quality_score(event)
            incident_type = self.normalize_incident_type(event.get('type', ''))
            severity_level = self.calculate_severity_level(event, incident_type)

            # Create geography point if coordinates available
            location_geography = None
            if latitude is not None and longitude is not None:
                location_geography = f"POINT({longitude} {latitude})"

            # Insert into raw_incidents
            if location_geography:
                insert_query = """
                    INSERT INTO data_lake.raw_incidents (
                        source, source_id, raw_data, location, location_address,
                        incident_timestamp, reported_timestamp, incident_type,
                        severity_level, data_quality_score, processing_status
                    ) VALUES (
                        $1, $2, $3, ST_GeogFromText($4), $5, $6, $7, $8, $9, $10, $11
                    ) RETURNING id
                """
                params = [
                    'polisen',
                    polisen_id,
                    json.dumps(event),
                    location_geography,
                    location.get('name', ''),
                    incident_time,
                    datetime.now(timezone.utc),
                    incident_type,
                    severity_level,
                    data_quality_score,
                    'pending'
                ]
            else:
                insert_query = """
                    INSERT INTO data_lake.raw_incidents (
                        source, source_id, raw_data, location, location_address,
                        incident_timestamp, reported_timestamp, incident_type,
                        severity_level, data_quality_score, processing_status
                    ) VALUES (
                        $1, $2, $3, NULL, $4, $5, $6, $7, $8, $9, $10
                    ) RETURNING id
                """
                params = [
                    'polisen',
                    polisen_id,
                    json.dumps(event),
                    location.get('name', ''),
                    incident_time,
                    datetime.now(timezone.utc),
                    incident_type,
                    severity_level,
                    data_quality_score,
                    'pending'
                ]

            result = await self.db.execute_query(insert_query, *params)
            if result:
                raw_incident_id = result[0]['id']
                logger.debug(f"✅ Stored raw incident {polisen_id} as {raw_incident_id}")

                # Process the raw incident immediately
                await self.process_raw_incident(raw_incident_id)

                return raw_incident_id

        except Exception as e:
            logger.error(f"❌ Error storing event {polisen_id}: {e}")
            return None

    async def process_raw_incident(self, raw_incident_id: str) -> bool:
        """Process a raw incident into the processed incidents table"""

        try:
            # Call the processing function
            process_query = "SELECT data_lake.process_raw_incident($1)"
            result = await self.db.execute_query(process_query, raw_incident_id)

            if result and len(result) > 0:
                # The result is a list with a dict containing the function result
                process_result = result[0].get('process_raw_incident', False)
                if process_result is True:
                    logger.debug(f"✅ Processed raw incident {raw_incident_id}")
                    return True
                else:
                    logger.warning(f"⚠️ Failed to process raw incident {raw_incident_id}: function returned {process_result}")
                    return False
            else:
                logger.warning(f"⚠️ No result from processing function for raw incident {raw_incident_id}")
                return False

        except Exception as e:
            logger.error(f"❌ Error processing raw incident {raw_incident_id}: {e}")
            return False

    async def ingest_current_data(self, limit: int = 500) -> Dict[str, int]:
        """Ingest current data from Polisen API"""

        logger.info("🚀 Starting Polisen.se data lake ingestion")

        try:
            # Fetch events from API
            events = await self.fetch_polisen_events(limit=limit)

            if not events:
                logger.warning("⚠️ No events fetched from API")
                return {"fetched": 0, "stored": 0, "processed": 0}

            # Store events in data lake
            stored_count = 0
            processed_count = 0

            for event in events:
                raw_id = await self.store_raw_incident(event)
                if raw_id:
                    stored_count += 1
                    processed_count += 1

            logger.info(f"✅ Ingestion complete: fetched={len(events)}, stored={stored_count}, processed={processed_count}")

            return {
                "fetched": len(events),
                "stored": stored_count,
                "processed": processed_count
            }

        except Exception as e:
            logger.error(f"❌ Data ingestion failed: {e}")
            return {"fetched": 0, "stored": 0, "processed": 0, "error": str(e)}

    async def continuous_ingestion(self, interval_minutes: int = 15):
        """Run continuous data ingestion"""

        logger.info(f"🔄 Starting continuous ingestion every {interval_minutes} minutes")

        while True:
            try:
                start_time = time.time()

                # Perform ingestion
                results = await self.ingest_current_data()

                # Log results
                elapsed = time.time() - start_time
                logger.info(f"📊 Ingestion cycle complete in {elapsed:.1f}s: {results}")

                # Wait for next cycle
                await asyncio.sleep(interval_minutes * 60)

            except KeyboardInterrupt:
                logger.info("🛑 Continuous ingestion stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in continuous ingestion: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry


async def get_polisen_ingestion() -> DataLakePolisenIngestion:
    """Get Polisen ingestion instance"""
    db = await get_database()
    return DataLakePolisenIngestion(db)


async def run_single_ingestion():
    """Run a single ingestion cycle"""
    ingestion = await get_polisen_ingestion()
    return await ingestion.ingest_current_data()


async def run_continuous_ingestion(interval_minutes: int = 15):
    """Run continuous ingestion"""
    ingestion = await get_polisen_ingestion()
    await ingestion.continuous_ingestion(interval_minutes)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "continuous":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 15
        asyncio.run(run_continuous_ingestion(interval))
    else:
        asyncio.run(run_single_ingestion())