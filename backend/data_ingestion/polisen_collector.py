"""
Polisen.se Data Collector
Fetches crime incidents from the Swedish Police public API
"""
import asyncio
import aiohttp
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import logging
import re

logger = logging.getLogger(__name__)


class PolisenCollector:
    """Collects incident data from polisen.se API"""

    BASE_URL = "https://polisen.se/api/events"

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_recent_incidents(self, hours_back: int = 24) -> List[Dict]:
        """
        Fetch recent incidents from polisen.se API

        Args:
            hours_back: How many hours back to fetch incidents

        Returns:
            List of incident dictionaries
        """
        if not self.session:
            raise RuntimeError("Collector not initialized. Use async with statement.")

        try:
            params = {}
            url = self.BASE_URL

            logger.info(f"Fetching incidents from polisen.se (last {hours_back} hours)")

            async with self.session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch from polisen.se: HTTP {response.status}")
                    return []

                data = await response.json()

                # Filter by time (use timezone-aware datetime)
                cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
                recent_incidents = []

                for incident in data:
                    # Parse datetime from polisen.se format
                    try:
                        dt_str = incident['datetime']
                        # Fix format: '2025-10-14 7:48:03 +02:00' -> '2025-10-14T07:48:03+02:00'
                        dt_str = re.sub(r'(\d{4}-\d{2}-\d{2}) (\d+):(\d{2}):(\d{2}) ([+-]\d{2}:\d{2})',
                                       r'\1T\2:\3:\4\5', dt_str)
                        # Ensure hour is zero-padded
                        dt_str = re.sub(r'T(\d):',r'T0\1:', dt_str)

                        incident_time = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
                        if incident_time >= cutoff_time:
                            recent_incidents.append(incident)
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Could not parse incident datetime: {e}")
                        continue

                logger.info(f"✅ Fetched {len(recent_incidents)} recent incidents from polisen.se")
                return recent_incidents

        except asyncio.TimeoutError:
            logger.error("Timeout fetching from polisen.se")
            return []
        except Exception as e:
            logger.error(f"Error fetching from polisen.se: {e}")
            return []

    def normalize_incident(self, raw_incident: Dict) -> Optional[Dict]:
        """
        Normalize a polisen.se incident to our schema

        Args:
            raw_incident: Raw incident from polisen.se API

        Returns:
            Normalized incident dict or None if invalid
        """
        try:
            # Extract location
            location = raw_incident.get('location', {})
            lat = location.get('gps') and float(location['gps'].split(',')[0])
            lon = location.get('gps') and float(location['gps'].split(',')[1])

            if not lat or not lon:
                logger.warning(f"Incident missing GPS coordinates: {raw_incident.get('id')}")
                return None

            # Map polisen.se type to our incident type
            polisen_type = raw_incident.get('type', 'Övrigt').lower()
            incident_type = self._map_incident_type(polisen_type)

            # Calculate severity based on type
            severity = self._calculate_severity(polisen_type, raw_incident.get('summary', ''))

            normalized = {
                'external_id': str(raw_incident.get('id')),
                'source': 'polisen',
                'incident_type': incident_type,
                'severity': severity,
                'latitude': lat,
                'longitude': lon,
                'location_name': location.get('name', 'Unknown'),
                'description': raw_incident.get('summary', ''),
                'occurred_at': raw_incident.get('datetime'),
                'raw_data': raw_incident,
                'status': 'verified'
            }

            return normalized

        except Exception as e:
            logger.error(f"Error normalizing incident: {e}")
            return None

    def _map_incident_type(self, polisen_type: str) -> str:
        """Map polisen.se incident type to our schema"""
        type_mapping = {
            'rån': 'robbery',
            'skottlossning': 'shooting',
            'brand': 'fire',
            'trafikolycka': 'accident',
            'misshandel': 'assault',
            'stöld': 'theft',
            'narkotikabrott': 'drug_activity',
            'hot': 'threat',
            'våld': 'violence'
        }

        for key, value in type_mapping.items():
            if key in polisen_type:
                return value

        return 'other'

    def _calculate_severity(self, polisen_type: str, summary: str) -> str:
        """Calculate severity based on incident type and description"""
        high_severity_keywords = ['skjuten', 'död', 'skottlossning', 'mord', 'rån', 'grov']
        moderate_severity_keywords = ['misshandel', 'hot', 'våld', 'brand']

        summary_lower = summary.lower()
        polisen_lower = polisen_type.lower()

        if any(kw in summary_lower or kw in polisen_lower for kw in high_severity_keywords):
            return 'high'
        elif any(kw in summary_lower or kw in polisen_lower for kw in moderate_severity_keywords):
            return 'moderate'
        else:
            return 'low'


async def collect_polisen_data() -> List[Dict]:
    """
    Main function to collect and normalize polisen.se data

    Returns:
        List of normalized incidents ready for database insertion
    """
    async with PolisenCollector() as collector:
        # Fetch last 1 hour of incidents (runs every 15 minutes, so catch anything new)
        raw_incidents = await collector.fetch_recent_incidents(hours_back=1)

        # Normalize incidents
        normalized_incidents = []
        for raw in raw_incidents:
            normalized = collector.normalize_incident(raw)
            if normalized:
                normalized_incidents.append(normalized)

        logger.info(f"Collected and normalized {len(normalized_incidents)} incidents")
        return normalized_incidents


if __name__ == "__main__":
    # Test the collector
    logging.basicConfig(level=logging.INFO)
    asyncio.run(collect_polisen_data())
