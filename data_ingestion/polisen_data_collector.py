"""
Atlas AI - Swedish Police Data Collector
Real-time data ingestion from polisen.se API and other Swedish crime data sources
"""

import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import time
from dataclasses import dataclass, asdict
import pandas as pd
from urllib.parse import urlencode

from ..database.postgis_database import get_database
from ..caching.redis_cache import get_cache
from ..observability.metrics import metrics

logger = logging.getLogger(__name__)

@dataclass
class PolisenIncident:
    """Swedish police incident data structure"""
    id: str
    datetime: str
    name: str
    summary: str
    type: str
    location: Dict[str, Any]  # name, gps coordinates
    url: str
    raw_data: Dict[str, Any]

@dataclass
class DataCollectionStats:
    """Statistics for data collection operations"""
    incidents_collected: int
    incidents_processed: int
    incidents_stored: int
    errors: int
    collection_start: datetime
    collection_end: Optional[datetime] = None
    processing_time_seconds: float = 0.0

class PolisenDataCollector:
    """
    Collects real crime data from Swedish Police (polisen.se) API
    """
    
    def __init__(self):
        self.base_url = "https://polisen.se/api"
        self.session: Optional[aiohttp.ClientSession] = None
        self.is_collecting = False
        self.collection_stats = DataCollectionStats(0, 0, 0, 0, datetime.now())
        
        # Rate limiting
        self.rate_limit_calls = 0
        self.rate_limit_window = 60  # seconds
        self.max_calls_per_window = 100
        self.last_rate_limit_reset = time.time()
        
        # Incident type mapping
        self.incident_type_mapping = {
            'Anträffad död': 'death',
            'Anträffat gods': 'found_property',
            'Arbetsplatsolycka': 'workplace_accident',
            'Bedrägeri': 'fraud',
            'Bilbrand': 'vehicle_fire',
            'Bomhot': 'bomb_threat',
            'Brand': 'fire',
            'Brand automatlarm': 'fire_alarm',
            'Bråk': 'disturbance',
            'Butiksstöld': 'shoplifting',
            'Cykelstöld': 'bicycle_theft',
            'Detonation': 'explosion',
            'Djur skadat/dött': 'animal_incident',
            'Drogbrott': 'drug_crime',
            'Ekobrott': 'economic_crime',
            'Fjällräddning': 'mountain_rescue',
            'Fylleri': 'public_intoxication',
            'Förfalskningsbrott': 'forgery',
            'Grov stöld': 'grand_theft',
            'Grovt vårdslöshetsbrott': 'gross_negligence',
            'Hets mot folkgrupp': 'hate_crime',
            'Inbrott': 'burglary',
            'Inbrott, försök': 'attempted_burglary',
            'Knivlagen': 'knife_law_violation',
            'Kontroll person/fordon': 'person_vehicle_check',
            'Kreatursstöld': 'livestock_theft',
            'Kreditkortsbedrägeri': 'credit_card_fraud',
            'Kuststöld': 'coastal_theft',
            'Luftfartsolycka': 'aviation_accident',
            'Miljöbrott': 'environmental_crime',
            'Misshandel': 'assault',
            'Misshandel, grov': 'aggravated_assault',
            'Mord/dråp': 'murder_manslaughter',
            'Mord/dråp, försök': 'attempted_murder',
            'Motorfordon, anträffat stulet': 'recovered_stolen_vehicle',
            'Motorfordon, stöld': 'vehicle_theft',
            'Motorfordon, tillgrepp': 'vehicle_taking',
            'Narkotikabrott': 'narcotics_crime',
            'Naturkatastrof': 'natural_disaster',
            'Ofredande': 'harassment',
            'Olaga frihetsberövande': 'unlawful_detention',
            'Olaga hot': 'unlawful_threat',
            'Olaga intrång': 'trespassing',
            'Olaga vapenbehav': 'illegal_weapons',
            'Olovlig körning': 'unlicensed_driving',
            'Ordningslagen': 'public_order_violation',
            'Personrån': 'robbery',
            'Rattfylleri': 'drunk_driving',
            'Rån': 'robbery',
            'Rån mot butik': 'shop_robbery',
            'Rån, övrigt': 'other_robbery',
            'Rån övrigt': 'other_robbery',
            'Räddningsinsats': 'rescue_operation',
            'Sammanfattning dag': 'daily_summary',
            'Sammanfattning helg': 'weekend_summary',
            'Sammanfattning kväll': 'evening_summary',
            'Sammanfattning natt': 'night_summary',
            'Sammanfattning vecka': 'weekly_summary',
            'Sedlighetsbrott': 'sexual_crime',
            'Sjukdom/olycksfall': 'illness_accident',
            'Sjölagen': 'maritime_law_violation',
            'Skadegörelse': 'vandalism',
            'Skottlossning': 'shooting',
            'Skottlossning, misstänkt': 'suspected_shooting',
            'Spritkörning': 'drunk_driving',
            'Stöld': 'theft',
            'Stöld/inbrott': 'theft_burglary',
            'Stöld, försök': 'attempted_theft',
            'Stöld, ringa': 'petty_theft',
            'Trafikbrott': 'traffic_crime',
            'Trafikolycka': 'traffic_accident',
            'Trafikolycka, personskada': 'traffic_accident_injury',
            'Trafikolycka, singel': 'single_vehicle_accident',
            'Trafikolycka, smitning': 'hit_and_run',
            'Trafikolycka, vilt': 'wildlife_accident',
            'Upphittad avliden person': 'found_deceased',
            'Våld/hot mot tjänsteman': 'violence_threat_official',
            'Våldtäkt': 'rape',
            'Våldtäkt, försök': 'attempted_rape',
            'Vårdslöshet i trafik': 'traffic_negligence',
            'Vållande till kroppsskada': 'causing_bodily_harm',
            'Vållande till annans död': 'causing_death',
            'Åldringsbrott': 'elderly_crime'
        }
    
    async def initialize(self):
        """Initialize the data collector"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'Atlas-AI/1.0 Crime Prevention System'}
        )
        logger.info("✅ Polisen data collector initialized")
    
    async def collect_events(
        self,
        location_name: Optional[str] = None,
        type_filter: Optional[List[str]] = None,
        max_events: int = 1000,
        days_back: int = 30
    ) -> List[PolisenIncident]:
        """
        Collect events from Polisen API

        Args:
            location_name: Filter by location (e.g., "Stockholm", "Göteborg")
            type_filter: Filter by incident types
            max_events: Maximum number of events to collect
            days_back: How many days back to collect data
        """
        if self.is_collecting:
            logger.warning("⚠️ Data collection already in progress")
            return []

        # Ensure session is initialized
        if not self.session:
            await self.initialize()

        try:
            self.is_collecting = True
            self.collection_stats = DataCollectionStats(0, 0, 0, 0, datetime.now())
            
            logger.info(f"🚀 Starting data collection from Polisen API...")
            logger.info(f"📍 Location filter: {location_name or 'All Sweden'}")
            logger.info(f"📅 Collecting data from last {days_back} days")
            
            all_incidents = []
            
            # Collect data day by day to avoid overwhelming the API
            # Start from yesterday to avoid API issues with current/future dates
            for day_offset in range(1, days_back + 1):
                if len(all_incidents) >= max_events:
                    break

                collection_date = datetime.now() - timedelta(days=day_offset)
                date_str = collection_date.strftime("%Y-%m-%d")
                
                logger.info(f"📅 Collecting data for {date_str}...")
                
                daily_incidents = await self.collect_events_for_date(
                    date_str, location_name, type_filter
                )
                
                all_incidents.extend(daily_incidents)
                self.collection_stats.incidents_collected += len(daily_incidents)
                
                # Rate limiting: wait between requests
                await asyncio.sleep(1.0)
                
                if len(daily_incidents) == 0:
                    logger.info(f"📅 No incidents found for {date_str}")
                else:
                    logger.info(f"📅 Collected {len(daily_incidents)} incidents for {date_str}")
            
            self.collection_stats.collection_end = datetime.now()
            self.collection_stats.processing_time_seconds = (
                self.collection_stats.collection_end - self.collection_stats.collection_start
            ).total_seconds()
            
            logger.info(f"✅ Data collection completed!")
            logger.info(f"📊 Collected {len(all_incidents)} incidents in {self.collection_stats.processing_time_seconds:.1f}s")
            
            return all_incidents[:max_events]
            
        except Exception as e:
            logger.error(f"❌ Error during data collection: {e}")
            self.collection_stats.errors += 1
            return []
        finally:
            self.is_collecting = False
    
    async def collect_events_for_date(
        self,
        date: str,
        location_name: Optional[str] = None,
        type_filter: Optional[List[str]] = None
    ) -> List[PolisenIncident]:
        """Collect events for a specific date"""
        try:
            # Ensure session is initialized
            if not self.session:
                await self.initialize()

            await self.check_rate_limit()

            # Build query parameters - don't use DateTime as API doesn't support it
            params = {}
            if location_name:
                params['locationname'] = location_name

            # Make API request
            url = f"{self.base_url}/events"

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    # Handle case where API returns None or empty
                    if data is None:
                        logger.debug(f"ℹ️  API returned None for {date} (date may be in future or no data available)")
                        return []

                    if not isinstance(data, list):
                        logger.warning(f"⚠️ API returned invalid data type: {type(data)}")
                        return []

                    if len(data) == 0:
                        logger.debug(f"ℹ️  No incidents found for {date}")
                        return []

                    incidents = []

                    for event_data in data:
                        # Skip null/None events
                        if not event_data or not isinstance(event_data, dict):
                            continue

                        try:
                            incident = await self.parse_polisen_event(event_data)

                            # Apply type filter if specified
                            if type_filter and incident.type not in type_filter:
                                continue

                            incidents.append(incident)

                        except Exception as e:
                            logger.debug(f"⚠️ Error parsing event {event_data.get('id', 'unknown') if isinstance(event_data, dict) else 'unknown'}: {e}")
                            self.collection_stats.errors += 1

                    return incidents
                else:
                    logger.error(f"❌ API request failed: {response.status}")
                    self.collection_stats.errors += 1
                    return []
                    
        except Exception as e:
            logger.error(f"❌ Error collecting data for {date}: {e}")
            self.collection_stats.errors += 1
            return []
    
    async def parse_polisen_event(self, event_data: Dict[str, Any]) -> PolisenIncident:
        """Parse a single event from Polisen API response"""
        # Extract basic information
        event_id = str(event_data.get('id', ''))
        datetime_str = event_data.get('datetime', '')
        name = event_data.get('name', '')
        summary = event_data.get('summary', '')
        url = event_data.get('url', '')
        
        # Extract and normalize incident type
        raw_type = event_data.get('type', '')
        normalized_type = self.incident_type_mapping.get(raw_type, 'other')
        
        # Extract location information (handle None case)
        location_data = event_data.get('location') or {}
        location = {
            'name': location_data.get('name', '') if location_data else '',
            'gps': location_data.get('gps', '') if location_data else ''
        }
        
        # Parse GPS coordinates if available
        if location['gps']:
            try:
                lat_str, lng_str = location['gps'].split(',')
                location['latitude'] = float(lat_str.strip())
                location['longitude'] = float(lng_str.strip())
            except (ValueError, AttributeError):
                logger.warning(f"⚠️ Invalid GPS coordinates: {location['gps']}")
                location['latitude'] = None
                location['longitude'] = None
        else:
            location['latitude'] = None
            location['longitude'] = None
        
        return PolisenIncident(
            id=event_id,
            datetime=datetime_str,
            name=name,
            summary=summary,
            type=normalized_type,
            location=location,
            url=url,
            raw_data=event_data
        )
    
    async def store_incidents_to_database(self, incidents: List[PolisenIncident]) -> int:
        """Store collected incidents in the database"""
        if not incidents:
            return 0
        
        try:
            db = await get_database()
            stored_count = 0
            
            logger.info(f"💾 Storing {len(incidents)} incidents to database...")
            
            for incident in incidents:
                try:
                    # Skip incidents without valid coordinates
                    if not incident.location.get('latitude') or not incident.location.get('longitude'):
                        continue
                    
                    # Convert datetime string to datetime object (remove timezone for database)
                    try:
                        occurred_at = datetime.fromisoformat(incident.datetime.replace('Z', '+00:00'))
                        # Remove timezone info for database storage
                        occurred_at = occurred_at.replace(tzinfo=None)
                    except ValueError:
                        # Try alternative parsing
                        occurred_at = datetime.now()
                    
                    # Determine severity based on incident type
                    severity = self.determine_incident_severity(incident.type)
                    
                    # Prepare incident data for database
                    incident_data = {
                        'incident_type': incident.type,
                        'severity': severity,
                        'description': f"{incident.name}: {incident.summary}",
                        'latitude': incident.location['latitude'],
                        'longitude': incident.location['longitude'],
                        'occurred_at': occurred_at,
                        'source': 'polisen.se',
                        'source_id': incident.id,
                        'confidence_score': 1.0  # High confidence for official police data
                    }
                    
                    # Store in database
                    await db.store_incident(incident_data)
                    stored_count += 1
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error storing incident {incident.id}: {e}")
                    self.collection_stats.errors += 1
            
            self.collection_stats.incidents_stored = stored_count
            
            logger.info(f"✅ Successfully stored {stored_count} incidents to database")
            
            # Update metrics
            incidents_counter = metrics.counter(
                "polisen_incidents_collected", 
                "Incidents collected from Polisen", 
                ("type",)
            )
            
            # Count by type
            type_counts = {}
            for incident in incidents:
                type_counts[incident.type] = type_counts.get(incident.type, 0) + 1
            
            for incident_type, count in type_counts.items():
                incidents_counter.labels(incident_type).inc(count)
            
            return stored_count
            
        except Exception as e:
            logger.error(f"❌ Error storing incidents to database: {e}")
            return 0
    
    def determine_incident_severity(self, incident_type: str) -> int:
        """Determine severity level (1-5) based on incident type"""
        severity_mapping = {
            # Critical severity (5)
            'murder_manslaughter': 5,
            'attempted_murder': 5,
            'shooting': 5,
            'suspected_shooting': 5,
            'bomb_threat': 5,
            'explosion': 5,
            'rape': 5,
            'attempted_rape': 5,
            'terrorism': 5,
            
            # High severity (4)
            'robbery': 4,
            'aggravated_assault': 4,
            'unlawful_detention': 4,
            'grand_theft': 4,
            'vehicle_theft': 4,
            'burglary': 4,
            'arson': 4,
            'hate_crime': 4,
            
            # Medium severity (3)
            'assault': 3,
            'theft': 3,
            'fraud': 3,
            'drug_crime': 3,
            'vandalism': 3,
            'domestic_violence': 3,
            'drunk_driving': 3,
            'unlawful_threat': 3,
            
            # Low severity (2)
            'shoplifting': 2,
            'bicycle_theft': 2,
            'petty_theft': 2,
            'public_intoxication': 2,
            'traffic_accident': 2,
            'harassment': 2,
            'trespassing': 2,
            
            # Very low severity (1)
            'traffic_crime': 1,
            'public_order_violation': 1,
            'found_property': 1,
            'animal_incident': 1,
            'noise_complaint': 1
        }
        
        return severity_mapping.get(incident_type, 3)  # Default to medium severity
    
    async def collect_and_store_batch(
        self, 
        location_name: Optional[str] = None,
        days_back: int = 7,
        max_events: int = 1000
    ) -> DataCollectionStats:
        """Collect incidents and store them in one operation"""
        try:
            # Collect incidents
            incidents = await self.collect_events(
                location_name=location_name,
                max_events=max_events,
                days_back=days_back
            )
            
            if incidents:
                # Store to database
                stored_count = await self.store_incidents_to_database(incidents)
                self.collection_stats.incidents_processed = len(incidents)
                self.collection_stats.incidents_stored = stored_count
                
                # Cache collection summary
                cache = await get_cache()
                cache_key = f"polisen_collection_{datetime.now().strftime('%Y%m%d_%H')}"
                await cache.set(cache_key, asdict(self.collection_stats), ttl=3600)
                
                logger.info(f"📊 Collection Summary:")
                logger.info(f"   • Collected: {self.collection_stats.incidents_collected}")
                logger.info(f"   • Processed: {self.collection_stats.incidents_processed}")
                logger.info(f"   • Stored: {self.collection_stats.incidents_stored}")
                logger.info(f"   • Errors: {self.collection_stats.errors}")
                logger.info(f"   • Time: {self.collection_stats.processing_time_seconds:.1f}s")
            
            return self.collection_stats
            
        except Exception as e:
            logger.error(f"❌ Error in batch collection: {e}")
            self.collection_stats.errors += 1
            return self.collection_stats
    
    async def collect_stockholm_data(self, days_back: int = 30) -> DataCollectionStats:
        """Collect data specifically for Stockholm region"""
        stockholm_areas = [
            "Stockholm",
            "Stockholms län",
            "Södermalm", 
            "Östermalm",
            "Vasastan",
            "Gamla stan",
            "Norrmalm"
        ]
        
        all_stats = []
        
        for area in stockholm_areas:
            logger.info(f"🏙️ Collecting data for {area}...")
            stats = await self.collect_and_store_batch(
                location_name=area,
                days_back=days_back,
                max_events=500
            )
            all_stats.append(stats)
            
            # Wait between areas to respect rate limits
            await asyncio.sleep(2.0)
        
        # Combine statistics
        total_stats = DataCollectionStats(
            incidents_collected=sum(s.incidents_collected for s in all_stats),
            incidents_processed=sum(s.incidents_processed for s in all_stats),
            incidents_stored=sum(s.incidents_stored for s in all_stats),
            errors=sum(s.errors for s in all_stats),
            collection_start=min(s.collection_start for s in all_stats),
            collection_end=max(s.collection_end or datetime.now() for s in all_stats)
        )
        
        logger.info(f"🏙️ Stockholm data collection completed: {total_stats.incidents_stored} incidents stored")
        
        return total_stats
    
    async def collect_major_cities_data(self, days_back: int = 14) -> DataCollectionStats:
        """Collect data for major Swedish cities"""
        major_cities = [
            "Stockholm",
            "Göteborg", 
            "Malmö",
            "Uppsala",
            "Västerås",
            "Örebro",
            "Linköping",
            "Helsingborg",
            "Jönköping",
            "Norrköping"
        ]
        
        all_stats = []
        
        for city in major_cities:
            logger.info(f"🏙️ Collecting data for {city}...")
            stats = await self.collect_and_store_batch(
                location_name=city,
                days_back=days_back,
                max_events=300
            )
            all_stats.append(stats)
            
            # Wait between cities to respect rate limits
            await asyncio.sleep(3.0)
        
        # Combine statistics
        total_stats = DataCollectionStats(
            incidents_collected=sum(s.incidents_collected for s in all_stats),
            incidents_processed=sum(s.incidents_processed for s in all_stats),
            incidents_stored=sum(s.incidents_stored for s in all_stats),
            errors=sum(s.errors for s in all_stats),
            collection_start=min(s.collection_start for s in all_stats),
            collection_end=max(s.collection_end or datetime.now() for s in all_stats)
        )
        
        logger.info(f"🇸🇪 Major cities data collection completed: {total_stats.incidents_stored} incidents stored")
        
        return total_stats
    
    async def check_rate_limit(self):
        """Check and enforce rate limiting"""
        current_time = time.time()
        
        # Reset rate limit window if needed
        if current_time - self.last_rate_limit_reset > self.rate_limit_window:
            self.rate_limit_calls = 0
            self.last_rate_limit_reset = current_time
        
        # Check if we've exceeded the rate limit
        if self.rate_limit_calls >= self.max_calls_per_window:
            wait_time = self.rate_limit_window - (current_time - self.last_rate_limit_reset)
            if wait_time > 0:
                logger.info(f"⏳ Rate limit reached, waiting {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
                self.rate_limit_calls = 0
                self.last_rate_limit_reset = time.time()
        
        self.rate_limit_calls += 1
    
    async def get_collection_stats(self) -> DataCollectionStats:
        """Get current collection statistics"""
        return self.collection_stats
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        logger.info("🧹 Polisen data collector cleaned up")

# Global instance
polisen_collector = PolisenDataCollector()

async def get_polisen_collector() -> PolisenDataCollector:
    """Get the global Polisen data collector instance"""
    return polisen_collector

# Scheduled data collection
async def scheduled_data_collection():
    """Background task for scheduled data collection"""
    logger.info("⏰ Starting scheduled data collection service...")
    
    collector = await get_polisen_collector()
    await collector.initialize()
    
    while True:
        try:
            logger.info("🔄 Starting scheduled data collection...")
            
            # Collect data for major cities every 6 hours
            stats = await collector.collect_major_cities_data(days_back=1)
            
            logger.info(f"✅ Scheduled collection completed: {stats.incidents_stored} new incidents")
            
            # Wait 6 hours before next collection
            await asyncio.sleep(6 * 3600)
            
        except Exception as e:
            logger.error(f"❌ Error in scheduled collection: {e}")
            # Wait 1 hour before retrying
            await asyncio.sleep(3600)