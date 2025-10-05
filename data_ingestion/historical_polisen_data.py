"""
Historical Crime Data Ingestion from Polisen.se
Fetches and processes historical crime data for AI training
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
import pandas as pd
from dataclasses import dataclass, asdict
import hashlib

from ..analytics import h3_utils
from ..config.settings import get_settings
from ..database.postgis_database import PostGISDatabase, get_database
from sqlalchemy import text

logger = logging.getLogger(__name__)


@dataclass
class HistoricalCrimeRecord:
    """Historical crime record for training"""

    # Core identification
    polisen_id: str

    # Location data
    latitude: float
    longitude: float
    location_name: str

    # Time data
    datetime_occurred: datetime
    datetime_reported: datetime

    # Crime details
    crime_type: str
    crime_category: str
    summary: str

    # Data quality
    data_quality_score: float  # 0-1 based on completeness

    # Optional fields with defaults
    h3_8: Optional[str] = None
    h3_9: Optional[str] = None
    has_coordinates: bool = False
    has_precise_time: bool = False

    # Training labels
    severity_score: int = 3  # 1-5, default to moderate
    outcome: Optional[str] = None  # If known
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        data = asdict(self)
        data['datetime_occurred'] = self.datetime_occurred.isoformat()
        data['datetime_reported'] = self.datetime_reported.isoformat()
        return data


class HistoricalPolisenDataIngestion:
    """
    Comprehensive historical crime data ingestion from Polisen.se
    """
    
    def __init__(self, database: PostGISDatabase = None):
        self.database = database
        self.session: Optional[aiohttp.ClientSession] = None
        settings = get_settings()
        self.base_url = settings.polisen_base_url.rstrip("/") + settings.polisen_events_path
        
        # Policy-based rate limiting (Polisen Open Data rules)
        # - At least 10 seconds between requests
        # - Max 60 requests per hour
        # - Max 1440 requests per day
        self.policy_min_interval_seconds = settings.polisen_min_interval_seconds
        self.last_request_time = 0.0
        self.hour_window_start: datetime | None = None
        self.requests_this_hour = 0
        self.day_window_date: datetime | None = None
        self.requests_today = 0
        self.hourly_cap = settings.polisen_hourly_cap
        self.daily_cap = settings.polisen_daily_cap
        # Track permanent 404 resources (e.g., date+location combos)
        self.permanent_404_keys = set()
        
        # Data storage
        self.data_cache_dir = Path("data_lake/historical_crime_data")
        self.data_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Swedish location mappings for better coverage
        self.major_cities = {
            "Stockholm": {
                "county": "Stockholms län",
                "center": (59.3293, 18.0686),
                "radius_km": 30
            },
            "Gothenburg": {
                "county": "Västra Götalands län", 
                "center": (57.7089, 11.9746),
                "radius_km": 25
            },
            "Malmö": {
                "county": "Skåne län",
                "center": (55.6050, 13.0038),
                "radius_km": 20
            },
            "Uppsala": {
                "county": "Uppsala län",
                "center": (59.8586, 17.6389),
                "radius_km": 15
            },
            "Linköping": {
                "county": "Östergötlands län",
                "center": (58.4108, 15.6214),
                "radius_km": 15
            }
        }
        
        # Crime type mapping for training labels
        self.crime_severity_mapping = {
            "Mord": 5,
            "Dråp": 5,
            "Misshandel": 4,
            "Rån": 4,
            "Våldtäkt": 5,
            "Inbrott": 3,
            "Stöld": 2,
            "Skadegörelse": 2,
            "Narkotikabrott": 3,
            "Bedrägeri": 2,
            "Trafikolycka": 3,
            "Brand": 4
        }
    
    async def initialize(self):
        """Initialize the historical data ingestion system"""
        
        if not self.database:
            self.database = await get_database()
        
        # Create HTTP session with proper headers
        settings = get_settings()
        timeout = aiohttp.ClientTimeout(total=settings.polisen_timeout_seconds, connect=10)
        headers = {
            # Per Polisen Open Data rules, identify app/org/contact
            'User-Agent': settings.polisen_user_agent,
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate'
        }
        
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers=headers,
            connector=aiohttp.TCPConnector(limit=10)
        )
        
        # Create historical data table
        await self._create_historical_data_table()
        
        logger.info("🚀 Historical Polisen data ingestion system initialized")
    
    async def _create_historical_data_table(self):
        """Create table for historical crime data"""
        
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS historical_crime_data (
            id SERIAL PRIMARY KEY,
            polisen_id VARCHAR(100) UNIQUE NOT NULL,
            location GEOGRAPHY(POINT, 4326) NOT NULL,
            latitude FLOAT NOT NULL,
            longitude FLOAT NOT NULL,
            location_name TEXT,
            datetime_occurred TIMESTAMPTZ NOT NULL,
            datetime_reported TIMESTAMPTZ NOT NULL,
            crime_type VARCHAR(100) NOT NULL,
            crime_category VARCHAR(100),
            summary TEXT,
            data_quality_score FLOAT DEFAULT 0.5,
            has_coordinates BOOLEAN DEFAULT false,
            has_precise_time BOOLEAN DEFAULT false,
            severity_score INTEGER DEFAULT 3,
            outcome VARCHAR(100),
            h3_8 VARCHAR(16),
            h3_9 VARCHAR(16),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );

        ALTER TABLE historical_crime_data
            ADD COLUMN IF NOT EXISTS h3_8 VARCHAR(16);

        ALTER TABLE historical_crime_data
            ADD COLUMN IF NOT EXISTS h3_9 VARCHAR(16);
        
        -- Create indexes for efficient querying
        CREATE INDEX IF NOT EXISTS idx_historical_crime_location 
        ON historical_crime_data USING GIST (location);
        
        CREATE INDEX IF NOT EXISTS idx_historical_crime_datetime 
        ON historical_crime_data (datetime_occurred);
        
        CREATE INDEX IF NOT EXISTS idx_historical_crime_type 
        ON historical_crime_data (crime_type);
        
        CREATE INDEX IF NOT EXISTS idx_historical_polisen_id 
        ON historical_crime_data (polisen_id);

        CREATE INDEX IF NOT EXISTS idx_historical_crime_h3_8 
        ON historical_crime_data (h3_8);

        CREATE INDEX IF NOT EXISTS idx_historical_crime_h3_9 
        ON historical_crime_data (h3_9);
        """
        
        # Execute DDL statements sequentially (asyncpg disallows multi-statements)
        statements = [s.strip() for s in create_table_sql.split(';') if s.strip()]
        async with self.database.get_session() as session:
            for stmt in statements:
                await session.execute(text(stmt))
            await session.commit()
    
    async def _rate_limit(self):
        """Rate limiting per Polisen policy with jitter and hourly/daily caps."""
        now = datetime.now(timezone.utc)

        # Reset hourly window
        if self.hour_window_start is None or (now - self.hour_window_start) >= timedelta(hours=1):
            self.hour_window_start = now
            self.requests_this_hour = 0

        # Reset daily window at UTC midnight
        if self.day_window_date is None or now.date() != self.day_window_date.date():
            self.day_window_date = now
            self.requests_today = 0

        # Enforce daily cap
        if self.requests_today >= 1440:
            raise RuntimeError("Daily API request limit (1440) reached; stopping for the day")

        # Enforce hourly cap
        if self.requests_this_hour >= 60:
            # Sleep until next hour window
            wait = 3600 - (now - self.hour_window_start).total_seconds()
            wait = max(0.0, wait)
            logger.info(f"Hourly cap reached (60). Sleeping {wait:.0f}s until next hour window")
            await asyncio.sleep(wait)
            # Reset after sleep
            self.hour_window_start = datetime.now(timezone.utc)
            self.requests_this_hour = 0

        # Enforce minimum 10s between requests with small jitter
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        min_interval = float(self.policy_min_interval_seconds)
        if elapsed < min_interval:
            sleep_time = (min_interval - elapsed) + random.uniform(0.0, 1.0)
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)
        # Do not update counters here; call _record_request() after request

    def _record_request(self):
        now = datetime.now(timezone.utc)
        if self.hour_window_start is None or (now - self.hour_window_start) >= timedelta(hours=1):
            self.hour_window_start = now
            self.requests_this_hour = 0
        if self.day_window_date is None or now.date() != self.day_window_date.date():
            self.day_window_date = now
            self.requests_today = 0
        self.requests_this_hour += 1
        self.requests_today += 1
        self.last_request_time = time.time()
    
    async def fetch_events_by_location_and_time(
        self,
        location_name: str,
        start_date: datetime,
        end_date: datetime,
        empty_day_bailout: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch events for a location and time period.

        Strategy:
        1) Try the historical range call (legacy 'datetime' param). If empty, 
           fall back to daily 'date=YYYY-MM-DD' calls which are known to work.
        """

        await self._rate_limit()

        # Attempt range-limited query first using documented 'DateTime' (case-sensitive)
        settings = get_settings()
        url = f"{settings.polisen_base_url}{settings.polisen_events_path}"
        params_range = {
            "locationname": location_name,
            # 'DateTime' supports YYYY, YYYY-MM, YYYY-MM-DD, or YYYY-MM-DD HH
            "DateTime": start_date.strftime('%Y-%m-%d'),
            "type": "all",
        }
        try:
            async with self.session.get(url, params=params_range) as response:
                self._record_request()
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, list) and data:
                        logger.info(
                            f"Fetched {len(data)} events for {location_name} via range {params_range['DateTime']}"
                        )
                        return data
                elif response.status == 429:
                    logger.warning("Rate limited by API on range call, backing off 60s...")
                    await asyncio.sleep(60)
        except Exception as e:
            logger.debug(f"Range fetch error for {location_name}: {e}")

        # Fallback: daily fetch loop using 'DateTime' parameter (documented)
        events: List[Dict[str, Any]] = []
        cursor = start_date
        empty_streak = 0
        while cursor < end_date:
            await self._rate_limit()
            day_str = cursor.strftime("%Y-%m-%d")
            params_day = {
                "locationname": location_name,
                "DateTime": day_str,
                "type": "all",
            }
            try:
                async with self.session.get(url, params=params_day) as resp:
                    self._record_request()
                    if resp.status == 200:
                        data = await resp.json()
                        if isinstance(data, list) and data:
                            events.extend(data)
                            empty_streak = 0
                        else:
                            empty_streak += 1
                    elif resp.status == 429:
                        logger.warning("Rate limited by API on daily call, sleeping 60s...")
                        await asyncio.sleep(60)
                        empty_streak = 0
                    elif resp.status == 404:
                        key = (location_name.lower(), day_str)
                        self.permanent_404_keys.add(key)
                        logger.info(f"Permanent 404 for {location_name} {day_str}; skipping further attempts of this resource")
                    # else ignore silently to keep progress
            except Exception as e:
                logger.debug(f"Daily fetch error for {location_name} {day_str}: {e}")
                empty_streak += 1
            finally:
                # Small delay and step to next day
                await asyncio.sleep(0.2)
                cursor += timedelta(days=1)

            # Empty-day bailout to avoid long runs during outages
            if empty_day_bailout and empty_streak >= empty_day_bailout:
                logger.info(
                    f"Bailing out daily loop for {location_name} after {empty_streak} empty days in a row starting {day_str}"
                )
                break

        logger.info(
            f"Fetched {len(events)} events for {location_name} via daily loop from {start_date:%Y-%m-%d} to {end_date:%Y-%m-%d}"
        )
        return events
    
    def _process_raw_event(self, raw_event: Dict[str, Any]) -> Optional[HistoricalCrimeRecord]:
        """Process raw API event into structured crime record"""
        
        try:
            # Extract basic info
            event_id = str(raw_event.get('id', ''))
            if not event_id:
                return None
            
            # Extract location data
            location_data = raw_event.get('location', {})
            
            if isinstance(location_data, dict):
                gps = location_data.get('gps', '')
                name = location_data.get('name', '')
            else:
                # Sometimes location is just a string
                gps = ''
                name = str(location_data) if location_data else ''
            
            # Parse coordinates
            lat, lon = None, None
            has_coordinates = False
            
            if gps and ',' in gps:
                try:
                    lat_str, lon_str = gps.split(',')
                    lat = float(lat_str.strip())
                    lon = float(lon_str.strip())
                    
                    # Validate Swedish coordinates
                    if 55.0 <= lat <= 70.0 and 10.0 <= lon <= 25.0:
                        has_coordinates = True
                    else:
                        lat, lon = None, None
                
                except (ValueError, IndexError):
                    pass
            
            # If no coordinates, try to geocode major cities
            if not has_coordinates and name:
                for city, info in self.major_cities.items():
                    if city.lower() in name.lower():
                        lat, lon = info['center']
                        has_coordinates = True
                        break
            
            # Skip events without any location data
            if not has_coordinates:
                return None
            
            # Extract time data
            datetime_str = raw_event.get('datetime', '')
            datetime_occurred = None
            has_precise_time = False
            
            if datetime_str:
                try:
                    # Handle different date formats from Polisen API
                    if 'T' in datetime_str:
                        datetime_occurred = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                        has_precise_time = True
                    else:
                        # Date only, set to midday
                        date_part = datetime_str.split(' ')[0]
                        datetime_occurred = datetime.strptime(date_part, '%Y-%m-%d')
                        datetime_occurred = datetime_occurred.replace(hour=12, tzinfo=timezone.utc)
                        has_precise_time = False
                        
                except (ValueError, TypeError):
                    # Default to current time if parsing fails
                    datetime_occurred = datetime.now(timezone.utc)
                    has_precise_time = False
            else:
                datetime_occurred = datetime.now(timezone.utc)
                has_precise_time = False
            
            # Extract crime details
            crime_type = raw_event.get('type', 'Unknown')
            summary = raw_event.get('summary', '')
            
            # Map to standardized crime categories
            crime_category = self._standardize_crime_type(crime_type)
            
            # Calculate severity score
            severity_score = self.crime_severity_mapping.get(crime_type, 3)
            
            # Calculate data quality score
            quality_factors = [
                1.0 if has_coordinates else 0.0,
                1.0 if has_precise_time else 0.5,
                1.0 if len(summary) > 10 else 0.3,
                1.0 if crime_type != 'Unknown' else 0.0,
                1.0 if name else 0.0
            ]
            data_quality_score = sum(quality_factors) / len(quality_factors)
            
            h3_indexes = h3_utils.event_h3_indexes(lat, lon) if has_coordinates else {}

            # Create record
            record = HistoricalCrimeRecord(
                polisen_id=event_id,
                latitude=lat,
                longitude=lon,
                location_name=name,
                h3_8=h3_indexes.get(8),
                h3_9=h3_indexes.get(9),
                datetime_occurred=datetime_occurred,
                datetime_reported=datetime_occurred,  # Same as occurred if not specified
                crime_type=crime_type,
                crime_category=crime_category,
                summary=summary,
                data_quality_score=data_quality_score,
                has_coordinates=has_coordinates,
                has_precise_time=has_precise_time,
                severity_score=severity_score,
                outcome=None  # Not available from API
            )
            
            return record
            
        except Exception as e:
            logger.error(f"Error processing raw event: {e}")
            return None

    def process_raw_event(self, raw_event: Dict[str, Any]) -> Optional[HistoricalCrimeRecord]:
        """Public wrapper so external callers avoid depending on private APIs."""
        return self._process_raw_event(raw_event)
    
    def _standardize_crime_type(self, crime_type: str) -> str:
        """Standardize Swedish crime types to English categories"""
        
        crime_type_lower = crime_type.lower()
        
        # Map Swedish terms to standard categories
        if any(term in crime_type_lower for term in ['mord', 'dråp']):
            return 'homicide'
        elif any(term in crime_type_lower for term in ['misshandel', 'våld']):
            return 'assault'
        elif any(term in crime_type_lower for term in ['rån']):
            return 'robbery'
        elif any(term in crime_type_lower for term in ['våldtäkt', 'sexualbrott']):
            return 'sexual_offense'
        elif any(term in crime_type_lower for term in ['inbrott', 'tillgrepp']):
            return 'burglary'
        elif any(term in crime_type_lower for term in ['stöld']):
            return 'theft'
        elif any(term in crime_type_lower for term in ['skadegörelse', 'vandalism']):
            return 'vandalism'
        elif any(term in crime_type_lower for term in ['narkotika', 'drog']):
            return 'drug_offense'
        elif any(term in crime_type_lower for term in ['bedrägeri', 'bedräg']):
            return 'fraud'
        elif any(term in crime_type_lower for term in ['trafikolycka', 'trafikbrott']):
            return 'traffic_accident'
        elif any(term in crime_type_lower for term in ['brand']):
            return 'fire'
        else:
            return 'other'
    
    async def store_historical_records(self, records: List[HistoricalCrimeRecord]) -> int:
        """Store historical records in database"""
        
        if not records:
            return 0
        
        stored_count = 0
        
        async with self.database.get_session() as session:
            for record in records:
                try:
                    # Use PostgreSQL UPSERT to handle duplicates
                    insert_sql = """
                    INSERT INTO public.historical_crime_data (
                        polisen_id, location, latitude, longitude, location_name,
                        datetime_occurred, datetime_reported, crime_type, crime_category,
                        summary, data_quality_score, has_coordinates, has_precise_time,
                        severity_score, outcome, h3_8, h3_9
                    ) VALUES (
                        :polisen_id,
                        public.ST_SetSRID(public.ST_Point(:longitude, :latitude), 4326)::public.geography,
                        :latitude, :longitude, :location_name,
                        :datetime_occurred, :datetime_reported, :crime_type, :crime_category,
                        :summary, :data_quality_score, :has_coordinates, :has_precise_time,
                        :severity_score, :outcome, :h3_8, :h3_9
                    )
                    ON CONFLICT (polisen_id) DO UPDATE SET
                        updated_at = NOW(),
                        data_quality_score = EXCLUDED.data_quality_score,
                        summary = EXCLUDED.summary,
                        h3_8 = COALESCE(EXCLUDED.h3_8, historical_crime_data.h3_8),
                        h3_9 = COALESCE(EXCLUDED.h3_9, historical_crime_data.h3_9)
                    """
                    
                    params = {
                        'polisen_id': record.polisen_id,
                        'latitude': record.latitude,
                        'longitude': record.longitude,
                        'location_name': record.location_name,
                        'datetime_occurred': record.datetime_occurred,
                        'datetime_reported': record.datetime_reported,
                        'crime_type': record.crime_type,
                        'crime_category': record.crime_category,
                        'summary': record.summary,
                        'data_quality_score': record.data_quality_score,
                        'has_coordinates': record.has_coordinates,
                        'has_precise_time': record.has_precise_time,
                        'severity_score': record.severity_score,
                        'outcome': record.outcome,
                        'h3_8': record.h3_8,
                        'h3_9': record.h3_9,
                    }
                    await session.execute(text(insert_sql), params)
                    stored_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to store record {record.polisen_id}: {e}")
                    # Roll back the failed statement so subsequent inserts can proceed
                    try:
                        await session.rollback()
                    except Exception:
                        pass
                    continue
            
            await session.commit()
        
        logger.info(f"Stored {stored_count} historical crime records")
        return stored_count
    
    async def ingest_historical_data_comprehensive(
        self,
        months_back: int = 12,
        cities: List[str] = None
    ) -> Dict[str, Any]:
        """Comprehensive historical data ingestion"""
        
        if cities is None:
            cities = list(self.major_cities.keys())
        
        logger.info(f"🚀 Starting comprehensive historical data ingestion for {len(cities)} cities, {months_back} months back")
        
        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=months_back * 30)
        
        ingestion_results = {
            'start_time': datetime.now(timezone.utc),
            'cities_processed': [],
            'total_records_fetched': 0,
            'total_records_stored': 0,
            'errors': [],
            'data_quality': {}
        }
        
        for city in cities:
            logger.info(f"📍 Processing historical data for {city}...")
            
            try:
                # Fetch events in monthly chunks to avoid API limits
                city_records = []
                
                current_start = start_date
                while current_start < end_date:
                    current_end = min(current_start + timedelta(days=30), end_date)
                    
                    raw_events = await self.fetch_events_by_location_and_time(
                        city, current_start, current_end
                    )
                    
                    # Process raw events
                    for raw_event in raw_events:
                        record = self._process_raw_event(raw_event)
                        if record and record.data_quality_score >= 0.3:  # Minimum quality threshold
                            city_records.append(record)
                    
                    current_start = current_end
                    
                    # Small delay between requests
                    await asyncio.sleep(1)
                
                # Store city records
                stored_count = await self.store_historical_records(city_records)
                
                # Calculate data quality metrics
                if city_records:
                    avg_quality = sum(r.data_quality_score for r in city_records) / len(city_records)
                    precise_time_pct = sum(1 for r in city_records if r.has_precise_time) / len(city_records) * 100
                    coordinates_pct = sum(1 for r in city_records if r.has_coordinates) / len(city_records) * 100
                else:
                    avg_quality = 0
                    precise_time_pct = 0
                    coordinates_pct = 0
                
                ingestion_results['cities_processed'].append({
                    'city': city,
                    'records_fetched': len(city_records),
                    'records_stored': stored_count,
                    'avg_data_quality': avg_quality,
                    'precise_time_percentage': precise_time_pct,
                    'coordinates_percentage': coordinates_pct
                })
                
                ingestion_results['total_records_fetched'] += len(city_records)
                ingestion_results['total_records_stored'] += stored_count
                
                logger.info(f"✅ {city}: {len(city_records)} fetched, {stored_count} stored (quality: {avg_quality:.2f})")
                
            except Exception as e:
                error_msg = f"Failed to process {city}: {str(e)}"
                logger.error(error_msg)
                ingestion_results['errors'].append(error_msg)
        
        ingestion_results['end_time'] = datetime.now(timezone.utc)
        ingestion_results['duration_minutes'] = (
            ingestion_results['end_time'] - ingestion_results['start_time']
        ).total_seconds() / 60
        
        # Save ingestion report
        report_path = self.data_cache_dir / f"ingestion_report_{int(time.time())}.json"
        with open(report_path, 'w') as f:
            json.dump(ingestion_results, f, indent=2, default=str)
        
        logger.info(f"🎉 Historical data ingestion completed!")
        logger.info(f"   - Total records: {ingestion_results['total_records_stored']}")
        logger.info(f"   - Duration: {ingestion_results['duration_minutes']:.1f} minutes")
        logger.info(f"   - Report saved: {report_path}")
        
        return ingestion_results

    async def ingest_historical_range(
        self,
        start_date: datetime,
        end_date: datetime,
        cities: List[str] | None = None,
    ) -> Dict[str, Any]:
        """Ingest historical data for an explicit date range, sequentially.

        Splits the range into ~30-day chunks per city and uses the resilient daily
        fallback fetch to ensure coverage.
        """
        if cities is None:
            cities = list(self.major_cities.keys())

        results: Dict[str, Any] = {
            'start_time': datetime.now(timezone.utc),
            'range_start': start_date.isoformat(),
            'range_end': end_date.isoformat(),
            'cities_processed': [],
            'total_records_fetched': 0,
            'total_records_stored': 0,
            'errors': [],
        }

        for city in cities:
            try:
                city_records: List[HistoricalCrimeRecord] = []
                cursor = start_date
                while cursor < end_date:
                    chunk_end = min(cursor + timedelta(days=30), end_date)
                    raw_events = await self.fetch_events_by_location_and_time(
                        city, cursor, chunk_end, empty_day_bailout=14
                    )
                    for raw in raw_events:
                        rec = self._process_raw_event(raw)
                        if not rec:
                            continue
                        # Filter: ensure location text mentions the requested city (best-effort)
                        if rec.location_name and city.lower() not in rec.location_name.lower():
                            # keep but lower quality unless coordinates are in city area (future enhancement)
                            pass
                        if rec.data_quality_score >= 0.3:
                            city_records.append(rec)
                    cursor = chunk_end
                    await asyncio.sleep(0.2)

                # Deduplicate by polisen_id (keep highest quality)
                by_id: Dict[str, HistoricalCrimeRecord] = {}
                for rec in city_records:
                    if rec.polisen_id in by_id:
                        if rec.data_quality_score > by_id[rec.polisen_id].data_quality_score:
                            by_id[rec.polisen_id] = rec
                    else:
                        by_id[rec.polisen_id] = rec
                unique_records = list(by_id.values())

                stored = await self.store_historical_records(unique_records)

                if city_records:
                    avg_quality = sum(r.data_quality_score for r in city_records) / len(city_records)
                    precise_time_pct = sum(1 for r in city_records if r.has_precise_time) / len(city_records) * 100
                    coordinates_pct = sum(1 for r in city_records if r.has_coordinates) / len(city_records) * 100
                else:
                    avg_quality = 0.0
                    precise_time_pct = 0.0
                    coordinates_pct = 0.0

                results['cities_processed'].append({
                    'city': city,
                    'records_fetched': len(city_records),
                    'records_unique': len(unique_records),
                    'records_stored': stored,
                    'avg_data_quality': avg_quality,
                    'precise_time_percentage': precise_time_pct,
                    'coordinates_percentage': coordinates_pct,
                })

                results['total_records_fetched'] += len(city_records)
                results['total_records_stored'] += stored

                logger.info(f"✅ {city}: {len(city_records)} fetched, {stored} stored for range {start_date:%Y-%m-%d}..{end_date:%Y-%m-%d}")

            except Exception as e:
                msg = f"Failed ingest for {city} in range {start_date:%Y-%m-%d}..{end_date:%Y-%m-%d}: {e}"
                logger.error(msg)
                results['errors'].append(msg)

        results['end_time'] = datetime.now(timezone.utc)
        results['duration_minutes'] = (results['end_time'] - results['start_time']).total_seconds() / 60
        return results
    
    async def get_training_dataset(
        self,
        min_quality_score: float = 0.5,
        start_date: datetime = None,
        end_date: datetime = None,
        max_records: int = 10000
    ) -> List[Dict[str, Any]]:
        """Get clean training dataset from historical data"""
        
        logger.info(f"📊 Fetching training dataset (min_quality: {min_quality_score}, max_records: {max_records})")
        
        # Default to last 6 months if no dates specified
        if not end_date:
            end_date = datetime.now(timezone.utc)
        if not start_date:
            start_date = end_date - timedelta(days=180)
        
        query_sql = """
        SELECT 
            polisen_id,
            latitude,
            longitude,
            location_name,
            datetime_occurred,
            crime_type,
            crime_category,
            severity_score,
            data_quality_score,
            has_coordinates,
            has_precise_time,
            summary
        FROM historical_crime_data
        WHERE 
            data_quality_score >= :min_quality
            AND datetime_occurred >= :start_date
            AND datetime_occurred <= :end_date
            AND has_coordinates = true
        ORDER BY datetime_occurred DESC, data_quality_score DESC
        LIMIT :max_records
        """
        
        async with self.database.get_session() as session:
            result = await session.execute(query_sql, {
                'min_quality': min_quality_score,
                'start_date': start_date,
                'end_date': end_date,
                'max_records': max_records
            })
            
            records = result.fetchall()
            
            # Convert to list of dictionaries
            dataset = []
            for record in records:
                dataset.append({
                    'polisen_id': record[0],
                    'latitude': record[1],
                    'longitude': record[2],
                    'location_name': record[3],
                    'datetime_occurred': record[4],
                    'crime_type': record[5],
                    'crime_category': record[6],
                    'severity_score': record[7],
                    'data_quality_score': record[8],
                    'has_coordinates': record[9],
                    'has_precise_time': record[10],
                    'summary': record[11]
                })
        
        logger.info(f"✅ Retrieved {len(dataset)} records for training")
        
        return dataset
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()


# Utility functions
async def run_historical_ingestion():
    """Main function to run historical data ingestion"""
    
    ingestion = HistoricalPolisenDataIngestion()
    
    try:
        await ingestion.initialize()
        
        # Run comprehensive ingestion
        results = await ingestion.ingest_historical_data_comprehensive(
            months_back=6,  # Last 6 months
            cities=['Stockholm', 'Gothenburg', 'Malmö', 'Uppsala']
        )
        
        print("📊 Historical Data Ingestion Results:")
        print(f"   - Cities processed: {len(results['cities_processed'])}")
        print(f"   - Total records: {results['total_records_stored']}")
        print(f"   - Duration: {results['duration_minutes']:.1f} minutes")
        
        if results['errors']:
            print(f"   - Errors: {len(results['errors'])}")
        
        return results
        
    finally:
        await ingestion.cleanup()


if __name__ == "__main__":
    asyncio.run(run_historical_ingestion())
