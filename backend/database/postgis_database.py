"""
PostgreSQL with PostGIS Database Implementation
Replaces in-memory storage with persistent, geospatial-enabled database
Critical infrastructure for Atlas AI production deployment
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Sequence
import json
import uuid
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import os
from pathlib import Path
from urllib.parse import quote_plus

# asyncpg imports (replacing psycopg_pool for Scaleway compatibility)
import asyncpg

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Integer, String, DateTime, Date, Float, Boolean, Text, ForeignKey, Index, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB

# Optional geospatial dependencies
try:
    from geoalchemy2 import Geometry, Geography
    from geoalchemy2.functions import ST_Distance, ST_DWithin, ST_Point, ST_AsGeoJSON
    import geoalchemy2.functions as geofunc
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    GEOSPATIAL_AVAILABLE = False
    Geometry = Geography = ST_Distance = ST_DWithin = ST_Point = ST_AsGeoJSON = None
    geofunc = None
    logging.warning("⚠️ geoalchemy2 not available - geospatial features disabled")

from ..analytics import h3_utils
from ..common.performance import performance_tracked
from ..observability.metrics import metrics

logger = logging.getLogger(__name__)

# Database Models with PostGIS Support
Base = declarative_base()


def _maybe_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    """Best-effort conversion to float that tolerates ``None``/empty values."""

    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _location_column(nullable: bool = False):
    """Create a location column - Geography if available, otherwise Text placeholder"""
    if GEOSPATIAL_AVAILABLE:
        return Column(Geography('POINT', srid=4326), nullable=nullable)
    else:
        # Fallback: Store as text "lat,lon" when PostGIS unavailable
        return Column(Text, nullable=nullable)

class CrimeIncident(Base):
    """Crime incident with geospatial indexing"""
    __tablename__ = 'crime_incidents'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    incident_type = Column(String, nullable=False)
    severity = Column(Integer, nullable=False)
    description = Column(Text)
    
    # PostGIS geometry column for efficient spatial queries
    location = _location_column(nullable=False)
    latitude = Column(Float, nullable=False)  # Redundant for easy access
    longitude = Column(Float, nullable=False)
    
    # Temporal data with indexing
    occurred_at = Column(DateTime, nullable=False)
    reported_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Source tracking
    source = Column(String, nullable=False)  # 'polisen', 'user_report', etc.
    source_id = Column(String)  # External ID from source system
    confidence_score = Column(Float, default=1.0)
    
    # Status and processing
    status = Column(String, default='active')
    is_verified = Column(Boolean, default=False)

    # Additional data for user_id, media_ids, etc. (using incident_metadata to avoid SQLAlchemy reserved name)
    incident_metadata = Column(JSONB, default=dict, name='metadata')

    # Spatial and temporal indexes
    def __table_args__(cls):
        args = [
            Index('idx_crime_occurred_at', 'occurred_at'),
            Index('idx_crime_type_severity', 'incident_type', 'severity'),
            Index('idx_crime_source', 'source'),
            Index('idx_crime_status', 'status'),
            UniqueConstraint('source', 'source_id', name='uq_crime_source_id'),
        ]
        # Only add GIST index if PostGIS is available
        if GEOSPATIAL_AVAILABLE:
            args.insert(0, Index('idx_crime_location_gist', 'location', postgresql_using='gist'))
        return tuple(args)

    __table_args__ = __table_args__(None)

class User(Base):
    """User accounts with authentication data"""
    __tablename__ = 'users'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    
    # User profile
    full_name = Column(String)
    role = Column(String, default='citizen')  # 'citizen', 'law_enforcement', 'admin'
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime)
    
    # User preferences and location
    home_location = _location_column(nullable=True)
    notification_radius = Column(Float, default=5000)  # meters
    
    # Relationships
    alerts = relationship("UserAlert", back_populates="user")
    reports = relationship("IncidentReport", back_populates="user")

class UserAlert(Base):
    """Geofenced alerts for users"""
    __tablename__ = 'user_alerts'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    incident_id = Column(String, ForeignKey('crime_incidents.id'), nullable=False)
    
    alert_type = Column(String, nullable=False)  # 'proximity', 'area', 'route'
    message = Column(Text)
    is_read = Column(Boolean, default=False)
    is_dismissed = Column(Boolean, default=False)
    
    # Geospatial alert data
    trigger_location = _location_column(nullable=True)
    distance_from_user = Column(Float)  # meters
    
    created_at = Column(DateTime, default=datetime.utcnow)
    read_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="alerts")

class IncidentReport(Base):
    """User-submitted incident reports"""
    __tablename__ = 'incident_reports'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    
    # Report content
    incident_type = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    severity = Column(Integer, nullable=False)
    
    # Location data
    location = _location_column(nullable=False)
    location_accuracy = Column(Float)  # GPS accuracy in meters
    
    # Media attachments
    has_photo = Column(Boolean, default=False)
    has_audio = Column(Boolean, default=False)
    has_video = Column(Boolean, default=False)
    
    # Processing status
    status = Column(String, default='pending')  # 'pending', 'verified', 'dismissed'
    processed_at = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="reports")

class PredictionCache(Base):
    """Cache for AI predictions with spatial indexing"""
    __tablename__ = 'prediction_cache'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Prediction area (could be point or polygon)
    location = _location_column(nullable=False)
    grid_cell_id = Column(String)  # For grid-based predictions
    
    # Prediction data
    crime_type = Column(String, nullable=False)
    predicted_probability = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)
    
    # Temporal window
    prediction_start = Column(DateTime, nullable=False)
    prediction_end = Column(DateTime, nullable=False)
    
    # Model metadata
    model_version = Column(String, nullable=False)
    features_used = Column(Text)  # JSON string
    contributing_factors = Column(Text)  # JSON string
    
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)

    # Spatial index for fast location queries
    def __table_args__(cls):
        args = [
            Index('idx_prediction_temporal', 'prediction_start', 'prediction_end'),
            Index('idx_prediction_expires', 'expires_at'),
        ]
        if GEOSPATIAL_AVAILABLE:
            args.insert(0, Index('idx_prediction_location_gist', 'location', postgresql_using='gist'))
        return tuple(args)

    __table_args__ = __table_args__(None)

class BraStatistic(Base):
    """Brå PxWeb aggregate statistics (stored in public schema).

    Stores monthly/periodic aggregates such as reported offences. We keep the
    PxWeb table identifier and dimension codes for traceability.
    """
    __tablename__ = 'bra_statistics'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    # PxWeb table identifier, e.g. 'Nationella_brottsstatistik/b1201'
    table_id = Column(String, nullable=False)

    # Dimensions
    region_code = Column(String, nullable=False)   # e.g. '00' (Sweden total) or municipal/region code
    region_name = Column(String, nullable=True)
    offence_code = Column(String, nullable=True)   # depends on table
    offence_name = Column(String, nullable=True)

    # Period as PxWeb code (e.g. '2023M01'), plus a parsed date start when available
    period = Column(String, nullable=False)
    period_date = Column(DateTime, nullable=True)

    # Value and metadata
    value = Column(Float, nullable=False)
    unit = Column(String, nullable=True)
    extra = Column(Text)  # JSON string for any additional dimensions/metadata

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('table_id', 'region_code', 'period', 'offence_code', name='uq_bra_statistics_unique'),
        Index('idx_bra_region_period_offence', region_code, period, offence_code),
        Index('idx_bra_table_period', table_id, period),
    )


class RegionDimension(Base):
    """Cross-walk between spatial cells and administrative regions."""

    __tablename__ = 'region_dim'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    h3 = Column(String, nullable=False)
    h3_resolution = Column(Integer, nullable=False)
    municipality_code = Column(String)
    municipality_name = Column(String)
    county_code = Column(String)
    county_name = Column(String)
    polisen_region = Column(String)
    centroid = _location_column(nullable=True)
    area_sq_km = Column(Float)
    attributes = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('h3', 'h3_resolution', name='uq_region_dim_h3_resolution'),
        Index('idx_region_dim_h3', 'h3'),
        Index('idx_region_dim_municipality', 'municipality_code'),
    )


class WeatherObservation(Base):
    """Hourly weather observation mapped to H3 cells."""

    __tablename__ = 'weather_hourly'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    observation_time = Column(DateTime, nullable=False)
    source = Column(String, default='smhi')
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    h3_8 = Column(String, nullable=False)
    temperature_c = Column(Float)
    feels_like_c = Column(Float)
    precipitation_mm = Column(Float)
    wind_speed_mps = Column(Float)
    wind_direction_deg = Column(Float)
    humidity = Column(Float)
    pressure_hpa = Column(Float)
    snow_depth_cm = Column(Float)
    conditions = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('h3_8', 'observation_time', 'source', name='uq_weather_cell_time_source'),
        Index('idx_weather_h3_time', 'h3_8', 'observation_time'),
        Index('idx_weather_time', 'observation_time'),
    )


class GridFeatureDaily(Base):
    """Pre-computed daily features per H3 cell and crime family."""

    __tablename__ = 'grid_features_daily'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    feature_date = Column(Date, nullable=False)
    h3 = Column(String, nullable=False)
    crime_family = Column(String, nullable=False)
    incident_count = Column(Float, default=0.0)
    incident_rate = Column(Float, default=0.0)
    avg_severity = Column(Float, default=0.0)
    rolling_7d = Column(Float, default=0.0)
    rolling_30d = Column(Float, default=0.0)
    population = Column(Float, default=0.0)
    features = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('h3', 'feature_date', 'crime_family', name='uq_grid_features_daily'),
        Index('idx_grid_features_h3_date', 'h3', 'feature_date'),
        Index('idx_grid_features_crime_family', 'crime_family'),
    )


class CalendarDimension(Base):
    """Calendar table capturing holidays and school breaks."""

    __tablename__ = 'calendar_dim'

    calendar_date = Column(Date, primary_key=True)
    is_holiday = Column(Boolean, default=False)
    holiday_name = Column(String)
    is_school_break = Column(Boolean, default=False)
    break_name = Column(String)
    week_of_year = Column(Integer)
    month = Column(Integer)
    weekday = Column(Integer)
    extras = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class PoiDimension(Base):
    """Points of interest enriched with H3 cell identifiers."""

    __tablename__ = 'poi_dim'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    source = Column(String, nullable=False)
    external_id = Column(String)
    name = Column(String)
    category = Column(String)
    subcategory = Column(String)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    h3_8 = Column(String, nullable=False)
    h3_9 = Column(String)
    attributes = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('source', 'external_id', name='uq_poi_source_external'),
        Index('idx_poi_h3_8', 'h3_8'),
        Index('idx_poi_category', 'category'),
    )

class ScbPopulation(Base):
    """SCB population by region and period (year/month)."""
    __tablename__ = 'scb_population'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    region_code = Column(String, nullable=False)
    region_name = Column(String, nullable=True)
    period = Column(String, nullable=False)  # e.g., '2024' or '2024M01'
    population = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_scb_pop_region_period', region_code, period),
    )

class BraFeatures(Base):
    """Aggregated Brå features for modeling."""
    __tablename__ = 'bra_features'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    region_code = Column(String, nullable=True)
    region_name = Column(String, nullable=False)
    period = Column(String, nullable=False)
    total_offences = Column(Float, nullable=True)
    offences_per_100k = Column(Float, nullable=True)
    clearance_rate = Column(Float, nullable=True)
    sources = Column(Text)  # JSON string of contributing table_ids
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index('idx_bra_feat_region_period', region_code, period),
    )

@dataclass
class DatabaseConfig:
    """Database connection configuration - supports Railway DATABASE_URL or individual vars"""
    host: str = None
    port: int = None
    database: str = None
    username: str = None
    password: str = None
    pool_size: int = None
    max_overflow: int = None

    def __post_init__(self):
        """Read environment variables at initialization time, not at class definition time"""
        if self.host is None:
            self.host = os.getenv('POSTGRES_HOST', 'localhost')
            logger.info(f"DatabaseConfig: POSTGRES_HOST = {self.host}")
        if self.port is None:
            self.port = int(os.getenv('POSTGRES_PORT', '5432'))
            logger.info(f"DatabaseConfig: POSTGRES_PORT = {self.port}")
        if self.database is None:
            self.database = os.getenv('POSTGRES_DB', 'atlas_ai')
            logger.info(f"DatabaseConfig: POSTGRES_DB = {self.database}")
        if self.username is None:
            self.username = os.getenv('POSTGRES_USER', 'atlas_user')
        if self.password is None:
            self.password = os.getenv('POSTGRES_PASSWORD', 'secure_password')
            logger.info(f"DatabaseConfig: password length = {len(self.password)}")
        if self.pool_size is None:
            self.pool_size = int(os.getenv('POSTGRES_POOL_SIZE', '10'))
        if self.max_overflow is None:
            self.max_overflow = int(os.getenv('POSTGRES_MAX_OVERFLOW', '20'))

    @classmethod
    def from_url(cls, database_url: str):
        """Parse Railway's DATABASE_URL format: postgres://user:pass@host:port/db or postgresql://..."""
        import re
        # Strip SSL/query parameters for parsing (psycopg3 handles SSL in conninfo)
        if '?' in database_url:
            database_url = database_url.split('?')[0]

        # Support both postgres:// and postgresql:// schemes
        match = re.match(r'postgres(?:ql)?://([^:]+):([^@]+)@([^:]+):(\d+)/(.+)', database_url)
        if match:
            username, password, host, port, database = match.groups()
            return cls(
                host=host,
                port=int(port),
                database=database,
                username=username,
                password=password
            )
        return cls()

class PostGISDatabase:
    """Production PostgreSQL + PostGIS database implementation"""
    
    def __init__(self, config: DatabaseConfig = None):
        self.config = config or DatabaseConfig()
        self.engine = None
        self.session_factory = None
        self.pool = None
        
    async def initialize(self):
        """Initialize database connection and setup"""
        try:
            # Create SQLAlchemy async engine with psycopg
            # URL-encode password to handle special characters like @ : [ ] / etc.
            encoded_password = quote_plus(self.config.password)
            database_url = (
                f"postgresql+psycopg://{self.config.username}:{encoded_password}"
                f"@{self.config.host}:{self.config.port}/{self.config.database}"
            )

            # Scaleway managed databases require SSL
            connect_args = {"sslmode": "require"}
            self.engine = create_async_engine(
                database_url,
                connect_args=connect_args,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                echo=False,  # Set to True for SQL debugging
                future=True
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                self.engine, 
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Create asyncpg pool for raw queries (Scaleway-compatible, same as Atlas Intelligence)
            # Scaleway managed databases require SSL
            logger.info(f"🔄 Creating asyncpg pool for {self.config.host}:{self.config.port}/{self.config.database}")

            async def init_conn(conn):
                """Configure each connection with search_path"""
                await conn.execute("SET search_path TO public")

            self.pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                user=self.config.username,
                password=self.config.password,
                database=self.config.database,
                ssl='require',  # Required for Scaleway Managed PostgreSQL
                min_size=5,
                max_size=20,
                timeout=30,
                command_timeout=60,
                max_inactive_connection_lifetime=300,
                init=init_conn
            )

            logger.info("✅ asyncpg pool created successfully")
            
            # Ensure PostGIS extension is enabled
            await self._ensure_postgis_extension()
            
            # Create tables if they don't exist
            await self._create_tables()
            
            logger.info("✅ PostgreSQL + PostGIS database initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    async def _ensure_postgis_extension(self):
        """Ensure PostGIS extension is installed"""
        async with self.pool.acquire() as conn:
            try:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
                await conn.execute("CREATE EXTENSION IF NOT EXISTS postgis_topology;")
                logger.info("✅ PostGIS extensions enabled")
            except Exception as e:
                logger.warning(f"⚠️ Could not enable PostGIS extensions: {e}")
    
    async def _create_tables(self):
        """Create database tables"""
        await self._prepare_bra_statistics_for_constraints()
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ Database tables created/verified")

    async def create_tables(self):
        """Public method to create database tables"""
        await self._create_tables()

    async def _prepare_bra_statistics_for_constraints(self):
        """Normalize legacy Brå rows so unique constraints can be applied safely."""
        if not self.pool:
            return
        async with self.pool.acquire() as conn:
            exists = await conn.fetchrow("SELECT to_regclass('public.bra_statistics') IS NOT NULL")
            if not exists or not exists[0]:
                return

            result = await conn.execute("UPDATE bra_statistics SET offence_code = 'ALL' WHERE offence_code IS NULL")
            updated = int(result.split()[-1]) if result and result.split() else 0

            result = await conn.execute(
                """
                DELETE FROM bra_statistics a
                USING bra_statistics b
                WHERE a.ctid < b.ctid
                  AND a.table_id = b.table_id
                  AND a.region_code = b.region_code
                  AND a.period = b.period
                  AND COALESCE(a.offence_code, 'ALL') = COALESCE(b.offence_code, 'ALL')
                """
            )
            deduped = int(result.split()[-1]) if result and result.split() else 0

            await conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_bra_statistics_unique
                ON bra_statistics (table_id, region_code, period, offence_code)
                """
            )

            if updated or deduped:
                logger.info(
                    "🧹 Normalized existing Brå statistics rows (offence_code updated=%s, duplicates removed=%s)",
                    updated,
                    deduped,
                )
    
    @asynccontextmanager
    async def get_session(self):
        """Get database session with automatic cleanup"""
        async with self.session_factory() as session:
            try:
                # Ensure we use the public schema to avoid search_path privilege issues
                try:
                    await session.execute(text("SET search_path TO public"))
                except Exception:
                    pass
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    @performance_tracked("database.store_incident")
    async def store_incident(self, incident_data: Dict[str, Any]) -> Optional[str]:
        """
        Store crime incident with geospatial indexing and duplicate detection.
        Returns incident ID if stored/updated, None if duplicate skipped.
        """
        async with self.get_session() as session:
            # Check for duplicate if source_id is provided
            if incident_data.get('source_id'):
                existing = await session.execute(
                    text("SELECT id FROM crime_incidents WHERE source = :source AND source_id = :source_id"),
                    {
                        'source': incident_data.get('source', 'unknown'),
                        'source_id': incident_data['source_id']
                    }
                )
                existing_row = existing.fetchone()

                if existing_row:
                    # Duplicate found, update confidence score if higher
                    new_confidence = incident_data.get('confidence_score', 1.0)
                    await session.execute(
                        text("""
                            UPDATE crime_incidents
                            SET confidence_score = GREATEST(confidence_score, :new_confidence),
                                updated_at = NOW()
                            WHERE id = :incident_id
                        """),
                        {'new_confidence': new_confidence, 'incident_id': existing_row[0]}
                    )
                    logger.debug(f"🔄 Updated existing incident {existing_row[0]}")
                    return existing_row[0]

            # Create PostGIS POINT from lat/lng
            location_point = func.ST_Point(
                incident_data['longitude'],
                incident_data['latitude']
            )

            incident = CrimeIncident(
                incident_type=incident_data['incident_type'],
                severity=incident_data.get('severity', 3),
                description=incident_data.get('description', ''),
                location=location_point,
                latitude=incident_data['latitude'],
                longitude=incident_data['longitude'],
                occurred_at=incident_data.get('occurred_at', datetime.utcnow()),
                source=incident_data.get('source', 'unknown'),
                source_id=incident_data.get('source_id'),
                confidence_score=incident_data.get('confidence_score', 1.0),
                incident_metadata=incident_data.get('metadata', {})
            )

            session.add(incident)
            await session.flush()

            # Track database metrics
            db_counter = metrics.counter("database_operations_total", "Database operations", ("operation",))
            db_counter.labels("incident_stored").inc()
            logger.info(f"📍 Stored new incident {incident.id} at ({incident.latitude}, {incident.longitude})")

            # Broadcast new incident in real-time (non-blocking)
            try:
                from ..websockets.incident_broadcaster import broadcast_new_incident
                broadcast_data = {
                    'id': incident.id,
                    'incident_type': incident.incident_type,
                    'latitude': incident.latitude,
                    'longitude': incident.longitude,
                    'severity': incident.severity,
                    'description': incident.description,
                    'occurred_at': incident.occurred_at,
                    'reported_at': incident.reported_at,
                    'source': incident.source,
                    'confidence_score': incident.confidence_score
                }
                # Fire and forget - don't block on broadcast
                asyncio.create_task(
                    broadcast_new_incident(broadcast_data, incident_data.get('metadata'))
                )
            except Exception as e:
                logger.warning(f"Failed to broadcast incident: {e}")

            return incident.id
    
    @performance_tracked("database.get_incidents_near")
    async def get_incidents_near(
        self, 
        latitude: float, 
        longitude: float, 
        radius_meters: float = 5000,
        hours_back: int = 24,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get incidents near a location using spatial indexing"""
        async with self.get_session() as session:
            # Create point for distance calculation
            user_point = func.ST_Point(longitude, latitude)
            
            # Time filter
            time_threshold = datetime.utcnow() - timedelta(hours=hours_back)
            
            # Spatial query with PostGIS
            query = """
            SELECT
                id, incident_type, severity, description,
                latitude, longitude,
                occurred_at, source, confidence_score,
                ST_Distance(location, ST_Point(%s, %s)::geography) as distance_meters
            FROM crime_incidents
            WHERE
                ST_DWithin(location, ST_Point(%s, %s)::geography, %s)
                AND occurred_at >= %s
                AND status = 'active'
            ORDER BY distance_meters ASC
            LIMIT %s;
            """

            async with self.pool.acquire() as conn:
                await conn.execute(
                    query,
                    (longitude, latitude, longitude, latitude, radius_meters, time_threshold, limit)
                )
                rows = await conn.fetch()

                incidents = []
                for row in rows:
                    incidents.append({
                        'id': row['id'],
                        'incident_type': row['incident_type'],
                        'severity': row['severity'],
                        'description': row['description'],
                        'latitude': row['latitude'],
                        'longitude': row['longitude'],
                        'occurred_at': row['occurred_at'].isoformat(),
                        'source': row['source'],
                        'confidence_score': row['confidence_score'],
                        'distance_meters': float(row['distance_meters'])
                    })
                
                # Track spatial query metrics
                db_counter = metrics.counter("database_operations_total", "Database operations", ("operation",))
                db_counter.labels("spatial_query").inc()
                logger.info(f"🔍 Found {len(incidents)} incidents within {radius_meters}m")
                
                return incidents
    
    @performance_tracked("database.store_user")
    async def store_user(self, user_data: Dict[str, Any]) -> str:
        """Store user with authentication data"""
        async with self.get_session() as session:
            user = User(
                username=user_data['username'],
                email=user_data['email'],
                hashed_password=user_data['hashed_password'],
                full_name=user_data.get('full_name'),
                role=user_data.get('role', 'citizen')
            )
            
            # Set home location if provided
            if 'home_lat' in user_data and 'home_lng' in user_data:
                user.home_location = func.ST_Point(
                    user_data['home_lng'], 
                    user_data['home_lat']
                )
            
            session.add(user)
            await session.flush()
            
            logger.info(f"👤 Created user {user.username} with ID {user.id}")
            return user.id
    
    @performance_tracked("database.get_user_by_username")
    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username for authentication"""
        query = """
        SELECT id, username, email, hashed_password, full_name, role,
               is_active, is_verified, created_at, last_login
        FROM users
        WHERE username = %s AND is_active = true;
        """

        async with self.pool.acquire() as conn:
            await conn.execute(query, (username,))
            row = await conn.fetchrow()
            if row:
                return dict(row)
            return None
    
    @performance_tracked("database.create_user_alert")
    async def create_user_alert(
        self, 
        user_id: str, 
        incident_id: str, 
        alert_type: str,
        message: str,
        user_lat: float,
        user_lng: float,
        distance: float
    ) -> str:
        """Create geofenced alert for user"""
        async with self.get_session() as session:
            alert = UserAlert(
                user_id=user_id,
                incident_id=incident_id,
                alert_type=alert_type,
                message=message,
                trigger_location=func.ST_Point(user_lng, user_lat),
                distance_from_user=distance
            )
            
            session.add(alert)
            await session.flush()
            
            logger.info(f"🚨 Created alert {alert.id} for user {user_id}")
            return alert.id
    
    @performance_tracked("database.cache_prediction")
    async def cache_prediction(self, prediction_data: Dict[str, Any]) -> str:
        """Cache AI prediction with spatial indexing"""
        async with self.get_session() as session:
            expires_at = datetime.utcnow() + timedelta(hours=prediction_data.get('ttl_hours', 24))
            
            prediction = PredictionCache(
                location=func.ST_Point(
                    prediction_data['longitude'],
                    prediction_data['latitude']
                ),
                crime_type=prediction_data['crime_type'],
                predicted_probability=prediction_data['probability'],
                confidence_score=prediction_data['confidence'],
                prediction_start=prediction_data['prediction_start'],
                prediction_end=prediction_data['prediction_end'],
                model_version=prediction_data['model_version'],
                features_used=json.dumps(prediction_data.get('features', [])),
                contributing_factors=json.dumps(prediction_data.get('factors', [])),
                expires_at=expires_at
            )
            
            session.add(prediction)
            await session.flush()
            
            return prediction.id
    
    @performance_tracked("database.get_cached_predictions")
    async def get_cached_predictions(
        self,
        latitude: float,
        longitude: float,
        radius_meters: float = 1000
    ) -> List[Dict[str, Any]]:
        """Get cached predictions near location"""
        query = """
        SELECT
            id, crime_type, predicted_probability, confidence_score,
            prediction_start, prediction_end, model_version,
            ST_Distance(location, ST_Point(%s, %s)::geography) as distance_meters
        FROM prediction_cache
        WHERE
            ST_DWithin(location, ST_Point(%s, %s)::geography, %s)
            AND expires_at > NOW()
        ORDER BY distance_meters ASC;
        """

        async with self.pool.acquire() as conn:
            await conn.execute(query, (longitude, latitude, longitude, latitude, radius_meters))
            rows = await conn.fetch()

            predictions = []
            for row in rows:
                predictions.append({
                    'id': row['id'],
                    'crime_type': row['crime_type'],
                    'probability': row['predicted_probability'],
                    'confidence': row['confidence_score'],
                    'prediction_start': row['prediction_start'].isoformat(),
                    'prediction_end': row['prediction_end'].isoformat(),
                    'model_version': row['model_version'],
                    'distance_meters': float(row['distance_meters'])
                })

            return predictions
    
    async def cleanup_expired_data(self):
        """Clean up expired predictions and old data"""
        async with self.pool.acquire() as conn:
            # Clean expired predictions
            result = await conn.execute("DELETE FROM prediction_cache WHERE expires_at < NOW();")
            deleted_predictions = int(result.split()[-1]) if result and result.split() else 0

            # Clean old alerts (older than 30 days)
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            result = await conn.execute(
                "DELETE FROM user_alerts WHERE created_at < $1;",
                thirty_days_ago
            )
            deleted_alerts = int(result.split()[-1]) if result and result.split() else 0

            logger.info(f"🧹 Cleaned {deleted_predictions} predictions, {deleted_alerts} alerts")

    # -------------- SCB population support --------------
    @performance_tracked("database.store_scb_population_rows")
    async def store_scb_population_rows(self, rows: List[Dict[str, Any]]) -> int:
        if not rows:
            return 0
        prepared = []
        for r in rows:
            prepared.append((
                str(uuid.uuid4()),
                r.get('region_code',''),
                r.get('region_name'),
                r.get('period',''),
                float(r.get('population',0.0)),
            ))
        insert_sql = """
            INSERT INTO scb_population (id, region_code, region_name, period, population, created_at)
            VALUES (%s,%s,%s,%s,%s,NOW())
        """
        async with self.pool.acquire() as conn:
            await conn.executemany(insert_sql, prepared)
        return len(prepared)

    # -------------- Brå feature support --------------
    @performance_tracked("database.store_bra_features_rows")
    async def store_bra_features_rows(self, rows: List[Dict[str, Any]]) -> int:
        if not rows:
            return 0
        prepared = []
        for r in rows:
            prepared.append((
                str(uuid.uuid4()),
                r.get('region_code'),
                r.get('region_name'),
                r.get('period'),
                r.get('total_offences'),
                r.get('offences_per_100k'),
                r.get('clearance_rate'),
                json.dumps(r.get('sources', [])),
            ))
        insert_sql = """
            INSERT INTO bra_features (
                id, region_code, region_name, period, total_offences, offences_per_100k, clearance_rate, sources, created_at
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,NOW())
        """
        async with self.pool.acquire() as conn:
            await conn.executemany(insert_sql, prepared)
        return len(prepared)

    @performance_tracked("database.get_bra_aggregated_features")
    async def get_bra_aggregated_features(
        self,
        period: str,
        region_code: str = "00",
    ) -> Dict[str, Any]:
        """Fetch aggregated Brå features from bra_features table for a region/period.

        Returns keys: total_offences, offences_per_100k, clearance_rate. If missing,
        returns zeros.
        """
        sql = """
            SELECT total_offences, offences_per_100k, clearance_rate
            FROM bra_features
            WHERE COALESCE(region_code,'') = %s AND period = %s
            LIMIT 1
        """
        async with self.pool.acquire() as conn:
            await conn.execute(sql, (region_code, period))
            row = await conn.fetchrow()
        if not row:
            return {
                'total_offences': 0.0,
                'offences_per_100k': 0.0,
                'clearance_rate': 0.0,
            }
        return {
            'total_offences': float(row['total_offences'] or 0.0),
            'offences_per_100k': float(row['offences_per_100k'] or 0.0),
            'clearance_rate': float(row['clearance_rate'] or 0.0),
        }

    # -------------- Brå (PxWeb) support --------------
    @staticmethod
    def _parse_pxweb_period_to_date(period: str) -> Optional[datetime]:
        """Parse PxWeb period strings like '2023M01' or '2023' to a datetime.

        Returns first day of the month (UTC) for monthly data, or Jan 1 for yearly.
        """
        try:
            if 'M' in period:
                year, month = period.split('M')
                return datetime(int(year), int(month), 1)
            # Yearly
            if len(period) == 4 and period.isdigit():
                return datetime(int(period), 1, 1)
        except Exception:
            return None
        return None

    @performance_tracked("database.store_bra_statistics_rows")
    async def store_bra_statistics_rows(self, rows: List[Dict[str, Any]]) -> int:
        """Bulk insert Brå statistic rows.

        Expected row keys: table_id, region_code, region_name, offence_code,
        offence_name, period, value, unit, extra (dict optional).
        """
        if not rows:
            return 0

        # Prepare rows for executemany
        prepared = []
        for r in rows:
            period_date = self._parse_pxweb_period_to_date(r.get('period', ''))
            extra_json = json.dumps(r.get('extra', {})) if isinstance(r.get('extra'), dict) else (r.get('extra') or None)
            offence_code = r.get('offence_code')
            offence_code_str = str(offence_code).strip() if offence_code is not None else ''
            if not offence_code_str:
                offence_code_str = 'ALL'
            prepared.append(
                (
                    str(uuid.uuid4()),
                    r.get('table_id'),
                    r.get('region_code'),
                    r.get('region_name'),
                    offence_code_str,
                    r.get('offence_name'),
                    r.get('period'),
                    period_date,
                    float(r.get('value', 0.0)),
                    r.get('unit'),
                    extra_json,
                )
            )

        insert_sql = """
            INSERT INTO bra_statistics (
                id, table_id, region_code, region_name, offence_code, offence_name,
                period, period_date, value, unit, extra, created_at
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, NOW())
            ON CONFLICT (table_id, region_code, period, offence_code)
            DO UPDATE SET
                region_name = COALESCE(EXCLUDED.region_name, bra_statistics.region_name),
                offence_name = COALESCE(EXCLUDED.offence_name, bra_statistics.offence_name),
                period_date = EXCLUDED.period_date,
                value = EXCLUDED.value,
                unit = EXCLUDED.unit,
                extra = EXCLUDED.extra
        """

        async with self.pool.acquire() as conn:
            await conn.executemany(insert_sql, prepared)

        db_counter = metrics.counter("database_operations_total", "Database operations", ("operation",))
        db_counter.labels("bra_rows_inserted").inc(len(prepared))
        logger.info(f"📈 Inserted {len(prepared)} Brå statistic rows")
        return len(prepared)

    async def upsert_weather_observations(self, observations: List[Dict[str, Any]]) -> int:
        """Insert or update hourly weather observations mapped to H3 cells."""

        if not observations:
            return 0

        prepared = []
        for obs in observations:
            lat = obs.get('latitude')
            lon = obs.get('longitude')
            if lat is None or lon is None:
                logger.debug("Skipping weather record without coordinates: %s", obs)
                continue
            h3_8 = obs.get('h3_8') or h3_utils.latlon_to_h3(float(lat), float(lon), 8)
            if not h3_8:
                logger.debug("Unable to compute H3 index for weather record: %s", obs)
                continue

            prepared.append(
                (
                    str(obs.get('id') or uuid.uuid4()),
                    obs['observation_time'],
                    obs.get('source', 'smhi'),
                    float(lat),
                    float(lon),
                    h3_8,
                    _maybe_float(obs.get('temperature_c')),
                    _maybe_float(obs.get('feels_like_c')),
                    _maybe_float(obs.get('precipitation_mm')),
                    _maybe_float(obs.get('wind_speed_mps')),
                    _maybe_float(obs.get('wind_direction_deg')),
                    _maybe_float(obs.get('humidity')),
                    _maybe_float(obs.get('pressure_hpa')),
                    _maybe_float(obs.get('snow_depth_cm')),
                    obs.get('conditions'),
                )
            )

        if not prepared:
            return 0

        insert_sql = """
            INSERT INTO weather_hourly (
                id, observation_time, source, latitude, longitude, h3_8,
                temperature_c, feels_like_c, precipitation_mm, wind_speed_mps,
                wind_direction_deg, humidity, pressure_hpa, snow_depth_cm,
                conditions, created_at, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, NOW(), NOW()
            )
            ON CONFLICT (h3_8, observation_time, source)
            DO UPDATE SET
                temperature_c = EXCLUDED.temperature_c,
                feels_like_c = EXCLUDED.feels_like_c,
                precipitation_mm = EXCLUDED.precipitation_mm,
                wind_speed_mps = EXCLUDED.wind_speed_mps,
                wind_direction_deg = EXCLUDED.wind_direction_deg,
                humidity = EXCLUDED.humidity,
                pressure_hpa = EXCLUDED.pressure_hpa,
                snow_depth_cm = EXCLUDED.snow_depth_cm,
                conditions = EXCLUDED.conditions,
                updated_at = NOW()
        """

        async with self.pool.acquire() as conn:
            await conn.executemany(insert_sql, prepared)

        logger.info("Upserted %s weather observations", len(prepared))
        return len(prepared)

    async def upsert_grid_daily_features(self, rows: List[Dict[str, Any]]) -> int:
        """Insert or update daily grid features."""

        if not rows:
            return 0

        prepared = []
        for row in rows:
            feature_date = row.get('feature_date')
            h3_cell = row.get('h3')
            crime_family = row.get('crime_family')
            if not feature_date or not h3_cell or not crime_family:
                logger.debug("Skipping incomplete grid feature row: %s", row)
                continue

            features_payload = row.get('features') or {}
            prepared.append(
                (
                    str(row.get('id') or uuid.uuid4()),
                    h3_cell,
                    feature_date,
                    crime_family,
                    _maybe_float(row.get('incident_count', 0.0)),
                    _maybe_float(row.get('incident_rate', 0.0)),
                    _maybe_float(row.get('avg_severity', 0.0)),
                    _maybe_float(row.get('rolling_7d', 0.0)),
                    _maybe_float(row.get('rolling_30d', 0.0)),
                    _maybe_float(row.get('population', 0.0)),
                    json.dumps(features_payload or {}),
                )
            )

        if not prepared:
            return 0

        insert_sql = """
            INSERT INTO grid_features_daily (
                id, h3, feature_date, crime_family, incident_count, incident_rate,
                avg_severity, rolling_7d, rolling_30d, population, features,
                created_at, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                NOW(), NOW()
            )
            ON CONFLICT (h3, feature_date, crime_family)
            DO UPDATE SET
                incident_count = EXCLUDED.incident_count,
                incident_rate = EXCLUDED.incident_rate,
                avg_severity = EXCLUDED.avg_severity,
                rolling_7d = EXCLUDED.rolling_7d,
                rolling_30d = EXCLUDED.rolling_30d,
                population = EXCLUDED.population,
                features = EXCLUDED.features,
                updated_at = NOW()
        """

        async with self.pool.acquire() as conn:
            await conn.executemany(insert_sql, prepared)

        logger.info("Upserted %s grid daily feature rows", len(prepared))
        return len(prepared)

    async def backfill_h3_indexes(
        self,
        resolutions: Sequence[int] = (8, 9),
        chunk_size: int = 2000,
    ) -> int:
        """Populate missing H3 columns for existing historical crime records."""

        if not self.pool:
            raise RuntimeError("Database pool is not initialised")

        column_map = {8: 'h3_8', 9: 'h3_9'}
        target_columns = [column_map[res] for res in resolutions if res in column_map]
        if not target_columns:
            return 0

        updated_rows = 0
        async with self.pool.acquire() as conn:
            while True:
                where_clause = " OR ".join(f"{col} IS NULL" for col in target_columns)
                await conn.execute(
                    f"""
                        SELECT id, latitude, longitude
                        FROM historical_crime_data
                        WHERE {where_clause}
                        LIMIT %s
                    """,
                    (chunk_size,)
                )
                rows = await conn.fetch()
                if not rows:
                    break

                    updates = []
                    for row in rows:
                        lat = row['latitude']
                        lon = row['longitude']
                        indexes = h3_utils.event_h3_indexes(lat, lon, resolutions)
                        updates.append(
                            (
                                indexes.get(8),
                                indexes.get(9),
                                row['id'],
                            )
                        )

                    await conn.executemany(
                        """
                            UPDATE historical_crime_data
                            SET
                                h3_8 = COALESCE(%s, h3_8),
                                h3_9 = COALESCE(%s, h3_9)
                            WHERE id = %s
                        """,
                        updates,
                    )
                    updated_rows += len(updates)

        if updated_rows:
            logger.info("Backfilled H3 indexes for %s records", updated_rows)
        return updated_rows

    async def upsert_calendar_entries(self, rows: List[Dict[str, Any]]) -> int:
        """Insert or update calendar records (holidays, school breaks)."""

        if not rows:
            return 0

        payload = []
        for row in rows:
            calendar_date = row.get('calendar_date')
            if not calendar_date:
                logger.debug("Skipping calendar row without date: %s", row)
                continue
            payload.append(
                (
                    calendar_date,
                    bool(row.get('is_holiday', False)),
                    row.get('holiday_name'),
                    bool(row.get('is_school_break', False)),
                    row.get('break_name'),
                    int(row.get('week_of_year') or calendar_date.isocalendar().week),
                    int(row.get('month') or calendar_date.month),
                    int(row.get('weekday') or calendar_date.weekday()),
                    json.dumps(row.get('extras') or {}),
                )
            )

        if not payload:
            return 0

        insert_sql = """
            INSERT INTO calendar_dim (
                calendar_date, is_holiday, holiday_name, is_school_break,
                break_name, week_of_year, month, weekday, extras,
                created_at, updated_at
            ) VALUES (
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                NOW(), NOW()
            )
            ON CONFLICT (calendar_date)
            DO UPDATE SET
                is_holiday = EXCLUDED.is_holiday,
                holiday_name = EXCLUDED.holiday_name,
                is_school_break = EXCLUDED.is_school_break,
                break_name = EXCLUDED.break_name,
                week_of_year = EXCLUDED.week_of_year,
                month = EXCLUDED.month,
                weekday = EXCLUDED.weekday,
                extras = EXCLUDED.extras,
                updated_at = NOW()
        """

        async with self.pool.acquire() as conn:
            await conn.executemany(insert_sql, payload)

        logger.info("Upserted %s calendar entries", len(payload))
        return len(payload)

    async def upsert_pois(self, rows: List[Dict[str, Any]]) -> int:
        """Insert or update points of interest with H3 indexes."""

        if not rows:
            return 0

        payload = []
        for row in rows:
            lat = row.get('latitude')
            lon = row.get('longitude')
            if lat is None or lon is None:
                continue
            h3_8 = row.get('h3_8') or h3_utils.latlon_to_h3(float(lat), float(lon), 8)
            h3_9 = row.get('h3_9')
            if h3_9 is None:
                h3_9 = h3_utils.latlon_to_h3(float(lat), float(lon), 9)

            payload.append(
                (
                    str(row.get('id') or uuid.uuid4()),
                    row.get('source', 'osm'),
                    row.get('external_id'),
                    row.get('name'),
                    row.get('category'),
                    row.get('subcategory'),
                    float(lat),
                    float(lon),
                    h3_8,
                    h3_9,
                    json.dumps(row.get('attributes') or {}),
                )
            )

        if not payload:
            return 0

        insert_sql = """
            INSERT INTO poi_dim (
                id, source, external_id, name, category, subcategory,
                latitude, longitude, h3_8, h3_9, attributes,
                created_at, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                NOW(), NOW()
            )
            ON CONFLICT (source, external_id)
            DO UPDATE SET
                name = EXCLUDED.name,
                category = EXCLUDED.category,
                subcategory = EXCLUDED.subcategory,
                latitude = EXCLUDED.latitude,
                longitude = EXCLUDED.longitude,
                h3_8 = EXCLUDED.h3_8,
                h3_9 = EXCLUDED.h3_9,
                attributes = EXCLUDED.attributes,
                updated_at = NOW()
        """

        async with self.pool.acquire() as conn:
            await conn.executemany(insert_sql, payload)

        logger.info("Upserted %s POI records", len(payload))
        return len(payload)

    @performance_tracked("database.get_bra_monthly_offence_features")
    async def get_bra_monthly_offence_features(
        self,
        period: str,
        region_code: str = "00",
        table_id: str = "Nationella_brottsstatistik/b1201",
    ) -> Dict[str, Any]:
        """Compute simple, robust features from Brå monthly reported-offences table.

        Returns a small set of aggregate features usable in models even without
        detailed mapping of offence codes: total, distinct categories, and
        top-category share.
        """
        query = """
            SELECT offence_code, offence_name, SUM(value) AS v
            FROM bra_statistics
            WHERE table_id = %s AND region_code = %s AND period = %s
            GROUP BY offence_code, offence_name
            ORDER BY v DESC
        """
        async with self.pool.acquire() as conn:
            await conn.execute(query, (table_id, region_code, period))
            rows = await conn.fetch()

        if not rows:
            return {
                'bra_total_offences': 0.0,
                'bra_offence_categories': 0.0,
                'bra_top_offence_share': 0.0,
            }

        values = [float(r['v']) for r in rows]
        total = sum(values)
        categories = len(values)
        top_share = (values[0] / total) if total > 0 and values else 0.0
        return {
            'bra_total_offences': total,
            'bra_offence_categories': float(categories),
            'bra_top_offence_share': top_share,
        }
    
    async def get_health_metrics(self) -> Dict[str, Any]:
        """Get database health and performance metrics"""
        async with self.pool.acquire() as conn:
            # Database size
            await conn.execute(
                "SELECT pg_size_pretty(pg_database_size(%s)) as size;",
                (self.config.database,)
            )
            db_size_row = await conn.fetchrow()

            # Table row counts
            await conn.execute("""
                SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del
                FROM pg_stat_user_tables
                WHERE schemaname = 'public';
            """)
            tables_info = await conn.fetch()

            # Connection stats
            await conn.execute("""
                SELECT count(*) as total_connections,
                       count(*) FILTER (WHERE state = 'active') as active_connections
                FROM pg_stat_activity
                WHERE datname = $1;
            """, self.config.database)
            conn_stats = await conn.fetchrow()

        return {
            'database_size': db_size_row['size'],
            'total_connections': conn_stats['total_connections'],
            'active_connections': conn_stats['active_connections'],
            'tables': [dict(row) for row in tables_info],
            'pool_size': self.pool.get_size() if self.pool else 0
        }
    
    async def store_crime_incident(self, incident_data: Dict[str, Any]) -> str:
        """Store crime incident - alias for store_incident"""
        return await self.store_incident(incident_data)
    
    async def get_incidents_near_location(
        self,
        latitude: float,
        longitude: float,
        radius_km: float = 5.0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get incidents near a location"""
        return await self.get_incidents_near(
            latitude=latitude,
            longitude=longitude, 
            radius_meters=radius_km * 1000,  # Convert km to meters
            limit=limit
        )
    
    async def get_recent_incidents(self, hours_back: int = 24, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent incidents within specified hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)

        query = """
        SELECT id, incident_type, severity, description, latitude, longitude,
               occurred_at, source, confidence_score
        FROM crime_incidents
        WHERE occurred_at >= %s AND status = 'active'
        ORDER BY occurred_at DESC
        LIMIT %s;
        """

        async with self.pool.acquire() as conn:
            await conn.execute(query, (cutoff_time, limit))
            rows = await conn.fetch()
            return [dict(row) for row in rows]
    
    async def get_incidents_in_area_timeframe(
        self,
        center_lat: float,
        center_lng: float,
        radius_km: float,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get incidents in area within timeframe"""
        query = """
        SELECT id, incident_type, severity, description, latitude, longitude,
               occurred_at, source, confidence_score,
               ST_Distance(location, ST_Point(%s, %s)) as distance_meters
        FROM crime_incidents
        WHERE ST_DWithin(location, ST_Point(%s, %s), %s)
          AND occurred_at BETWEEN %s AND %s
          AND status = 'active'
        ORDER BY distance_meters ASC, occurred_at DESC
        LIMIT %s;
        """

        async with self.pool.acquire() as conn:
            await conn.execute(
                query,
                (center_lng, center_lat, center_lng, center_lat, radius_km * 1000,
                 start_time, end_time, limit)
            )
            rows = await conn.fetch()
            return [dict(row) for row in rows]
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        async with self.pool.acquire() as conn:
            # Count incidents
            await conn.execute("SELECT COUNT(*) FROM crime_incidents WHERE status = 'active';")
            incident_count = (await conn.fetchrow())[0]

            # Count users
            await conn.execute("SELECT COUNT(*) FROM users WHERE is_active = true;")
            user_count = (await conn.fetchrow())[0]

            # Count cached predictions
            await conn.execute("SELECT COUNT(*) FROM prediction_cache WHERE expires_at > NOW();")
            prediction_count = (await conn.fetchrow())[0]

            return {
                'total_incidents': incident_count or 0,
                'active_users': user_count or 0,
                'cached_predictions': prediction_count or 0
            }
    
    async def cleanup(self):
        """Cleanup database connections"""
        await self.close()

    async def execute_query(self, query: str, *params) -> List[Dict[str, Any]]:
        """Execute raw SQL query and return results as list of dictionaries"""
        async with self.pool.acquire() as conn:
            if params:
                rows = await conn.fetch(query, *params)
            else:
                rows = await conn.fetch(query)
            return [dict(row) for row in rows]

    async def execute_query_single(self, query: str, *params) -> Optional[Dict[str, Any]]:
        """Execute raw SQL query and return single result as dictionary"""
        async with self.pool.acquire() as conn:
            await conn.execute(query, params if params else None)
            row = await conn.fetchrow()
            return dict(row) if row else None

    async def execute_non_query(self, query: str, *params) -> int:
        """Execute non-query SQL statement and return affected row count"""
        async with self.pool.acquire() as conn:
            result = await conn.execute(query, *params)
            return int(result.split()[-1]) if result and result.split() else 0

    async def close(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()
        if self.engine:
            await self.engine.dispose()
        logger.info("Database connections closed")

# Singleton instance
_database = None

async def get_database() -> PostGISDatabase:
    """Get database instance - Railway compatible"""
    global _database
    if _database is None:
        # Create config at runtime, not at module import time
        # This ensures environment variables are available
        if os.getenv('DATABASE_URL'):
            config = DatabaseConfig.from_url(os.getenv('DATABASE_URL'))
        else:
            config = DatabaseConfig()
        _database = PostGISDatabase(config)

    if not _database.engine:
        await _database.initialize()
    return _database
