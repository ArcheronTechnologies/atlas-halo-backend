"""
Mobile Database Manager
Handles all database operations for the mobile_app schema
"""

import asyncio
import asyncpg
import logging
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone
from uuid import UUID, uuid4
import json
from decimal import Decimal

from .postgis_database import PostGISDatabase

logger = logging.getLogger(__name__)


class MobileDatabaseManager:
    """Database manager specifically for mobile_app schema operations"""

    def __init__(self, database: PostGISDatabase):
        self.db = database
        self.schema = "mobile_app"

    async def get_incidents(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        radius_km: Optional[float] = None,
        incident_types: Optional[List[str]] = None,
        severity_levels: Optional[List[str]] = None,
        hours_back: Optional[int] = 24,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get incidents with spatial and temporal filtering"""

        conditions = []
        params = []
        param_count = 0

        # Time filter
        if hours_back:
            param_count += 1
            conditions.append(f"incident_time >= NOW() - INTERVAL '{hours_back} hours'")

        # Spatial filter
        if lat is not None and lon is not None and radius_km is not None:
            param_count += 1
            conditions.append(f"ST_DWithin(location, ST_SetSRID(ST_MakePoint(${param_count}, ${param_count + 1}), 4326), ${param_count + 2})")
            params.extend([lon, lat, radius_km * 1000])  # Convert km to meters
            param_count += 2

        # Type filter
        if incident_types:
            param_count += 1
            placeholders = ", ".join([f"${param_count + i}" for i in range(len(incident_types))])
            conditions.append(f"incident_type = ANY(ARRAY[{placeholders}])")
            params.extend(incident_types)
            param_count += len(incident_types) - 1

        # Severity filter
        if severity_levels:
            param_count += 1
            placeholders = ", ".join([f"${param_count + i}" for i in range(len(severity_levels))])
            conditions.append(f"severity_level = ANY(ARRAY[{placeholders}])")
            params.extend(severity_levels)
            param_count += len(severity_levels) - 1

        # Exclude false alarms
        conditions.append("resolution_status != 'false_alarm'")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"""
        SELECT
            id,
            incident_type,
            ST_Y(location::geometry) as latitude,
            ST_X(location::geometry) as longitude,
            location_address,
            incident_time,
            reported_time,
            severity_level,
            description,
            source,
            verification_status,
            resolution_status,
            data_quality_score,
            metadata,
            EXTRACT(EPOCH FROM (NOW() - incident_time))/3600 as hours_ago
        FROM {self.schema}.incidents
        WHERE {where_clause}
        ORDER BY incident_time DESC
        LIMIT ${param_count + 1} OFFSET ${param_count + 2}
        """

        params.extend([limit, offset])

        try:
            records = await self.db.execute_query(query, *params)
            return [dict(record) for record in records]
        except Exception as e:
            logger.error(f"Error fetching incidents: {e}")
            raise

    async def create_incident(
        self,
        incident_type: str,
        latitude: float,
        longitude: float,
        description: str,
        user_id: UUID,
        severity_level: str = "moderate",
        location_address: Optional[str] = None,
        incident_time: Optional[datetime] = None,
        media_files: Optional[List[str]] = None
    ) -> UUID:
        """Create a new incident report"""

        incident_id = uuid4()
        incident_time = incident_time or datetime.now(timezone.utc)

        query = f"""
        INSERT INTO {self.schema}.incidents (
            id, incident_type, location, location_address, incident_time,
            reported_time, severity_level, description, source, source_id,
            verification_status, resolution_status, data_quality_score,
            metadata, created_at, updated_at
        ) VALUES (
            $1, $2, ST_SetSRID(ST_MakePoint($3, $4), 4326), $5, $6,
            $7, $8, $9, 'citizen_report', $10,
            'unverified', 'open', 0.7,
            $11, $12, $13
        ) RETURNING id
        """

        metadata = {
            "media_files": media_files or [],
            "reported_via": "mobile_app",
            "user_id": str(user_id)
        }

        params = [
            incident_id, incident_type, longitude, latitude, location_address,
            incident_time, datetime.now(timezone.utc), severity_level, description,
            str(user_id), json.dumps(metadata), datetime.now(timezone.utc),
            datetime.now(timezone.utc)
        ]

        try:
            result = await self.db.execute_query(query, *params)
            logger.info(f"Created incident {incident_id} by user {user_id}")
            return incident_id
        except Exception as e:
            logger.error(f"Error creating incident: {e}")
            raise

    async def get_safety_zones(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        radius_km: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get safety zones with optional spatial filtering"""

        if lat is not None and lon is not None and radius_km is not None:
            query = f"""
            SELECT
                id, zone_name,
                ST_Y(center_location::geometry) as center_latitude,
                ST_X(center_location::geometry) as center_longitude,
                radius_meters, current_risk_level, risk_score,
                incident_count_24h, incident_count_7d, last_incident_time,
                area_type, created_at, updated_at
            FROM {self.schema}.safety_zones
            WHERE ST_DWithin(center_location, ST_SetSRID(ST_MakePoint($1, $2), 4326), $3)
            ORDER BY risk_score DESC
            """
            params = [lon, lat, radius_km * 1000]
        else:
            query = f"""
            SELECT
                id, zone_name,
                ST_Y(center_location::geometry) as center_latitude,
                ST_X(center_location::geometry) as center_longitude,
                radius_meters, current_risk_level, risk_score,
                incident_count_24h, incident_count_7d, last_incident_time,
                area_type, created_at, updated_at
            FROM {self.schema}.safety_zones
            ORDER BY risk_score DESC
            LIMIT 50
            """
            params = []

        try:
            records = await self.db.execute_query(query, params)
            return [dict(record) for record in records]
        except Exception as e:
            logger.error(f"Error fetching safety zones: {e}")
            raise

    async def calculate_location_risk(self, latitude: float, longitude: float, radius_meters: int = 500) -> float:
        """Calculate risk score for a specific location"""

        query = f"""
        SELECT {self.schema}.calculate_risk_score($1, $2, $3) as risk_score
        """

        try:
            result = await self.db.execute_query(query, latitude, longitude, radius_meters)
            return float(result[0]['risk_score']) if result else 0.0
        except Exception as e:
            logger.error(f"Error calculating location risk: {e}")
            return 0.0

    async def create_user(
        self,
        username: str,
        email: str,
        password_hash: str,
        user_type: str = "citizen",
        phone_number: Optional[str] = None
    ) -> UUID:
        """Create a new user account"""

        user_id = uuid4()

        query = f"""
        INSERT INTO {self.schema}.users (
            id, username, email, password_hash, user_type, phone_number,
            is_active, is_verified, created_at, updated_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, true, false, $7, $8
        ) RETURNING id
        """

        params = [
            user_id, username, email, password_hash, user_type, phone_number,
            datetime.now(timezone.utc), datetime.now(timezone.utc)
        ]

        try:
            result = await self.db.execute_query(query, *params)
            logger.info(f"Created user {user_id} with username {username}")
            return user_id
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            raise

    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email address"""

        query = f"""
        SELECT id, username, email, password_hash, user_type, phone_number,
               is_active, email_verified_at, last_login, created_at, updated_at
        FROM {self.schema}.users
        WHERE email = $1 AND is_active = true
        """

        try:
            records = await self.db.execute_query(query, email)
            return dict(records[0]) if records else None
        except Exception as e:
            logger.error(f"Error fetching user by email: {e}")
            return None

    async def get_user_by_id(self, user_id: UUID) -> Optional[Dict[str, Any]]:
        """Get user by ID"""

        query = f"""
        SELECT id, username, email, user_type, phone_number,
               is_active, email_verified_at, last_login, created_at, updated_at,
               personal_number, full_name, given_name, surname
        FROM {self.schema}.users
        WHERE id = $1 AND is_active = true
        """

        try:
            records = await self.db.execute_query(query, user_id)
            return dict(records[0]) if records else None
        except Exception as e:
            logger.error(f"Error fetching user by ID: {e}")
            return None

    async def get_user_by_personal_number(self, personal_number: str) -> Optional[Dict[str, Any]]:
        """Get user by Swedish personal number"""

        query = f"""
        SELECT id, username, email, user_type, phone_number,
               is_active, email_verified_at, last_login, created_at, updated_at,
               personal_number, full_name, given_name, surname
        FROM {self.schema}.users
        WHERE personal_number = $1 AND is_active = true
        """

        try:
            records = await self.db.execute_query(query, personal_number)
            return dict(records[0]) if records else None
        except Exception as e:
            logger.error(f"Error fetching user by personal number: {e}")
            return None

    async def create_user_with_bankid(
        self,
        username: str,
        email: str,
        personal_number: str,
        full_name: str,
        given_name: str = None,
        surname: str = None,
        user_type: str = "citizen"
    ) -> UUID:
        """Create a new user account with BankID data"""

        user_id = uuid4()

        query = f"""
        INSERT INTO {self.schema}.users (
            id, username, email, user_type, is_active, is_verified,
            personal_number, full_name, given_name, surname,
            created_at, updated_at, email_verified_at
        ) VALUES (
            $1, $2, $3, $4, true, true,
            $5, $6, $7, $8,
            $9, $10, $11
        ) RETURNING id
        """

        params = [
            user_id, username, email, user_type,
            personal_number, full_name, given_name, surname,
            datetime.now(timezone.utc), datetime.now(timezone.utc), datetime.now(timezone.utc)
        ]

        try:
            result = await self.db.execute_query(query, *params)
            logger.info(f"Created BankID user {user_id} with personal number")
            return user_id
        except Exception as e:
            logger.error(f"Error creating BankID user: {e}")
            raise

    async def update_user_last_login(self, user_id: UUID) -> None:
        """Update user's last login timestamp"""

        query = f"""
        UPDATE {self.schema}.users
        SET last_login = $1, updated_at = $2
        WHERE id = $3
        """

        try:
            await self.db.execute_query(query,
                datetime.now(timezone.utc),
                datetime.now(timezone.utc),
                user_id
            )
        except Exception as e:
            logger.error(f"Error updating user last login: {e}")

    async def add_watched_location(
        self,
        user_id: UUID,
        latitude: float,
        longitude: float,
        name: str,
        radius_meters: int = 500
    ) -> UUID:
        """Add a watched location for a user"""

        location_id = uuid4()

        query = f"""
        INSERT INTO {self.schema}.watched_locations (
            id, user_id, location, location_name, radius_meters,
            alert_threshold, is_active, created_at, updated_at
        ) VALUES (
            $1, $2, ST_SetSRID(ST_MakePoint($3, $4), 4326), $5, $6,
            'moderate', true, $7, $8
        ) RETURNING id
        """

        params = [
            location_id, user_id, longitude, latitude, name, radius_meters,
            datetime.now(timezone.utc), datetime.now(timezone.utc)
        ]

        try:
            result = await self.db.execute_query(query, params)
            logger.info(f"Added watched location {location_id} for user {user_id}")
            return location_id
        except Exception as e:
            logger.error(f"Error adding watched location: {e}")
            raise

    async def get_user_watched_locations(self, user_id: UUID) -> List[Dict[str, Any]]:
        """Get all watched locations for a user"""

        query = f"""
        SELECT
            id,
            ST_Y(location::geometry) as latitude,
            ST_X(location::geometry) as longitude,
            location_name, radius_meters, alert_threshold,
            is_active, created_at, updated_at
        FROM {self.schema}.watched_locations
        WHERE user_id = $1 AND is_active = true
        ORDER BY created_at DESC
        """

        try:
            records = await self.db.execute_query(query, user_id)
            return [dict(record) for record in records]
        except Exception as e:
            logger.error(f"Error fetching watched locations: {e}")
            raise

    async def log_user_activity(
        self,
        user_id: UUID,
        activity_type: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log user activity for audit trail"""

        query = f"""
        INSERT INTO {self.schema}.user_activity_logs (
            id, user_id, activity_type, description, metadata,
            ip_address, created_at
        ) VALUES (
            $1, $2, $3, $4, $5, $6, $7
        )
        """

        params = [
            uuid4(), user_id, activity_type, description,
            json.dumps(metadata or {}), None,  # IP will be set by middleware
            datetime.now(timezone.utc)
        ]

        try:
            await self.db.execute_query(query, params)
        except Exception as e:
            logger.error(f"Error logging user activity: {e}")

    async def get_incident_statistics(
        self,
        hours_back: int = 24,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        radius_km: Optional[float] = None
    ) -> Dict[str, Any]:
        """Get incident statistics for the mobile dashboard"""

        spatial_filter = ""
        params = [hours_back]

        if lat is not None and lon is not None and radius_km is not None:
            spatial_filter = "AND ST_DWithin(location, ST_SetSRID(ST_MakePoint($2, $3), 4326), $4)"
            params.extend([lon, lat, radius_km * 1000])

        query = f"""
        WITH stats AS (
            SELECT
                COUNT(*) as total_incidents,
                COUNT(CASE WHEN severity_level = 'critical' THEN 1 END) as critical_count,
                COUNT(CASE WHEN severity_level = 'high' THEN 1 END) as high_count,
                COUNT(CASE WHEN severity_level = 'moderate' THEN 1 END) as moderate_count,
                COUNT(CASE WHEN severity_level = 'low' THEN 1 END) as low_count,
                COUNT(CASE WHEN incident_time >= NOW() - INTERVAL '1 hour' THEN 1 END) as last_hour,
                COUNT(CASE WHEN verification_status = 'verified' THEN 1 END) as verified_count,
                AVG(data_quality_score) as avg_quality_score
            FROM {self.schema}.incidents
            WHERE incident_time >= NOW() - INTERVAL '%s hours'
            AND resolution_status != 'false_alarm'
            {spatial_filter}
        ),
        type_breakdown AS (
            SELECT
                incident_type,
                COUNT(*) as count
            FROM {self.schema}.incidents
            WHERE incident_time >= NOW() - INTERVAL '%s hours'
            AND resolution_status != 'false_alarm'
            {spatial_filter}
            GROUP BY incident_type
            ORDER BY count DESC
            LIMIT 10
        )
        SELECT
            s.*,
            json_agg(
                json_build_object('type', t.incident_type, 'count', t.count)
            ) as type_breakdown
        FROM stats s
        CROSS JOIN type_breakdown t
        GROUP BY s.total_incidents, s.critical_count, s.high_count,
                 s.moderate_count, s.low_count, s.last_hour,
                 s.verified_count, s.avg_quality_score
        """ % (hours_back, hours_back)

        try:
            records = await self.db.execute_query(query, params)
            if records:
                result = dict(records[0])
                return result
            else:
                return {
                    'total_incidents': 0,
                    'critical_count': 0,
                    'high_count': 0,
                    'moderate_count': 0,
                    'low_count': 0,
                    'last_hour': 0,
                    'verified_count': 0,
                    'avg_quality_score': 0.0,
                    'type_breakdown': []
                }
        except Exception as e:
            logger.error(f"Error fetching incident statistics: {e}")
            raise


async def get_mobile_database() -> MobileDatabaseManager:
    """Get mobile database manager instance"""
    from .postgis_database import get_database
    db = await get_database()
    return MobileDatabaseManager(db)