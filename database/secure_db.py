"""
Secure Database Interface with Privacy Controls
All database operations enforced through privacy and compliance frameworks
"""

import asyncpg
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import uuid
import logging
import hashlib
import re
import os
from dataclasses import dataclass
from enum import Enum

from ..compliance.privacy_framework import (
    privacy_framework,
    DataCategory,
    ProcessingPurpose,
)
from ..audit.audit_system import audit_logger, AuditEventType, AccessResult, RiskLevel
from ..auth.access_control import AccessContext, AccessRequest
from ..data_management.retention_policy import retention_engine
from .query_builder import build_select_sql
from ..common.performance import performance_tracked, memoize_with_ttl


class QueryType(Enum):
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    BULK_SELECT = "bulk_select"
    AGGREGATE = "aggregate"


@dataclass
class DatabaseQuery:
    query_id: str
    sql: str
    query_type: QueryType
    tables: List[str]
    data_categories: List[DataCategory]
    purpose: ProcessingPurpose
    context: AccessContext
    parameters: Optional[Dict[str, Any]] = None
    estimated_records: int = 0


class SecureDatabase:
    def __init__(self, connection_params: Dict[str, str]):
        self.connection_params = connection_params
        self.pool = None
        self.logger = logging.getLogger(__name__)
        self.query_cache: Dict[str, Any] = {}
        self.table_metadata = self._load_table_metadata()

    @memoize_with_ttl(ttl_seconds=3600)  # Cache for 1 hour
    def _load_table_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load metadata about tables and their data classifications"""
        return {
            "persons": {
                "data_categories": [DataCategory.PERSONAL, DataCategory.SENSITIVE],
                "retention_category": "personal_data",
                "classification_level": "restricted",
                "pii_fields": [
                    "first_name",
                    "last_name",
                    "national_id",
                    "phone",
                    "email",
                    "address",
                ],
            },
            "incidents": {
                "data_categories": [DataCategory.PERSONAL],
                "retention_category": "incident_data",
                "classification_level": "official",
                "pii_fields": ["involved_persons", "witness_statements"],
            },
            "criminal_records": {
                "data_categories": [DataCategory.CRIMINAL, DataCategory.SENSITIVE],
                "retention_category": "criminal_data",
                "classification_level": "secret",
                "pii_fields": ["subject_id", "charges", "conviction_details"],
            },
            "biometric_data": {
                "data_categories": [DataCategory.BIOMETRIC],
                "retention_category": "biometric_data",
                "classification_level": "top_secret",
                "pii_fields": ["fingerprints", "dna_profile", "facial_features"],
            },
            "location_tracking": {
                "data_categories": [DataCategory.LOCATION],
                "retention_category": "location_data",
                "classification_level": "restricted",
                "pii_fields": ["coordinates", "timestamp", "device_id"],
            },
            "communications": {
                "data_categories": [
                    DataCategory.COMMUNICATIONS,
                    DataCategory.SENSITIVE,
                ],
                "retention_category": "communications_data",
                "classification_level": "secret",
                "pii_fields": ["sender", "recipient", "content", "metadata"],
            },
            "surveillance_footage": {
                "data_categories": [DataCategory.BIOMETRIC, DataCategory.LOCATION],
                "retention_category": "surveillance_data",
                "classification_level": "restricted",
                "pii_fields": ["camera_location", "timestamp", "detected_faces"],
            },
        }

    async def initialize(self):
        """Initialize database connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                **self.connection_params,
                min_size=int(os.getenv("DB_POOL_MIN_SIZE", "5")),
                max_size=int(os.getenv("DB_POOL_MAX_SIZE", "20")),
                command_timeout=int(os.getenv("DB_COMMAND_TIMEOUT", "30")),
            )
            self.logger.info("Database pool initialized successfully")

            # Create audit tables if they don't exist
            await self._create_audit_tables()

        except Exception as e:
            self.logger.error(f"Failed to initialize database pool: {e}")
            raise

    async def _create_audit_tables(self):
        """Create tables for audit and compliance tracking"""

        audit_schema = """
        CREATE TABLE IF NOT EXISTS data_access_log (
            log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
            user_id VARCHAR(255) NOT NULL,
            session_id VARCHAR(255) NOT NULL,
            table_name VARCHAR(255) NOT NULL,
            operation VARCHAR(50) NOT NULL,
            record_ids TEXT[],
            data_categories VARCHAR(100)[],
            purpose VARCHAR(100) NOT NULL,
            authorization_id VARCHAR(255),
            risk_level VARCHAR(20) NOT NULL,
            ip_address INET,
            user_agent TEXT,
            query_hash VARCHAR(64) NOT NULL,
            affected_subjects TEXT[]
        );
        
        CREATE INDEX IF NOT EXISTS idx_data_access_log_timestamp ON data_access_log(timestamp);
        CREATE INDEX IF NOT EXISTS idx_data_access_log_user ON data_access_log(user_id);
        CREATE INDEX IF NOT EXISTS idx_data_access_log_table ON data_access_log(table_name);
        
        CREATE TABLE IF NOT EXISTS data_retention_tracking (
            record_id UUID PRIMARY KEY,
            table_name VARCHAR(255) NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
            last_accessed TIMESTAMP WITH TIME ZONE,
            data_category VARCHAR(100) NOT NULL,
            purpose VARCHAR(100) NOT NULL,
            retention_status VARCHAR(50) NOT NULL DEFAULT 'active',
            anonymization_date TIMESTAMP WITH TIME ZONE,
            deletion_date TIMESTAMP WITH TIME ZONE,
            legal_hold_reason TEXT,
            subject_id VARCHAR(255),
            source_system VARCHAR(255) NOT NULL,
            legal_basis VARCHAR(100) NOT NULL,
            authorization_id VARCHAR(255)
        );
        
        CREATE INDEX IF NOT EXISTS idx_retention_tracking_status ON data_retention_tracking(retention_status);
        CREATE INDEX IF NOT EXISTS idx_retention_tracking_category ON data_retention_tracking(data_category);
        """

        async with self.pool.acquire() as conn:
            await conn.execute(audit_schema)

    @performance_tracked("secure_db.execute_secure_query")
    async def execute_secure_query(self, query: DatabaseQuery) -> Tuple[bool, Any, str]:
        """Execute query with full security and compliance checks"""

        try:
            # 1. Validate query security
            security_check = await self._validate_query_security(query)
            if not security_check[0]:
                return False, None, security_check[1]

            # 2. Check access permissions
            access_granted = await self._check_database_access(query)
            if not access_granted[0]:
                return False, None, access_granted[1]

            # 3. Apply data minimization
            minimized_query = await self._apply_data_minimization(query)

            # 4. Execute query with monitoring
            result = await self._execute_monitored_query(minimized_query)

            # 5. Apply post-processing filters
            filtered_result = await self._apply_result_filters(result, query)

            # 6. Log access for audit
            await self._log_database_access(
                query,
                True,
                len(filtered_result) if isinstance(filtered_result, list) else 1,
            )

            # 7. Update retention tracking
            await self._update_retention_tracking(query, filtered_result)

            return True, filtered_result, "Query executed successfully"

        except Exception as e:
            await self._log_database_access(query, False, 0, str(e))
            self.logger.error(f"Query execution failed: {e}")
            return False, None, f"Query execution failed: {str(e)}"

    async def _validate_query_security(self, query: DatabaseQuery) -> Tuple[bool, str]:
        """Validate query for security risks"""

        sql_lower = query.sql.lower().strip()

        # Block dangerous SQL patterns
        dangerous_patterns = [
            "drop table",
            "drop database",
            "truncate",
            "alter table",
            "create user",
            "grant",
            "revoke",
            "delete from",
            "update.*set.*=.*select",
            "union.*select",
            "information_schema",
            "pg_catalog",
            "pg_user",
        ]

        for pattern in dangerous_patterns:
            if pattern in sql_lower:
                audit_logger.log_event(
                    event_type=AuditEventType.AUTHORIZATION_CHECK,
                    user_id=query.context.user_id,
                    action="dangerous_sql_blocked",
                    session_id=query.context.session_id,
                    ip_address=query.context.ip_address,
                    user_agent=query.context.user_agent,
                    result=AccessResult.BLOCKED,
                    details={"pattern": pattern, "query_id": query.query_id},
                    risk_level=RiskLevel.CRITICAL,
                )
                return False, f"Dangerous SQL pattern detected: {pattern}"

        # Validate table access
        for table in query.tables:
            if table not in self.table_metadata:
                return False, f"Access to table '{table}' not permitted"

        # Check for bulk operations without proper authorization
        if query.query_type == QueryType.BULK_SELECT:
            if not query.context.authorization_id:
                return False, "Bulk operations require explicit authorization"

            if query.estimated_records > 10000:
                return False, "Bulk operations limited to 10,000 records"

        return True, "Query security validated"

    async def _check_database_access(self, query: DatabaseQuery) -> Tuple[bool, str]:
        """Check access permissions for database query"""

        for table in query.tables:
            table_meta = self.table_metadata.get(table, {})
            table_categories = table_meta.get("data_categories", [])

            # Create access request for each data category
            for category in table_categories:
                access_request = AccessRequest(
                    resource_type="database_table",
                    resource_id=table,
                    action=f"database_{query.query_type.value}",
                    data_category=category,
                    purpose=query.purpose,
                    context=query.context,
                    additional_params={
                        "table": table,
                        "estimated_records": query.estimated_records,
                        "classification": table_meta.get(
                            "classification_level", "restricted"
                        ),
                    },
                )

                # Use access control system
                from ..auth.access_control import access_control

                access_granted, message, risk_level = access_control.check_access(
                    access_request
                )

                if not access_granted:
                    return False, f"Access denied for table {table}: {message}"

        return True, "Database access authorized"

    async def _apply_data_minimization(self, query: DatabaseQuery) -> DatabaseQuery:
        """Apply data minimization to query"""

        # For SELECT queries, limit columns based on purpose and add retention constraints safely
        if query.query_type == QueryType.SELECT:
            query.sql = build_select_sql(query.sql, query.tables, query.purpose)

        # Add row limits for bulk queries
        if query.query_type == QueryType.BULK_SELECT:
            if "limit" not in query.sql.lower():
                query.sql += " LIMIT 1000"  # Default safety limit

        # Retention constraints applied by builder; keep legacy function no-op

        return query

    async def _minimize_select_columns(
        self, sql: str, purpose: ProcessingPurpose, tables: List[str]
    ) -> str:
        """Minimize SELECT columns based on processing purpose"""

        # Define minimal field sets for different purposes
        minimal_fields: Dict[ProcessingPurpose, Dict[str, List[str]]] = {
            ProcessingPurpose.INVESTIGATION: {
                "persons": [
                    "id",
                    "first_name",
                    "last_name",
                    "date_of_birth",
                    "address",
                    "phone",
                ],
                "incidents": [
                    "id",
                    "incident_type",
                    "timestamp",
                    "location",
                    "severity_level",
                ],
                "criminal_records": [
                    "id",
                    "subject_id",
                    "charges",
                    "conviction_date",
                    "status",
                ],
            },
            ProcessingPurpose.PREVENTION: {
                "persons": ["id", "risk_score", "last_known_location", "alert_flags"],
                "incidents": [
                    "id",
                    "incident_type",
                    "location_general",
                    "risk_indicators",
                ],
                "criminal_records": ["id", "risk_category", "recidivism_score"],
            },
            ProcessingPurpose.PUBLIC_SAFETY: {
                "persons": [
                    "id",
                    "emergency_contacts",
                    "medical_alerts",
                    "location_current",
                ],
                "incidents": ["id", "incident_type", "location", "emergency_level"],
                "criminal_records": ["id", "public_safety_flags", "restrictions"],
            },
            ProcessingPurpose.EMERGENCY_RESPONSE: {
                "persons": [
                    "id",
                    "emergency_contacts",
                    "medical_info",
                    "current_location",
                ],
                "incidents": ["id", "emergency_type", "location", "response_required"],
                "criminal_records": [],  # Usually not needed for emergency response
            },
        }

        # Check if SQL is a simple SELECT that can be minimized
        if sql.strip().lower().startswith("select *"):
            allowed_fields = []
            for table in tables:
                purpose_fields = minimal_fields.get(purpose, {})
                table_fields = purpose_fields.get(table, [])
                allowed_fields.extend([f"{table}.{field}" for field in table_fields])

            if allowed_fields:
                field_list = ", ".join(allowed_fields)
                # Replace leading SELECT * (case-insensitive, tolerant of whitespace)
                sql = re.sub(
                    r"(?i)^\s*select\s*\*", f"SELECT {field_list}", sql, count=1
                )

        return sql

    async def _add_retention_constraints(self, sql: str, tables: List[str]) -> str:
        """Deprecated: retention constraints handled by build_select_sql"""
        return sql

    async def _execute_monitored_query(self, query: DatabaseQuery) -> Any:
        """Execute query with performance and security monitoring"""

        start_time = datetime.now()
        query_hash = hashlib.sha256(query.sql.encode()).hexdigest()

        try:
            if self.pool is None:
                raise Exception("Database pool not initialized")
            async with self.pool.acquire() as conn:
                # Set query timeout based on query type
                timeout = 30  # Default 30 seconds
                if query.query_type == QueryType.BULK_SELECT:
                    timeout = 120  # 2 minutes for bulk operations
                elif query.query_type == QueryType.AGGREGATE:
                    timeout = 60  # 1 minute for aggregations

                # Execute with timeout
                if query.parameters:
                    result = await asyncio.wait_for(
                        conn.fetch(query.sql, *query.parameters.values()),
                        timeout=timeout,
                    )
                else:
                    result = await asyncio.wait_for(
                        conn.fetch(query.sql), timeout=timeout
                    )

                execution_time = (datetime.now() - start_time).total_seconds()
                slow_threshold = float(os.getenv("SLOW_QUERY_SECONDS", "1.0"))
                if execution_time > slow_threshold:
                    self.logger.warning(
                        f"Slow query ({execution_time:.3f}s): {query.sql[:200]}"
                    )

                # Log performance metrics
                if execution_time > 10:  # Queries taking more than 10 seconds
                    audit_logger.log_event(
                        event_type=AuditEventType.SYSTEM_ADMIN,
                        user_id=query.context.user_id,
                        action="slow_query_detected",
                        session_id=query.context.session_id,
                        ip_address=query.context.ip_address,
                        user_agent=query.context.user_agent,
                        result=AccessResult.SUCCESS,
                        details={
                            "execution_time": execution_time,
                            "query_hash": query_hash,
                            "result_count": len(result),
                        },
                        risk_level=RiskLevel.MEDIUM,
                    )

                # Metrics (optional)
                try:
                    from ..observability.metrics import (
                        db_queries_total,
                        db_query_duration_seconds,
                    )
                    db_queries_total.labels(type=query.query_type.value).inc()
                    db_query_duration_seconds.labels(type=query.query_type.value).observe(
                        execution_time
                    )
                except Exception:
                    pass

                return [dict(record) for record in result]

        except asyncio.TimeoutError:
            timeout_value = 30  # Default timeout value
            if query.query_type == QueryType.BULK_SELECT:
                timeout_value = 120
            elif query.query_type == QueryType.AGGREGATE:
                timeout_value = 60
            raise Exception(f"Query timeout after {timeout_value} seconds")
        except Exception as e:
            self.logger.error(f"Query execution error: {e}")
            raise

    async def _apply_result_filters(
        self, results: List[Dict[str, Any]], query: DatabaseQuery
    ) -> List[Dict[str, Any]]:
        """Apply post-query filtering and anonymization"""

        if not results:
            return results

        filtered_results = []

        for record in results:
            # Apply privacy framework filters
            filtered_record = privacy_framework.apply_data_minimization(
                record, query.purpose
            )

            # Apply anonymization if required
            anonymization_level = self._determine_anonymization_level(query)
            if anonymization_level != "none":
                filtered_record = privacy_framework.anonymize_data(
                    filtered_record, anonymization_level
                )

            # Remove fields that user doesn't have permission for
            cleaned_record = await self._remove_unauthorized_fields(
                filtered_record, query
            )

            filtered_results.append(cleaned_record)

        return filtered_results

    def _determine_anonymization_level(self, query: DatabaseQuery) -> str:
        """Determine required anonymization level"""

        # Emergency response typically needs full data
        if query.purpose == ProcessingPurpose.EMERGENCY_RESPONSE:
            return "none"

        # Prevention work can use pseudonymized data
        if query.purpose == ProcessingPurpose.PREVENTION:
            return "standard"

        # Public safety often needs identifiable data
        if query.purpose == ProcessingPurpose.PUBLIC_SAFETY:
            return "none"

        # Investigation needs full data with proper authorization
        if query.purpose == ProcessingPurpose.INVESTIGATION:
            if query.context.authorization_id:
                return "none"
            else:
                return "standard"

        return "standard"  # Default to some anonymization

    async def _remove_unauthorized_fields(
        self, record: Dict[str, Any], query: DatabaseQuery
    ) -> Dict[str, Any]:
        """Remove fields user doesn't have permission to see"""

        # This would integrate with the access control system
        # to check field-level permissions

        cleaned_record = record.copy()

        # Remove sensitive fields based on user clearance
        # This is a simplified example - real implementation would be more sophisticated

        sensitive_fields = ["national_id", "phone", "email", "address"]

        # Check if user has permission for sensitive data
        from ..auth.access_control import access_control, Permission

        session = access_control.active_sessions.get(query.context.session_id)
        if session:
            user = access_control.users.get(session["user_id"])
            if user and Permission.READ_SENSITIVE_DATA not in user.permissions:
                for field in sensitive_fields:
                    if field in cleaned_record:
                        cleaned_record[field] = "[REDACTED]"

        return cleaned_record

    async def _log_database_access(
        self,
        query: DatabaseQuery,
        success: bool,
        record_count: int,
        error_message: Optional[str] = None,
    ):
        """Log database access for audit purposes"""

        query_hash = hashlib.sha256(query.sql.encode()).hexdigest()

        # Extract potential subject IDs from results
        affected_subjects: List[str] = []  # Would extract from actual results

        # Log to audit system
        audit_logger.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            user_id=query.context.user_id,
            action=f"database_{query.query_type.value}",
            session_id=query.context.session_id,
            ip_address=query.context.ip_address,
            user_agent=query.context.user_agent,
            result=AccessResult.SUCCESS if success else AccessResult.ERROR,
            resource=",".join(query.tables),
            authorization_id=query.context.authorization_id,
            data_subjects=affected_subjects,
            details={
                "query_id": query.query_id,
                "query_hash": query_hash,
                "tables": query.tables,
                "purpose": query.purpose.value,
                "record_count": record_count,
                "error": error_message,
            },
            risk_level=self._assess_query_risk(query, record_count),
        )

        # Log to database audit table
        try:
            if self.pool is None:
                raise Exception("Database pool not initialized")
            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO data_access_log (
                        user_id, session_id, table_name, operation, record_ids,
                        data_categories, purpose, authorization_id, risk_level,
                        ip_address, user_agent, query_hash, affected_subjects
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                    query.context.user_id,
                    query.context.session_id,
                    ",".join(query.tables),
                    query.query_type.value,
                    [],  # Would contain actual record IDs accessed
                    [cat.value for cat in query.data_categories],
                    query.purpose.value,
                    query.context.authorization_id,
                    self._assess_query_risk(query, record_count).value,
                    query.context.ip_address,
                    query.context.user_agent,
                    query_hash,
                    affected_subjects,
                )
        except Exception as e:
            self.logger.error(f"Failed to log database access: {e}")

    def _assess_query_risk(self, query: DatabaseQuery, record_count: int) -> RiskLevel:
        """Assess risk level of database query"""

        risk_score = 0

        # Table sensitivity
        for table in query.tables:
            table_meta = self.table_metadata.get(table, {})
            classification = table_meta.get("classification_level", "restricted")

            if classification == "top_secret":
                risk_score += 4
            elif classification == "secret":
                risk_score += 3
            elif classification == "restricted":
                risk_score += 2
            else:
                risk_score += 1

        # Data categories
        for category in query.data_categories:
            if category in [DataCategory.BIOMETRIC, DataCategory.SENSITIVE]:
                risk_score += 3
            elif category in [DataCategory.CRIMINAL, DataCategory.COMMUNICATIONS]:
                risk_score += 2
            else:
                risk_score += 1

        # Query type
        if query.query_type == QueryType.BULK_SELECT:
            risk_score += 3
        elif query.query_type == QueryType.DELETE:
            risk_score += 4
        elif query.query_type == QueryType.UPDATE:
            risk_score += 2

        # Record count
        if record_count > 1000:
            risk_score += 3
        elif record_count > 100:
            risk_score += 2
        elif record_count > 10:
            risk_score += 1

        # Map to risk levels
        if risk_score >= 10:
            return RiskLevel.CRITICAL
        elif risk_score >= 7:
            return RiskLevel.HIGH
        elif risk_score >= 4:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    async def _update_retention_tracking(
        self, query: DatabaseQuery, results: List[Dict[str, Any]]
    ):
        """Update retention tracking for accessed records"""

        if not results or query.query_type != QueryType.SELECT:
            return

        # Update access times for retention tracking
        for result in results:
            record_id = result.get("id")
            if record_id:
                # Update in retention engine
                retention_engine.update_access_time(str(record_id))

                # Update in database
                try:
                    if self.pool is None:
                        continue
                    async with self.pool.acquire() as conn:
                        await conn.execute(
                            """
                            UPDATE data_retention_tracking 
                            SET last_accessed = NOW() 
                            WHERE record_id = $1
                        """,
                            record_id,
                        )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to update retention tracking for {record_id}: {e}"
                    )


# Factory function to create secure database instance
async def create_secure_database(connection_params: Dict[str, str]) -> SecureDatabase:
    """Create and initialize secure database instance"""

    db = SecureDatabase(connection_params)
    await db.initialize()
    return db
