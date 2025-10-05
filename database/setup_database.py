#!/usr/bin/env python3
"""
Atlas AI Database Setup and Migration Script
Initializes PostgreSQL database with PostGIS extension and all required tables
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional
import asyncpg
from datetime import datetime

# Add backend to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from postgis_database import DatabaseConfig, PostGISDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseSetup:
    """Database setup and migration manager."""
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self.db = PostGISDatabase(self.config)
        
    async def setup_database(self, force_recreate: bool = False) -> bool:
        """
        Complete database setup including extensions, tables, and initial data.
        
        Args:
            force_recreate: If True, drops and recreates all tables
            
        Returns:
            bool: True if successful, False otherwise
        """
        
        try:
            logger.info("🚀 Starting Atlas AI database setup...")
            
            # Test connection
            await self._test_connection()
            
            # Setup extensions
            await self._setup_extensions()
            
            # Create tables
            if force_recreate:
                logger.info("⚠️ Force recreate enabled - dropping existing tables")
                await self._drop_all_tables()
            
            await self._create_core_tables()
            await self._create_sensor_fusion_tables()
            await self._create_threat_alert_tables()
            await self._create_ai_training_tables()
            await self._create_user_management_tables()
            await self._create_analytics_tables()
            
            # Setup indexes
            await self._create_indexes()
            
            # Setup functions and triggers
            await self._create_functions_and_triggers()
            
            # Insert initial data
            await self._insert_initial_data()
            
            # Verify setup
            await self._verify_setup()
            
            logger.info("✅ Database setup completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            return False
    
    async def _test_connection(self):
        """Test database connection."""
        
        logger.info("Testing database connection...")
        try:
            result = await self.db.execute_query("SELECT version()")
            if result:
                version = result[0]['version']
                logger.info(f"✅ Connected to PostgreSQL: {version}")
            else:
                raise Exception("No version information returned")
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            raise
    
    async def _setup_extensions(self):
        """Setup required PostgreSQL extensions."""
        
        logger.info("Setting up database extensions...")
        
        extensions = [
            'postgis',
            'postgis_topology', 
            'uuid-ossp',
            'btree_gin',
            'pg_trgm'
        ]
        
        for ext in extensions:
            try:
                await self.db.execute_query(f"CREATE EXTENSION IF NOT EXISTS \"{ext}\"")
                logger.info(f"✅ Extension enabled: {ext}")
            except Exception as e:
                logger.warning(f"⚠️ Could not enable extension {ext}: {e}")
    
    async def _drop_all_tables(self):
        """Drop all Atlas AI tables."""
        
        logger.info("Dropping existing tables...")
        
        # Get all Atlas AI tables
        tables_query = """
        SELECT tablename FROM pg_tables 
        WHERE schemaname = 'public' 
        AND (tablename LIKE 'atlas_%' 
             OR tablename IN (
                'threat_detections', 'threat_alerts', 'alert_dismissals', 
                'alert_recipients', 'alert_statistics', 'user_alert_preferences',
                'alert_zones', 'behavior_incidents', 'person_identities',
                'cross_user_alerts', 'training_jobs', 'model_artifacts',
                'user_feedback', 'prediction_accuracy', 'hotspot_areas',
                'incidents', 'users', 'sensors', 'sensor_data'
             ))
        ORDER BY tablename
        """
        
        result = await self.db.execute_query(tables_query)
        
        if result:
            for row in result:
                table_name = row['tablename']
                try:
                    await self.db.execute_query(f"DROP TABLE IF EXISTS {table_name} CASCADE")
                    logger.info(f"🗑️ Dropped table: {table_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not drop table {table_name}: {e}")
    
    async def _create_core_tables(self):
        """Create core Atlas AI tables."""
        
        logger.info("Creating core tables...")
        
        # Users table
        await self.db.execute_query("""
            CREATE TABLE IF NOT EXISTS users (
                user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                email VARCHAR(255) UNIQUE NOT NULL,
                username VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                first_name VARCHAR(100),
                last_name VARCHAR(100),
                role VARCHAR(50) NOT NULL DEFAULT 'citizen' CHECK (role IN ('citizen', 'law_enforcement', 'admin', 'system')),
                permissions TEXT[] DEFAULT ARRAY[]::TEXT[],
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                is_verified BOOLEAN NOT NULL DEFAULT FALSE,
                phone_number VARCHAR(20),
                location GEOMETRY(POINT, 4326),
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                last_login TIMESTAMP,
                failed_login_attempts INTEGER DEFAULT 0,
                locked_until TIMESTAMP,
                metadata JSONB DEFAULT '{}'::JSONB
            )
        """)
        
        # Incidents table
        await self.db.execute_query("""
            CREATE TABLE IF NOT EXISTS incidents (
                incident_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                incident_type VARCHAR(100) NOT NULL,
                severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
                status VARCHAR(20) NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'investigating', 'closed', 'resolved')),
                title VARCHAR(255) NOT NULL,
                description TEXT,
                location GEOMETRY(POINT, 4326) NOT NULL,
                address TEXT,
                reported_by UUID REFERENCES users(user_id),
                assigned_to UUID REFERENCES users(user_id),
                occurred_at TIMESTAMP NOT NULL,
                reported_at TIMESTAMP NOT NULL DEFAULT NOW(),
                resolved_at TIMESTAMP,
                evidence JSONB DEFAULT '[]'::JSONB,
                metadata JSONB DEFAULT '{}'::JSONB,
                tags TEXT[] DEFAULT ARRAY[]::TEXT[]
            )
        """)
        
        # Sensors table
        await self.db.execute_query("""
            CREATE TABLE IF NOT EXISTS sensors (
                sensor_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                sensor_external_id VARCHAR(255) UNIQUE NOT NULL,
                sensor_type VARCHAR(50) NOT NULL CHECK (sensor_type IN ('camera', 'audio', 'motion', 'temperature', 'air_quality', 'traffic')),
                status VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'maintenance', 'error')),
                location GEOMETRY(POINT, 4326) NOT NULL,
                address TEXT,
                installed_by UUID REFERENCES users(user_id),
                installed_at TIMESTAMP NOT NULL DEFAULT NOW(),
                last_ping TIMESTAMP,
                configuration JSONB DEFAULT '{}'::JSONB,
                metadata JSONB DEFAULT '{}'::JSONB,
                capabilities TEXT[] DEFAULT ARRAY[]::TEXT[]
            )
        """)
        
        # Sensor data table
        await self.db.execute_query("""
            CREATE TABLE IF NOT EXISTS sensor_data (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                sensor_id UUID NOT NULL REFERENCES sensors(sensor_id) ON DELETE CASCADE,
                data_type VARCHAR(50) NOT NULL,
                timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                location GEOMETRY(POINT, 4326),
                raw_data JSONB NOT NULL,
                processed_data JSONB DEFAULT '{}'::JSONB,
                quality_score REAL DEFAULT 1.0 CHECK (quality_score >= 0 AND quality_score <= 1),
                file_path TEXT,
                file_size BIGINT,
                file_hash VARCHAR(64),
                metadata JSONB DEFAULT '{}'::JSONB
            )
        """)
        
        logger.info("✅ Core tables created")
    
    async def _create_sensor_fusion_tables(self):
        """Create sensor fusion and threat detection tables."""
        
        logger.info("Creating sensor fusion tables...")
        
        # Threat detections table
        await self.db.execute_query("""
            CREATE TABLE IF NOT EXISTS threat_detections (
                detection_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID REFERENCES users(user_id),
                threat_level VARCHAR(20) NOT NULL CHECK (threat_level IN ('low', 'medium', 'high', 'critical')),
                threat_score REAL NOT NULL CHECK (threat_score >= 0 AND threat_score <= 1),
                confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
                threat_types TEXT[] NOT NULL,
                location GEOMETRY(POINT, 4326) NOT NULL,
                timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                processing_time_ms REAL,
                alert_radius_meters REAL DEFAULT 250.0,
                requires_emergency BOOLEAN DEFAULT FALSE,
                features JSONB DEFAULT '{}'::JSONB,
                video_analysis JSONB DEFAULT '{}'::JSONB,
                audio_analysis JSONB DEFAULT '{}'::JSONB,
                ground_truth_label VARCHAR(50),
                verified_by UUID REFERENCES users(user_id),
                verified_at TIMESTAMP,
                metadata JSONB DEFAULT '{}'::JSONB
            )
        """)
        
        # Person identities table for cross-user tracking
        await self.db.execute_query("""
            CREATE TABLE IF NOT EXISTS person_identities (
                person_hash VARCHAR(64) PRIMARY KEY,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                last_seen TIMESTAMP NOT NULL DEFAULT NOW(),
                detection_count INTEGER NOT NULL DEFAULT 1,
                locations GEOMETRY(POINT, 4326)[] DEFAULT ARRAY[]::GEOMETRY[],
                threat_escalation_score REAL DEFAULT 0.0,
                behavioral_profile JSONB DEFAULT '{}'::JSONB,
                is_anonymous BOOLEAN NOT NULL DEFAULT TRUE,
                metadata JSONB DEFAULT '{}'::JSONB
            )
        """)
        
        # Behavior incidents table
        await self.db.execute_query("""
            CREATE TABLE IF NOT EXISTS behavior_incidents (
                incident_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                person_hash VARCHAR(64) REFERENCES person_identities(person_hash),
                primary_detection_id UUID NOT NULL REFERENCES threat_detections(detection_id),
                related_detection_ids UUID[] DEFAULT ARRAY[]::UUID[],
                incident_type VARCHAR(50) NOT NULL,
                escalation_level VARCHAR(20) NOT NULL CHECK (escalation_level IN ('low', 'medium', 'high', 'critical')),
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                location_cluster GEOMETRY(POINT, 4326)[] NOT NULL,
                progression_analysis JSONB DEFAULT '{}'::JSONB,
                intervention_recommended BOOLEAN DEFAULT FALSE,
                intervention_type VARCHAR(50),
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                metadata JSONB DEFAULT '{}'::JSONB
            )
        """)
        
        # Cross-user alerts table
        await self.db.execute_query("""
            CREATE TABLE IF NOT EXISTS cross_user_alerts (
                alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                person_hash VARCHAR(64) NOT NULL REFERENCES person_identities(person_hash),
                alert_type VARCHAR(50) NOT NULL,
                severity VARCHAR(20) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
                location GEOMETRY(POINT, 4326) NOT NULL,
                radius_meters REAL NOT NULL DEFAULT 500.0,
                message TEXT NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                created_by UUID NOT NULL REFERENCES users(user_id),
                alert_data JSONB DEFAULT '{}'::JSONB,
                is_active BOOLEAN NOT NULL DEFAULT TRUE
            )
        """)
        
        logger.info("✅ Sensor fusion tables created")
    
    async def _create_threat_alert_tables(self):
        """Create threat alert system tables."""
        
        logger.info("Creating threat alert tables...")
        
        # Execute the SQL from our existing schema file
        schema_file = Path(__file__).parent / "threat_alerts_schema.sql"
        if schema_file.exists():
            with open(schema_file, 'r') as f:
                sql_content = f.read()
                
                # Split by ; and execute each statement
                statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
                
                for stmt in statements:
                    if stmt and not stmt.startswith('--'):
                        try:
                            await self.db.execute_query(stmt)
                        except Exception as e:
                            logger.warning(f"⚠️ Could not execute statement: {e}")
        
        logger.info("✅ Threat alert tables created")
    
    async def _create_ai_training_tables(self):
        """Create AI training and model management tables."""
        
        logger.info("Creating AI training tables...")
        
        # Training jobs table
        await self.db.execute_query("""
            CREATE TABLE IF NOT EXISTS training_jobs (
                job_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                model_type VARCHAR(100) NOT NULL,
                status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
                dataset_size INTEGER NOT NULL DEFAULT 0,
                epochs INTEGER NOT NULL DEFAULT 100,
                batch_size INTEGER NOT NULL DEFAULT 32,
                learning_rate REAL NOT NULL DEFAULT 0.001,
                validation_split REAL NOT NULL DEFAULT 0.2,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                created_by UUID NOT NULL REFERENCES users(user_id),
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                config JSONB DEFAULT '{}'::JSONB,
                metrics JSONB DEFAULT '{}'::JSONB,
                model_path TEXT,
                error_message TEXT,
                logs TEXT
            )
        """)
        
        # Model artifacts table
        await self.db.execute_query("""
            CREATE TABLE IF NOT EXISTS model_artifacts (
                artifact_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                training_job_id UUID NOT NULL REFERENCES training_jobs(job_id) ON DELETE CASCADE,
                model_type VARCHAR(100) NOT NULL,
                version VARCHAR(50) NOT NULL,
                file_path TEXT NOT NULL,
                file_size BIGINT,
                file_hash VARCHAR(64),
                is_active BOOLEAN NOT NULL DEFAULT FALSE,
                performance_metrics JSONB DEFAULT '{}'::JSONB,
                deployment_config JSONB DEFAULT '{}'::JSONB,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                deployed_at TIMESTAMP,
                deprecated_at TIMESTAMP,
                metadata JSONB DEFAULT '{}'::JSONB
            )
        """)
        
        # User feedback table
        await self.db.execute_query("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                feedback_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                detection_id UUID NOT NULL REFERENCES threat_detections(detection_id),
                user_id UUID NOT NULL REFERENCES users(user_id),
                feedback_type VARCHAR(50) NOT NULL CHECK (feedback_type IN ('accuracy', 'false_positive', 'false_negative', 'severity', 'general')),
                accuracy_score REAL CHECK (accuracy_score >= 0 AND accuracy_score <= 1),
                false_positive BOOLEAN,
                threat_confirmed BOOLEAN,
                correct_threat_level VARCHAR(20) CHECK (correct_threat_level IN ('low', 'medium', 'high', 'critical')),
                comments TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                is_verified BOOLEAN NOT NULL DEFAULT FALSE,
                verified_by UUID REFERENCES users(user_id),
                verified_at TIMESTAMP,
                metadata JSONB DEFAULT '{}'::JSONB
            )
        """)
        
        # Prediction accuracy tracking
        await self.db.execute_query("""
            CREATE TABLE IF NOT EXISTS prediction_accuracy (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                model_type VARCHAR(100) NOT NULL,
                model_version VARCHAR(50) NOT NULL,
                prediction_type VARCHAR(100) NOT NULL,
                actual_outcome VARCHAR(100),
                predicted_outcome VARCHAR(100),
                confidence_score REAL,
                accuracy_score REAL,
                prediction_time TIMESTAMP NOT NULL,
                verification_time TIMESTAMP,
                location GEOMETRY(POINT, 4326),
                metadata JSONB DEFAULT '{}'::JSONB
            )
        """)
        
        logger.info("✅ AI training tables created")
    
    async def _create_user_management_tables(self):
        """Create user management and authentication tables."""
        
        logger.info("Creating user management tables...")
        
        # User sessions table
        await self.db.execute_query("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                token_hash VARCHAR(255) NOT NULL,
                refresh_token_hash VARCHAR(255),
                ip_address INET,
                user_agent TEXT,
                location GEOMETRY(POINT, 4326),
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                expires_at TIMESTAMP NOT NULL,
                last_activity TIMESTAMP NOT NULL DEFAULT NOW(),
                is_active BOOLEAN NOT NULL DEFAULT TRUE,
                device_info JSONB DEFAULT '{}'::JSONB
            )
        """)
        
        # User preferences table
        await self.db.execute_query("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id UUID PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
                notification_settings JSONB DEFAULT '{
                    "email_notifications": true,
                    "push_notifications": true,
                    "sms_notifications": false,
                    "alert_radius_meters": 1000,
                    "min_threat_level": "medium"
                }'::JSONB,
                privacy_settings JSONB DEFAULT '{
                    "location_sharing": true,
                    "anonymous_reporting": false,
                    "data_retention_days": 365
                }'::JSONB,
                ui_preferences JSONB DEFAULT '{
                    "theme": "light",
                    "language": "en",
                    "map_style": "standard"
                }'::JSONB,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """)
        
        # Audit log table
        await self.db.execute_query("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID REFERENCES users(user_id),
                session_id UUID REFERENCES user_sessions(session_id),
                event_type VARCHAR(100) NOT NULL,
                action VARCHAR(255) NOT NULL,
                resource_type VARCHAR(100),
                resource_id VARCHAR(255),
                ip_address INET,
                user_agent TEXT,
                location GEOMETRY(POINT, 4326),
                success BOOLEAN NOT NULL DEFAULT TRUE,
                error_message TEXT,
                request_data JSONB DEFAULT '{}'::JSONB,
                response_data JSONB DEFAULT '{}'::JSONB,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                risk_level VARCHAR(20) DEFAULT 'low' CHECK (risk_level IN ('low', 'medium', 'high', 'critical'))
            )
        """)
        
        logger.info("✅ User management tables created")
    
    async def _create_analytics_tables(self):
        """Create analytics and reporting tables."""
        
        logger.info("Creating analytics tables...")
        
        # Hotspot areas table
        await self.db.execute_query("""
            CREATE TABLE IF NOT EXISTS hotspot_areas (
                hotspot_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                area_name VARCHAR(255) NOT NULL,
                boundary GEOMETRY(POLYGON, 4326) NOT NULL,
                center_point GEOMETRY(POINT, 4326) NOT NULL,
                risk_level VARCHAR(20) NOT NULL CHECK (risk_level IN ('low', 'medium', 'high', 'critical')),
                incident_count INTEGER NOT NULL DEFAULT 0,
                threat_detection_count INTEGER NOT NULL DEFAULT 0,
                population_density REAL,
                crime_types TEXT[] DEFAULT ARRAY[]::TEXT[],
                peak_hours INTEGER[] DEFAULT ARRAY[]::INTEGER[],
                seasonal_patterns JSONB DEFAULT '{}'::JSONB,
                last_incident TIMESTAMP,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                metadata JSONB DEFAULT '{}'::JSONB,
                is_active BOOLEAN NOT NULL DEFAULT TRUE
            )
        """)
        
        # System metrics table
        await self.db.execute_query("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                metric_name VARCHAR(100) NOT NULL,
                metric_type VARCHAR(50) NOT NULL CHECK (metric_type IN ('counter', 'gauge', 'histogram', 'summary')),
                value REAL NOT NULL,
                labels JSONB DEFAULT '{}'::JSONB,
                timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                source VARCHAR(100) NOT NULL DEFAULT 'atlas_ai',
                metadata JSONB DEFAULT '{}'::JSONB
            )
        """)
        
        # API usage statistics
        await self.db.execute_query("""
            CREATE TABLE IF NOT EXISTS api_usage_stats (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                endpoint VARCHAR(255) NOT NULL,
                method VARCHAR(10) NOT NULL,
                user_id UUID REFERENCES users(user_id),
                response_status INTEGER NOT NULL,
                response_time_ms REAL NOT NULL,
                request_size BIGINT,
                response_size BIGINT,
                ip_address INET,
                user_agent TEXT,
                timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                date DATE NOT NULL DEFAULT CURRENT_DATE,
                hour INTEGER NOT NULL DEFAULT EXTRACT(HOUR FROM NOW())
            )
        """)
        
        logger.info("✅ Analytics tables created")
    
    async def _create_indexes(self):
        """Create database indexes for performance."""
        
        logger.info("Creating database indexes...")
        
        indexes = [
            # Core tables
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
            "CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)",
            "CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)",
            "CREATE INDEX IF NOT EXISTS idx_users_location ON users USING GIST(location)",
            "CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at)",
            
            # Incidents
            "CREATE INDEX IF NOT EXISTS idx_incidents_type ON incidents(incident_type)",
            "CREATE INDEX IF NOT EXISTS idx_incidents_severity ON incidents(severity)",
            "CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status)",
            "CREATE INDEX IF NOT EXISTS idx_incidents_location ON incidents USING GIST(location)",
            "CREATE INDEX IF NOT EXISTS idx_incidents_occurred_at ON incidents(occurred_at)",
            "CREATE INDEX IF NOT EXISTS idx_incidents_reported_by ON incidents(reported_by)",
            
            # Sensors
            "CREATE INDEX IF NOT EXISTS idx_sensors_type ON sensors(sensor_type)",
            "CREATE INDEX IF NOT EXISTS idx_sensors_status ON sensors(status)",
            "CREATE INDEX IF NOT EXISTS idx_sensors_location ON sensors USING GIST(location)",
            "CREATE INDEX IF NOT EXISTS idx_sensors_external_id ON sensors(sensor_external_id)",
            
            # Sensor data
            "CREATE INDEX IF NOT EXISTS idx_sensor_data_sensor_id ON sensor_data(sensor_id)",
            "CREATE INDEX IF NOT EXISTS idx_sensor_data_timestamp ON sensor_data(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_sensor_data_type ON sensor_data(data_type)",
            "CREATE INDEX IF NOT EXISTS idx_sensor_data_location ON sensor_data USING GIST(location)",
            
            # Threat detections
            "CREATE INDEX IF NOT EXISTS idx_threat_detections_user_id ON threat_detections(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_threat_detections_level ON threat_detections(threat_level)",
            "CREATE INDEX IF NOT EXISTS idx_threat_detections_location ON threat_detections USING GIST(location)",
            "CREATE INDEX IF NOT EXISTS idx_threat_detections_timestamp ON threat_detections(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_threat_detections_score ON threat_detections(threat_score DESC)",
            "CREATE INDEX IF NOT EXISTS idx_threat_detections_types ON threat_detections USING GIN(threat_types)",
            
            # Person identities
            "CREATE INDEX IF NOT EXISTS idx_person_identities_last_seen ON person_identities(last_seen)",
            "CREATE INDEX IF NOT EXISTS idx_person_identities_escalation ON person_identities(threat_escalation_score DESC)",
            
            # Behavior incidents
            "CREATE INDEX IF NOT EXISTS idx_behavior_incidents_person ON behavior_incidents(person_hash)",
            "CREATE INDEX IF NOT EXISTS idx_behavior_incidents_type ON behavior_incidents(incident_type)",
            "CREATE INDEX IF NOT EXISTS idx_behavior_incidents_escalation ON behavior_incidents(escalation_level)",
            "CREATE INDEX IF NOT EXISTS idx_behavior_incidents_start_time ON behavior_incidents(start_time)",
            
            # User sessions
            "CREATE INDEX IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions(token_hash)",
            "CREATE INDEX IF NOT EXISTS idx_user_sessions_active ON user_sessions(is_active, expires_at)",
            
            # Audit logs
            "CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_audit_logs_event_type ON audit_logs(event_type)",
            "CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_audit_logs_risk_level ON audit_logs(risk_level)",
            
            # Training jobs
            "CREATE INDEX IF NOT EXISTS idx_training_jobs_status ON training_jobs(status)",
            "CREATE INDEX IF NOT EXISTS idx_training_jobs_created_by ON training_jobs(created_by)",
            "CREATE INDEX IF NOT EXISTS idx_training_jobs_created_at ON training_jobs(created_at)",
            
            # User feedback
            "CREATE INDEX IF NOT EXISTS idx_user_feedback_detection_id ON user_feedback(detection_id)",
            "CREATE INDEX IF NOT EXISTS idx_user_feedback_user_id ON user_feedback(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_user_feedback_type ON user_feedback(feedback_type)",
            "CREATE INDEX IF NOT EXISTS idx_user_feedback_created_at ON user_feedback(created_at)",
            
            # Hotspot areas
            "CREATE INDEX IF NOT EXISTS idx_hotspot_areas_boundary ON hotspot_areas USING GIST(boundary)",
            "CREATE INDEX IF NOT EXISTS idx_hotspot_areas_center ON hotspot_areas USING GIST(center_point)",
            "CREATE INDEX IF NOT EXISTS idx_hotspot_areas_risk_level ON hotspot_areas(risk_level)",
            "CREATE INDEX IF NOT EXISTS idx_hotspot_areas_active ON hotspot_areas(is_active)",
            
            # System metrics
            "CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON system_metrics(metric_name)",
            "CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_system_metrics_type ON system_metrics(metric_type)",
            
            # API usage stats
            "CREATE INDEX IF NOT EXISTS idx_api_usage_endpoint ON api_usage_stats(endpoint)",
            "CREATE INDEX IF NOT EXISTS idx_api_usage_timestamp ON api_usage_stats(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_api_usage_date_hour ON api_usage_stats(date, hour)",
            "CREATE INDEX IF NOT EXISTS idx_api_usage_user_id ON api_usage_stats(user_id)",
        ]
        
        for index_sql in indexes:
            try:
                await self.db.execute_query(index_sql)
            except Exception as e:
                logger.warning(f"⚠️ Could not create index: {e}")
        
        logger.info("✅ Database indexes created")
    
    async def _create_functions_and_triggers(self):
        """Create database functions and triggers."""
        
        logger.info("Creating database functions and triggers...")
        
        # Update timestamp trigger function
        await self.db.execute_query("""
            CREATE OR REPLACE FUNCTION update_updated_at_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at = NOW();
                RETURN NEW;
            END;
            $$ language 'plpgsql';
        """)
        
        # Triggers for updated_at columns
        triggers = [
            "CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column()",
            "CREATE TRIGGER update_user_preferences_updated_at BEFORE UPDATE ON user_preferences FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column()",
            "CREATE TRIGGER update_hotspot_areas_updated_at BEFORE UPDATE ON hotspot_areas FOR EACH ROW EXECUTE PROCEDURE update_updated_at_column()"
        ]
        
        for trigger_sql in triggers:
            try:
                await self.db.execute_query(trigger_sql)
            except Exception as e:
                logger.warning(f"⚠️ Could not create trigger: {e}")
        
        logger.info("✅ Database functions and triggers created")
    
    async def _insert_initial_data(self):
        """Insert initial data and configuration."""
        
        logger.info("Inserting initial data...")
        
        # Create default admin user if not exists
        admin_exists = await self.db.execute_query(
            "SELECT user_id FROM users WHERE role = 'admin' LIMIT 1"
        )
        
        if not admin_exists:
            await self.db.execute_query("""
                INSERT INTO users (email, username, password_hash, first_name, last_name, role, is_active, is_verified)
                VALUES ('admin@atlas-ai.com', 'admin', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewGY5AkrHxjkfHRe', 
                       'System', 'Administrator', 'admin', true, true)
                ON CONFLICT (email) DO NOTHING
            """)
            logger.info("✅ Default admin user created (password: admin123)")
        
        # Create default user preferences for admin
        await self.db.execute_query("""
            INSERT INTO user_preferences (user_id)
            SELECT user_id FROM users WHERE username = 'admin'
            ON CONFLICT (user_id) DO NOTHING
        """)
        
        logger.info("✅ Initial data inserted")
    
    async def _verify_setup(self):
        """Verify database setup was successful."""
        
        logger.info("Verifying database setup...")
        
        # Check that all tables exist
        tables_query = """
            SELECT COUNT(*) as table_count
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
        """
        
        result = await self.db.execute_query(tables_query)
        table_count = result[0]['table_count']
        
        if table_count < 20:  # We expect at least 20 tables
            logger.warning(f"⚠️ Only {table_count} tables found, expected more")
        else:
            logger.info(f"✅ {table_count} tables verified")
        
        # Check PostGIS extension
        postgis_check = await self.db.execute_query(
            "SELECT PostGIS_Version() as version"
        )
        
        if postgis_check:
            logger.info(f"✅ PostGIS extension verified: {postgis_check[0]['version']}")
        
        # Check admin user
        admin_check = await self.db.execute_query(
            "SELECT username FROM users WHERE role = 'admin' LIMIT 1"
        )
        
        if admin_check:
            logger.info(f"✅ Admin user verified: {admin_check[0]['username']}")
        
        logger.info("✅ Database verification completed")


async def main():
    """Main setup function."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Atlas AI Database Setup')
    parser.add_argument('--force-recreate', action='store_true', 
                       help='Drop and recreate all tables')
    parser.add_argument('--config', type=str, 
                       help='Database configuration file')
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config:
        # Load from config file
        pass
    else:
        # Use environment variables or defaults
        config = DatabaseConfig()
    
    # Setup database
    setup = DatabaseSetup(config)
    success = await setup.setup_database(force_recreate=args.force_recreate)
    
    if success:
        logger.info("🎉 Atlas AI database setup completed successfully!")
        return 0
    else:
        logger.error("❌ Database setup failed!")
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))