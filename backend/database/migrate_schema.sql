-- Migration script to update existing Atlas AI database to production schema
-- This preserves existing data while adding new mobile app tables

BEGIN;

-- Check and enable extensions if not already present
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_topology;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Backup existing users table structure if it exists
DO $$
BEGIN
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'users' AND table_schema = 'public') THEN
        -- Check if our production users table structure exists
        IF NOT EXISTS (SELECT FROM information_schema.columns WHERE table_name = 'users' AND column_name = 'email' AND table_schema = 'public') THEN
            -- Rename existing users table to backup
            ALTER TABLE users RENAME TO users_backup_legacy;
        END IF;
    END IF;
END $$;

-- Create or update users table for production
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    phone_number VARCHAR(20),
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    is_verified BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    email_verified_at TIMESTAMP WITH TIME ZONE,
    phone_verified_at TIMESTAMP WITH TIME ZONE
);

-- User sessions for JWT token management
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    device_id VARCHAR(255),
    device_info JSONB,
    ip_address INET,
    user_agent TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_used_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Migrate existing crime_incidents to new incidents table
DO $$
BEGIN
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'crime_incidents' AND table_schema = 'public') THEN
        -- Create new incidents table
        CREATE TABLE IF NOT EXISTS incidents (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            incident_type VARCHAR(50) NOT NULL,
            location GEOGRAPHY(POINT, 4326) NOT NULL,
            location_address TEXT,
            location_details JSONB,
            incident_time TIMESTAMP WITH TIME ZONE NOT NULL,
            reported_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            severity_level VARCHAR(20) NOT NULL CHECK (severity_level IN ('safe', 'low', 'moderate', 'high', 'critical')),
            description TEXT,
            source VARCHAR(50) NOT NULL DEFAULT 'legacy',
            source_id VARCHAR(255),
            verification_status VARCHAR(20) DEFAULT 'unverified' CHECK (verification_status IN ('unverified', 'verified', 'disputed', 'false_alarm')),
            resolution_status VARCHAR(20) DEFAULT 'open' CHECK (resolution_status IN ('open', 'investigating', 'resolved', 'closed', 'false_alarm')),
            reporter_id UUID REFERENCES users(id) ON DELETE SET NULL,
            assigned_officer VARCHAR(255),
            evidence_urls JSONB DEFAULT '[]'::jsonb,
            metadata JSONB DEFAULT '{}'::jsonb,
            weather_conditions JSONB,
            crowd_density_estimate INTEGER,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        -- Migrate data from crime_incidents if it has data
        INSERT INTO incidents (
            incident_type,
            location,
            incident_time,
            severity_level,
            description,
            source,
            created_at
        )
        SELECT
            COALESCE(incident_type, 'unknown'),
            ST_GeogFromText('POINT(' || longitude || ' ' || latitude || ')'),
            COALESCE(incident_date, NOW()),
            CASE
                WHEN severity IN ('critical', 'high', 'moderate', 'low', 'safe') THEN severity
                ELSE 'moderate'
            END,
            description,
            'legacy_migration',
            COALESCE(created_at, NOW())
        FROM crime_incidents
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        ON CONFLICT DO NOTHING;
    ELSE
        -- Create incidents table if crime_incidents doesn't exist
        CREATE TABLE IF NOT EXISTS incidents (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            incident_type VARCHAR(50) NOT NULL,
            location GEOGRAPHY(POINT, 4326) NOT NULL,
            location_address TEXT,
            location_details JSONB,
            incident_time TIMESTAMP WITH TIME ZONE NOT NULL,
            reported_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            severity_level VARCHAR(20) NOT NULL CHECK (severity_level IN ('safe', 'low', 'moderate', 'high', 'critical')),
            description TEXT,
            source VARCHAR(50) NOT NULL,
            source_id VARCHAR(255),
            verification_status VARCHAR(20) DEFAULT 'unverified' CHECK (verification_status IN ('unverified', 'verified', 'disputed', 'false_alarm')),
            resolution_status VARCHAR(20) DEFAULT 'open' CHECK (resolution_status IN ('open', 'investigating', 'resolved', 'closed', 'false_alarm')),
            reporter_id UUID REFERENCES users(id) ON DELETE SET NULL,
            assigned_officer VARCHAR(255),
            evidence_urls JSONB DEFAULT '[]'::jsonb,
            metadata JSONB DEFAULT '{}'::jsonb,
            weather_conditions JSONB,
            crowd_density_estimate INTEGER,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
    END IF;
END $$;

-- Create all other production tables
CREATE TABLE IF NOT EXISTS crime_data_sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_name VARCHAR(100) NOT NULL,
    source_type VARCHAR(50) NOT NULL,
    api_endpoint TEXT,
    api_key_encrypted TEXT,
    update_frequency_minutes INTEGER DEFAULT 60,
    last_sync TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT TRUE,
    data_mapping JSONB,
    geographical_coverage GEOGRAPHY(POLYGON, 4326),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS safety_zones (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    zone_name VARCHAR(255),
    center_location GEOGRAPHY(POINT, 4326) NOT NULL,
    radius_meters INTEGER NOT NULL,
    polygon_bounds GEOGRAPHY(POLYGON, 4326),
    current_risk_level VARCHAR(20) NOT NULL CHECK (current_risk_level IN ('safe', 'low', 'moderate', 'high', 'critical')),
    risk_score DECIMAL(5,2) DEFAULT 0.0,
    incident_count_24h INTEGER DEFAULT 0,
    incident_count_7d INTEGER DEFAULT 0,
    last_incident_time TIMESTAMP WITH TIME ZONE,
    population_density INTEGER,
    area_type VARCHAR(50),
    lighting_quality VARCHAR(20),
    police_presence_level VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS watched_locations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    alias VARCHAR(100) NOT NULL,
    location_data JSONB NOT NULL,
    alerts_enabled BOOLEAN DEFAULT TRUE,
    alert_threshold VARCHAR(20) DEFAULT 'moderate' CHECK (alert_threshold IN ('safe', 'low', 'moderate', 'high', 'critical')),
    current_risk_level VARCHAR(20) DEFAULT 'safe' CHECK (current_risk_level IN ('safe', 'low', 'moderate', 'high', 'critical')),
    last_checked TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    notification_preferences JSONB DEFAULT '{}'::jsonb,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS watched_locations_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    watched_location_id UUID NOT NULL REFERENCES watched_locations(id) ON DELETE CASCADE,
    previous_risk_level VARCHAR(20),
    current_risk_level VARCHAR(20) NOT NULL,
    risk_score DECIMAL(5,2),
    contributing_factors JSONB,
    last_checked TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS safety_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    alert_type VARCHAR(50) NOT NULL,
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('safe', 'low', 'moderate', 'high', 'critical')),
    location GEOGRAPHY(POINT, 4326),
    related_incident_id UUID REFERENCES incidents(id) ON DELETE SET NULL,
    related_watched_location_id UUID REFERENCES watched_locations(id) ON DELETE SET NULL,
    is_read BOOLEAN DEFAULT FALSE,
    is_dismissed BOOLEAN DEFAULT FALSE,
    delivery_status VARCHAR(20) DEFAULT 'pending' CHECK (delivery_status IN ('pending', 'sent', 'delivered', 'failed')),
    delivery_method VARCHAR(50) DEFAULT 'push' CHECK (delivery_method IN ('push', 'sms', 'email', 'websocket')),
    scheduled_for TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    sent_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS watchlist_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    watched_location_id UUID NOT NULL REFERENCES watched_locations(id) ON DELETE CASCADE,
    alert_type VARCHAR(50) NOT NULL,
    alert_message TEXT NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    is_acknowledged BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS user_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    incident_type VARCHAR(50) NOT NULL,
    location GEOGRAPHY(POINT, 4326) NOT NULL,
    location_address TEXT,
    description TEXT NOT NULL,
    severity_estimate VARCHAR(20) CHECK (severity_estimate IN ('safe', 'low', 'moderate', 'high', 'critical')),
    is_anonymous BOOLEAN DEFAULT FALSE,
    media_urls JSONB DEFAULT '[]'::jsonb,
    audio_url TEXT,
    ai_analysis_result JSONB,
    verification_status VARCHAR(20) DEFAULT 'pending' CHECK (verification_status IN ('pending', 'verified', 'disputed', 'rejected')),
    verified_by UUID REFERENCES users(id) ON DELETE SET NULL,
    verified_at TIMESTAMP WITH TIME ZONE,
    created_incident_id UUID REFERENCES incidents(id) ON DELETE SET NULL,
    device_info JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS media_files (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    incident_id UUID REFERENCES incidents(id) ON DELETE CASCADE,
    user_report_id UUID REFERENCES user_reports(id) ON DELETE CASCADE,
    file_type VARCHAR(20) NOT NULL CHECK (file_type IN ('image', 'audio', 'video')),
    file_url TEXT NOT NULL,
    file_size_bytes BIGINT,
    mime_type VARCHAR(100),
    original_filename TEXT,
    ai_analysis_result JSONB,
    threat_detected BOOLEAN DEFAULT FALSE,
    confidence_score DECIMAL(5,4),
    processing_status VARCHAR(20) DEFAULT 'pending' CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed')),
    storage_provider VARCHAR(50) DEFAULT 'local',
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS location_search_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    search_query VARCHAR(500) NOT NULL,
    search_hash VARCHAR(64) UNIQUE NOT NULL,
    results JSONB NOT NULL,
    result_count INTEGER DEFAULT 0,
    search_location GEOGRAPHY(POINT, 4326),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS risk_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    location GEOGRAPHY(POINT, 4326) NOT NULL,
    prediction_radius_meters INTEGER NOT NULL,
    current_risk_level VARCHAR(20) NOT NULL,
    predicted_risk_level VARCHAR(20) NOT NULL,
    prediction_horizon_minutes INTEGER NOT NULL,
    confidence_score DECIMAL(5,4) NOT NULL,
    contributing_factors JSONB,
    historical_data_points INTEGER,
    weather_factors JSONB,
    event_factors JSONB,
    prediction_algorithm VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    valid_until TIMESTAMP WITH TIME ZONE NOT NULL
);

CREATE TABLE IF NOT EXISTS system_config (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    config_key VARCHAR(100) UNIQUE NOT NULL,
    config_value JSONB NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    requires_restart BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS api_rate_limits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    identifier VARCHAR(255) NOT NULL,
    identifier_type VARCHAR(20) NOT NULL CHECK (identifier_type IN ('ip', 'user', 'api_key')),
    endpoint VARCHAR(255) NOT NULL,
    request_count INTEGER DEFAULT 0,
    window_start TIMESTAMP WITH TIME ZONE NOT NULL,
    window_duration_seconds INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(identifier, endpoint, window_start)
);

CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id UUID,
    old_values JSONB,
    new_values JSONB,
    ip_address INET,
    user_agent TEXT,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS device_registrations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    device_id VARCHAR(255) NOT NULL,
    device_type VARCHAR(20) NOT NULL CHECK (device_type IN ('ios', 'android')),
    push_token VARCHAR(500) NOT NULL,
    device_info JSONB,
    app_version VARCHAR(50),
    os_version VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, device_id)
);

CREATE TABLE IF NOT EXISTS emergency_contacts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    contact_name VARCHAR(100) NOT NULL,
    phone_number VARCHAR(20) NOT NULL,
    relationship VARCHAR(50),
    is_primary BOOLEAN DEFAULT FALSE,
    auto_notify_on_alert BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create all indexes
CREATE INDEX IF NOT EXISTS idx_incidents_location ON incidents USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_incidents_time ON incidents (incident_time);
CREATE INDEX IF NOT EXISTS idx_incidents_type ON incidents (incident_type);
CREATE INDEX IF NOT EXISTS idx_incidents_severity ON incidents (severity_level);
CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents (resolution_status);

CREATE INDEX IF NOT EXISTS idx_safety_zones_location ON safety_zones USING GIST (center_location);
CREATE INDEX IF NOT EXISTS idx_safety_zones_risk ON safety_zones (current_risk_level);

CREATE INDEX IF NOT EXISTS idx_watched_locations_user ON watched_locations (user_id);
CREATE INDEX IF NOT EXISTS idx_watched_locations_active ON watched_locations (is_active);

CREATE INDEX IF NOT EXISTS idx_watched_locations_history_location ON watched_locations_history (watched_location_id);
CREATE INDEX IF NOT EXISTS idx_watched_locations_history_time ON watched_locations_history (last_checked);

CREATE INDEX IF NOT EXISTS idx_safety_alerts_user ON safety_alerts (user_id);
CREATE INDEX IF NOT EXISTS idx_safety_alerts_type ON safety_alerts (alert_type);
CREATE INDEX IF NOT EXISTS idx_safety_alerts_severity ON safety_alerts (severity);
CREATE INDEX IF NOT EXISTS idx_safety_alerts_unread ON safety_alerts (user_id, is_read) WHERE is_read = FALSE;

CREATE INDEX IF NOT EXISTS idx_watchlist_alerts_location ON watchlist_alerts (watched_location_id);

CREATE INDEX IF NOT EXISTS idx_user_reports_location ON user_reports USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_user_reports_user ON user_reports (user_id);
CREATE INDEX IF NOT EXISTS idx_user_reports_status ON user_reports (verification_status);

CREATE INDEX IF NOT EXISTS idx_media_files_incident ON media_files (incident_id);
CREATE INDEX IF NOT EXISTS idx_media_files_report ON media_files (user_report_id);
CREATE INDEX IF NOT EXISTS idx_media_files_type ON media_files (file_type);

CREATE INDEX IF NOT EXISTS idx_location_search_hash ON location_search_cache (search_hash);
CREATE INDEX IF NOT EXISTS idx_location_search_expires ON location_search_cache (expires_at);

CREATE INDEX IF NOT EXISTS idx_risk_predictions_location ON risk_predictions USING GIST (location);
CREATE INDEX IF NOT EXISTS idx_risk_predictions_valid ON risk_predictions (valid_until);

CREATE INDEX IF NOT EXISTS idx_api_rate_limits_identifier ON api_rate_limits (identifier, endpoint);
CREATE INDEX IF NOT EXISTS idx_api_rate_limits_window ON api_rate_limits (window_start);

CREATE INDEX IF NOT EXISTS idx_audit_log_user ON audit_log (user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log (action);
CREATE INDEX IF NOT EXISTS idx_audit_log_time ON audit_log (created_at);

CREATE INDEX IF NOT EXISTS idx_device_registrations_user ON device_registrations (user_id);
CREATE INDEX IF NOT EXISTS idx_device_registrations_active ON device_registrations (is_active);

CREATE INDEX IF NOT EXISTS idx_emergency_contacts_user ON emergency_contacts (user_id);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_incidents_recent ON incidents (incident_time DESC) WHERE incident_time >= NOW() - INTERVAL '7 days';
CREATE INDEX IF NOT EXISTS idx_incidents_location_time ON incidents USING GIST (location, incident_time);
CREATE INDEX IF NOT EXISTS idx_safety_alerts_recent ON safety_alerts (created_at DESC) WHERE created_at >= NOW() - INTERVAL '30 days';
CREATE INDEX IF NOT EXISTS idx_user_reports_recent ON user_reports (created_at DESC) WHERE created_at >= NOW() - INTERVAL '7 days';

-- Partial indexes for active records
CREATE INDEX IF NOT EXISTS idx_users_active ON users (id) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_watched_locations_active_user ON watched_locations (user_id) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_device_registrations_active_user ON device_registrations (user_id) WHERE is_active = TRUE;

-- Create functions
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE OR REPLACE FUNCTION calculate_distance_meters(lat1 DOUBLE PRECISION, lon1 DOUBLE PRECISION, lat2 DOUBLE PRECISION, lon2 DOUBLE PRECISION)
RETURNS DOUBLE PRECISION AS $$
BEGIN
    RETURN ST_Distance(
        ST_GeogFromText('POINT(' || lon1 || ' ' || lat1 || ')'),
        ST_GeogFromText('POINT(' || lon2 || ' ' || lat2 || ')')
    );
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION get_incidents_within_radius(center_lat DOUBLE PRECISION, center_lon DOUBLE PRECISION, radius_meters INTEGER, time_hours INTEGER DEFAULT 24)
RETURNS TABLE(
    incident_id UUID,
    incident_type VARCHAR(50),
    severity_level VARCHAR(20),
    incident_time TIMESTAMP WITH TIME ZONE,
    distance_meters DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        i.id,
        i.incident_type,
        i.severity_level,
        i.incident_time,
        ST_Distance(
            i.location,
            ST_GeogFromText('POINT(' || center_lon || ' ' || center_lat || ')')
        )::DOUBLE PRECISION as distance_meters
    FROM incidents i
    WHERE ST_DWithin(
        i.location,
        ST_GeogFromText('POINT(' || center_lon || ' ' || center_lat || ')'),
        radius_meters
    )
    AND i.incident_time >= NOW() - INTERVAL '1 hour' * time_hours
    AND i.resolution_status != 'false_alarm'
    ORDER BY distance_meters;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION calculate_risk_score(center_lat DOUBLE PRECISION, center_lon DOUBLE PRECISION, radius_meters INTEGER DEFAULT 1000)
RETURNS DECIMAL(5,2) AS $$
DECLARE
    incident_count INTEGER;
    avg_severity DECIMAL(5,2);
    recent_critical INTEGER;
    risk_score DECIMAL(5,2);
BEGIN
    SELECT COUNT(*) INTO incident_count
    FROM incidents
    WHERE ST_DWithin(
        location,
        ST_GeogFromText('POINT(' || center_lon || ' ' || center_lat || ')'),
        radius_meters
    )
    AND incident_time >= NOW() - INTERVAL '24 hours'
    AND resolution_status != 'false_alarm';

    SELECT AVG(
        CASE severity_level
            WHEN 'critical' THEN 5
            WHEN 'high' THEN 4
            WHEN 'moderate' THEN 3
            WHEN 'low' THEN 2
            WHEN 'safe' THEN 1
            ELSE 0
        END
    ) INTO avg_severity
    FROM incidents
    WHERE ST_DWithin(
        location,
        ST_GeogFromText('POINT(' || center_lon || ' ' || center_lat || ')'),
        radius_meters
    )
    AND incident_time >= NOW() - INTERVAL '7 days'
    AND resolution_status != 'false_alarm';

    SELECT COUNT(*) INTO recent_critical
    FROM incidents
    WHERE ST_DWithin(
        location,
        ST_GeogFromText('POINT(' || center_lon || ' ' || center_lat || ')'),
        radius_meters
    )
    AND incident_time >= NOW() - INTERVAL '2 hours'
    AND severity_level IN ('critical', 'high')
    AND resolution_status != 'false_alarm';

    risk_score := (
        (incident_count * 0.3) +
        (COALESCE(avg_severity, 1) * 0.5) +
        (recent_critical * 1.5)
    );

    IF risk_score > 10.0 THEN
        risk_score := 10.0;
    END IF;

    RETURN risk_score;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for updated_at columns
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_incidents_updated_at ON incidents;
CREATE TRIGGER update_incidents_updated_at BEFORE UPDATE ON incidents FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_crime_data_sources_updated_at ON crime_data_sources;
CREATE TRIGGER update_crime_data_sources_updated_at BEFORE UPDATE ON crime_data_sources FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_safety_zones_updated_at ON safety_zones;
CREATE TRIGGER update_safety_zones_updated_at BEFORE UPDATE ON safety_zones FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_watched_locations_updated_at ON watched_locations;
CREATE TRIGGER update_watched_locations_updated_at BEFORE UPDATE ON watched_locations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_user_reports_updated_at ON user_reports;
CREATE TRIGGER update_user_reports_updated_at BEFORE UPDATE ON user_reports FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_media_files_updated_at ON media_files;
CREATE TRIGGER update_media_files_updated_at BEFORE UPDATE ON media_files FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_system_config_updated_at ON system_config;
CREATE TRIGGER update_system_config_updated_at BEFORE UPDATE ON system_config FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_api_rate_limits_updated_at ON api_rate_limits;
CREATE TRIGGER update_api_rate_limits_updated_at BEFORE UPDATE ON api_rate_limits FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_device_registrations_updated_at ON device_registrations;
CREATE TRIGGER update_device_registrations_updated_at BEFORE UPDATE ON device_registrations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_emergency_contacts_updated_at ON emergency_contacts;
CREATE TRIGGER update_emergency_contacts_updated_at BEFORE UPDATE ON emergency_contacts FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert default system configuration if not exists
INSERT INTO system_config (config_key, config_value, description) VALUES
('crime_data_sync_interval_minutes', '15', 'How often to sync crime data from external sources'),
('max_alert_distance_meters', '5000', 'Maximum distance for location-based alerts'),
('risk_calculation_radius_meters', '1000', 'Default radius for risk calculations'),
('user_report_auto_verify_threshold', '0.85', 'AI confidence threshold for auto-verifying user reports'),
('websocket_heartbeat_interval_seconds', '30', 'WebSocket heartbeat interval'),
('rate_limit_requests_per_minute', '100', 'Default API rate limit per minute'),
('alert_cooldown_minutes', '5', 'Minimum time between alerts of same type'),
('emergency_alert_radius_meters', '10000', 'Radius for emergency broadcast alerts'),
('ai_threat_detection_enabled', 'true', 'Enable AI threat detection for images/audio'),
('geofencing_check_interval_seconds', '30', 'How often to check geofencing rules'),
('prediction_algorithm_version', 'v2.1', 'Current risk prediction algorithm version'),
('max_watched_locations_per_user', '10', 'Maximum watched locations per user'),
('incident_verification_required', 'true', 'Require verification for user-reported incidents'),
('push_notification_enabled', 'true', 'Enable push notifications'),
('sms_alerts_enabled', 'false', 'Enable SMS alerts (requires SMS provider)')
ON CONFLICT (config_key) DO NOTHING;

COMMIT;