-- Threat alerts database schema
-- Creates tables for storing threat alerts and related data

-- Create threat alerts table
CREATE TABLE IF NOT EXISTS threat_alerts (
    id SERIAL PRIMARY KEY,
    alert_id VARCHAR(255) UNIQUE NOT NULL,
    alert_type VARCHAR(50) NOT NULL CHECK (alert_type IN ('immediate_threat', 'escalating_behavior', 'suspicious_activity', 'area_warning')),
    threat_level VARCHAR(20) NOT NULL CHECK (threat_level IN ('low', 'medium', 'high', 'critical')),
    location GEOMETRY(POINT, 4326) NOT NULL,
    radius_meters FLOAT NOT NULL DEFAULT 250.0,
    message TEXT NOT NULL,
    threat_details JSONB NOT NULL DEFAULT '{}',
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    source_user_id VARCHAR(255),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    dismissed_count INTEGER NOT NULL DEFAULT 0,
    
    -- Indexes for efficient queries
    CONSTRAINT valid_radius CHECK (radius_meters > 0 AND radius_meters <= 5000)
);

-- Create spatial index for location-based queries
CREATE INDEX IF NOT EXISTS idx_threat_alerts_location ON threat_alerts USING GIST (location);

-- Create index for active alerts
CREATE INDEX IF NOT EXISTS idx_threat_alerts_active ON threat_alerts (is_active, expires_at) WHERE is_active = TRUE;

-- Create index for alert type and level
CREATE INDEX IF NOT EXISTS idx_threat_alerts_type_level ON threat_alerts (alert_type, threat_level);

-- Create index for time-based queries
CREATE INDEX IF NOT EXISTS idx_threat_alerts_created_at ON threat_alerts (created_at);

-- Create index for source user
CREATE INDEX IF NOT EXISTS idx_threat_alerts_source_user ON threat_alerts (source_user_id);


-- Create alert dismissals table (tracks which users have dismissed which alerts)
CREATE TABLE IF NOT EXISTS alert_dismissals (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    alert_id VARCHAR(255) NOT NULL,
    dismissed_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    -- Ensure user can only dismiss an alert once
    UNIQUE(user_id, alert_id)
);

-- Create index for user dismissals
CREATE INDEX IF NOT EXISTS idx_alert_dismissals_user ON alert_dismissals (user_id);

-- Create index for alert dismissals
CREATE INDEX IF NOT EXISTS idx_alert_dismissals_alert ON alert_dismissals (alert_id);


-- Create alert recipients table (tracks which users received which alerts)
CREATE TABLE IF NOT EXISTS alert_recipients (
    id SERIAL PRIMARY KEY,
    alert_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    sent_at TIMESTAMP NOT NULL DEFAULT NOW(),
    delivery_status VARCHAR(20) NOT NULL DEFAULT 'sent' CHECK (delivery_status IN ('sent', 'delivered', 'failed', 'dismissed')),
    user_location GEOMETRY(POINT, 4326),
    distance_meters FLOAT,
    
    -- Ensure unique alert-user combination
    UNIQUE(alert_id, user_id)
);

-- Create index for alert recipients
CREATE INDEX IF NOT EXISTS idx_alert_recipients_alert ON alert_recipients (alert_id);

-- Create index for user recipients
CREATE INDEX IF NOT EXISTS idx_alert_recipients_user ON alert_recipients (user_id);

-- Create index for delivery status
CREATE INDEX IF NOT EXISTS idx_alert_recipients_status ON alert_recipients (delivery_status);


-- Create alert statistics table (for analytics and reporting)
CREATE TABLE IF NOT EXISTS alert_statistics (
    id SERIAL PRIMARY KEY,
    alert_id VARCHAR(255) NOT NULL,
    total_recipients INTEGER NOT NULL DEFAULT 0,
    total_dismissed INTEGER NOT NULL DEFAULT 0,
    total_delivered INTEGER NOT NULL DEFAULT 0,
    total_failed INTEGER NOT NULL DEFAULT 0,
    avg_response_time_seconds FLOAT,
    area_coverage_km2 FLOAT,
    effectiveness_score FLOAT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    
    UNIQUE(alert_id)
);

-- Create index for alert statistics
CREATE INDEX IF NOT EXISTS idx_alert_statistics_alert ON alert_statistics (alert_id);


-- Create user alert preferences table
CREATE TABLE IF NOT EXISTS user_alert_preferences (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) UNIQUE NOT NULL,
    
    -- Alert type preferences
    immediate_threat_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    escalating_behavior_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    suspicious_activity_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    area_warning_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    
    -- Notification settings
    push_notifications_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    sound_alerts_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    vibration_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    
    -- Distance preferences
    max_alert_distance_meters INTEGER NOT NULL DEFAULT 1000,
    min_threat_level VARCHAR(20) NOT NULL DEFAULT 'medium' CHECK (min_threat_level IN ('low', 'medium', 'high', 'critical')),
    
    -- Timing preferences
    quiet_hours_start TIME,
    quiet_hours_end TIME,
    quiet_hours_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create index for user preferences
CREATE INDEX IF NOT EXISTS idx_user_alert_preferences_user ON user_alert_preferences (user_id);


-- Create alert zones table (for predefined alert areas)
CREATE TABLE IF NOT EXISTS alert_zones (
    id SERIAL PRIMARY KEY,
    zone_id VARCHAR(255) UNIQUE NOT NULL,
    zone_name VARCHAR(255) NOT NULL,
    zone_type VARCHAR(50) NOT NULL CHECK (zone_type IN ('school', 'hospital', 'government', 'transport', 'residential', 'commercial', 'custom')),
    boundary GEOMETRY(POLYGON, 4326) NOT NULL,
    alert_multiplier FLOAT NOT NULL DEFAULT 1.0,
    special_instructions TEXT,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_by VARCHAR(255),
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Create spatial index for zones
CREATE INDEX IF NOT EXISTS idx_alert_zones_boundary ON alert_zones USING GIST (boundary);

-- Create index for zone type
CREATE INDEX IF NOT EXISTS idx_alert_zones_type ON alert_zones (zone_type);


-- Function to automatically update statistics when alerts are dismissed
CREATE OR REPLACE FUNCTION update_alert_statistics()
RETURNS TRIGGER AS $$
BEGIN
    -- Update dismissal count in alert statistics
    INSERT INTO alert_statistics (alert_id, total_dismissed)
    VALUES (NEW.alert_id, 1)
    ON CONFLICT (alert_id) 
    DO UPDATE SET 
        total_dismissed = alert_statistics.total_dismissed + 1,
        updated_at = NOW();
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for alert dismissals
CREATE TRIGGER trigger_update_alert_statistics
    AFTER INSERT ON alert_dismissals
    FOR EACH ROW
    EXECUTE FUNCTION update_alert_statistics();


-- Function to automatically expire old alerts
CREATE OR REPLACE FUNCTION expire_old_alerts()
RETURNS INTEGER AS $$
DECLARE
    expired_count INTEGER;
BEGIN
    UPDATE threat_alerts 
    SET is_active = FALSE 
    WHERE is_active = TRUE 
    AND expires_at <= NOW();
    
    GET DIAGNOSTICS expired_count = ROW_COUNT;
    
    RETURN expired_count;
END;
$$ LANGUAGE plpgsql;


-- Function to get active alerts in an area
CREATE OR REPLACE FUNCTION get_active_alerts_in_area(
    center_lat FLOAT,
    center_lng FLOAT,
    radius_meters FLOAT DEFAULT 1000,
    min_threat_level VARCHAR(20) DEFAULT 'low'
)
RETURNS TABLE (
    alert_id VARCHAR(255),
    alert_type VARCHAR(50),
    threat_level VARCHAR(20),
    lat FLOAT,
    lng FLOAT,
    radius_meters FLOAT,
    message TEXT,
    threat_details JSONB,
    created_at TIMESTAMP,
    expires_at TIMESTAMP,
    distance_meters FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ta.alert_id,
        ta.alert_type,
        ta.threat_level,
        ST_Y(ta.location) as lat,
        ST_X(ta.location) as lng,
        ta.radius_meters,
        ta.message,
        ta.threat_details,
        ta.created_at,
        ta.expires_at,
        ST_Distance(ta.location::geography, ST_Point(center_lng, center_lat)::geography) as distance_meters
    FROM threat_alerts ta
    WHERE ta.is_active = TRUE
    AND ta.expires_at > NOW()
    AND ST_DWithin(ta.location::geography, ST_Point(center_lng, center_lat)::geography, radius_meters)
    AND CASE min_threat_level
        WHEN 'critical' THEN ta.threat_level = 'critical'
        WHEN 'high' THEN ta.threat_level IN ('high', 'critical')
        WHEN 'medium' THEN ta.threat_level IN ('medium', 'high', 'critical')
        ELSE TRUE
    END
    ORDER BY distance_meters ASC;
END;
$$ LANGUAGE plpgsql;


-- Function to get alert statistics for a specific alert
CREATE OR REPLACE FUNCTION get_alert_effectiveness(alert_id_param VARCHAR(255))
RETURNS TABLE (
    alert_id VARCHAR(255),
    total_recipients INTEGER,
    dismissal_rate FLOAT,
    avg_distance_meters FLOAT,
    coverage_area_km2 FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ar.alert_id,
        COUNT(ar.user_id)::INTEGER as total_recipients,
        COALESCE(COUNT(ad.user_id)::FLOAT / NULLIF(COUNT(ar.user_id), 0), 0) as dismissal_rate,
        AVG(ar.distance_meters) as avg_distance_meters,
        (pi() * power(ta.radius_meters / 1000.0, 2)) as coverage_area_km2
    FROM alert_recipients ar
    LEFT JOIN alert_dismissals ad ON ar.alert_id = ad.alert_id AND ar.user_id = ad.user_id
    JOIN threat_alerts ta ON ar.alert_id = ta.alert_id
    WHERE ar.alert_id = alert_id_param
    GROUP BY ar.alert_id, ta.radius_meters;
END;
$$ LANGUAGE plpgsql;


-- Insert default alert preferences for new users
CREATE OR REPLACE FUNCTION create_default_alert_preferences()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO user_alert_preferences (user_id)
    VALUES (NEW.user_id)
    ON CONFLICT (user_id) DO NOTHING;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Note: This trigger would be created on the users table when it exists
-- CREATE TRIGGER trigger_create_default_alert_preferences
--     AFTER INSERT ON users
--     FOR EACH ROW
--     EXECUTE FUNCTION create_default_alert_preferences();


-- Grant permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE ON threat_alerts TO atlas_ai_app;
-- GRANT SELECT, INSERT, UPDATE ON alert_dismissals TO atlas_ai_app;
-- GRANT SELECT, INSERT, UPDATE ON alert_recipients TO atlas_ai_app;
-- GRANT SELECT, INSERT, UPDATE ON alert_statistics TO atlas_ai_app;
-- GRANT SELECT, INSERT, UPDATE ON user_alert_preferences TO atlas_ai_app;
-- GRANT SELECT, INSERT, UPDATE ON alert_zones TO atlas_ai_app;