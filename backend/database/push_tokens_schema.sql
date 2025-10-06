-- Push Notification Tokens Table
-- Stores Expo push tokens for mobile devices

CREATE TABLE IF NOT EXISTS push_tokens (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id VARCHAR(255) UNIQUE NOT NULL,  -- Anonymous device ID
    push_token TEXT NOT NULL,                -- Expo push token (ExponentPushToken[...])
    user_id UUID,                            -- Optional: linked user (if authenticated)
    active BOOLEAN DEFAULT true,             -- Can be disabled by user or on error
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_push_tokens_device_id ON push_tokens(device_id);
CREATE INDEX IF NOT EXISTS idx_push_tokens_active ON push_tokens(active) WHERE active = true;
CREATE INDEX IF NOT EXISTS idx_push_tokens_user_id ON push_tokens(user_id) WHERE user_id IS NOT NULL;

-- Optional: Device location tracking (requires user opt-in)
-- This enables nearby device notifications for incidents
CREATE TABLE IF NOT EXISTS device_locations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id VARCHAR(255) NOT NULL,
    location GEOGRAPHY(Point, 4326) NOT NULL,  -- PostGIS geography
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    accuracy FLOAT,                            -- GPS accuracy in meters
    updated_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT fk_device FOREIGN KEY (device_id) REFERENCES push_tokens(device_id) ON DELETE CASCADE
);

-- Spatial index for nearby queries
CREATE INDEX IF NOT EXISTS idx_device_locations_geography ON device_locations USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_device_locations_device_id ON device_locations(device_id);
CREATE INDEX IF NOT EXISTS idx_device_locations_updated_at ON device_locations(updated_at);

-- Function to update device location
CREATE OR REPLACE FUNCTION update_device_location(
    p_device_id VARCHAR,
    p_latitude DOUBLE PRECISION,
    p_longitude DOUBLE PRECISION,
    p_accuracy FLOAT DEFAULT NULL
) RETURNS BOOLEAN AS $$
BEGIN
    INSERT INTO device_locations (device_id, location, latitude, longitude, accuracy, updated_at)
    VALUES (
        p_device_id,
        ST_SetSRID(ST_MakePoint(p_longitude, p_latitude), 4326)::geography,
        p_latitude,
        p_longitude,
        p_accuracy,
        NOW()
    )
    ON CONFLICT (device_id)
    DO UPDATE SET
        location = ST_SetSRID(ST_MakePoint(p_longitude, p_latitude), 4326)::geography,
        latitude = p_latitude,
        longitude = p_longitude,
        accuracy = p_accuracy,
        updated_at = NOW();

    RETURN true;
END;
$$ LANGUAGE plpgsql;

-- Function to find nearby devices
CREATE OR REPLACE FUNCTION get_nearby_devices(
    p_latitude DOUBLE PRECISION,
    p_longitude DOUBLE PRECISION,
    p_radius_km FLOAT DEFAULT 2.0,
    p_max_age_minutes INT DEFAULT 60  -- Only consider recent locations
) RETURNS TABLE (
    device_id VARCHAR,
    distance_km FLOAT,
    minutes_ago INT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        dl.device_id,
        ST_Distance(
            dl.location,
            ST_SetSRID(ST_MakePoint(p_longitude, p_latitude), 4326)::geography
        ) / 1000.0 AS distance_km,
        EXTRACT(EPOCH FROM (NOW() - dl.updated_at))::INT / 60 AS minutes_ago
    FROM device_locations dl
    INNER JOIN push_tokens pt ON dl.device_id = pt.device_id
    WHERE
        pt.active = true
        AND ST_DWithin(
            dl.location,
            ST_SetSRID(ST_MakePoint(p_longitude, p_latitude), 4326)::geography,
            p_radius_km * 1000  -- Convert km to meters
        )
        AND dl.updated_at > NOW() - INTERVAL '1 minute' * p_max_age_minutes
    ORDER BY distance_km ASC;
END;
$$ LANGUAGE plpgsql;

-- Notification log (track what's been sent to avoid spam)
CREATE TABLE IF NOT EXISTS notification_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    device_id VARCHAR(255) NOT NULL,
    notification_type VARCHAR(50) NOT NULL,  -- 'risk_alert', 'incident', 'predictive'
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    data JSONB,
    sent_at TIMESTAMP DEFAULT NOW(),
    success BOOLEAN DEFAULT true
);

-- Index for deduplication checks
CREATE INDEX IF NOT EXISTS idx_notification_log_device_type_time
    ON notification_log(device_id, notification_type, sent_at);

-- Function to check if notification was recently sent (prevent spam)
CREATE OR REPLACE FUNCTION was_recently_notified(
    p_device_id VARCHAR,
    p_notification_type VARCHAR,
    p_cooldown_minutes INT DEFAULT 30
) RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1
        FROM notification_log
        WHERE
            device_id = p_device_id
            AND notification_type = p_notification_type
            AND sent_at > NOW() - INTERVAL '1 minute' * p_cooldown_minutes
    );
END;
$$ LANGUAGE plpgsql;

-- Log notification sent
CREATE OR REPLACE FUNCTION log_notification(
    p_device_id VARCHAR,
    p_notification_type VARCHAR,
    p_title TEXT,
    p_body TEXT,
    p_data JSONB DEFAULT NULL,
    p_success BOOLEAN DEFAULT true
) RETURNS UUID AS $$
DECLARE
    v_id UUID;
BEGIN
    INSERT INTO notification_log (device_id, notification_type, title, body, data, success)
    VALUES (p_device_id, p_notification_type, p_title, p_body, p_data, p_success)
    RETURNING id INTO v_id;

    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

-- Cleanup old notification logs (run weekly)
CREATE OR REPLACE FUNCTION cleanup_old_notification_logs(
    p_days_to_keep INT DEFAULT 30
) RETURNS INT AS $$
DECLARE
    v_deleted_count INT;
BEGIN
    DELETE FROM notification_log
    WHERE sent_at < NOW() - INTERVAL '1 day' * p_days_to_keep;

    GET DIAGNOSTICS v_deleted_count = ROW_COUNT;
    RETURN v_deleted_count;
END;
$$ LANGUAGE plpgsql;
