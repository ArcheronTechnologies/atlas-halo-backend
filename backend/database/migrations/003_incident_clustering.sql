-- Incident Report Clustering System
-- Created: 2025-10-02
-- Purpose: Automatically combine duplicate reports from multiple users into verified incidents

-- Drop old verification system
DROP TABLE IF EXISTS incident_verifications CASCADE;
ALTER TABLE crime_incidents DROP COLUMN IF EXISTS verification_count_up CASCADE;
ALTER TABLE crime_incidents DROP COLUMN IF EXISTS verification_count_down CASCADE;
ALTER TABLE crime_incidents DROP COLUMN IF EXISTS verification_score CASCADE;

-- Add clustering fields to crime_incidents
ALTER TABLE crime_incidents
ADD COLUMN IF NOT EXISTS report_count INTEGER DEFAULT 1,
ADD COLUMN IF NOT EXISTS cluster_id UUID,
ADD COLUMN IF NOT EXISTS is_cluster_primary BOOLEAN DEFAULT true,
ADD COLUMN IF NOT EXISTS cluster_confidence FLOAT DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS unique_reporters INTEGER DEFAULT 1;

-- Create incident reports table (tracks individual submissions)
CREATE TABLE IF NOT EXISTS incident_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    incident_id VARCHAR NOT NULL REFERENCES crime_incidents(id) ON DELETE CASCADE,
    device_fingerprint VARCHAR(64) NOT NULL, -- Hashed device ID (anonymous)
    report_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    location geography(Point,4326) NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    media_ids TEXT[], -- Associated media files
    description TEXT,
    confidence_score FLOAT DEFAULT 1.0,

    -- Metadata about the report
    metadata JSONB DEFAULT '{}'::jsonb,

    UNIQUE(incident_id, device_fingerprint) -- One report per device per incident cluster
);

-- Indexes for fast clustering
CREATE INDEX IF NOT EXISTS idx_reports_incident ON incident_reports(incident_id);
CREATE INDEX IF NOT EXISTS idx_reports_location ON incident_reports USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_reports_time ON incident_reports(report_time);
CREATE INDEX IF NOT EXISTS idx_incident_cluster ON crime_incidents(cluster_id) WHERE cluster_id IS NOT NULL;

-- Function to calculate spatial-temporal distance between two reports
CREATE OR REPLACE FUNCTION calculate_report_similarity(
    lat1 DOUBLE PRECISION,
    lon1 DOUBLE PRECISION,
    time1 TIMESTAMP,
    lat2 DOUBLE PRECISION,
    lon2 DOUBLE PRECISION,
    time2 TIMESTAMP,
    type1 VARCHAR,
    type2 VARCHAR
)
RETURNS FLOAT AS $$
DECLARE
    distance_km FLOAT;
    time_diff_hours FLOAT;
    spatial_score FLOAT;
    temporal_score FLOAT;
    type_score FLOAT;
    final_score FLOAT;
BEGIN
    -- Calculate spatial distance (Haversine)
    distance_km := ST_Distance(
        ST_SetSRID(ST_MakePoint(lon1, lat1), 4326)::geography,
        ST_SetSRID(ST_MakePoint(lon2, lat2), 4326)::geography
    ) / 1000.0;

    -- Calculate temporal distance
    time_diff_hours := EXTRACT(EPOCH FROM (time1 - time2)) / 3600.0;
    time_diff_hours := ABS(time_diff_hours);

    -- Spatial similarity (1.0 if within 200m, decreases to 0 at 2km)
    spatial_score := CASE
        WHEN distance_km <= 0.2 THEN 1.0
        WHEN distance_km >= 2.0 THEN 0.0
        ELSE 1.0 - ((distance_km - 0.2) / 1.8)
    END;

    -- Temporal similarity (1.0 if within 1 hour, decreases to 0 at 24 hours)
    temporal_score := CASE
        WHEN time_diff_hours <= 1.0 THEN 1.0
        WHEN time_diff_hours >= 24.0 THEN 0.0
        ELSE 1.0 - ((time_diff_hours - 1.0) / 23.0)
    END;

    -- Type similarity (exact match or closely related)
    type_score := CASE
        WHEN type1 = type2 THEN 1.0
        -- Related types also match (e.g., theft and robbery)
        WHEN (type1 IN ('theft', 'robbery') AND type2 IN ('theft', 'robbery')) THEN 0.8
        WHEN (type1 IN ('assault', 'violence') AND type2 IN ('assault', 'violence')) THEN 0.8
        WHEN (type1 IN ('vandalism', 'property_damage') AND type2 IN ('vandalism', 'property_damage')) THEN 0.8
        ELSE 0.3
    END;

    -- Combined score (weighted average)
    final_score := (spatial_score * 0.5) + (temporal_score * 0.3) + (type_score * 0.2);

    RETURN final_score;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to find matching incident cluster
CREATE OR REPLACE FUNCTION find_matching_cluster(
    p_lat DOUBLE PRECISION,
    p_lon DOUBLE PRECISION,
    p_time TIMESTAMP,
    p_type VARCHAR,
    p_device VARCHAR
)
RETURNS UUID AS $$
DECLARE
    best_match UUID;
    best_score FLOAT := 0.0;
    current_score FLOAT;
    incident RECORD;
BEGIN
    -- Find incidents within 2km and 24 hours
    FOR incident IN
        SELECT
            id,
            cluster_id,
            latitude,
            longitude,
            occurred_at,
            incident_type
        FROM crime_incidents
        WHERE
            occurred_at >= (p_time - INTERVAL '24 hours')
            AND occurred_at <= (p_time + INTERVAL '24 hours')
            AND ST_DWithin(
                location,
                ST_SetSRID(ST_MakePoint(p_lon, p_lat), 4326)::geography,
                2000  -- 2km radius
            )
            -- Exclude if same device already reported this incident
            AND NOT EXISTS (
                SELECT 1 FROM incident_reports
                WHERE incident_id = crime_incidents.id
                AND device_fingerprint = p_device
            )
    LOOP
        -- Calculate similarity score
        current_score := calculate_report_similarity(
            p_lat, p_lon, p_time,
            incident.latitude, incident.longitude, incident.occurred_at,
            p_type, incident.incident_type
        );

        -- If similarity > 0.6, consider it a match
        IF current_score > 0.6 AND current_score > best_score THEN
            best_score := current_score;
            -- Return the cluster_id if exists, otherwise the incident id
            best_match := COALESCE(incident.cluster_id, incident.id::UUID);
        END IF;
    END LOOP;

    RETURN best_match;
END;
$$ LANGUAGE plpgsql;

-- Function to update cluster confidence based on report count
CREATE OR REPLACE FUNCTION update_cluster_confidence(p_incident_id VARCHAR)
RETURNS VOID AS $$
DECLARE
    report_count INTEGER;
    unique_reporters INTEGER;
    confidence FLOAT;
BEGIN
    -- Count total reports for this incident
    SELECT COUNT(*), COUNT(DISTINCT device_fingerprint)
    INTO report_count, unique_reporters
    FROM incident_reports
    WHERE incident_id = p_incident_id;

    -- Calculate confidence: starts at 0.5 for 1 report, approaches 1.0 asymptotically
    -- Formula: confidence = 1 - (1 / (unique_reporters + 1))
    confidence := 1.0 - (1.0 / (unique_reporters + 1.0));

    -- Update incident
    UPDATE crime_incidents
    SET
        report_count = report_count,
        unique_reporters = unique_reporters,
        cluster_confidence = confidence,
        is_verified = (unique_reporters >= 2) -- Auto-verify with 2+ reports
    WHERE id = p_incident_id;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update cluster confidence when reports are added
CREATE OR REPLACE FUNCTION trigger_update_cluster_confidence()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM update_cluster_confidence(NEW.incident_id);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_report_added
AFTER INSERT ON incident_reports
FOR EACH ROW
EXECUTE FUNCTION trigger_update_cluster_confidence();

-- Comments
COMMENT ON TABLE incident_reports IS 'Individual anonymous incident reports (before clustering)';
COMMENT ON COLUMN incident_reports.device_fingerprint IS 'Hashed device ID - anonymous, prevents spam';
COMMENT ON COLUMN crime_incidents.report_count IS 'Total number of individual reports for this incident';
COMMENT ON COLUMN crime_incidents.unique_reporters IS 'Number of unique devices that reported this incident';
COMMENT ON COLUMN crime_incidents.cluster_confidence IS 'Confidence score: 0.5 (1 report) to 1.0 (many reports)';
COMMENT ON COLUMN crime_incidents.cluster_id IS 'UUID linking clustered incidents together';
COMMENT ON FUNCTION find_matching_cluster IS 'Finds existing incident cluster that matches new report (spatial + temporal + semantic matching)';
