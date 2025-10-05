-- Cross-Source Incident Clustering
-- Created: 2025-10-02
-- Purpose: Enable clustering between user reports and Polisen.se official incidents

-- Add source type tracking to distinguish official vs user reports
ALTER TABLE crime_incidents
ADD COLUMN IF NOT EXISTS source_type VARCHAR DEFAULT 'official' CHECK (source_type IN ('official', 'user', 'clustered'));

-- Update existing data
UPDATE crime_incidents
SET source_type = CASE
    WHEN source = 'user_report' THEN 'user'
    WHEN source LIKE 'polisen.se%' THEN 'official'
    ELSE 'official'
END
WHERE source_type IS NULL;

-- Add verification metadata for cross-source validation
ALTER TABLE crime_incidents
ADD COLUMN IF NOT EXISTS verification_metadata JSONB DEFAULT '{}'::jsonb;

-- Update the clustering function to handle cross-source matching
CREATE OR REPLACE FUNCTION find_matching_cluster_cross_source(
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
    -- Find incidents within 2km and 24 hours (BOTH official and user reports)
    FOR incident IN
        SELECT
            id,
            cluster_id,
            latitude,
            longitude,
            occurred_at,
            incident_type,
            source_type,
            unique_reporters
        FROM crime_incidents
        WHERE
            occurred_at >= (p_time - INTERVAL '24 hours')
            AND occurred_at <= (p_time + INTERVAL '24 hours')
            AND ST_DWithin(
                location,
                ST_SetSRID(ST_MakePoint(p_lon, p_lat), 4326)::geography,
                2000  -- 2km radius
            )
            -- Don't match if same device already reported this incident
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

        -- Higher threshold for official incidents (0.7 vs 0.6 for user reports)
        -- This prevents false matches with official data
        IF incident.source_type = 'official' THEN
            -- Official Polisen.se incident
            IF current_score > 0.7 AND current_score > best_score THEN
                best_score := current_score;
                best_match := COALESCE(incident.cluster_id, incident.id::UUID);
            END IF;
        ELSE
            -- User report
            IF current_score > 0.6 AND current_score > best_score THEN
                best_score := current_score;
                best_match := COALESCE(incident.cluster_id, incident.id::UUID);
            END IF;
        END IF;
    END LOOP;

    RETURN best_match;
END;
$$ LANGUAGE plpgsql;

-- Function to update incident when user report validates official incident
CREATE OR REPLACE FUNCTION enrich_official_incident_with_user_data(
    p_incident_id VARCHAR,
    p_media_ids TEXT[],
    p_description TEXT
)
RETURNS VOID AS $$
DECLARE
    current_metadata JSONB;
    official_incident RECORD;
BEGIN
    -- Get incident details
    SELECT source_type, verification_metadata, unique_reporters
    INTO official_incident
    FROM crime_incidents
    WHERE id = p_incident_id;

    -- Only enrich if this is an official incident
    IF official_incident.source_type = 'official' THEN
        -- Update source_type to 'clustered' (official + user verification)
        UPDATE crime_incidents
        SET
            source_type = 'clustered',
            verification_metadata = verification_metadata || jsonb_build_object(
                'has_user_media', COALESCE(array_length(p_media_ids, 1), 0) > 0,
                'user_descriptions', COALESCE(
                    (verification_metadata->'user_descriptions')::jsonb,
                    '[]'::jsonb
                ) || jsonb_build_array(p_description),
                'enriched_at', NOW()
            )
        WHERE id = p_incident_id;

        -- Log enrichment
        RAISE NOTICE 'Official incident % enriched with user report (% total reporters)',
            p_incident_id, official_incident.unique_reporters + 1;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically enrich official incidents when user reports cluster with them
CREATE OR REPLACE FUNCTION trigger_enrich_official_incident()
RETURNS TRIGGER AS $$
DECLARE
    incident_source_type VARCHAR;
BEGIN
    -- Check if this report is clustering with an official incident
    SELECT source_type INTO incident_source_type
    FROM crime_incidents
    WHERE id = NEW.incident_id;

    -- If clustering with official incident, enrich it
    IF incident_source_type IN ('official', 'clustered') THEN
        PERFORM enrich_official_incident_with_user_data(
            NEW.incident_id,
            NEW.media_ids,
            NEW.description
        );
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_enrich_on_user_report
AFTER INSERT ON incident_reports
FOR EACH ROW
EXECUTE FUNCTION trigger_enrich_official_incident();

-- Create view for cross-source validated incidents
CREATE OR REPLACE VIEW verified_incidents AS
SELECT
    ci.id,
    ci.incident_type,
    ci.severity,
    ci.latitude,
    ci.longitude,
    ci.occurred_at,
    ci.source,
    ci.source_type,
    ci.is_verified,
    ci.report_count,
    ci.unique_reporters,
    ci.cluster_confidence,

    -- Cross-source validation flags
    CASE
        WHEN ci.source_type = 'clustered' THEN true
        WHEN ci.source_type = 'user' AND ci.unique_reporters >= 2 THEN true
        ELSE false
    END AS is_cross_validated,

    -- Combine official + user confidence
    CASE
        WHEN ci.source_type = 'clustered' THEN
            LEAST(1.0, ci.cluster_confidence + 0.2)  -- Bonus for official validation
        ELSE ci.cluster_confidence
    END AS combined_confidence,

    -- Media availability
    (ci.verification_metadata->>'has_user_media')::boolean as has_media,

    -- User descriptions (for official incidents enriched by users)
    ci.verification_metadata->'user_descriptions' as user_descriptions

FROM crime_incidents ci
WHERE
    -- Either verified by multiple users OR validated by official + user combo
    (ci.is_verified = true OR ci.source_type = 'clustered');

-- Add indexes for cross-source queries
CREATE INDEX IF NOT EXISTS idx_incidents_source_type
ON crime_incidents(source_type);

CREATE INDEX IF NOT EXISTS idx_incidents_clustered
ON crime_incidents(source_type, cluster_id)
WHERE source_type = 'clustered';

-- Comments
COMMENT ON COLUMN crime_incidents.source_type IS 'official: Polisen.se only, user: citizen reports only, clustered: official + user validation';
COMMENT ON COLUMN crime_incidents.verification_metadata IS 'JSON metadata for cross-source validation (media availability, user descriptions, etc.)';
COMMENT ON FUNCTION find_matching_cluster_cross_source IS 'Enhanced clustering that matches user reports with official Polisen.se incidents';
COMMENT ON VIEW verified_incidents IS 'All verified incidents (either multi-user verified OR official+user validated)';

-- Statistics function for cross-source validation
CREATE OR REPLACE FUNCTION get_cross_source_stats()
RETURNS TABLE(
    total_incidents BIGINT,
    official_only BIGINT,
    user_only BIGINT,
    cross_validated BIGINT,
    validation_rate NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*) as total_incidents,
        COUNT(*) FILTER (WHERE source_type = 'official') as official_only,
        COUNT(*) FILTER (WHERE source_type = 'user') as user_only,
        COUNT(*) FILTER (WHERE source_type = 'clustered') as cross_validated,
        ROUND(
            COUNT(*) FILTER (WHERE source_type = 'clustered')::numeric /
            NULLIF(COUNT(*) FILTER (WHERE source_type = 'official'), 0) * 100,
            2
        ) as validation_rate
    FROM crime_incidents
    WHERE created_at >= NOW() - INTERVAL '30 days';
END;
$$ LANGUAGE plpgsql;
