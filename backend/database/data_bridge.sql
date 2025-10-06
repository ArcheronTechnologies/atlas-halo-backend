-- Data Bridge Functions - Sync existing polisen.se data to mobile_app schema
-- Preserves all historical crime data while making it available to mobile APIs

-- Function to convert severity integer to string
CREATE OR REPLACE FUNCTION mobile_app.convert_severity_to_string(severity_int INTEGER)
RETURNS VARCHAR(20) AS $$
BEGIN
    CASE severity_int
        WHEN 1 THEN RETURN 'low';
        WHEN 2 THEN RETURN 'low';
        WHEN 3 THEN RETURN 'moderate';
        WHEN 4 THEN RETURN 'high';
        WHEN 5 THEN RETURN 'critical';
        ELSE RETURN 'moderate';
    END CASE;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to normalize crime type from polisen.se data
CREATE OR REPLACE FUNCTION mobile_app.normalize_crime_type(polisen_type VARCHAR(100))
RETURNS VARCHAR(50) AS $$
BEGIN
    -- Normalize Swedish crime types to standard categories
    CASE LOWER(polisen_type)
        WHEN 'stöld' THEN RETURN 'theft';
        WHEN 'theft' THEN RETURN 'theft';
        WHEN 'rån' THEN RETURN 'robbery';
        WHEN 'robbery' THEN RETURN 'robbery';
        WHEN 'misshandel' THEN RETURN 'assault';
        WHEN 'assault' THEN RETURN 'assault';
        WHEN 'våldtäkt' THEN RETURN 'violence';
        WHEN 'violence' THEN RETURN 'violence';
        WHEN 'skadegörelse' THEN RETURN 'vandalism';
        WHEN 'vandalism' THEN RETURN 'vandalism';
        WHEN 'narkotika' THEN RETURN 'drug_activity';
        WHEN 'drug_activity' THEN RETURN 'drug_activity';
        WHEN 'misstänkt brott' THEN RETURN 'suspicious_activity';
        WHEN 'suspicious_activity' THEN RETURN 'suspicious_activity';
        WHEN 'trafficking' THEN RETURN 'trafficking';
        WHEN 'störning av allmän ordning' THEN RETURN 'public_disturbance';
        WHEN 'public_disturbance' THEN RETURN 'public_disturbance';
        WHEN 'gränskontroll' THEN RETURN 'border_control';
        WHEN 'trafikolycka' THEN RETURN 'traffic_accident';
        WHEN 'traffic_accident' THEN RETURN 'traffic_accident';
        WHEN 'brand' THEN RETURN 'fire';
        WHEN 'fire' THEN RETURN 'fire';
        WHEN 'övrigt' THEN RETURN 'other';
        WHEN 'other' THEN RETURN 'other';
        ELSE RETURN 'other';
    END CASE;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to sync existing historical_crime_data to mobile_app.incidents
CREATE OR REPLACE FUNCTION mobile_app.sync_historical_crime_data()
RETURNS INTEGER AS $$
DECLARE
    sync_count INTEGER := 0;
    record_data RECORD;
BEGIN
    -- Insert historical crime data into mobile incidents table
    INSERT INTO mobile_app.incidents (
        legacy_polisen_id,
        incident_type,
        location,
        location_address,
        incident_time,
        reported_time,
        severity_level,
        description,
        source,
        source_id,
        verification_status,
        resolution_status,
        metadata,
        data_quality_score,
        created_at,
        updated_at
    )
    SELECT
        hcd.polisen_id,
        mobile_app.normalize_crime_type(hcd.crime_type),
        hcd.location,
        hcd.location_name,
        hcd.datetime_occurred,
        hcd.datetime_reported,
        mobile_app.convert_severity_to_string(hcd.severity_score),
        hcd.summary,
        'polisen',
        hcd.polisen_id,
        CASE
            WHEN hcd.data_quality_score >= 0.8 THEN 'verified'
            WHEN hcd.data_quality_score >= 0.5 THEN 'unverified'
            ELSE 'disputed'
        END,
        CASE
            WHEN hcd.outcome IS NOT NULL AND LENGTH(hcd.outcome) > 0 THEN 'resolved'
            ELSE 'open'
        END,
        jsonb_build_object(
            'crime_category', hcd.crime_category,
            'outcome', hcd.outcome,
            'has_coordinates', hcd.has_coordinates,
            'has_precise_time', hcd.has_precise_time,
            'original_severity_score', hcd.severity_score
        ),
        hcd.data_quality_score,
        hcd.created_at,
        hcd.updated_at
    FROM public.historical_crime_data hcd
    WHERE NOT EXISTS (
        SELECT 1 FROM mobile_app.incidents mi
        WHERE mi.legacy_polisen_id = hcd.polisen_id
    );

    GET DIAGNOSTICS sync_count = ROW_COUNT;

    -- Update legacy_id_mapping table
    INSERT INTO mobile_app.legacy_id_mapping (mobile_uuid, legacy_table, legacy_id)
    SELECT
        mi.id,
        'historical_crime_data',
        mi.legacy_polisen_id
    FROM mobile_app.incidents mi
    WHERE mi.legacy_polisen_id IS NOT NULL
    AND NOT EXISTS (
        SELECT 1 FROM mobile_app.legacy_id_mapping lim
        WHERE lim.legacy_table = 'historical_crime_data'
        AND lim.legacy_id = mi.legacy_polisen_id
    );

    RETURN sync_count;
END;
$$ LANGUAGE plpgsql;

-- Function to create a trigger for real-time sync
CREATE OR REPLACE FUNCTION mobile_app.sync_new_polisen_data()
RETURNS TRIGGER AS $$
DECLARE
    new_incident_id UUID;
BEGIN
    -- Insert new polisen data into mobile incidents
    INSERT INTO mobile_app.incidents (
        legacy_polisen_id,
        incident_type,
        location,
        location_address,
        incident_time,
        reported_time,
        severity_level,
        description,
        source,
        source_id,
        verification_status,
        resolution_status,
        metadata,
        data_quality_score,
        created_at,
        updated_at
    ) VALUES (
        NEW.polisen_id,
        mobile_app.normalize_crime_type(NEW.crime_type),
        NEW.location,
        NEW.location_name,
        NEW.datetime_occurred,
        NEW.datetime_reported,
        mobile_app.convert_severity_to_string(NEW.severity_score),
        NEW.summary,
        'polisen',
        NEW.polisen_id,
        CASE
            WHEN NEW.data_quality_score >= 0.8 THEN 'verified'
            WHEN NEW.data_quality_score >= 0.5 THEN 'unverified'
            ELSE 'disputed'
        END,
        CASE
            WHEN NEW.outcome IS NOT NULL AND LENGTH(NEW.outcome) > 0 THEN 'resolved'
            ELSE 'open'
        END,
        jsonb_build_object(
            'crime_category', NEW.crime_category,
            'outcome', NEW.outcome,
            'has_coordinates', NEW.has_coordinates,
            'has_precise_time', NEW.has_precise_time,
            'original_severity_score', NEW.severity_score
        ),
        NEW.data_quality_score,
        NEW.created_at,
        NEW.updated_at
    ) RETURNING id INTO new_incident_id;

    -- Add to legacy mapping
    INSERT INTO mobile_app.legacy_id_mapping (mobile_uuid, legacy_table, legacy_id)
    VALUES (new_incident_id, 'historical_crime_data', NEW.polisen_id);

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for real-time sync of new polisen data
DROP TRIGGER IF EXISTS sync_polisen_to_mobile ON public.historical_crime_data;
CREATE TRIGGER sync_polisen_to_mobile
    AFTER INSERT ON public.historical_crime_data
    FOR EACH ROW
    EXECUTE FUNCTION mobile_app.sync_new_polisen_data();

-- Function to update existing incident if polisen data changes
CREATE OR REPLACE FUNCTION mobile_app.update_polisen_data()
RETURNS TRIGGER AS $$
BEGIN
    -- Update corresponding mobile incident
    UPDATE mobile_app.incidents SET
        incident_type = mobile_app.normalize_crime_type(NEW.crime_type),
        location = NEW.location,
        location_address = NEW.location_name,
        incident_time = NEW.datetime_occurred,
        reported_time = NEW.datetime_reported,
        severity_level = mobile_app.convert_severity_to_string(NEW.severity_score),
        description = NEW.summary,
        resolution_status = CASE
            WHEN NEW.outcome IS NOT NULL AND LENGTH(NEW.outcome) > 0 THEN 'resolved'
            ELSE 'open'
        END,
        metadata = jsonb_build_object(
            'crime_category', NEW.crime_category,
            'outcome', NEW.outcome,
            'has_coordinates', NEW.has_coordinates,
            'has_precise_time', NEW.has_precise_time,
            'original_severity_score', NEW.severity_score
        ),
        data_quality_score = NEW.data_quality_score,
        updated_at = NEW.updated_at
    WHERE legacy_polisen_id = NEW.polisen_id;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for updating mobile incidents when polisen data changes
DROP TRIGGER IF EXISTS update_polisen_to_mobile ON public.historical_crime_data;
CREATE TRIGGER update_polisen_to_mobile
    AFTER UPDATE ON public.historical_crime_data
    FOR EACH ROW
    EXECUTE FUNCTION mobile_app.update_polisen_data();

-- Function to generate safety zones based on incident clustering
CREATE OR REPLACE FUNCTION mobile_app.generate_safety_zones_from_incidents()
RETURNS INTEGER AS $$
DECLARE
    zone_count INTEGER := 0;
    cluster_data RECORD;
BEGIN
    -- Generate safety zones based on incident density
    -- This uses spatial clustering to identify high-risk areas

    WITH incident_clusters AS (
        SELECT
            ST_ClusterKMeans(location, 20) OVER() as cluster_id,
            location,
            severity_level,
            incident_time
        FROM mobile_app.incidents
        WHERE incident_time >= NOW() - INTERVAL '30 days'
        AND resolution_status != 'false_alarm'
    ),
    cluster_centers AS (
        SELECT
            cluster_id,
            ST_Centroid(ST_Collect(location)) as center_location,
            COUNT(*) as incident_count,
            AVG(CASE severity_level
                WHEN 'critical' THEN 5
                WHEN 'high' THEN 4
                WHEN 'moderate' THEN 3
                WHEN 'low' THEN 2
                WHEN 'safe' THEN 1
                ELSE 0
            END) as avg_severity,
            COUNT(CASE WHEN incident_time >= NOW() - INTERVAL '24 hours' THEN 1 END) as recent_count
        FROM incident_clusters
        GROUP BY cluster_id
        HAVING COUNT(*) >= 3 -- At least 3 incidents to form a zone
    )
    INSERT INTO mobile_app.safety_zones (
        zone_name,
        center_location,
        radius_meters,
        current_risk_level,
        risk_score,
        incident_count_24h,
        incident_count_7d,
        last_incident_time,
        area_type,
        created_at,
        updated_at
    )
    SELECT
        'Auto-generated Zone ' || cluster_id,
        center_location,
        CASE
            WHEN incident_count >= 20 THEN 500
            WHEN incident_count >= 10 THEN 750
            ELSE 1000
        END as radius_meters,
        CASE
            WHEN avg_severity >= 4.0 OR recent_count >= 5 THEN 'critical'
            WHEN avg_severity >= 3.5 OR recent_count >= 3 THEN 'high'
            WHEN avg_severity >= 2.5 OR recent_count >= 2 THEN 'moderate'
            WHEN avg_severity >= 1.5 THEN 'low'
            ELSE 'safe'
        END as current_risk_level,
        LEAST(avg_severity * 2, 10.0) as risk_score,
        recent_count::INTEGER as incident_count_24h,
        incident_count::INTEGER as incident_count_7d,
        NOW() - INTERVAL '1 day' as last_incident_time,
        'mixed' as area_type,
        NOW(),
        NOW()
    FROM cluster_centers
    WHERE NOT EXISTS (
        SELECT 1 FROM mobile_app.safety_zones sz
        WHERE ST_DWithin(sz.center_location, cluster_centers.center_location, 1000)
    );

    GET DIAGNOSTICS zone_count = ROW_COUNT;
    RETURN zone_count;
END;
$$ LANGUAGE plpgsql;

-- Function to update safety zone risk levels based on recent incidents
CREATE OR REPLACE FUNCTION mobile_app.update_safety_zone_risks()
RETURNS INTEGER AS $$
DECLARE
    updated_count INTEGER := 0;
BEGIN
    UPDATE mobile_app.safety_zones SET
        risk_score = mobile_app.calculate_risk_score(
            ST_Y(center_location::geometry),
            ST_X(center_location::geometry),
            radius_meters
        ),
        current_risk_level = CASE
            WHEN mobile_app.calculate_risk_score(
                ST_Y(center_location::geometry),
                ST_X(center_location::geometry),
                radius_meters
            ) >= 8.0 THEN 'critical'
            WHEN mobile_app.calculate_risk_score(
                ST_Y(center_location::geometry),
                ST_X(center_location::geometry),
                radius_meters
            ) >= 6.0 THEN 'high'
            WHEN mobile_app.calculate_risk_score(
                ST_Y(center_location::geometry),
                ST_X(center_location::geometry),
                radius_meters
            ) >= 4.0 THEN 'moderate'
            WHEN mobile_app.calculate_risk_score(
                ST_Y(center_location::geometry),
                ST_X(center_location::geometry),
                radius_meters
            ) >= 2.0 THEN 'low'
            ELSE 'safe'
        END,
        incident_count_24h = (
            SELECT COUNT(*)::INTEGER
            FROM mobile_app.incidents i
            WHERE ST_DWithin(i.location, center_location, radius_meters)
            AND i.incident_time >= NOW() - INTERVAL '24 hours'
            AND i.resolution_status != 'false_alarm'
        ),
        incident_count_7d = (
            SELECT COUNT(*)::INTEGER
            FROM mobile_app.incidents i
            WHERE ST_DWithin(i.location, center_location, radius_meters)
            AND i.incident_time >= NOW() - INTERVAL '7 days'
            AND i.resolution_status != 'false_alarm'
        ),
        last_incident_time = (
            SELECT MAX(i.incident_time)
            FROM mobile_app.incidents i
            WHERE ST_DWithin(i.location, center_location, radius_meters)
            AND i.resolution_status != 'false_alarm'
        ),
        updated_at = NOW()
    WHERE created_at < NOW() - INTERVAL '5 minutes'; -- Only update zones older than 5 minutes

    GET DIAGNOSTICS updated_count = ROW_COUNT;
    RETURN updated_count;
END;
$$ LANGUAGE plpgsql;

-- View to present unified incident data to mobile APIs
CREATE OR REPLACE VIEW mobile_app.mobile_incidents_view AS
SELECT
    i.id,
    i.incident_type,
    i.location,
    i.location_address,
    i.incident_time,
    i.severity_level,
    i.description,
    i.source,
    i.verification_status,
    i.resolution_status,
    i.data_quality_score,
    -- Add computed fields for mobile app
    EXTRACT(EPOCH FROM (NOW() - i.incident_time))/3600 as hours_ago,
    ST_Y(i.location::geometry) as latitude,
    ST_X(i.location::geometry) as longitude,
    CASE
        WHEN i.incident_time >= NOW() - INTERVAL '2 hours' THEN 'very_recent'
        WHEN i.incident_time >= NOW() - INTERVAL '24 hours' THEN 'recent'
        WHEN i.incident_time >= NOW() - INTERVAL '7 days' THEN 'this_week'
        ELSE 'older'
    END as recency_category
FROM mobile_app.incidents i
WHERE i.resolution_status != 'false_alarm';

-- Grant permissions for mobile app user (will be created later)
-- GRANT USAGE ON SCHEMA mobile_app TO mobile_app_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA mobile_app TO mobile_app_user;
-- GRANT SELECT ON mobile_app.mobile_incidents_view TO mobile_app_user;

COMMIT;