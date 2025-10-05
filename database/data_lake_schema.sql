-- Atlas AI Data Lake Schema
-- Centralized data architecture for all platforms: mobile app, analytics, police investigations

BEGIN;

-- Create data lake schema for raw and processed data
CREATE SCHEMA IF NOT EXISTS data_lake;
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS investigation;

-- =============================================
-- DATA LAKE LAYER - Raw and processed data
-- =============================================

-- Raw incident data from all sources (polisen.se, citizen reports, etc.)
CREATE TABLE IF NOT EXISTS data_lake.raw_incidents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source VARCHAR(50) NOT NULL, -- 'polisen', 'citizen', 'sensor', 'ai_detection'
    source_id VARCHAR(100), -- original ID from source system
    raw_data JSONB NOT NULL, -- complete raw data from source
    ingestion_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processing_status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'processed', 'failed'
    data_quality_score DECIMAL(3,2),

    -- Spatial data (extracted from raw_data for indexing)
    location GEOGRAPHY(POINT, 4326),
    location_address TEXT,

    -- Temporal data (extracted from raw_data for indexing)
    incident_timestamp TIMESTAMP WITH TIME ZONE,
    reported_timestamp TIMESTAMP WITH TIME ZONE,

    -- Basic classification (extracted from raw_data for indexing)
    incident_type VARCHAR(50),
    severity_level VARCHAR(20),

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT valid_processing_status CHECK (processing_status IN ('pending', 'processed', 'failed', 'duplicate')),
    CONSTRAINT valid_severity CHECK (severity_level IN ('safe', 'low', 'moderate', 'high', 'critical'))
);

-- Processed/normalized incident data for consumption by apps
CREATE TABLE IF NOT EXISTS data_lake.processed_incidents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    raw_incident_id UUID REFERENCES data_lake.raw_incidents(id),

    -- Normalized fields
    incident_type VARCHAR(50) NOT NULL,
    location GEOGRAPHY(POINT, 4326) NOT NULL,
    location_address TEXT,
    incident_time TIMESTAMP WITH TIME ZONE NOT NULL,
    reported_time TIMESTAMP WITH TIME ZONE NOT NULL,
    severity_level VARCHAR(20) NOT NULL,
    description TEXT,

    -- Source tracking
    source VARCHAR(50) NOT NULL,
    source_id VARCHAR(100),

    -- Status and verification
    verification_status VARCHAR(20) DEFAULT 'unverified',
    resolution_status VARCHAR(20) DEFAULT 'open',
    confidence_score DECIMAL(3,2) DEFAULT 0.5,

    -- Enriched data
    area_classification VARCHAR(50), -- 'residential', 'commercial', 'industrial', etc.
    risk_indicators JSONB, -- computed risk factors
    related_incident_ids UUID[], -- array of related incident IDs

    -- Quality and processing metadata
    data_quality_score DECIMAL(3,2),
    processing_pipeline VARCHAR(100),
    ai_analysis_results JSONB,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Constraints
    CONSTRAINT valid_verification CHECK (verification_status IN ('verified', 'unverified', 'disputed', 'false_alarm')),
    CONSTRAINT valid_resolution CHECK (resolution_status IN ('open', 'investigating', 'resolved', 'closed', 'false_alarm')),
    CONSTRAINT valid_severity_processed CHECK (severity_level IN ('safe', 'low', 'moderate', 'high', 'critical'))
);

-- User activity and feedback data
CREATE TABLE IF NOT EXISTS data_lake.user_interactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    session_id UUID,
    interaction_type VARCHAR(50) NOT NULL, -- 'report', 'feedback', 'view', 'share', etc.
    target_incident_id UUID REFERENCES data_lake.processed_incidents(id),

    -- Interaction data
    interaction_data JSONB,
    device_info JSONB,
    location GEOGRAPHY(POINT, 4326),

    -- Privacy and compliance
    anonymized BOOLEAN DEFAULT FALSE,
    retention_expires_at TIMESTAMP WITH TIME ZONE,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Sensor and IoT data (for future ML and real-time detection)
CREATE TABLE IF NOT EXISTS data_lake.sensor_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sensor_id VARCHAR(100) NOT NULL,
    sensor_type VARCHAR(50) NOT NULL, -- 'camera', 'audio', 'environmental', 'traffic'
    location GEOGRAPHY(POINT, 4326),

    -- Raw sensor data
    raw_data JSONB NOT NULL,
    media_urls TEXT[], -- URLs to video/audio/image files

    -- Processing status
    processing_status VARCHAR(20) DEFAULT 'pending',
    ai_analysis_results JSONB,

    -- Temporal data
    sensor_timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    ingestion_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Quality metrics
    data_quality_score DECIMAL(3,2),
    anomaly_detected BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================
-- ANALYTICS LAYER - Aggregated data for insights
-- =============================================

-- Pre-computed spatial risk zones
CREATE TABLE IF NOT EXISTS analytics.risk_zones (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    zone_name VARCHAR(100),
    zone_geometry GEOGRAPHY(POLYGON, 4326),
    center_location GEOGRAPHY(POINT, 4326),

    -- Risk metrics
    current_risk_level VARCHAR(20),
    risk_score DECIMAL(4,2),
    confidence_interval DECIMAL(4,2),

    -- Statistical data
    incident_count_24h INTEGER DEFAULT 0,
    incident_count_7d INTEGER DEFAULT 0,
    incident_count_30d INTEGER DEFAULT 0,

    -- Temporal patterns
    peak_risk_hours INTEGER[], -- array of hours (0-23) with highest risk
    seasonal_patterns JSONB,

    -- Classification
    area_type VARCHAR(50), -- 'residential', 'commercial', 'entertainment', etc.
    population_density VARCHAR(20), -- 'low', 'medium', 'high'

    -- Metadata
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    data_sources TEXT[], -- which data sources contributed to this zone
    computation_method VARCHAR(50),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Time series aggregations for dashboards
CREATE TABLE IF NOT EXISTS analytics.incident_timeseries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    time_bucket TIMESTAMP WITH TIME ZONE NOT NULL,
    granularity VARCHAR(10) NOT NULL, -- 'hour', 'day', 'week', 'month'

    -- Geographical scope
    geographic_scope VARCHAR(50), -- 'global', 'country', 'region', 'city', 'district'
    geographic_id VARCHAR(100), -- identifier for the geographic area
    location_bounds GEOGRAPHY(POLYGON, 4326),

    -- Aggregated metrics
    total_incidents INTEGER DEFAULT 0,
    incidents_by_type JSONB, -- {"theft": 5, "assault": 2, ...}
    incidents_by_severity JSONB, -- {"high": 3, "moderate": 4, ...}
    avg_response_time INTERVAL,
    resolution_rate DECIMAL(3,2),

    -- Trends
    change_from_previous DECIMAL(5,2), -- percentage change
    trend_direction VARCHAR(10), -- 'increasing', 'decreasing', 'stable'

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Unique constraint for time series data
    UNIQUE (time_bucket, granularity, geographic_scope, geographic_id)
);

-- =============================================
-- INVESTIGATION LAYER - Police investigation tools
-- =============================================

-- Investigation cases
CREATE TABLE IF NOT EXISTS investigation.cases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    case_number VARCHAR(50) UNIQUE NOT NULL,
    case_title VARCHAR(200) NOT NULL,
    case_type VARCHAR(50), -- 'single_incident', 'pattern_analysis', 'ongoing_threat'
    priority_level VARCHAR(20),

    -- Related incidents
    primary_incident_id UUID REFERENCES data_lake.processed_incidents(id),
    related_incident_ids UUID[],

    -- Investigation details
    lead_investigator VARCHAR(100),
    assigned_team VARCHAR(100),
    investigation_status VARCHAR(20) DEFAULT 'open',

    -- Spatial and temporal scope
    investigation_area GEOGRAPHY(POLYGON, 4326),
    time_range_start TIMESTAMP WITH TIME ZONE,
    time_range_end TIMESTAMP WITH TIME ZONE,

    -- Evidence and analysis
    evidence_summary TEXT,
    pattern_analysis JSONB,
    ai_insights JSONB,

    -- Case metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    closed_at TIMESTAMP WITH TIME ZONE,

    CONSTRAINT valid_investigation_status CHECK (investigation_status IN ('open', 'active', 'pending', 'closed', 'cold'))
);

-- Pattern detection results for investigations
CREATE TABLE IF NOT EXISTS investigation.pattern_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pattern_type VARCHAR(50) NOT NULL, -- 'temporal', 'spatial', 'behavioral', 'network'
    pattern_name VARCHAR(100),

    -- Pattern details
    pattern_description TEXT,
    confidence_score DECIMAL(3,2),
    statistical_significance DECIMAL(5,4),

    -- Related data
    incident_ids UUID[] NOT NULL,
    geographic_areas GEOGRAPHY(MULTIPOLYGON, 4326),
    time_periods JSONB, -- array of time ranges

    -- Analysis metadata
    detection_algorithm VARCHAR(100),
    analysis_parameters JSONB,
    validation_status VARCHAR(20) DEFAULT 'pending',

    -- Investigation link
    case_id UUID REFERENCES investigation.cases(id),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    validated_at TIMESTAMP WITH TIME ZONE,
    validated_by VARCHAR(100)
);

-- =============================================
-- INDEXES FOR PERFORMANCE
-- =============================================

-- Raw incidents indexes
CREATE INDEX IF NOT EXISTS idx_raw_incidents_source ON data_lake.raw_incidents(source);
CREATE INDEX IF NOT EXISTS idx_raw_incidents_source_id ON data_lake.raw_incidents(source, source_id);
CREATE INDEX IF NOT EXISTS idx_raw_incidents_location ON data_lake.raw_incidents USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_raw_incidents_timestamp ON data_lake.raw_incidents(incident_timestamp);
CREATE INDEX IF NOT EXISTS idx_raw_incidents_processing_status ON data_lake.raw_incidents(processing_status);

-- Processed incidents indexes
CREATE INDEX IF NOT EXISTS idx_processed_incidents_location ON data_lake.processed_incidents USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_processed_incidents_time ON data_lake.processed_incidents(incident_time);
CREATE INDEX IF NOT EXISTS idx_processed_incidents_type ON data_lake.processed_incidents(incident_type);
CREATE INDEX IF NOT EXISTS idx_processed_incidents_severity ON data_lake.processed_incidents(severity_level);
CREATE INDEX IF NOT EXISTS idx_processed_incidents_source ON data_lake.processed_incidents(source, source_id);
CREATE INDEX IF NOT EXISTS idx_processed_incidents_status ON data_lake.processed_incidents(verification_status, resolution_status);

-- Risk zones indexes
CREATE INDEX IF NOT EXISTS idx_risk_zones_geometry ON analytics.risk_zones USING GIST(zone_geometry);
CREATE INDEX IF NOT EXISTS idx_risk_zones_center ON analytics.risk_zones USING GIST(center_location);
CREATE INDEX IF NOT EXISTS idx_risk_zones_risk_level ON analytics.risk_zones(current_risk_level);

-- Time series indexes
CREATE INDEX IF NOT EXISTS idx_timeseries_time_granularity ON analytics.incident_timeseries(time_bucket, granularity);
CREATE INDEX IF NOT EXISTS idx_timeseries_geographic ON analytics.incident_timeseries(geographic_scope, geographic_id);

-- Investigation indexes
CREATE INDEX IF NOT EXISTS idx_cases_status ON investigation.cases(investigation_status);
CREATE INDEX IF NOT EXISTS idx_cases_area ON investigation.cases USING GIST(investigation_area);
CREATE INDEX IF NOT EXISTS idx_pattern_analysis_confidence ON investigation.pattern_analysis(confidence_score);
CREATE INDEX IF NOT EXISTS idx_pattern_analysis_type ON investigation.pattern_analysis(pattern_type);

-- =============================================
-- FUNCTIONS FOR DATA PROCESSING
-- =============================================

-- Function to migrate existing historical_crime_data to data lake
CREATE OR REPLACE FUNCTION data_lake.migrate_historical_data()
RETURNS INTEGER AS $$
DECLARE
    migrated_count INTEGER := 0;
BEGIN
    -- Insert historical crime data into raw_incidents
    INSERT INTO data_lake.raw_incidents (
        source,
        source_id,
        raw_data,
        location,
        location_address,
        incident_timestamp,
        reported_timestamp,
        incident_type,
        severity_level,
        data_quality_score,
        processing_status
    )
    SELECT
        'polisen' as source,
        hcd.polisen_id as source_id,
        jsonb_build_object(
            'polisen_id', hcd.polisen_id,
            'crime_type', hcd.crime_type,
            'crime_category', hcd.crime_category,
            'summary', hcd.summary,
            'outcome', hcd.outcome,
            'location_name', hcd.location_name,
            'has_coordinates', hcd.has_coordinates,
            'has_precise_time', hcd.has_precise_time,
            'latitude', hcd.latitude,
            'longitude', hcd.longitude
        ) as raw_data,
        hcd.location,
        hcd.location_name as location_address,
        hcd.datetime_occurred as incident_timestamp,
        hcd.datetime_reported as reported_timestamp,
        CASE hcd.crime_category
            WHEN 'theft' THEN 'theft'
            WHEN 'robbery' THEN 'robbery'
            WHEN 'assault' THEN 'assault'
            WHEN 'violence' THEN 'violence'
            WHEN 'vandalism' THEN 'vandalism'
            WHEN 'drug_offense' THEN 'drug_activity'
            WHEN 'traffic_accident' THEN 'traffic_accident'
            WHEN 'fire' THEN 'fire'
            WHEN 'homicide' THEN 'violence'
            ELSE 'other'
        END as incident_type,
        CASE
            WHEN hcd.severity_score >= 4 THEN 'high'
            WHEN hcd.severity_score >= 3 THEN 'moderate'
            WHEN hcd.severity_score >= 2 THEN 'low'
            ELSE 'safe'
        END as severity_level,
        hcd.data_quality_score,
        'processed' as processing_status
    FROM public.historical_crime_data hcd
    WHERE NOT EXISTS (
        SELECT 1 FROM data_lake.raw_incidents ri
        WHERE ri.source = 'polisen' AND ri.source_id = hcd.polisen_id
    );

    GET DIAGNOSTICS migrated_count = ROW_COUNT;

    -- Process raw incidents into processed incidents
    INSERT INTO data_lake.processed_incidents (
        raw_incident_id,
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
        confidence_score,
        data_quality_score,
        processing_pipeline
    )
    SELECT
        ri.id as raw_incident_id,
        ri.incident_type,
        ri.location,
        ri.location_address,
        ri.incident_timestamp,
        ri.reported_timestamp,
        ri.severity_level,
        ri.raw_data->>'summary' as description,
        ri.source,
        ri.source_id,
        CASE
            WHEN ri.data_quality_score >= 0.8 THEN 'verified'
            WHEN ri.data_quality_score >= 0.5 THEN 'unverified'
            ELSE 'disputed'
        END as verification_status,
        CASE
            WHEN ri.raw_data->>'outcome' IS NOT NULL AND LENGTH(ri.raw_data->>'outcome') > 0 THEN 'resolved'
            ELSE 'open'
        END as resolution_status,
        GREATEST(ri.data_quality_score, 0.3) as confidence_score,
        ri.data_quality_score,
        'historical_migration_v1' as processing_pipeline
    FROM data_lake.raw_incidents ri
    WHERE ri.source = 'polisen'
    AND ri.processing_status = 'processed'
    AND NOT EXISTS (
        SELECT 1 FROM data_lake.processed_incidents pi
        WHERE pi.source = 'polisen' AND pi.source_id = ri.source_id
    );

    RETURN migrated_count;
END;
$$ LANGUAGE plpgsql;

-- Function to process new raw incidents
CREATE OR REPLACE FUNCTION data_lake.process_raw_incident(raw_incident_id UUID)
RETURNS BOOLEAN AS $$
DECLARE
    raw_record RECORD;
    processed_id UUID;
BEGIN
    -- Get the raw incident
    SELECT * INTO raw_record FROM data_lake.raw_incidents WHERE id = raw_incident_id;

    IF NOT FOUND THEN
        RETURN FALSE;
    END IF;

    -- Insert into processed incidents
    INSERT INTO data_lake.processed_incidents (
        raw_incident_id,
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
        confidence_score,
        data_quality_score,
        processing_pipeline
    ) VALUES (
        raw_record.id,
        raw_record.incident_type,
        raw_record.location,
        raw_record.location_address,
        raw_record.incident_timestamp,
        raw_record.reported_timestamp,
        raw_record.severity_level,
        COALESCE(raw_record.raw_data->>'summary', raw_record.raw_data->>'description', 'No description available'),
        raw_record.source,
        raw_record.source_id,
        CASE
            WHEN raw_record.data_quality_score >= 0.8 THEN 'verified'
            WHEN raw_record.data_quality_score >= 0.5 THEN 'unverified'
            ELSE 'disputed'
        END,
        'open',
        GREATEST(raw_record.data_quality_score, 0.3),
        raw_record.data_quality_score,
        'auto_processing_v1'
    ) RETURNING id INTO processed_id;

    -- Update raw incident status
    UPDATE data_lake.raw_incidents
    SET processing_status = 'processed', updated_at = NOW()
    WHERE id = raw_incident_id;

    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT USAGE ON SCHEMA data_lake TO atlas_user;
GRANT USAGE ON SCHEMA analytics TO atlas_user;
GRANT USAGE ON SCHEMA investigation TO atlas_user;

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA data_lake TO atlas_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA analytics TO atlas_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA investigation TO atlas_user;

GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA data_lake TO atlas_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA analytics TO atlas_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA investigation TO atlas_user;

COMMIT;