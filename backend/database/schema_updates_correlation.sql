-- Schema Updates for Sensor Fusion / Multi-User Correlation
-- Date: 2025-10-15
-- Purpose: Add fields to support incident correlation

-- Add correlation columns to incidents table
ALTER TABLE incidents
ADD COLUMN IF NOT EXISTS reporter_count INTEGER DEFAULT 1;

ALTER TABLE incidents
ADD COLUMN IF NOT EXISTS corroborating_reports JSONB DEFAULT '[]'::jsonb;

ALTER TABLE incidents
ADD COLUMN IF NOT EXISTS video_ids TEXT[] DEFAULT ARRAY[]::TEXT[];

-- Add index for faster correlation queries
CREATE INDEX IF NOT EXISTS idx_incidents_location_time
ON incidents USING GIST (
    ST_SetSRID(ST_MakePoint(longitude, latitude), 4326),
    occurred_at
);

-- Add index for incident type filtering
CREATE INDEX IF NOT EXISTS idx_incidents_type_time
ON incidents (incident_type, occurred_at DESC);

-- Comments
COMMENT ON COLUMN incidents.reporter_count IS 'Number of users who reported this incident';
COMMENT ON COLUMN incidents.corroborating_reports IS 'Array of {user_id, reported_at, video_id, confidence} objects';
COMMENT ON COLUMN incidents.video_ids IS 'Array of all video IDs associated with this incident';
