-- Migration: Consolidate mobile_app and public schemas
-- Date: 2025-10-03
-- Purpose: Eliminate duplicate schemas, merge all data into public schema
-- Impact: Single source of truth, eliminate data fragmentation

-- ==============================================================================
-- STEP 1: Verify schemas exist
-- ==============================================================================

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name = 'mobile_app') THEN
        RAISE NOTICE 'mobile_app schema does not exist - nothing to migrate';
        RETURN;
    END IF;

    RAISE NOTICE 'Starting schema consolidation migration...';
END $$;

-- ==============================================================================
-- STEP 2: Migrate data from mobile_app to public schema
-- ==============================================================================

-- 2.1 Migrate users
INSERT INTO public.users (
    id, username, email, hashed_password, full_name, phone_number,
    user_type, location, preferences, created_at, updated_at,
    is_active, email_verified, last_login_at
)
SELECT
    id, username, email, hashed_password, full_name, phone_number,
    user_type, location, preferences, created_at, updated_at,
    is_active, email_verified, last_login_at
FROM mobile_app.users
ON CONFLICT (id) DO NOTHING;

RAISE NOTICE 'Migrated users: %', (SELECT COUNT(*) FROM mobile_app.users);

-- 2.2 Migrate incidents
INSERT INTO public.crime_incidents (
    id, incident_type, severity, description, location, latitude, longitude,
    occurred_at, reported_at, created_at, updated_at, source, source_id,
    confidence_score, status, is_verified, incident_metadata
)
SELECT
    id,
    incident_type,
    CASE severity_level
        WHEN 'critical' THEN 5
        WHEN 'high' THEN 4
        WHEN 'moderate' THEN 3
        WHEN 'low' THEN 2
        ELSE 1
    END as severity,
    description,
    location,
    ST_Y(location::geometry) as latitude,
    ST_X(location::geometry) as longitude,
    incident_time as occurred_at,
    reported_time as reported_at,
    reported_time as created_at,
    reported_time as updated_at,
    source,
    NULL as source_id,
    data_quality_score as confidence_score,
    resolution_status as status,
    CASE verification_status
        WHEN 'verified' THEN true
        ELSE false
    END as is_verified,
    metadata as incident_metadata
FROM mobile_app.incidents
ON CONFLICT (id) DO NOTHING;

RAISE NOTICE 'Migrated incidents: %', (SELECT COUNT(*) FROM mobile_app.incidents);

-- 2.3 Migrate media files
INSERT INTO public.media_files (
    id, user_id, incident_id, file_type, file_path, file_size,
    mime_type, thumbnail_path, uploaded_at, processed_at, storage_location,
    is_evidence, metadata
)
SELECT
    id, user_id, incident_id, media_type as file_type, file_path, file_size,
    mime_type, thumbnail_path, uploaded_at, processed_at, storage_location,
    COALESCE(is_evidence, false) as is_evidence, metadata
FROM mobile_app.media_files
ON CONFLICT (id) DO NOTHING;

RAISE NOTICE 'Migrated media_files: %', (SELECT COUNT(*) FROM mobile_app.media_files);

-- 2.4 Migrate user settings
INSERT INTO public.user_settings (
    user_id, notification_enabled, notification_radius_km,
    notification_types, theme, language, privacy_mode, created_at, updated_at
)
SELECT
    user_id, notification_enabled, notification_radius_km,
    notification_types, theme, language, privacy_mode, created_at, updated_at
FROM mobile_app.user_settings
ON CONFLICT (user_id) DO NOTHING;

RAISE NOTICE 'Migrated user_settings: %', (SELECT COUNT(*) FROM mobile_app.user_settings);

-- 2.5 Migrate push tokens
INSERT INTO public.push_tokens (
    id, user_id, push_token, device_type, device_id, is_active, created_at
)
SELECT
    id, user_id, push_token, device_type, device_id, is_active, created_at
FROM mobile_app.push_tokens
ON CONFLICT (id) DO NOTHING;

RAISE NOTICE 'Migrated push_tokens: %', (SELECT COUNT(*) FROM mobile_app.push_tokens);

-- 2.6 Migrate user locations
INSERT INTO public.user_locations (
    id, user_id, name, latitude, longitude, location, radius_meters,
    is_home, created_at
)
SELECT
    id, user_id, location_name as name,
    ST_Y(location::geometry) as latitude,
    ST_X(location::geometry) as longitude,
    location, radius_meters, is_home, created_at
FROM mobile_app.user_locations
ON CONFLICT (id) DO NOTHING;

RAISE NOTICE 'Migrated user_locations: %', (SELECT COUNT(*) FROM mobile_app.user_locations);

-- ==============================================================================
-- STEP 3: Add mobile_source column to track origin
-- ==============================================================================

ALTER TABLE public.crime_incidents
ADD COLUMN IF NOT EXISTS mobile_source BOOLEAN DEFAULT FALSE;

UPDATE public.crime_incidents
SET mobile_source = TRUE
WHERE source = 'mobile_app' OR id IN (
    SELECT id FROM mobile_app.incidents
);

RAISE NOTICE 'Tagged mobile-sourced incidents';

-- ==============================================================================
-- STEP 4: Create backup of mobile_app schema before dropping
-- ==============================================================================

-- Export mobile_app schema to separate backup schema
CREATE SCHEMA IF NOT EXISTS mobile_app_backup_20251003;

-- Clone all tables to backup schema
DO $$
DECLARE
    table_name text;
BEGIN
    FOR table_name IN
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = 'mobile_app'
    LOOP
        EXECUTE format('CREATE TABLE mobile_app_backup_20251003.%I AS SELECT * FROM mobile_app.%I',
                      table_name, table_name);
        RAISE NOTICE 'Backed up table: %', table_name;
    END LOOP;
END $$;

-- ==============================================================================
-- STEP 5: Drop mobile_app schema
-- ==============================================================================

-- Uncomment after verifying migration
-- DROP SCHEMA mobile_app CASCADE;
-- RAISE NOTICE 'Dropped mobile_app schema';

-- ==============================================================================
-- STEP 6: Update sequences and constraints
-- ==============================================================================

-- Ensure all sequences are at correct value
SELECT setval('public.crime_incidents_id_seq',
              COALESCE((SELECT MAX(id::integer) FROM public.crime_incidents WHERE id ~ '^\d+$'), 1),
              false);

-- ==============================================================================
-- STEP 7: Verify migration
-- ==============================================================================

DO $$
DECLARE
    mobile_count INTEGER;
    public_count INTEGER;
BEGIN
    -- Count incidents
    SELECT COUNT(*) INTO mobile_count FROM mobile_app.incidents;
    SELECT COUNT(*) INTO public_count FROM public.crime_incidents WHERE mobile_source = true;

    RAISE NOTICE 'Verification - mobile_app incidents: %, public incidents (mobile): %',
                 mobile_count, public_count;

    IF mobile_count != public_count THEN
        RAISE WARNING 'Migration count mismatch! Check for errors.';
    ELSE
        RAISE NOTICE 'Migration verified successfully!';
    END IF;
END $$;

-- ==============================================================================
-- STEP 8: Create view for backward compatibility (temporary)
-- ==============================================================================

-- Create view so old queries still work during transition period
CREATE OR REPLACE VIEW mobile_app.incidents AS
SELECT
    id,
    incident_type,
    CASE severity
        WHEN 5 THEN 'critical'
        WHEN 4 THEN 'high'
        WHEN 3 THEN 'moderate'
        WHEN 2 THEN 'low'
        ELSE 'info'
    END as severity_level,
    description,
    location,
    occurred_at as incident_time,
    reported_at as reported_time,
    source,
    CASE is_verified WHEN true THEN 'verified' ELSE 'pending' END as verification_status,
    status as resolution_status,
    confidence_score as data_quality_score,
    incident_metadata as metadata
FROM public.crime_incidents
WHERE mobile_source = true;

COMMENT ON VIEW mobile_app.incidents IS
'Backward compatibility view - redirects to public.crime_incidents. Remove after code migration.';

-- ==============================================================================
-- MIGRATION COMPLETE
-- ==============================================================================

RAISE NOTICE '
================================================================================
Schema Consolidation Migration Complete!

Next Steps:
1. Verify data in public schema
2. Update application code to use public schema only
3. Test all endpoints
4. After 7 days, uncomment DROP SCHEMA command (line 112)
5. Remove backward compatibility views

Backup Location: mobile_app_backup_20251003 schema
================================================================================
';
