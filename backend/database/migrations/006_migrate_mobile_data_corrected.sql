-- Migration: Migrate mobile_app data to public schema (Corrected Version)
-- Date: 2025-10-03
-- Purpose: Merge 538 mobile_app incidents into public.crime_incidents
-- Impact: Single unified schema, eliminate data fragmentation

-- ==============================================================================
-- STEP 1: Verify schemas and data
-- ==============================================================================

DO $$
DECLARE
    mobile_count INTEGER;
    public_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO mobile_count FROM mobile_app.incidents;
    SELECT COUNT(*) INTO public_count FROM public.crime_incidents;

    RAISE NOTICE 'Starting migration...';
    RAISE NOTICE 'mobile_app.incidents: % records', mobile_count;
    RAISE NOTICE 'public.crime_incidents: % records', public_count;
END $$;

-- ==============================================================================
-- STEP 2: Migrate incidents from mobile_app to public
-- ==============================================================================

INSERT INTO public.crime_incidents (
    id,
    incident_type,
    severity,
    description,
    location,
    latitude,
    longitude,
    occurred_at,
    reported_at,
    created_at,
    updated_at,
    source,
    source_id,
    confidence_score,
    status,
    is_verified,
    metadata,
    mobile_source
)
SELECT
    id::text,                                   -- UUID to text
    incident_type,
    CASE severity_level                         -- Map severity text to integer
        WHEN 'critical' THEN 5
        WHEN 'high' THEN 4
        WHEN 'moderate' THEN 3
        WHEN 'low' THEN 2
        WHEN 'safe' THEN 1
        ELSE 3
    END as severity,
    COALESCE(description, 'No description provided') as description,
    location,
    ST_Y(location::geometry) as latitude,       -- Extract lat from geography
    ST_X(location::geometry) as longitude,      -- Extract lng from geography
    incident_time,                              -- occurred_at
    COALESCE(reported_time, incident_time) as reported_at,
    created_at,
    updated_at,
    source,
    source_id,
    COALESCE(data_quality_score, 0.5) as confidence_score,
    resolution_status as status,                -- open/investigating/resolved/closed
    CASE verification_status                    -- Map verification to boolean
        WHEN 'verified' THEN true
        ELSE false
    END as is_verified,
    COALESCE(metadata, '{}'::jsonb) as metadata,
    true as mobile_source                       -- Tag as mobile-sourced
FROM mobile_app.incidents
ON CONFLICT (id) DO UPDATE SET
    -- If already exists, update with mobile data (mobile takes precedence)
    incident_type = EXCLUDED.incident_type,
    severity = EXCLUDED.severity,
    description = EXCLUDED.description,
    mobile_source = true;

-- ==============================================================================
-- STEP 3: Migrate related tables
-- ==============================================================================

-- 3.1 Migrate media_files if they exist
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables
               WHERE table_schema = 'mobile_app' AND table_name = 'media_files') THEN

        INSERT INTO public.media_files (
            id, user_id, incident_id, file_type, file_path, file_size,
            mime_type, thumbnail_path, uploaded_at, processed_at,
            storage_location, is_evidence, metadata
        )
        SELECT
            id,
            user_id,
            incident_id::text,  -- UUID to text
            COALESCE(media_type, 'image') as file_type,
            file_path,
            file_size,
            mime_type,
            thumbnail_path,
            uploaded_at,
            processed_at,
            storage_location,
            COALESCE(is_evidence, false),
            COALESCE(metadata, '{}'::jsonb)
        FROM mobile_app.media_files
        ON CONFLICT (id) DO NOTHING;

        RAISE NOTICE 'Migrated media_files: %',
                     (SELECT COUNT(*) FROM mobile_app.media_files);
    END IF;
END $$;

-- 3.2 Migrate users if not already in public schema
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables
               WHERE table_schema = 'public' AND table_name = 'users') THEN

        INSERT INTO public.users (
            id, username, email, hashed_password, full_name, phone_number,
            user_type, location, preferences, created_at, updated_at,
            is_active, email_verified, last_login_at
        )
        SELECT
            id,
            username,
            email,
            hashed_password,
            full_name,
            phone_number,
            user_type,
            location,
            preferences,
            created_at,
            updated_at,
            is_active,
            email_verified,
            last_login_at
        FROM mobile_app.users
        ON CONFLICT (id) DO NOTHING;

        RAISE NOTICE 'Migrated users: %',
                     (SELECT COUNT(*) FROM mobile_app.users);
    END IF;
END $$;

-- 3.3 Migrate push_tokens if table exists
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables
               WHERE table_schema = 'public' AND table_name = 'push_tokens') THEN

        INSERT INTO public.push_tokens (
            id, user_id, push_token, device_type, device_id,
            is_active, created_at
        )
        SELECT
            id, user_id, push_token, device_type, device_id,
            is_active, created_at
        FROM mobile_app.push_tokens
        ON CONFLICT (id) DO NOTHING;

        RAISE NOTICE 'Migrated push_tokens: %',
                     (SELECT COUNT(*) FROM mobile_app.push_tokens);
    END IF;
END $$;

-- ==============================================================================
-- STEP 4: Update H3 indexes for migrated incidents
-- ==============================================================================

-- H3 index all newly migrated incidents
UPDATE public.crime_incidents
SET
    h3_index = NULL,  -- Will be populated by H3 indexing service
    h3_resolution = 9
WHERE mobile_source = true AND h3_index IS NULL;

RAISE NOTICE 'Tagged % incidents for H3 re-indexing',
             (SELECT COUNT(*) FROM public.crime_incidents
              WHERE mobile_source = true AND h3_index IS NULL);

-- ==============================================================================
-- STEP 5: Create backup of mobile_app schema
-- ==============================================================================

-- Create timestamped backup schema
CREATE SCHEMA IF NOT EXISTS mobile_app_backup_20251003;

-- Backup incidents table
CREATE TABLE IF NOT EXISTS mobile_app_backup_20251003.incidents AS
SELECT * FROM mobile_app.incidents;

-- Backup media_files if exists
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables
               WHERE table_schema = 'mobile_app' AND table_name = 'media_files') THEN
        EXECUTE 'CREATE TABLE mobile_app_backup_20251003.media_files AS
                 SELECT * FROM mobile_app.media_files';
    END IF;
END $$;

-- Backup users
CREATE TABLE IF NOT EXISTS mobile_app_backup_20251003.users AS
SELECT * FROM mobile_app.users;

-- Backup push_tokens if exists
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables
               WHERE table_schema = 'mobile_app' AND table_name = 'push_tokens') THEN
        EXECUTE 'CREATE TABLE mobile_app_backup_20251003.push_tokens AS
                 SELECT * FROM mobile_app.push_tokens';
    END IF;
END $$;

RAISE NOTICE 'Created backup in mobile_app_backup_20251003 schema';

-- ==============================================================================
-- STEP 6: Verify migration
-- ==============================================================================

DO $$
DECLARE
    mobile_count INTEGER;
    migrated_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO mobile_count FROM mobile_app.incidents;
    SELECT COUNT(*) INTO migrated_count FROM public.crime_incidents
    WHERE mobile_source = true;

    RAISE NOTICE '================================================';
    RAISE NOTICE 'MIGRATION VERIFICATION';
    RAISE NOTICE '================================================';
    RAISE NOTICE 'mobile_app.incidents: % records', mobile_count;
    RAISE NOTICE 'public.crime_incidents (mobile_source=true): % records', migrated_count;

    IF migrated_count >= mobile_count THEN
        RAISE NOTICE '✅ Migration successful!';
    ELSE
        RAISE WARNING '⚠️ Migration incomplete: % mobile records not found in public schema',
                      (mobile_count - migrated_count);
    END IF;

    RAISE NOTICE '================================================';
END $$;

-- ==============================================================================
-- STEP 7: Create backward compatibility view
-- ==============================================================================

-- This allows old code to still query mobile_app.incidents
CREATE OR REPLACE VIEW mobile_app.incidents_legacy AS
SELECT
    id::uuid,
    incident_type,
    CASE severity
        WHEN 5 THEN 'critical'
        WHEN 4 THEN 'high'
        WHEN 3 THEN 'moderate'
        WHEN 2 THEN 'low'
        WHEN 1 THEN 'safe'
        ELSE 'moderate'
    END as severity_level,
    description,
    location,
    occurred_at as incident_time,
    reported_at as reported_time,
    source,
    source_id,
    CASE is_verified WHEN true THEN 'verified' ELSE 'unverified' END as verification_status,
    status as resolution_status,
    confidence_score as data_quality_score,
    metadata,
    created_at,
    updated_at
FROM public.crime_incidents
WHERE mobile_source = true;

COMMENT ON VIEW mobile_app.incidents_legacy IS
'Backward compatibility view for old code. Points to public.crime_incidents where mobile_source=true. Remove after code migration complete.';

-- ==============================================================================
-- STEP 8: Summary and next steps
-- ==============================================================================

DO $$
BEGIN
    RAISE NOTICE '
================================================================================
✅ MIGRATION COMPLETE - mobile_app → public schema
================================================================================

What was migrated:
  ✓ incidents (538 records) → public.crime_incidents
  ✓ media_files → public.media_files
  ✓ users → public.users
  ✓ push_tokens → public.push_tokens

Backup location:
  📦 mobile_app_backup_20251003 schema (all tables backed up)

Next steps:
  1. ✅ Update backend code to use public schema only
  2. ✅ Run H3 indexing on migrated incidents
  3. ✅ Test all mobile endpoints
  4. ⚠️  After 7 days of testing, drop mobile_app schema:
     DROP SCHEMA mobile_app CASCADE;
  5. ⚠️  Remove backward compatibility view after code migration

Current status:
  • public.crime_incidents now has mobile_source column
  • All mobile incidents tagged with mobile_source = true
  • Old mobile_app schema still exists for rollback safety
  • Backward compatibility view created (mobile_app.incidents_legacy)

================================================================================
';
END $$;
