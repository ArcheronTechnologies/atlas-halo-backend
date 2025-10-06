-- User-Submitted Incidents Database Schema
-- Supports citizen reporting with proper validation, moderation, and privacy controls

-- Main user-submitted incidents table
CREATE TABLE IF NOT EXISTS user_submitted_incidents (
    incident_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    submission_id VARCHAR(255) UNIQUE NOT NULL, -- Public-facing ID

    -- Temporal data
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    incident_datetime TIMESTAMP WITH TIME ZONE, -- When incident occurred

    -- User information (anonymized for privacy)
    submitter_id VARCHAR(255), -- Hashed/anonymized user identifier
    submitter_type VARCHAR(50) NOT NULL DEFAULT 'citizen', -- citizen, visitor, anonymous
    submission_method VARCHAR(50) NOT NULL DEFAULT 'mobile_app', -- mobile_app, web, api

    -- Location data (with privacy controls)
    location_provided BOOLEAN NOT NULL DEFAULT FALSE,
    latitude DECIMAL(10, 8), -- Stored with reduced precision for privacy
    longitude DECIMAL(11, 8), -- Stored with reduced precision for privacy
    location_description TEXT, -- User-provided description
    municipality VARCHAR(255), -- Derived from coordinates
    region VARCHAR(255), -- Derived from coordinates

    -- Incident classification
    incident_type VARCHAR(100) NOT NULL,
    incident_category VARCHAR(100) NOT NULL,
    severity_reported INTEGER CHECK (severity_reported BETWEEN 1 AND 10),
    urgency_level VARCHAR(20) DEFAULT 'normal', -- low, normal, high, urgent

    -- Incident details
    title VARCHAR(500) NOT NULL,
    description TEXT NOT NULL,
    evidence_provided BOOLEAN NOT NULL DEFAULT FALSE,
    witness_count INTEGER DEFAULT 0,

    -- Content moderation
    moderation_status VARCHAR(50) NOT NULL DEFAULT 'pending', -- pending, approved, rejected, flagged
    moderation_reason TEXT,
    moderated_at TIMESTAMP WITH TIME ZONE,
    moderated_by VARCHAR(255),

    -- Data quality and verification
    verification_status VARCHAR(50) NOT NULL DEFAULT 'unverified', -- unverified, verified, disputed, false
    verification_score DECIMAL(5, 4) DEFAULT 0.0000, -- AI-generated confidence score
    quality_score DECIMAL(5, 4) DEFAULT 0.0000, -- Data quality assessment
    duplicate_check_status VARCHAR(50) DEFAULT 'unchecked', -- unchecked, unique, potential_duplicate, duplicate

    -- Privacy and consent
    consent_data_processing BOOLEAN NOT NULL DEFAULT FALSE,
    consent_law_enforcement BOOLEAN NOT NULL DEFAULT FALSE,
    consent_research BOOLEAN NOT NULL DEFAULT FALSE,
    anonymization_level VARCHAR(50) NOT NULL DEFAULT 'standard', -- minimal, standard, high, full

    -- Training data flags
    approved_for_training BOOLEAN NOT NULL DEFAULT FALSE,
    training_exclusion_reason TEXT,
    training_weight DECIMAL(5, 4) DEFAULT 1.0000, -- Weight for AI training

    -- System metadata
    source_ip_hash VARCHAR(64), -- Hashed IP for abuse prevention
    user_agent_hash VARCHAR(64), -- Hashed user agent
    session_id VARCHAR(255),
    api_version VARCHAR(20),

    -- Audit trail
    created_by VARCHAR(255) NOT NULL DEFAULT 'system',
    updated_by VARCHAR(255) NOT NULL DEFAULT 'system',

    -- Constraints
    CONSTRAINT valid_coordinates CHECK (
        (latitude IS NULL AND longitude IS NULL) OR
        (latitude IS NOT NULL AND longitude IS NOT NULL AND
         latitude BETWEEN -90 AND 90 AND longitude BETWEEN -180 AND 180)
    ),
    CONSTRAINT valid_moderation_status CHECK (
        moderation_status IN ('pending', 'approved', 'rejected', 'flagged', 'under_review')
    ),
    CONSTRAINT valid_verification_status CHECK (
        verification_status IN ('unverified', 'verified', 'disputed', 'false', 'investigating')
    )
);

-- Evidence attachments table
CREATE TABLE IF NOT EXISTS user_incident_evidence (
    evidence_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    incident_id UUID NOT NULL REFERENCES user_submitted_incidents(incident_id) ON DELETE CASCADE,

    -- Evidence metadata
    evidence_type VARCHAR(50) NOT NULL, -- photo, video, audio, document, other
    file_name VARCHAR(500),
    file_size_bytes BIGINT,
    mime_type VARCHAR(255),
    file_hash VARCHAR(64) UNIQUE, -- For deduplication

    -- Storage and access
    storage_path TEXT, -- Encrypted storage path
    access_url TEXT, -- Temporary access URL
    access_expires_at TIMESTAMP WITH TIME ZONE,

    -- Content analysis
    content_analysis JSONB, -- AI analysis results
    moderation_status VARCHAR(50) NOT NULL DEFAULT 'pending',
    contains_pii BOOLEAN DEFAULT NULL, -- Detected personally identifiable info
    contains_sensitive BOOLEAN DEFAULT NULL, -- Sensitive content detection

    -- Privacy controls
    anonymized BOOLEAN NOT NULL DEFAULT FALSE,
    blur_faces BOOLEAN NOT NULL DEFAULT TRUE,
    blur_plates BOOLEAN NOT NULL DEFAULT TRUE,
    redact_text BOOLEAN NOT NULL DEFAULT TRUE,

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE,

    CONSTRAINT valid_evidence_moderation CHECK (
        moderation_status IN ('pending', 'approved', 'rejected', 'processing', 'error')
    )
);

-- Incident tags for categorization and search
CREATE TABLE IF NOT EXISTS user_incident_tags (
    tag_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    incident_id UUID NOT NULL REFERENCES user_submitted_incidents(incident_id) ON DELETE CASCADE,
    tag_name VARCHAR(100) NOT NULL,
    tag_category VARCHAR(50), -- user_provided, system_generated, ai_detected
    confidence_score DECIMAL(5, 4) DEFAULT 1.0000,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    UNIQUE(incident_id, tag_name)
);

-- Incident follow-up and updates
CREATE TABLE IF NOT EXISTS user_incident_updates (
    update_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    incident_id UUID NOT NULL REFERENCES user_submitted_incidents(incident_id) ON DELETE CASCADE,

    -- Update details
    update_type VARCHAR(50) NOT NULL, -- status_change, additional_info, correction, official_response
    update_text TEXT NOT NULL,
    update_source VARCHAR(50) NOT NULL, -- submitter, moderator, law_enforcement, system

    -- Privacy and visibility
    public_visible BOOLEAN NOT NULL DEFAULT FALSE,
    submitter_visible BOOLEAN NOT NULL DEFAULT TRUE,

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_by VARCHAR(255) NOT NULL
);

-- Moderation queue and workflow
CREATE TABLE IF NOT EXISTS incident_moderation_queue (
    queue_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    incident_id UUID NOT NULL REFERENCES user_submitted_incidents(incident_id) ON DELETE CASCADE,

    -- Queue management
    priority_level INTEGER NOT NULL DEFAULT 5, -- 1 (highest) to 10 (lowest)
    assigned_to VARCHAR(255),
    queue_status VARCHAR(50) NOT NULL DEFAULT 'pending', -- pending, in_review, completed, escalated

    -- Moderation flags
    auto_flagged BOOLEAN NOT NULL DEFAULT FALSE,
    flag_reasons TEXT[],
    requires_human_review BOOLEAN NOT NULL DEFAULT TRUE,

    -- Workflow tracking
    queued_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    estimated_review_time INTEGER, -- minutes

    -- Decision tracking
    decision VARCHAR(50), -- approve, reject, request_more_info, escalate
    decision_reason TEXT,
    decision_confidence DECIMAL(5, 4),

    UNIQUE(incident_id)
);

-- Training data preparation tracking
CREATE TABLE IF NOT EXISTS incident_training_data (
    training_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    incident_id UUID NOT NULL REFERENCES user_submitted_incidents(incident_id) ON DELETE CASCADE,

    -- Training data preparation
    preprocessed BOOLEAN NOT NULL DEFAULT FALSE,
    preprocessing_version VARCHAR(50),
    feature_vector JSONB, -- Extracted features for AI training

    -- Data quality metrics
    completeness_score DECIMAL(5, 4),
    accuracy_score DECIMAL(5, 4),
    relevance_score DECIMAL(5, 4),

    -- Training usage
    included_in_training_sets TEXT[], -- Array of training set IDs
    training_weight DECIMAL(5, 4) DEFAULT 1.0000,
    last_used_for_training TIMESTAMP WITH TIME ZONE,

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_user_incidents_created_at ON user_submitted_incidents(created_at);
CREATE INDEX IF NOT EXISTS idx_user_incidents_municipality ON user_submitted_incidents(municipality);
CREATE INDEX IF NOT EXISTS idx_user_incidents_incident_type ON user_submitted_incidents(incident_type);
CREATE INDEX IF NOT EXISTS idx_user_incidents_moderation_status ON user_submitted_incidents(moderation_status);
CREATE INDEX IF NOT EXISTS idx_user_incidents_verification_status ON user_submitted_incidents(verification_status);
CREATE INDEX IF NOT EXISTS idx_user_incidents_training_approved ON user_submitted_incidents(approved_for_training);
CREATE INDEX IF NOT EXISTS idx_user_incidents_location ON user_submitted_incidents(latitude, longitude) WHERE latitude IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_evidence_incident_id ON user_incident_evidence(incident_id);
CREATE INDEX IF NOT EXISTS idx_evidence_moderation_status ON user_incident_evidence(moderation_status);
CREATE INDEX IF NOT EXISTS idx_evidence_file_hash ON user_incident_evidence(file_hash);

CREATE INDEX IF NOT EXISTS idx_tags_incident_id ON user_incident_tags(incident_id);
CREATE INDEX IF NOT EXISTS idx_tags_name ON user_incident_tags(tag_name);

CREATE INDEX IF NOT EXISTS idx_moderation_queue_status ON incident_moderation_queue(queue_status);
CREATE INDEX IF NOT EXISTS idx_moderation_queue_priority ON incident_moderation_queue(priority_level);
CREATE INDEX IF NOT EXISTS idx_moderation_queue_assigned ON incident_moderation_queue(assigned_to);

CREATE INDEX IF NOT EXISTS idx_training_data_incident_id ON incident_training_data(incident_id);
CREATE INDEX IF NOT EXISTS idx_training_data_preprocessed ON incident_training_data(preprocessed);

-- Views for common queries

-- View for approved incidents ready for training
CREATE OR REPLACE VIEW training_ready_incidents AS
SELECT
    i.*,
    t.feature_vector,
    t.completeness_score,
    t.accuracy_score,
    t.relevance_score,
    t.training_weight
FROM user_submitted_incidents i
JOIN incident_training_data t ON i.incident_id = t.incident_id
WHERE i.approved_for_training = TRUE
    AND i.moderation_status = 'approved'
    AND i.verification_status IN ('verified', 'unverified')
    AND t.preprocessed = TRUE;

-- View for moderation dashboard
CREATE OR REPLACE VIEW moderation_dashboard AS
SELECT
    i.incident_id,
    i.submission_id,
    i.created_at,
    i.incident_type,
    i.severity_reported,
    i.title,
    i.moderation_status,
    i.verification_status,
    q.priority_level,
    q.assigned_to,
    q.queue_status,
    q.auto_flagged,
    q.flag_reasons,
    (SELECT COUNT(*) FROM user_incident_evidence e WHERE e.incident_id = i.incident_id) as evidence_count
FROM user_submitted_incidents i
LEFT JOIN incident_moderation_queue q ON i.incident_id = q.incident_id
WHERE i.moderation_status IN ('pending', 'under_review', 'flagged');

-- Update trigger to maintain updated_at timestamp
CREATE OR REPLACE FUNCTION update_user_incident_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    NEW.updated_by = COALESCE(NEW.updated_by, OLD.updated_by, 'system');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER user_incidents_update_timestamp
    BEFORE UPDATE ON user_submitted_incidents
    FOR EACH ROW
    EXECUTE FUNCTION update_user_incident_timestamp();

-- Function to automatically queue incidents for moderation
CREATE OR REPLACE FUNCTION queue_incident_for_moderation()
RETURNS TRIGGER AS $$
BEGIN
    -- Determine priority based on incident characteristics
    DECLARE
        priority INTEGER := 5; -- Default priority
        auto_flagged BOOLEAN := FALSE;
        flag_reasons TEXT[] := ARRAY[]::TEXT[];
    BEGIN
        -- High priority for urgent incidents
        IF NEW.urgency_level = 'urgent' THEN
            priority := 1;
        ELSIF NEW.urgency_level = 'high' THEN
            priority := 2;
        ELSIF NEW.severity_reported >= 8 THEN
            priority := 2;
        ELSIF NEW.severity_reported >= 6 THEN
            priority := 3;
        END IF;

        -- Auto-flag based on content analysis
        IF LENGTH(NEW.description) < 50 THEN
            auto_flagged := TRUE;
            flag_reasons := array_append(flag_reasons, 'insufficient_description');
        END IF;

        IF NEW.incident_type = 'test' OR LOWER(NEW.title) LIKE '%test%' THEN
            auto_flagged := TRUE;
            flag_reasons := array_append(flag_reasons, 'potential_test_submission');
        END IF;

        -- Insert into moderation queue
        INSERT INTO incident_moderation_queue (
            incident_id,
            priority_level,
            auto_flagged,
            flag_reasons,
            requires_human_review
        ) VALUES (
            NEW.incident_id,
            priority,
            auto_flagged,
            flag_reasons,
            TRUE
        );

        RETURN NEW;
    END;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER auto_queue_moderation
    AFTER INSERT ON user_submitted_incidents
    FOR EACH ROW
    EXECUTE FUNCTION queue_incident_for_moderation();

-- Function to initialize training data record
CREATE OR REPLACE FUNCTION initialize_training_data()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO incident_training_data (
        incident_id,
        completeness_score,
        accuracy_score,
        relevance_score
    ) VALUES (
        NEW.incident_id,
        CASE
            WHEN NEW.location_provided AND LENGTH(NEW.description) > 100 THEN 0.8
            WHEN NEW.location_provided OR LENGTH(NEW.description) > 100 THEN 0.6
            ELSE 0.4
        END,
        0.5, -- Default, will be updated by AI analysis
        CASE
            WHEN NEW.incident_type IN ('crime', 'emergency', 'safety_concern') THEN 0.9
            WHEN NEW.incident_type IN ('noise_complaint', 'maintenance') THEN 0.6
            ELSE 0.7
        END
    );

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER auto_initialize_training_data
    AFTER INSERT ON user_submitted_incidents
    FOR EACH ROW
    EXECUTE FUNCTION initialize_training_data();

-- Grant appropriate permissions (adjust as needed for your user roles)
-- GRANT SELECT, INSERT, UPDATE ON user_submitted_incidents TO app_user;
-- GRANT SELECT, INSERT, UPDATE ON user_incident_evidence TO app_user;
-- GRANT SELECT, INSERT ON user_incident_tags TO app_user;
-- GRANT SELECT, INSERT ON user_incident_updates TO app_user;
-- GRANT SELECT ON training_ready_incidents TO ai_training_user;
-- GRANT SELECT, UPDATE ON incident_moderation_queue TO moderator_user;