-- Community Verification System (Anonymous, Privacy-Focused)
-- Created: 2025-10-02
-- Purpose: Enable anonymous incident verification without user identification

-- Add verification counts to incidents
ALTER TABLE crime_incidents
ADD COLUMN IF NOT EXISTS verification_count_up INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS verification_count_down INTEGER DEFAULT 0,
ADD COLUMN IF NOT EXISTS verification_score FLOAT DEFAULT 0.0;

-- Create verification tracking table (anonymous)
CREATE TABLE IF NOT EXISTS incident_verifications (
    id SERIAL PRIMARY KEY,
    incident_id VARCHAR NOT NULL REFERENCES crime_incidents(id) ON DELETE CASCADE,
    device_fingerprint VARCHAR(64) NOT NULL, -- Anonymous device ID (hashed)
    vote_type VARCHAR(10) NOT NULL CHECK (vote_type IN ('up', 'down')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Prevent double voting from same device
    UNIQUE(incident_id, device_fingerprint)
);

-- Create index for fast lookups
CREATE INDEX IF NOT EXISTS idx_verification_incident
ON incident_verifications(incident_id);

CREATE INDEX IF NOT EXISTS idx_verification_device
ON incident_verifications(device_fingerprint);

-- Function to calculate verification score
-- Score = (up_votes - down_votes) / (total_votes + 5)
-- The +5 prevents wild swings from first few votes
CREATE OR REPLACE FUNCTION calculate_verification_score(up_count INTEGER, down_count INTEGER)
RETURNS FLOAT AS $$
BEGIN
    RETURN CASE
        WHEN (up_count + down_count) = 0 THEN 0.0
        ELSE (up_count::FLOAT - down_count::FLOAT) / (up_count + down_count + 5)::FLOAT
    END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Trigger to update verification counts and score
CREATE OR REPLACE FUNCTION update_incident_verification_score()
RETURNS TRIGGER AS $$
BEGIN
    -- Update counts
    UPDATE crime_incidents
    SET
        verification_count_up = (
            SELECT COUNT(*) FROM incident_verifications
            WHERE incident_id = NEW.incident_id AND vote_type = 'up'
        ),
        verification_count_down = (
            SELECT COUNT(*) FROM incident_verifications
            WHERE incident_id = NEW.incident_id AND vote_type = 'down'
        )
    WHERE id = NEW.incident_id;

    -- Update score
    UPDATE crime_incidents
    SET verification_score = calculate_verification_score(verification_count_up, verification_count_down)
    WHERE id = NEW.incident_id;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_verification_score
AFTER INSERT OR UPDATE OR DELETE ON incident_verifications
FOR EACH ROW
EXECUTE FUNCTION update_incident_verification_score();

-- Comments
COMMENT ON TABLE incident_verifications IS 'Anonymous incident verification votes (no user identification)';
COMMENT ON COLUMN incident_verifications.device_fingerprint IS 'Hashed device ID to prevent double voting (anonymous)';
COMMENT ON COLUMN crime_incidents.verification_score IS 'Community trust score: -1 (unverified) to +1 (highly verified)';
