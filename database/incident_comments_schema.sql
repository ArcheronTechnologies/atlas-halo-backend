-- Incident Comments System
-- Allows users to discuss and provide additional information on incidents

CREATE TABLE IF NOT EXISTS incident_comments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    incident_id VARCHAR NOT NULL REFERENCES crime_incidents(id) ON DELETE CASCADE,
    user_id VARCHAR REFERENCES users(id) ON DELETE SET NULL,
    comment_text TEXT NOT NULL,
    is_verified BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    upvotes INTEGER DEFAULT 0,
    downvotes INTEGER DEFAULT 0,
    is_deleted BOOLEAN DEFAULT false
);

-- Index for fast incident lookups
CREATE INDEX IF NOT EXISTS idx_incident_comments_incident_id ON incident_comments(incident_id);

-- Index for user lookups
CREATE INDEX IF NOT EXISTS idx_incident_comments_user_id ON incident_comments(user_id);

-- Index for sorting by creation date
CREATE INDEX IF NOT EXISTS idx_incident_comments_created_at ON incident_comments(created_at DESC);

-- Updated timestamp trigger
CREATE OR REPLACE FUNCTION update_incident_comment_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER incident_comment_update_timestamp
BEFORE UPDATE ON incident_comments
FOR EACH ROW
EXECUTE FUNCTION update_incident_comment_timestamp();
