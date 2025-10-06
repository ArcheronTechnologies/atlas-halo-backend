-- AI Learning Database Schema
-- Stores AI predictions and user feedback for continuous learning

-- Table: AI Predictions
-- Stores every prediction the AI makes
CREATE TABLE IF NOT EXISTS ai_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    incident_id VARCHAR(255),  -- References crime_incidents(id) but as VARCHAR
    media_url TEXT NOT NULL,
    media_type VARCHAR(20) NOT NULL CHECK (media_type IN ('photo', 'video', 'audio')),

    -- AI Prediction
    predicted_category VARCHAR(50) NOT NULL,
    predicted_subcategory VARCHAR(50),
    confidence_score FLOAT NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
    full_analysis JSONB,

    -- User Feedback (ground truth)
    user_category VARCHAR(50),
    user_subcategory VARCHAR(50),
    was_correct BOOLEAN,

    -- Timestamps
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    feedback_received_at TIMESTAMP,

    -- Indexes
    CONSTRAINT valid_prediction CHECK (
        predicted_category IS NOT NULL AND
        confidence_score IS NOT NULL
    )
);

CREATE INDEX IF NOT EXISTS idx_ai_predictions_incident ON ai_predictions(incident_id);
CREATE INDEX IF NOT EXISTS idx_ai_predictions_category ON ai_predictions(predicted_category);
CREATE INDEX IF NOT EXISTS idx_ai_predictions_feedback ON ai_predictions(feedback_received_at) WHERE feedback_received_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ai_predictions_correctness ON ai_predictions(was_correct) WHERE was_correct IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ai_predictions_created ON ai_predictions(created_at DESC);


-- Table: Training Feedback
-- Simplified table for quick access to training data
CREATE TABLE IF NOT EXISTS training_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    incident_id VARCHAR(255) NOT NULL,
    prediction_id UUID REFERENCES ai_predictions(id) ON DELETE CASCADE,

    -- Ground truth
    actual_category VARCHAR(50) NOT NULL,
    actual_subcategory VARCHAR(50),

    -- Was AI correct?
    was_ai_correct BOOLEAN,

    -- For weighting samples
    user_confidence VARCHAR(20), -- 'high', 'medium', 'low'

    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_training_feedback_category ON training_feedback(actual_category);
CREATE INDEX IF NOT EXISTS idx_training_feedback_created ON training_feedback(created_at DESC);


-- Table: Model Training Runs
-- Track when models are retrained and their performance
CREATE TABLE IF NOT EXISTS model_training_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_type VARCHAR(50) NOT NULL, -- 'classifier', 'object_detector', etc.

    -- Training data
    training_samples_count INT NOT NULL,
    training_started_at TIMESTAMP NOT NULL DEFAULT NOW(),
    training_completed_at TIMESTAMP,

    -- Performance metrics
    accuracy FLOAT,
    precision FLOAT,
    recall FLOAT,
    f1_score FLOAT,

    -- Model artifact
    model_path TEXT,
    model_size_mb FLOAT,

    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'running', -- 'running', 'completed', 'failed'
    error_message TEXT,

    -- Configuration
    hyperparameters JSONB,

    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_model_training_runs_type ON model_training_runs(model_type);
CREATE INDEX IF NOT EXISTS idx_model_training_runs_status ON model_training_runs(status);
CREATE INDEX IF NOT EXISTS idx_model_training_runs_created ON model_training_runs(created_at DESC);


-- Table: User Corrections
-- Track when users correct AI suggestions (for UX analytics)
CREATE TABLE IF NOT EXISTS user_corrections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255), -- If we have user tracking
    incident_id VARCHAR(255) NOT NULL,
    prediction_id UUID REFERENCES ai_predictions(id) ON DELETE CASCADE,

    -- What was corrected
    field_corrected VARCHAR(50) NOT NULL, -- 'category', 'subcategory', 'location', etc.
    ai_value TEXT,
    user_value TEXT NOT NULL,

    -- Time to correct (UX metric)
    correction_time_seconds INT,

    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_corrections_field ON user_corrections(field_corrected);
CREATE INDEX IF NOT EXISTS idx_user_corrections_created ON user_corrections(created_at DESC);


-- View: AI Performance Summary
-- Quick overview of AI performance
CREATE OR REPLACE VIEW ai_performance_summary AS
SELECT
    COUNT(*) as total_predictions,
    SUM(CASE WHEN was_correct = TRUE THEN 1 ELSE 0 END) as correct_predictions,
    SUM(CASE WHEN was_correct = FALSE THEN 1 ELSE 0 END) as incorrect_predictions,
    SUM(CASE WHEN user_category IS NULL THEN 1 ELSE 0 END) as pending_feedback,
    AVG(CASE WHEN was_correct THEN 1.0 ELSE 0.0 END) as accuracy,
    AVG(confidence_score) as avg_confidence,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY confidence_score) as median_confidence
FROM ai_predictions
WHERE created_at > NOW() - INTERVAL '30 days';


-- View: Category Accuracy
-- Accuracy breakdown by category
CREATE OR REPLACE VIEW ai_category_accuracy AS
SELECT
    predicted_category,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN was_correct = TRUE THEN 1 ELSE 0 END) as correct,
    SUM(CASE WHEN was_correct = FALSE THEN 1 ELSE 0 END) as incorrect,
    AVG(CASE WHEN was_correct THEN 1.0 ELSE 0.0 END) as accuracy,
    AVG(confidence_score) as avg_confidence
FROM ai_predictions
WHERE user_category IS NOT NULL
  AND created_at > NOW() - INTERVAL '30 days'
GROUP BY predicted_category
ORDER BY total_predictions DESC;


-- View: Recent Training Data
-- Latest feedback for model retraining
CREATE OR REPLACE VIEW recent_training_data AS
SELECT
    ap.id,
    ap.incident_id,
    ap.media_url,
    ap.media_type,
    ap.predicted_category,
    ap.user_category as actual_category,
    ap.predicted_subcategory,
    ap.user_subcategory as actual_subcategory,
    ap.confidence_score,
    ap.was_correct,
    ap.full_analysis,
    ap.created_at
FROM ai_predictions ap
WHERE ap.user_category IS NOT NULL
ORDER BY ap.created_at DESC
LIMIT 1000;


-- Function: Get Training Dataset
-- Efficient function to fetch training data
CREATE OR REPLACE FUNCTION get_training_dataset(
    min_samples INT DEFAULT 100,
    max_samples INT DEFAULT 1000,
    category_filter TEXT DEFAULT NULL
)
RETURNS TABLE (
    id UUID,
    media_url TEXT,
    media_type VARCHAR(20),
    actual_category VARCHAR(50),
    actual_subcategory VARCHAR(50),
    full_analysis JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ap.id,
        ap.media_url,
        ap.media_type,
        ap.user_category as actual_category,
        ap.user_subcategory as actual_subcategory,
        ap.full_analysis
    FROM ai_predictions ap
    WHERE ap.user_category IS NOT NULL
      AND (category_filter IS NULL OR ap.user_category = category_filter)
    ORDER BY ap.created_at DESC
    LIMIT max_samples;
END;
$$ LANGUAGE plpgsql;


-- Function: Calculate Model Performance
-- Calculate accuracy metrics for a date range
CREATE OR REPLACE FUNCTION calculate_model_performance(
    start_date TIMESTAMP DEFAULT NOW() - INTERVAL '7 days',
    end_date TIMESTAMP DEFAULT NOW()
)
RETURNS TABLE (
    total_predictions BIGINT,
    correct_predictions BIGINT,
    accuracy NUMERIC,
    avg_confidence NUMERIC,
    predictions_per_day NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*) as total_predictions,
        SUM(CASE WHEN was_correct = TRUE THEN 1 ELSE 0 END) as correct_predictions,
        ROUND(AVG(CASE WHEN was_correct THEN 1.0 ELSE 0.0 END)::NUMERIC, 4) as accuracy,
        ROUND(AVG(confidence_score)::NUMERIC, 4) as avg_confidence,
        ROUND((COUNT(*) / GREATEST(EXTRACT(EPOCH FROM (end_date - start_date)) / 86400, 1))::NUMERIC, 2) as predictions_per_day
    FROM ai_predictions
    WHERE created_at BETWEEN start_date AND end_date
      AND user_category IS NOT NULL;
END;
$$ LANGUAGE plpgsql;


-- Comments
COMMENT ON TABLE ai_predictions IS 'Stores all AI predictions with user feedback for continuous learning';
COMMENT ON TABLE training_feedback IS 'Simplified training data for model retraining';
COMMENT ON TABLE model_training_runs IS 'Tracks model training runs and performance over time';
COMMENT ON TABLE user_corrections IS 'Tracks when users correct AI suggestions for UX analytics';
