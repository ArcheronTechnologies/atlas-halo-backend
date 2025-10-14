-- Pre-computed predictions table for scalable country-wide coverage
-- This table stores predictions for all of Sweden on a fixed grid
-- Predictions are regenerated every hour by a background worker

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,

    -- Grid cell identification
    grid_lat NUMERIC(10, 6) NOT NULL,  -- Rounded latitude (0.03° grid)
    grid_lon NUMERIC(10, 6) NOT NULL,  -- Rounded longitude (0.03° grid)

    -- Center point (weighted by severity)
    latitude NUMERIC(10, 6) NOT NULL,
    longitude NUMERIC(10, 6) NOT NULL,

    -- Geographic bounds for polygon rendering
    bounds_north NUMERIC(10, 6) NOT NULL,
    bounds_south NUMERIC(10, 6) NOT NULL,
    bounds_east NUMERIC(10, 6) NOT NULL,
    bounds_west NUMERIC(10, 6) NOT NULL,

    -- Neighborhood information
    neighborhood_name VARCHAR(255) NOT NULL,

    -- Prediction metadata (stored as JSON for 24 hours ahead)
    -- Structure: {"0": {"risk": 0.5, "confidence": 0.7, "incidents": 5}, "1": {...}, ...}
    hourly_predictions JSONB NOT NULL,

    -- Static metadata
    historical_count INTEGER NOT NULL,
    incident_types TEXT[] NOT NULL,
    avg_severity NUMERIC(3, 1) NOT NULL,

    -- Timestamps
    computed_at TIMESTAMP WITH TIME ZONE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,

    -- Constraints
    UNIQUE(grid_lat, grid_lon, computed_at)
);

-- Geospatial index for fast bounding box queries
CREATE INDEX IF NOT EXISTS idx_predictions_location
ON predictions USING GIST (
    ST_MakeEnvelope(bounds_west, bounds_south, bounds_east, bounds_north, 4326)
);

-- Index for expiration cleanup
CREATE INDEX IF NOT EXISTS idx_predictions_expires ON predictions(expires_at);

-- Index for grid lookups
CREATE INDEX IF NOT EXISTS idx_predictions_grid ON predictions(grid_lat, grid_lon);

-- Index for computed_at to get latest predictions
CREATE INDEX IF NOT EXISTS idx_predictions_computed ON predictions(computed_at DESC);
