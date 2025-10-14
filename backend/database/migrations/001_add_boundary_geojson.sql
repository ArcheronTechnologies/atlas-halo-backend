-- Migration: Add boundary_geojson column for OSM neighborhood boundaries
-- This allows storing actual neighborhood polygon shapes from OpenStreetMap
-- while maintaining backward compatibility with existing bounds_* columns

-- Add the new column (nullable to allow gradual migration)
ALTER TABLE predictions
ADD COLUMN IF NOT EXISTS boundary_geojson JSONB;

-- Add index for GeoJSON queries
CREATE INDEX IF NOT EXISTS idx_predictions_boundary_geojson
ON predictions USING GIN (boundary_geojson);

-- Add comment explaining the column
COMMENT ON COLUMN predictions.boundary_geojson IS
'GeoJSON polygon representing actual neighborhood boundaries from OSM.
Format: {"type": "Polygon", "coordinates": [[[lon, lat], [lon, lat], ...]]}
Falls back to bounds_* if null for backward compatibility.';
