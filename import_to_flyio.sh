#!/bin/bash
# Quick database migration script for Fly.io atlas-db
# Imports CSV data from Railway export

set -e

echo "🔄 Migrating Atlas Intelligence Database to Fly.io"
echo "=================================================="

# Get DATABASE_URL from Fly.io app
echo "📡 Getting DATABASE_URL from atlas-intelligence..."
DB_URL=$(flyctl ssh console --app atlas-intelligence --command "echo \$DATABASE_URL" | tail -1)

if [ -z "$DB_URL" ] || [ "$DB_URL" = "\$DATABASE_URL" ]; then
    echo "❌ Failed to get DATABASE_URL"
    echo "Using proxy connection instead..."
    DB_URL="postgresql://postgres@localhost:5434/postgres"
fi

echo "✅ Database URL obtained"

# Create tables using Railway schema
echo ""
echo "1️⃣  Creating incidents table..."
psql "$DB_URL" <<'SQL'
CREATE TABLE IF NOT EXISTS incidents (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    external_id VARCHAR(200) NOT NULL,
    source VARCHAR(50) NOT NULL,
    incident_type VARCHAR(100) NOT NULL,
    summary TEXT,
    location_name VARCHAR(200),
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    occurred_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    url TEXT,
    severity INTEGER CHECK (severity >= 1 AND severity <= 5),
    created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITHOUT TIME ZONE,
    CONSTRAINT uq_incident_external_id_source UNIQUE (external_id, source)
);

CREATE INDEX IF NOT EXISTS idx_incident_occurred_at ON incidents(occurred_at);
CREATE INDEX IF NOT EXISTS idx_incident_severity ON incidents(severity);
CREATE INDEX IF NOT EXISTS idx_incident_source ON incidents(source);
SQL

echo "✅ Incidents table created"

# Import CSV data
echo ""
echo "2️⃣  Importing 595 incidents from Railway..."
psql "$DB_URL" <<'SQL'
\copy incidents FROM '/tmp/incidents.csv' WITH (FORMAT CSV, HEADER);
SQL

echo "✅ Data imported"

# Verify
echo ""
echo "3️⃣  Verifying migration..."
psql "$DB_URL" -c "SELECT COUNT(*) as total_incidents, MAX(occurred_at) as latest_incident FROM incidents;"

echo ""
echo "✅ Migration completed successfully!"
echo "=================================================="
SQL
