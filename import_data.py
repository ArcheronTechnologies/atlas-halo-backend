#!/usr/bin/env python3
"""
Direct data import to Fly.io atlas-db via proxy
Bypasses all authentication issues by using psycopg directly
"""

import csv
import psycopg
from datetime import datetime

# Connect via proxy (no password needed for local proxy)
DB_URL = "postgresql://postgres@localhost:5434/postgres?password="

def main():
    print("🔄 Importing data to Fly.io atlas-db...")

    try:
        # Connect
        print("🔌 Connecting to database...")
        conn = psycopg.connect(DB_URL, autocommit=False)

        # Create table
        print("📝 Creating incidents table...")
        with conn.cursor() as cur:
            cur.execute("""
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
            """)
        conn.commit()
        print("✅ Table created")

        # Import CSV
        print("📦 Importing incidents from CSV...")
        with open('/tmp/incidents.csv', 'r') as f:
            reader = csv.DictReader(f)
            count = 0

            with conn.cursor() as cur:
                for row in reader:
                    try:
                        cur.execute("""
                            INSERT INTO incidents (
                                id, external_id, source, incident_type, summary,
                                location_name, latitude, longitude, occurred_at,
                                url, severity, created_at, updated_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (external_id, source) DO NOTHING
                        """, (
                            row['id'], row['external_id'], row['source'],
                            row['incident_type'], row['summary'] or None,
                            row['location_name'] or None,
                            float(row['latitude']), float(row['longitude']),
                            row['occurred_at'], row['url'] or None,
                            int(row['severity']) if row['severity'] else None,
                            row['created_at'], row['updated_at'] or None
                        ))
                        count += 1
                        if count % 100 == 0:
                            conn.commit()
                            print(f"   Imported {count} incidents...")
                    except Exception as e:
                        print(f"   ⚠️  Skipped row: {e}")
                        continue

            conn.commit()

        print(f"✅ Imported {count} total incidents")

        # Verify
        print("\n📊 Verifying data...")
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    MAX(occurred_at) as latest,
                    MIN(occurred_at) as earliest
                FROM incidents
            """)
            result = cur.fetchone()
            print(f"   Total incidents: {result[0]}")
            print(f"   Latest incident: {result[1]}")
            print(f"   Earliest incident: {result[2]}")

        conn.close()
        print("\n✅ Migration completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
