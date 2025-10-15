#!/usr/bin/env python3
"""
Quick database migration script using Python (avoids pg_dump version issues)
Migrates Atlas Intelligence data from Railway to Fly.io
"""

import psycopg
from psycopg.rows import dict_row

# Database URLs
RAILWAY_URL = "postgresql://postgres:JartxbAHvmbRYDzclcQVIcjNlJtqFkrH@caboose.proxy.rlwy.net:36478/railway"
FLYIO_URL = "postgresql://postgres@localhost:5434/postgres"

# Tables to migrate
TABLES = [
    "incidents",
    "intelligence_patterns",
    "model_registry",
    "threat_intelligence",
    "training_samples"
]

def get_table_schema(conn, table_name):
    """Get CREATE TABLE statement for a table"""
    with conn.cursor() as cur:
        # Get table definition using pg_dump-like query
        cur.execute(f"""
            SELECT
                'CREATE TABLE IF NOT EXISTS {table_name} (' ||
                string_agg(
                    column_name || ' ' ||
                    CASE
                        WHEN data_type = 'USER-DEFINED' THEN udt_name
                        WHEN character_maximum_length IS NOT NULL THEN data_type || '(' || character_maximum_length || ')'
                        ELSE data_type
                    END ||
                    CASE WHEN is_nullable = 'NO' THEN ' NOT NULL' ELSE '' END,
                    ', '
                ORDER BY ordinal_position
                ) ||
                ');'
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = '{table_name}'
            GROUP BY table_name;
        """)
        result = cur.fetchone()
        return result[0] if result else None

def migrate_table(source_conn, dest_conn, table_name):
    """Migrate a single table from source to destination"""
    print(f"\n📦 Migrating table: {table_name}")

    # Get row count
    with source_conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cur.fetchone()[0]
        print(f"   Found {count} rows")

    if count == 0:
        print(f"   ⏭️  Skipping empty table")
        return

    # Get schema and create table
    schema = get_table_schema(source_conn, table_name)
    if schema:
        print(f"   Creating table in destination...")
        with dest_conn.cursor() as cur:
            cur.execute(schema)
        dest_conn.commit()

    # Copy data in batches
    batch_size = 1000
    offset = 0

    while offset < count:
        with source_conn.cursor(row_factory=dict_row) as src_cur:
            src_cur.execute(f"SELECT * FROM {table_name} ORDER BY id LIMIT {batch_size} OFFSET {offset}")
            rows = src_cur.fetchall()

            if not rows:
                break

            # Insert into destination
            if rows:
                columns = list(rows[0].keys())
                placeholders = ', '.join(['%s'] * len(columns))
                insert_sql = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"

                with dest_conn.cursor() as dest_cur:
                    for row in rows:
                        dest_cur.execute(insert_sql, list(row.values()))

                dest_conn.commit()
                offset += len(rows)
                print(f"   Copied {offset}/{count} rows")

    print(f"   ✅ Completed {table_name}")

def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║     ATLAS INTELLIGENCE DB MIGRATION                       ║
║     Railway → Fly.io                                      ║
╚═══════════════════════════════════════════════════════════╝
""")

    try:
        print("🔌 Connecting to Railway...")
        source_conn = psycopg.connect(RAILWAY_URL)

        print("🔌 Connecting to Fly.io...")
        dest_conn = psycopg.connect(FLYIO_URL)

        print("✅ Connections established\n")

        for table in TABLES:
            try:
                migrate_table(source_conn, dest_conn, table)
            except Exception as e:
                print(f"   ❌ Error migrating {table}: {e}")
                continue

        source_conn.close()
        dest_conn.close()

        print("""
╔═══════════════════════════════════════════════════════════╗
║              ✅ MIGRATION COMPLETED!                      ║
╚═══════════════════════════════════════════════════════════╝

Next: Restart Atlas Intelligence on Fly.io
  flyctl machine restart --app atlas-intelligence
""")

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
