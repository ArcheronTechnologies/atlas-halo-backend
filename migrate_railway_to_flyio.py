"""
Database Migration Script: Railway → Fly.io
Migrates all data from Railway Postgres to Fly.io Postgres
"""

import asyncio
import os
import sys
from datetime import datetime
import psycopg

# Database URLs
RAILWAY_DATABASE_URL = os.getenv("RAILWAY_DATABASE_URL", "")
FLYIO_DATABASE_URL = os.getenv("FLYIO_DATABASE_URL", "")

# Tables to migrate (in dependency order)
TABLES_TO_MIGRATE = [
    "users",
    "incidents",
    "incident_media",
    "incident_comments",
    "predictions",
    "model_versions",
    "training_runs",
    # Add more tables as needed
]


async def get_table_count(conn, table_name):
    """Get row count for a table"""
    try:
        async with conn.cursor() as cur:
            await cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            result = await cur.fetchone()
            return result[0] if result else 0
    except Exception as e:
        print(f"  ⚠️  Table {table_name} doesn't exist or error: {e}")
        return 0


async def list_tables(conn):
    """List all tables in the database"""
    async with conn.cursor() as cur:
        await cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        tables = await cur.fetchall()
        return [table[0] for table in tables]


async def migrate_table(source_conn, dest_conn, table_name):
    """Migrate a single table from source to destination"""
    print(f"\n📋 Migrating table: {table_name}")

    try:
        # Get row count
        source_count = await get_table_count(source_conn, table_name)
        print(f"  Source rows: {source_count}")

        if source_count == 0:
            print(f"  ⏭️  Skipping {table_name} (empty)")
            return True

        # Get table schema
        async with source_conn.cursor() as cur:
            await cur.execute(f"""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = %s AND table_schema = 'public'
                ORDER BY ordinal_position
            """, (table_name,))
            columns_info = await cur.fetchall()

        column_names = [col[0] for col in columns_info]
        columns_str = ", ".join(column_names)
        placeholders = ", ".join(["%s"] * len(column_names))

        # Fetch all data
        async with source_conn.cursor() as cur:
            await cur.execute(f"SELECT {columns_str} FROM {table_name}")
            rows = await cur.fetchall()

        print(f"  Fetched {len(rows)} rows")

        # Insert into destination (with conflict handling)
        insert_query = f"""
            INSERT INTO {table_name} ({columns_str})
            VALUES ({placeholders})
            ON CONFLICT DO NOTHING
        """

        async with dest_conn.cursor() as cur:
            inserted = 0
            for row in rows:
                try:
                    await cur.execute(insert_query, row)
                    inserted += 1
                except Exception as e:
                    print(f"    ⚠️  Error inserting row: {e}")
                    continue

        print(f"  ✅ Migrated {inserted}/{len(rows)} rows")
        return True

    except Exception as e:
        print(f"  ❌ Error migrating {table_name}: {e}")
        return False


async def main():
    """Main migration function"""
    print("=" * 60)
    print("🚀 Railway → Fly.io Database Migration")
    print("=" * 60)

    # Validate URLs
    if not RAILWAY_DATABASE_URL:
        print("❌ RAILWAY_DATABASE_URL environment variable not set")
        print("\nUsage:")
        print("  export RAILWAY_DATABASE_URL='postgresql://...'")
        print("  export FLYIO_DATABASE_URL='postgresql://...'")
        print("  python3 migrate_railway_to_flyio.py")
        sys.exit(1)

    if not FLYIO_DATABASE_URL:
        print("❌ FLYIO_DATABASE_URL environment variable not set")
        sys.exit(1)

    print(f"\n📡 Source (Railway): {RAILWAY_DATABASE_URL[:50]}...")
    print(f"📡 Destination (Fly.io): {FLYIO_DATABASE_URL[:50]}...")

    try:
        # Connect to both databases
        print("\n🔌 Connecting to databases...")
        source_conn = await psycopg.AsyncConnection.connect(
            RAILWAY_DATABASE_URL,
            autocommit=True
        )
        print("  ✅ Connected to Railway")

        dest_conn = await psycopg.AsyncConnection.connect(
            FLYIO_DATABASE_URL,
            autocommit=True
        )
        print("  ✅ Connected to Fly.io")

        # List all tables
        print("\n📋 Discovering tables...")
        source_tables = await list_tables(source_conn)
        print(f"  Found {len(source_tables)} tables in source")
        print(f"  Tables: {', '.join(source_tables)}")

        # Migrate each table
        print("\n🚚 Starting migration...")
        start_time = datetime.now()

        success_count = 0
        for table in source_tables:
            if await migrate_table(source_conn, dest_conn, table):
                success_count += 1

        # Summary
        duration = (datetime.now() - start_time).total_seconds()
        print("\n" + "=" * 60)
        print("📊 Migration Summary")
        print("=" * 60)
        print(f"  Total tables: {len(source_tables)}")
        print(f"  Successfully migrated: {success_count}")
        print(f"  Failed: {len(source_tables) - success_count}")
        print(f"  Duration: {duration:.2f} seconds")
        print("\n✅ Migration complete!")

        # Close connections
        await source_conn.close()
        await dest_conn.close()

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
