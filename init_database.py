"""
Initialize Halo Database Schema on Fly.io
Run this script to create all necessary tables
"""

import asyncio
import os
import sys
from backend.database.postgis_database import get_database, Base

async def init_db():
    """Initialize database with all tables"""
    print("🚀 Initializing Halo database...")

    # Get database connection
    db = await get_database()

    if not db.engine:
        print("❌ Database engine not initialized")
        return False

    print(f"✅ Connected to database")

    # Create all tables defined in SQLAlchemy models
    try:
        async with db.engine.begin() as conn:
            print("📋 Creating tables...")
            await conn.run_sync(Base.metadata.create_all)

        print("✅ All tables created successfully")

        # Test connection
        test_result = await db.execute_query("SELECT 1 as test")
        print(f"✅ Test query successful: {test_result}")

        # Check what tables exist
        tables_query = """
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = 'public'
        ORDER BY tablename;
        """
        tables = await db.execute_query(tables_query)
        print(f"\n📊 Created tables ({len(tables)}):")
        for table in tables:
            print(f"  - {table['tablename']}")

        return True

    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(init_db())
    sys.exit(0 if success else 1)
