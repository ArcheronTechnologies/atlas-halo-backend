#!/usr/bin/env python3
"""Test Halo database connection"""
import asyncio
import os

# Set env vars
os.environ['POSTGRES_HOST'] = '51.159.27.120'
os.environ['POSTGRES_PORT'] = '19168'
os.environ['POSTGRES_USER'] = 'atlas_user'
os.environ['POSTGRES_PASSWORD'] = '@UijY:[e\\8_yy5>85Z/^a'
os.environ['POSTGRES_DB'] = 'rdb'

async def test():
    from backend.database.postgis_database import get_database

    print("Testing Halo database connection...")
    db = await get_database()

    print("✅ Database initialized!")
    print(f"Engine: {db.engine}")
    print(f"Pool: {db.pool}")

    # Try a query
    if db.pool:
        async with db.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT COUNT(*) FROM incidents")
                count = (await cur.fetchone())[0]
                print(f"✅ Found {count} incidents in database!")

if __name__ == "__main__":
    asyncio.run(test())
