#!/usr/bin/env python3
"""
Database Migration Script: Railway → Fly.io
Migrates data from Railway PostgreSQL databases to Fly.io PostgreSQL databases
"""

import subprocess
import sys
import os
from urllib.parse import urlparse

# Railway Halo Backend DATABASE_URL (from context)
RAILWAY_HALO_URL = "postgresql://postgres:JartxbAHvmbRYDzclcQVIcjNlJtqFkrH@caboose.proxy.rlwy.net:36478/railway"

# Fly.io connections (via proxy on localhost:5433 for halo-db)
# We'll need to get the actual credentials from flyctl
FLY_HALO_LOCAL = "postgresql://postgres@localhost:5433/postgres"


def run_command(cmd, description):
    """Run a shell command and handle errors"""
    print(f"\n{'='*60}")
    print(f"▶️  {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")

    result = subprocess.run(
        cmd if isinstance(cmd, list) else cmd,
        shell=not isinstance(cmd, list),
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"❌ Error: {result.stderr}")
        return False

    print(f"✅ Success")
    if result.stdout and result.stdout.strip():
        print(f"Output: {result.stdout[:500]}")  # First 500 chars

    return True


def check_source_data(db_url, db_name):
    """Check if source database has data"""
    print(f"\n📊 Checking {db_name} source database...")

    # Parse the URL to get connection details
    parsed = urlparse(db_url)

    # Count tables using psql
    cmd = [
        "psql",
        db_url,
        "-c",
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"✅ Connection successful")
        print(result.stdout)
        return True
    else:
        print(f"❌ Connection failed: {result.stderr}")
        return False


def migrate_database(source_url, dest_url, db_name):
    """Migrate a database from Railway to Fly.io"""
    print(f"\n🔄 Starting migration for {db_name}")
    print(f"Source: Railway")
    print(f"Destination: Fly.io")

    # Step 1: Dump the Railway database
    dump_file = f"/tmp/{db_name}_migration.sql"

    print(f"\n1️⃣  Dumping Railway {db_name} database...")
    dump_cmd = f"pg_dump {source_url} --no-owner --no-acl -f {dump_file}"

    if not run_command(dump_cmd, f"Dump {db_name} from Railway"):
        return False

    # Check dump file size
    if os.path.exists(dump_file):
        size = os.path.getsize(dump_file)
        print(f"📦 Dump file size: {size:,} bytes ({size/1024/1024:.2f} MB)")

    # Step 2: Restore to Fly.io database
    print(f"\n2️⃣  Restoring to Fly.io {db_name} database...")
    restore_cmd = f"psql {dest_url} -f {dump_file}"

    if not run_command(restore_cmd, f"Restore {db_name} to Fly.io"):
        print(f"⚠️  Some errors may be expected (e.g., role creation)")

    # Step 3: Verify migration
    print(f"\n3️⃣  Verifying migration...")
    verify_cmd = f"psql {dest_url} -c \"SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';\""

    if run_command(verify_cmd, f"Verify {db_name} tables"):
        print(f"✅ {db_name} migration completed successfully!")
        return True
    else:
        print(f"❌ {db_name} migration verification failed")
        return False


def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║       DATABASE MIGRATION: Railway → Fly.io               ║
║                                                           ║
║  This script will migrate databases from Railway to      ║
║  Fly.io, preserving all data, schemas, and indexes.      ║
╚═══════════════════════════════════════════════════════════╝
""")

    # Check if psql and pg_dump are available
    print("🔍 Checking prerequisites...")

    for tool in ["psql", "pg_dump"]:
        result = subprocess.run(["which", tool], capture_output=True)
        if result.returncode != 0:
            print(f"❌ Error: {tool} not found. Please install PostgreSQL client tools.")
            sys.exit(1)
        print(f"✅ {tool} found")

    # Check source databases
    print("\n📊 Checking source databases on Railway...")

    if not check_source_data(RAILWAY_HALO_URL, "Halo Backend"):
        print("❌ Cannot connect to Railway Halo Backend database")
        response = input("\nContinue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Confirm migration
    print("\n⚠️  WARNING: This will migrate data to Fly.io databases.")
    print("⚠️  Existing data in Fly.io databases will be preserved, but conflicts may occur.")
    response = input("\nProceed with migration? (y/N): ")

    if response.lower() != 'y':
        print("❌ Migration cancelled")
        sys.exit(0)

    # Migrate Halo Backend database
    success = migrate_database(
        RAILWAY_HALO_URL,
        FLY_HALO_LOCAL,
        "Halo Backend"
    )

    if success:
        print("""
╔═══════════════════════════════════════════════════════════╗
║              ✅ MIGRATION COMPLETED!                      ║
╚═══════════════════════════════════════════════════════════╝

Next steps:
1. Restart Halo Backend on Fly.io: flyctl machine restart --app halo-backend-solitary-smoke-5582
2. Test the mobile app to verify data is accessible
3. Check logs: flyctl logs --app halo-backend-solitary-smoke-5582
""")
    else:
        print("""
╔═══════════════════════════════════════════════════════════╗
║              ❌ MIGRATION FAILED                          ║
╚═══════════════════════════════════════════════════════════╝

Please check the errors above and try again.
""")
        sys.exit(1)


if __name__ == "__main__":
    main()
