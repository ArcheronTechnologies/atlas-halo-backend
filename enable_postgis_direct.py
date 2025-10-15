#!/usr/bin/env python3
"""
Enable PostGIS on Railway PostgreSQL database
This connects directly using the connection string
"""

import subprocess
import sys

# Railway connection details
DB_HOST = "postgres.railway.internal"
DB_PORT = "5432"
DB_NAME = "railway"
DB_USER = "postgres"
DB_PASS = "BwlsEEQZfzpwMRPsUNqBADwUMydWivKy"

print("🔌 Enabling PostGIS on Railway PostgreSQL...")
print("This will run psql commands via Railway CLI...")

# Run via railway environment where postgres.railway.internal is accessible
cmd = f"""railway run bash -c 'PGPASSWORD="{DB_PASS}" psql -h postgres.railway.internal -p 5432 -U postgres -d railway -c "CREATE EXTENSION IF NOT EXISTS postgis; CREATE EXTENSION IF NOT EXISTS postgis_topology; SELECT PostGIS_version();"'"""

print(f"Running command...")
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

sys.exit(result.returncode)
