#!/bin/sh
set -e

echo "🚀 Starting Halo Backend..."
echo "DATABASE_URL: ${DATABASE_URL:0:50}..."

# Test database connectivity before starting
echo "📡 Testing database connection..."
python3 -c "
import os
import psycopg
try:
    conn = psycopg.connect(os.environ['DATABASE_URL'], connect_timeout=5)
    conn.close()
    print('✅ Database connection successful')
except Exception as e:
    print(f'⚠️  Database connection failed: {e}')
    print('⚠️  Starting anyway (database might not be ready yet)')
"

echo "🎯 Starting uvicorn on 0.0.0.0:8000..."
exec uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info
