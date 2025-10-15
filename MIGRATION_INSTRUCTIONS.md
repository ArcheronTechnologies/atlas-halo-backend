# Database Migration: Railway → Fly.io

## Overview
This guide will help you migrate data from Railway Postgres to Fly.io Postgres for the Halo Backend.

## Prerequisites
- Railway account with access to `atlas-halo-backend-production`
- Fly.io CLI installed and authenticated
- Python 3.12+ with psycopg installed

## Step 1: Get Railway DATABASE_URL

### Option A: From Railway Dashboard
1. Go to Railway dashboard: https://railway.app
2. Select project: `atlas-halo-backend-production`
3. Click on the Postgres service
4. Go to "Variables" tab
5. Copy the `DATABASE_URL` value

### Option B: From Railway CLI
```bash
cd /Users/timothyaikenhead/Desktop/Halo
railway variables --service web | grep DATABASE_URL
```

## Step 2: Get Fly.io DATABASE_URL

The DATABASE_URL from the `flyctl postgres attach` command was:
```
postgres://halo_backend_solitary_smoke_5582:c2bxvoFkAUXhOii@halo-db.flycast:5432/halo_backend_solitary_smoke_5582?sslmode=disable
```

Or get it from Fly.io secrets:
```bash
flyctl secrets list --app halo-backend-solitary-smoke-5582
```

For external connection (from your local machine), use the proxy connection:
```bash
flyctl proxy 5432:5432 --app halo-db
```

Then use:
```
postgres://halo_backend_solitary_smoke_5582:c2bxvoFkAUXhOii@localhost:5432/halo_backend_solitary_smoke_5582
```

## Step 3: Run the Migration

```bash
cd /Users/timothyaikenhead/Desktop/Halo

# Set environment variables (replace with actual URLs from Step 1 & 2)
export RAILWAY_DATABASE_URL="postgresql://postgres:PASSWORD@HOST:PORT/railway"
export FLYIO_DATABASE_URL="postgres://halo_backend_solitary_smoke_5582:c2bxvoFkAUXhOii@localhost:5432/halo_backend_solitary_smoke_5582"

# Run migration
python3 migrate_railway_to_flyio.py
```

## Step 4: Verify Migration

After migration completes, verify the data:

```bash
# Check Fly.io database
flyctl postgres connect --app halo-db --database halo_backend_solitary_smoke_5582

# Inside psql:
\dt                  # List tables
SELECT COUNT(*) FROM users;
SELECT COUNT(*) FROM incidents;
\q                   # Exit
```

## Step 5: Test Halo Backend with Fly.io Database

```bash
# Test health endpoint
curl -s https://halo-backend-solitary-smoke-5582.fly.dev/health | python3 -m json.tool

# The mobile app is already configured to use Fly.io backend
# Test from mobile app after migration
```

## Expected Migration Output

```
============================================================
🚀 Railway → Fly.io Database Migration
============================================================

📡 Source (Railway): postgresql://postgres:...
📡 Destination (Fly.io): postgres://halo_backend...

🔌 Connecting to databases...
  ✅ Connected to Railway
  ✅ Connected to Fly.io

📋 Discovering tables...
  Found X tables in source
  Tables: users, incidents, incident_media, ...

🚚 Starting migration...

📋 Migrating table: users
  Source rows: 150
  Fetched 150 rows
  ✅ Migrated 150/150 rows

📋 Migrating table: incidents
  Source rows: 5234
  Fetched 5234 rows
  ✅ Migrated 5234/5234 rows

...

============================================================
📊 Migration Summary
============================================================
  Total tables: 15
  Successfully migrated: 15
  Failed: 0
  Duration: 45.23 seconds

✅ Migration complete!
```

## Rollback Plan

If you need to rollback:

1. The mobile app can be quickly reconfigured to point back to Railway:
   ```typescript
   // mobile/constants/config.ts
   PROD_BASE_URL: 'https://atlas-halo-backend-production.up.railway.app'
   ```

2. Railway database remains untouched (migration is read-only from source)

## Post-Migration

1. ✅ Mobile app already configured for Fly.io
2. ✅ Halo Backend deployed on Fly.io with psycopg3
3. ✅ Database connection pooling configured
4. Monitor Fly.io backend for 24-48 hours
5. Consider decommissioning Railway resources after verification

## Troubleshooting

### Connection timeout to Railway
- Check Railway service is running: `railway status`
- Verify DATABASE_URL is correct
- Ensure no firewall blocking connection

### Connection timeout to Fly.io
- Start proxy: `flyctl proxy 5432:5432 --app halo-db`
- Use localhost in DATABASE_URL
- Check halo-db app is running: `flyctl status --app halo-db`

### Migration script errors
- Ensure psycopg is installed: `pip install psycopg[binary]`
- Check Python version: `python3 --version` (need 3.12+)
- Review error messages for specific table/row issues

## Support

If you encounter issues:
1. Check logs: `python3 migrate_railway_to_flyio.py 2>&1 | tee migration.log`
2. Review Fly.io status: `flyctl status --app halo-backend-solitary-smoke-5582`
3. Test database connectivity independently with `psql`
