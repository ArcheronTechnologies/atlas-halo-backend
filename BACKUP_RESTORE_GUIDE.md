# Database Backup & Restore Guide

**Created:** October 18, 2025
**Status:** Automated backups configured

---

## Backup Schedule

**Automated Daily Backups:**
- **Time:** 2:00 AM UTC daily
- **Location:** Scaleway Object Storage (s3://halo-backups/)
- **Retention:** 30 days (automatic cleanup)
- **Format:** PostgreSQL custom format (.dump)

---

## List Available Backups

### Using AWS CLI

```bash
# Configure AWS CLI for Scaleway (one-time setup)
export AWS_ACCESS_KEY_ID="YOUR_SCW_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="YOUR_SCW_SECRET_KEY"

# List all backups
aws s3 ls s3://halo-backups/ --endpoint-url https://s3.fr-par.scw.cloud

# Example output:
# 2025-10-18 02:00:00   12345678 20251018.dump
# 2025-10-19 02:00:00   12345679 20251019.dump
```

### Via Scaleway Console

1. Go to https://console.scaleway.com/object-storage/buckets
2. Select region: fr-par
3. Find bucket: halo-backups
4. Browse files

---

## Download a Backup

### Latest Backup

```bash
# Find latest backup
LATEST=$(aws s3 ls s3://halo-backups/ --endpoint-url https://s3.fr-par.scw.cloud | tail -1 | awk '{print $4}')

# Download it
aws s3 cp s3://halo-backups/$LATEST ./halo_backup_latest.dump \
  --endpoint-url https://s3.fr-par.scw.cloud

echo "Downloaded: $LATEST"
```

### Specific Date

```bash
# Download backup from specific date (YYYYMMDD format)
DATE="20251018"

aws s3 cp s3://halo-backups/${DATE}.dump ./halo_backup_${DATE}.dump \
  --endpoint-url https://s3.fr-par.scw.cloud
```

---

## Restore Database

### ⚠️ WARNING: Restoring will overwrite existing data!

**ALWAYS test restore on a local database first!**

### Test Restore (Local Database - SAFE)

```bash
# 1. Download backup
aws s3 cp s3://halo-backups/20251018.dump ./restore_test.dump \
  --endpoint-url https://s3.fr-par.scw.cloud

# 2. Create local test database
createdb halo_test

# 3. Restore to test database
pg_restore \
  -h localhost \
  -p 5432 \
  -U postgres \
  -d halo_test \
  --clean \
  --if-exists \
  restore_test.dump

# 4. Verify data
psql -h localhost -U postgres -d halo_test -c "SELECT COUNT(*) FROM incidents;"
psql -h localhost -U postgres -d halo_test -c "SELECT COUNT(*) FROM predictions;"

# 5. If looks good, proceed to production restore (carefully!)
```

### Production Restore (DANGEROUS - Use with caution!)

```bash
# 1. Download backup
aws s3 cp s3://halo-backups/20251018.dump ./production_restore.dump \
  --endpoint-url https://s3.fr-par.scw.cloud

# 2. STOP the backend to prevent new writes
scw container container update 35a73370-0199-42de-862c-88b67af1890d min-scale=0 max-scale=0

# 3. Wait 1 minute for connections to close
sleep 60

# 4. Restore to production database
PGPASSWORD='@UijY:[e\8_yy5>85Z/^a' pg_restore \
  -h 51.159.27.120 \
  -p 19168 \
  -U atlas_user \
  -d rdb \
  --clean \
  --if-exists \
  production_restore.dump

# 5. Verify restoration
PGPASSWORD='@UijY:[e\8_yy5>85Z/^a' psql \
  -h 51.159.27.120 \
  -p 19168 \
  -U atlas_user \
  -d rdb \
  -c "SELECT COUNT(*) FROM incidents;"

# 6. Restart backend
scw container container update 35a73370-0199-42de-862c-88b67af1890d min-scale=1 max-scale=3
scw container container deploy 35a73370-0199-42de-862c-88b67af1890d

# 7. Test backend health
curl https://halobackend4k1irws6-halo-backend.functions.fnc.fr-par.scw.cloud/health
```

---

## Manual Backup (Ad-Hoc)

### Create Immediate Backup

```bash
# Create backup with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

PGPASSWORD='@UijY:[e\8_yy5>85Z/^a' pg_dump \
  -h 51.159.27.120 \
  -p 19168 \
  -U atlas_user \
  -d rdb \
  -F c \
  -f halo_backup_${TIMESTAMP}.dump

# Upload to S3
aws s3 cp halo_backup_${TIMESTAMP}.dump \
  s3://halo-backups/manual/${TIMESTAMP}.dump \
  --endpoint-url https://s3.fr-par.scw.cloud

echo "✅ Manual backup created: manual/${TIMESTAMP}.dump"
```

### Trigger Automated Backup (GitHub Actions)

```bash
# Via GitHub CLI (if installed)
gh workflow run database-backup.yml

# Or via web interface:
# Go to: https://github.com/ArcheronTechnologies/atlas-halo-backend/actions/workflows/database-backup.yml
# Click: "Run workflow" → Run workflow
```

---

## Verify Backup Integrity

### Check Backup File

```bash
# Download backup
aws s3 cp s3://halo-backups/20251018.dump ./verify.dump \
  --endpoint-url https://s3.fr-par.scw.cloud

# List contents without restoring
pg_restore --list verify.dump | head -20

# Should show:
# - Database: rdb
# - Tables: incidents, predictions, crime_clusters, etc.
# - Indexes
# - Constraints
```

### Test Restore (Recommended Weekly)

1. Download latest backup
2. Restore to local test database
3. Run test queries to verify data integrity
4. Document test results

---

## Backup Monitoring

### Check Last Backup

```bash
# Get most recent backup
aws s3 ls s3://halo-backups/ --endpoint-url https://s3.fr-par.scw.cloud | tail -1

# Should show today's date if running daily
```

### GitHub Actions Status

1. Go to https://github.com/ArcheronTechnologies/atlas-halo-backend/actions
2. Check "Daily Database Backup" workflow
3. Verify last run was successful
4. Check run logs for errors

### Set Up Alerts (Optional)

**UptimeRobot for GitHub Actions:**
- Monitor workflow status via GitHub API
- Get notified if backup fails

**Email Alerts:**
- GitHub Actions sends email on workflow failure (if enabled in settings)

---

## Disaster Recovery Plan

### Scenario 1: Database Corruption

1. Identify corruption (errors in queries, missing data)
2. Stop backend immediately
3. Download most recent good backup
4. Restore to production (follow production restore steps)
5. Restart backend
6. Verify data integrity
7. Investigate root cause

**Recovery Time Objective (RTO):** 15-30 minutes
**Recovery Point Objective (RPO):** 24 hours (last nightly backup)

### Scenario 2: Accidental Data Deletion

1. Don't panic - backups exist!
2. Identify when deletion occurred
3. Download backup from before deletion
4. Restore to local database
5. Export only the deleted data
6. Import back to production
7. Verify restoration

### Scenario 3: Complete Database Loss

1. Provision new PostgreSQL instance
2. Download latest backup
3. Restore to new instance
4. Update backend environment variables
5. Deploy backend
6. Verify all services operational

---

## Backup Size Estimates

**Current Database Size:** ~50-100 MB
**Backup Size (compressed):** ~10-20 MB
**30 days retention:** ~300-600 MB total
**Cost:** ~$0.01/month (Scaleway Object Storage)

### Monitor Backup Growth

```bash
# List backups with sizes
aws s3 ls s3://halo-backups/ --endpoint-url https://s3.fr-par.scw.cloud --human-readable

# Total bucket size
aws s3 ls s3://halo-backups/ --endpoint-url https://s3.fr-par.scw.cloud --summarize
```

---

## Troubleshooting

### Backup Failed: Connection Timeout

**Solution:**
- Check database is accessible: `psql -h 51.159.27.120 -p 19168 -U atlas_user -d rdb`
- Verify password in GitHub Secrets
- Check Railway database status

### Backup Failed: Permission Denied (S3)

**Solution:**
- Verify SCW_ACCESS_KEY and SCW_SECRET_KEY in GitHub Secrets
- Check Scaleway IAM permissions for Object Storage

### Restore Failed: Database in Use

**Solution:**
- Stop backend: `scw container container update ... min-scale=0 max-scale=0`
- Wait for connections to close
- Retry restore

### Backup File Corrupted

**Solution:**
- Try previous day's backup
- Verify file integrity: `pg_restore --list backup.dump`
- Download again (may have been network error)

---

## Best Practices

✅ **DO:**
- Test restore procedure monthly
- Keep backups for at least 30 days
- Monitor backup success via GitHub Actions
- Document any manual restores
- Verify backup integrity regularly

❌ **DON'T:**
- Delete backups manually (automatic cleanup configured)
- Restore to production without testing first
- Ignore backup failure alerts
- Store passwords in plain text (use GitHub Secrets)

---

## GitHub Secrets Required

For automated backups to work, these secrets must be set in GitHub:

1. **DB_PASSWORD:** `@UijY:[e\8_yy5>85Z/^a`
2. **SCW_ACCESS_KEY:** Your Scaleway access key
3. **SCW_SECRET_KEY:** Your Scaleway secret key

**To add secrets:**
1. Go to https://github.com/ArcheronTechnologies/atlas-halo-backend/settings/secrets/actions
2. Click "New repository secret"
3. Add each secret

---

## Next Steps

After setting up backups:
1. ✅ Add GitHub Secrets (DB_PASSWORD, SCW keys)
2. ✅ Commit backup workflow file
3. ✅ Test manual backup run
4. ✅ Verify backup appears in S3
5. ✅ Test restore procedure (locally)
6. ✅ Document restore process
7. ✅ Set up monitoring/alerts

---

**Status:** Ready to deploy
**Automation:** GitHub Actions
**Storage:** Scaleway Object Storage
**Retention:** 30 days
**Schedule:** Daily at 2 AM UTC

**Last Updated:** October 18, 2025
