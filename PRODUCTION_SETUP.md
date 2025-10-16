# Production Setup Guide

## Task 4: CI/CD Setup

### GitHub Actions Secrets

Add these 3 secrets to enable automatic deployment:

1. Go to: https://github.com/ArcheronTechnologies/atlas-halo-backend/settings/secrets/actions
2. Click **New repository secret** for each:

| Secret Name | Value |
|-------------|-------|
| `SCW_ACCESS_KEY` | `SCWS7RSF5GCWHZMWC4E8` |
| `SCW_SECRET_KEY` | `44d40591-1a92-4848-9b09-baf2f69c0f64` |
| `SCW_DEFAULT_ORGANIZATION_ID` | `d70661ef-9ec0-4c64-9c0c-108fa0d0ce51` |

### How It Works

Once secrets are added:
1. Push code to `main` branch
2. GitHub Actions automatically builds AMD64 Docker image
3. Pushes to Scaleway Container Registry
4. Updates production container
5. Runs health checks

**To trigger first deployment:**
```bash
git push backend-origin main
```

Monitor at: https://github.com/ArcheronTechnologies/atlas-halo-backend/actions

---

## Task 5: Production Monitoring

### Database Schema Migration

Run these SQL updates on production database:

```bash
# Connect to production database
PGPASSWORD='@UijY:[e\8_yy5>85Z/^a' psql -h 51.159.27.120 -p 19168 -U atlas_user -d rdb

# Run schema updates
\i backend/database/schema_updates_correlation.sql
```

**Or via backend API** (recommended):

```bash
curl -X POST "https://halobackend4k1irws6-halo-backend.functions.fnc.fr-par.scw.cloud/api/v1/admin/migrate"
```

### Health Monitoring Endpoints

**Primary Health Check:**
```bash
curl https://halobackend4k1irws6-halo-backend.functions.fnc.fr-par.scw.cloud/health
```

**AI Services Health:**
```bash
curl https://halobackend4k1irws6-halo-backend.functions.fnc.fr-par.scw.cloud/api/v1/ai/health
```

**Database Connectivity:**
```bash
curl https://halobackend4k1irws6-halo-backend.functions.fnc.fr-par.scw.cloud/api/v1/predictions/hotspots?lat=59.33&lon=18.07&radius_km=1&hours_ahead=0&min_risk=0&limit=1
```

### Recommended Monitoring Tools

#### 1. UptimeRobot (Free)
- URL: https://uptimerobot.com
- Monitor: `/health` endpoint every 5 minutes
- Alert via email/SMS on downtime

#### 2. Sentry Error Tracking
```bash
# Add to backend
pip install sentry-sdk

# In main.py
import sentry_sdk
sentry_sdk.init(
    dsn="YOUR_SENTRY_DSN",
    traces_sample_rate=0.1,
)
```

#### 3. Scaleway Cockpit
```bash
# View container logs
scw container container logs 35a73370-0199-42de-862c-88b67af1890d --follow

# View metrics in Scaleway Console
# https://console.scaleway.com/containers/namespaces/fr-par/6beff714-75f0-44f6-9326-9441f5ce6b63/containers/35a73370-0199-42de-862c-88b67af1890d/metrics
```

### Performance Monitoring

**Check API Response Times:**
```bash
time curl -s "https://halobackend4k1irws6-halo-backend.functions.fnc.fr-par.scw.cloud/health" > /dev/null
```

**Monitor Database Queries:**
```sql
-- Connect to database
PGPASSWORD='@UijY:[e\8_yy5>85Z/^a' psql -h 51.159.27.120 -p 19168 -U atlas_user -d rdb

-- Check slow queries
SELECT query, calls, mean_exec_time, max_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Check table sizes
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

### Backup Procedures

**Database Backup:**
```bash
# Full backup
PGPASSWORD='@UijY:[e\8_yy5>85Z/^a' pg_dump -h 51.159.27.120 -p 19168 -U atlas_user -d rdb -F c -f halo_backup_$(date +%Y%m%d).dump

# Restore from backup
PGPASSWORD='@UijY:[e\8_yy5>85Z/^a' pg_restore -h 51.159.27.120 -p 19168 -U atlas_user -d rdb halo_backup_20251016.dump
```

**Container Image Backup:**
```bash
# Images are stored in Scaleway registry with git SHA tags
# View all tags:
scw registry tag list image-id=69be41ca-f518-4fab-b72d-80fd6e02c6dd

# Rollback to previous version:
scw container container update 35a73370-0199-42de-862c-88b67af1890d \
  registry-image="rg.fr-par.scw.cloud/funcscwhalobackend4k1irws6/halo-backend:PREVIOUS_TAG" \
  redeploy=true
```

### Daily Retraining Verification

**Check if scheduler is running:**
```bash
curl -s "https://halobackend4k1irws6-halo-backend.functions.fnc.fr-par.scw.cloud/health" | grep -i scheduler
```

**View logs at 02:00 UTC:**
```bash
# Monitor logs during retraining time
scw container container logs 35a73370-0199-42de-862c-88b67af1890d \
  --follow \
  --since "2025-10-16T02:00:00Z"
```

**Manual trigger (for testing):**
```bash
curl -X POST "https://halobackend4k1irws6-halo-backend.functions.fnc.fr-par.scw.cloud/api/v1/admin/retrain"
```

### Security Checklist

- [ ] Database credentials in environment variables only
- [ ] HTTPS enforced on all endpoints
- [ ] Rate limiting configured
- [ ] Input validation on all endpoints
- [ ] SQL injection prevention (parameterized queries)
- [ ] CORS configured properly
- [ ] Secrets not in git history
- [ ] Container runs as non-root user
- [ ] Dependencies regularly updated

### Scaling Configuration

**Current Limits:**
- Min instances: 1
- Max instances: 3
- Memory: 2GB
- CPU: 1000m (1 vCPU)
- Timeout: 5 minutes

**To increase capacity:**
```bash
scw container container update 35a73370-0199-42de-862c-88b67af1890d \
  max-scale=5 \
  memory-limit=4096 \
  cpu-limit=2000
```

### Incident Response

**If backend is down:**
1. Check health endpoint
2. View container logs
3. Check Scaleway status page
4. Rollback to previous image if needed
5. Check database connectivity

**If sensor fusion not working:**
1. Check correlation query performance
2. Verify Atlas Intelligence is responding
3. Check database indexes exist
4. Review logs for errors

**If daily retraining fails:**
1. Check scheduler logs at 02:00 UTC
2. Verify Atlas Intelligence endpoint
3. Check database has recent incidents
4. Manually trigger retraining

### Support Contacts

- **Scaleway Support**: https://console.scaleway.com/support
- **GitHub Issues**: https://github.com/ArcheronTechnologies/atlas-halo-backend/issues
- **Documentation**: See SENSOR_FUSION_IMPLEMENTATION.md

---

## Production Checklist

### Before Launch
- [ ] Add GitHub Actions secrets
- [ ] Run database migrations
- [ ] Set up UptimeRobot monitoring
- [ ] Configure Sentry error tracking
- [ ] Test sensor fusion with real incidents
- [ ] Verify daily retraining at 02:00 UTC
- [ ] Test push notifications on mobile
- [ ] Test offline mode
- [ ] Load test API endpoints
- [ ] Review security checklist

### After Launch
- [ ] Monitor health endpoints daily
- [ ] Review logs weekly
- [ ] Check database backups weekly
- [ ] Update dependencies monthly
- [ ] Review error rates in Sentry
- [ ] Monitor API response times
- [ ] Check sensor fusion correlation accuracy

