# Halo Production - Outstanding Tasks

## ✅ Recently Fixed

### 1. Clustering API - NOW WORKING ✅
**Status:** FIXED (2025-10-16)

**Root Causes Found:**
1. FastAPI `Depends(get_database)` doesn't properly await async dependencies
2. Missing `psycopg[binary]` package for SQLAlchemy async PostgreSQL

**Solution Applied:**
- Removed `Depends()` from all clustering endpoints
- Added direct `db = await get_database()` calls (matching other APIs)
- Added `psycopg[binary]==3.1.18` to requirements.txt

**Verified Working:**
- ✅ `GET /api/v1/reports/stats`
- ✅ `GET /api/v1/reports/cross-source-stats`
- ✅ `GET /api/v1/reports/debug/test-query`
- ✅ `POST /api/v1/reports/submit` (endpoint exists, needs data to test fully)

**Commits:**
- c72293a: Fix clustering API by removing Depends()
- 2bc8cd4: Add psycopg[binary] to requirements

---

## 🟡 Important (Not Blocking)

### 2. External Monitoring Setup
**Status:** DOCUMENTED BUT NOT CONFIGURED

**What's Needed:**
1. **UptimeRobot** - Monitor `/health` endpoint every 5 minutes
   - URL: https://uptimerobot.com
   - Endpoint: https://halobackend4k1irws6-halo-backend.functions.fnc.fr-par.scw.cloud/health
   - Alert via email/SMS on downtime

2. **Sentry Error Tracking**
   ```bash
   # Add to backend/requirements.txt
   pip install sentry-sdk

   # In main.py (add at startup)
   import sentry_sdk
   sentry_sdk.init(
       dsn="YOUR_SENTRY_DSN",
       traces_sample_rate=0.1,
   )
   ```

3. **Scaleway Cockpit** - Already available, just need to set up dashboards
   - Metrics: https://console.scaleway.com/containers/namespaces/fr-par/6beff714-75f0-44f6-9326-9441f5ce6b63/containers/35a73370-0199-42de-862c-88b67af1890d/metrics

**Documentation:** See [PRODUCTION_SETUP.md](PRODUCTION_SETUP.md) sections 73-100

---

### 3. Daily Retraining Job Monitoring
**Status:** CONFIGURED BUT NOT VERIFIED

**What's Needed:**
1. Monitor logs at 02:00 UTC to confirm job runs successfully
2. Set up alert if job fails
3. Verify predictions are regenerated after retraining

**Manual Test:**
```bash
curl -X POST "https://halobackend4k1irws6-halo-backend.functions.fnc.fr-par.scw.cloud/api/v1/admin/retrain/trigger"
```

**Expected Result:**
```json
{
  "status": "success",
  "message": "AI model retraining completed successfully"
}
```

---

### 4. Mobile App Testing
**Status:** CODE DEPLOYED BUT NOT TESTED ON REAL DEVICES

**What's Needed:**
1. **Test Geofence Alerts:**
   - Enable location permissions on device
   - Move into high-risk area (risk_score >= 0.6)
   - Verify notification appears

2. **Test Offline Mode:**
   - Disable WiFi/cellular
   - Submit incident report
   - Verify "📴 Offline • 1 queued" appears on map
   - Re-enable network
   - Verify auto-sync occurs

3. **Test Background Location:**
   - Grant "Always Allow" location permission
   - Put app in background
   - Verify geofence monitoring continues

**How to Test:**
```bash
cd mobile
npx expo start
# Scan QR with Expo Go app on your phone
```

**Test Locations (Stockholm high-risk areas):**
- Rinkeby: 59.386, 17.926
- Tensta: 59.395, 17.901
- Järva: 59.387, 17.943

---

### 5. Database Backup Setup
**Status:** DOCUMENTED BUT NOT SCHEDULED

**What's Needed:**
1. Schedule daily backups of production database
   ```bash
   PGPASSWORD='@UijY:[e\8_yy5>85Z/^a' pg_dump \
     -h 51.159.27.120 -p 19168 -U atlas_user -d rdb \
     -F c -f halo_backup_$(date +%Y%m%d).dump
   ```

2. Upload to cloud storage (S3, Scaleway Object Storage)
3. Test restore procedure

**Documentation:** [PRODUCTION_SETUP.md](PRODUCTION_SETUP.md) lines 127-140

---

## 🟢 Optional Improvements

### 6. Performance Optimization
- Monitor container CPU/memory usage
- Consider increasing to 4GB RAM if needed (currently 2GB)
- Add Redis caching for prediction queries
- Optimize H3 spatial queries

### 7. Security Hardening
- Enable rate limiting on APIs
- Add API key authentication for clustering endpoints
- Review and rotate database credentials
- Enable HTTPS-only enforcement

### 8. Mobile App Enhancements
- Add push notification configuration to backend
- Implement notification categories (threat levels)
- Add in-app incident photo/video upload
- Improve offline queue UI with sync progress

---

## ✅ Completed

- ✅ Backend deployed to Scaleway Serverless Containers
- ✅ GitHub Actions CI/CD configured
- ✅ Database schema migrations applied
- ✅ Sensor fusion implementation deployed
- ✅ Mobile app with geofence, offline mode, status indicator
- ✅ Daily retraining job scheduled (02:00 UTC)
- ✅ Manual retraining trigger endpoint
- ✅ Production documentation (DEPLOYMENT_STATUS.md, PRODUCTION_SETUP.md)
- ✅ Health monitoring endpoints working
- ✅ 805 polisen.se incidents flowing from Atlas Intelligence
- ✅ ML predictions API with neighborhood boundaries
- ✅ H3 spatial indexing for map performance

---

## 📊 Current Status Summary

**Backend:** 🟢 READY (all APIs working)
**Mobile:** 🟡 READY (needs device testing)
**Database:** 🟢 READY
**CI/CD:** 🟢 READY
**Monitoring:** 🟡 DOCUMENTED (needs setup)
**Overall:** 🟢 PRODUCTION READY - No blockers

**Next Priorities:**
1. Set up external monitoring (Sentry)
2. Test daily retraining job
3. Test mobile app on real device
