# Sentry Setup Guide for Halo Backend

## Step 1: Create Sentry Account

1. Go to https://sentry.io/signup/
2. Sign up with your email or GitHub account
3. Choose the **Free** plan (sufficient for this project)

## Step 2: Create New Project

1. After login, click **"Create Project"**
2. Select **Python** as the platform
3. Set alert frequency: **On every new issue** (recommended)
4. Name your project: **halo-backend**
5. Click **Create Project**

## Step 3: Get Your DSN

After creating the project, you'll see a page with your DSN. It looks like:
```
https://1234567890abcdef1234567890abcdef@o123456.ingest.sentry.io/123456
```

**Copy this DSN** - you'll need it for the next step.

## Step 4: Add DSN to Scaleway Environment Variables

Run this command (replace `YOUR_SENTRY_DSN` with the actual DSN):

```bash
scw container container update 35a73370-0199-42de-862c-88b67af1890d \
  environment-variables.SENTRY_DSN="YOUR_SENTRY_DSN"
```

Example:
```bash
scw container container update 35a73370-0199-42de-862c-88b67af1890d \
  environment-variables.SENTRY_DSN="https://1234567890abcdef1234567890abcdef@o123456.ingest.sentry.io/123456"
```

## Step 5: Redeploy Container

```bash
scw container container deploy 35a73370-0199-42de-862c-88b67af1890d
```

## Step 6: Verify Sentry is Working

1. Wait 30 seconds for deployment
2. Check backend logs - you should see: `✅ Sentry error tracking initialized`
3. Trigger a test error:
```bash
# This endpoint doesn't exist, will create a 404 error in Sentry
curl https://halobackend4k1irws6-halo-backend.functions.fnc.fr-par.scw.cloud/api/v1/test-error
```
4. Go to your Sentry project dashboard - you should see the error appear

## Step 7: Configure Alerts (Optional but Recommended)

1. In Sentry project, go to **Settings → Alerts**
2. Click **Create Alert Rule**
3. Select **Issues**
4. Configure:
   - **When:** An event is seen
   - **If:** None (alert on all errors)
   - **Then:** Send a notification via Email
5. Save the alert rule

## Sentry Benefits for Halo

✅ **Automatic Error Tracking** - All unhandled exceptions captured
✅ **Performance Monitoring** - 10% of requests sampled for performance insights
✅ **Stack Traces** - Full Python stack traces with line numbers
✅ **Environment Context** - Know if error is from production/staging
✅ **User Context** - See which API endpoints are failing
✅ **Email Alerts** - Get notified immediately when errors occur

## Troubleshooting

**Issue:** Sentry not logging errors
**Solution:** Check that SENTRY_DSN environment variable is set correctly:
```bash
scw container container get 35a73370-0199-42de-862c-88b67af1890d | grep SENTRY_DSN
```

**Issue:** Too many alerts
**Solution:** Adjust alert rules in Sentry dashboard to only alert on critical errors

---

## Quick Setup Commands

```bash
# 1. Add DSN to environment
scw container container update 35a73370-0199-42de-862c-88b67af1890d \
  environment-variables.SENTRY_DSN="YOUR_DSN_HERE"

# 2. Redeploy
scw container container deploy 35a73370-0199-42de-862c-88b67af1890d

# 3. Wait 30 seconds, then verify
sleep 30 && curl https://halobackend4k1irws6-halo-backend.functions.fnc.fr-par.scw.cloud/health
```

---

**Next:** After Sentry is set up, proceed to [UptimeRobot Setup](UPTIMEROBOT_SETUP_GUIDE.md)
