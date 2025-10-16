# UptimeRobot Setup Guide for Halo Backend

## What is UptimeRobot?

UptimeRobot monitors your backend API and sends alerts if it goes down. It's free for up to 50 monitors with 5-minute check intervals.

## Step 1: Create UptimeRobot Account

1. Go to https://uptimerobot.com/
2. Click **"Sign Up Free"**
3. Enter your email and create a password
4. Verify your email address

## Step 2: Create Monitor for Halo Backend

1. After login, click **"+ Add New Monitor"**
2. Fill in the details:

**Monitor Type:** HTTP(s)

**Friendly Name:** Halo Backend Health

**URL (or IP):**
```
https://halobackend4k1irws6-halo-backend.functions.fnc.fr-par.scw.cloud/health
```

**Monitoring Interval:** 5 minutes (free tier)

**Monitor Timeout:** 30 seconds

**Advanced Settings (click to expand):**
- **HTTP Method:** GET
- **HTTP Status Code:** 200
- **Keyword Monitoring:** Enable
  - **Keyword Type:** Should exist
  - **Keyword Value:** `healthy`

3. Click **"Create Monitor"**

## Step 3: Set Up Alert Contacts

1. Go to **"My Settings"** (top right menu)
2. Click **"Add Alert Contact"**
3. Choose notification methods:

### Option A: Email Alerts
- **Type:** Email
- **Email Address:** Your email
- Click **"Create Alert Contact"**

### Option B: SMS Alerts (Optional - Premium)
- **Type:** SMS
- **Phone Number:** Your phone number
- Note: SMS requires paid plan

### Option C: Webhook/Slack (Optional)
- **Type:** Webhook
- **URL:** Your webhook endpoint
- Useful for integration with Slack, Discord, etc.

## Step 4: Configure Monitor to Use Alert Contact

1. Go back to **"Dashboard"**
2. Click on the **"Halo Backend Health"** monitor
3. Click **"Edit"**
4. Scroll to **"Alert Contacts to Notify"**
5. Select your alert contact (email)
6. Click **"Save Changes"**

## Step 5: Test the Monitor

The monitor will start checking your backend every 5 minutes. To verify it's working:

1. Check the **Dashboard** - you should see:
   - **Status:** Up (green)
   - **Uptime:** 100%
   - **Response Time:** ~100-300ms

2. You can manually trigger a check by clicking **"Quick Stats"** → **"Force Check"**

## Step 6: Create Additional Monitors (Recommended)

### Monitor 2: Predictions API
```
Monitor Type: HTTP(s)
Name: Halo Predictions API
URL: https://halobackend4k1irws6-halo-backend.functions.fnc.fr-par.scw.cloud/api/v1/predictions/hotspots?lat=59.33&lon=18.07&radius_km=1&hours_ahead=0&min_risk=0&limit=1
Keyword: "predictions"
```

### Monitor 3: Clustering API
```
Monitor Type: HTTP(s)
Name: Halo Clustering API
URL: https://halobackend4k1irws6-halo-backend.functions.fnc.fr-par.scw.cloud/api/v1/reports/stats
Keyword: "verified_incidents"
```

## Alert Configuration Best Practices

**When to Alert:**
- ✅ Server is down (no response)
- ✅ Response time > 5 seconds
- ✅ HTTP status code is not 200
- ✅ Missing expected keyword in response

**Alert Thresholds:**
- Send alert after **2 consecutive failures** (10 minutes)
- This prevents false alarms from temporary network issues

To configure:
1. Edit monitor → Advanced Settings
2. Set **"Alert After"** to 2 checks

## Expected Metrics

**Normal Operation:**
- **Uptime:** 99.5%+ (target)
- **Response Time:** 100-500ms
- **Status:** Up (green)

**What to Watch:**
- Response time > 2 seconds = performance issue
- Downtime > 5 minutes = critical issue
- Uptime < 99% = recurring problems

## Monitoring Dashboard

UptimeRobot provides:
- ✅ **Real-time status** of all monitors
- ✅ **30-day uptime history**
- ✅ **Response time graphs**
- ✅ **Public status page** (optional - share with users)

### Create Public Status Page (Optional)

1. Go to **"Status Pages"**
2. Click **"Add Status Page"**
3. Select monitors to display
4. Get public URL like: `https://stats.uptimerobot.com/ABC123`
5. Share with users to show system status

## Integration with Sentry

You can connect UptimeRobot downtime alerts to Sentry:

1. In UptimeRobot, create a **Webhook** alert contact
2. Use Sentry webhook URL as the destination
3. Downtime events will appear in Sentry dashboard

## Troubleshooting

**Issue:** Monitor shows "Down" but API is working
**Solution:**
- Check if keyword monitoring is too strict
- Verify URL is correct
- Test URL manually: `curl -v https://...`

**Issue:** Too many alerts
**Solution:**
- Increase "Alert After" to 3-5 checks
- Adjust timeout from 30s to 60s

**Issue:** Email alerts not received
**Solution:**
- Check spam folder
- Verify alert contact is enabled
- Test with "Send Test Alert" button

## Cost

- **Free Plan:** 50 monitors, 5-minute intervals
- **Paid Plans:**
  - Pro: $7/month - 1-minute intervals, SMS alerts
  - Enterprise: Custom pricing

For Halo Backend, the **free plan is sufficient**.

---

## Quick Setup Checklist

- [ ] Create UptimeRobot account
- [ ] Add "Halo Backend Health" monitor
- [ ] Configure email alert contact
- [ ] Verify monitor shows "Up" status
- [ ] (Optional) Add monitors for other APIs
- [ ] (Optional) Create public status page

---

## Summary

After setup, you'll receive:
- 📧 **Email alerts** when backend goes down
- 📊 **Uptime reports** showing 99.5%+ availability
- 📈 **Performance metrics** to track response times
- 🔍 **Early warning** of issues before users notice

**Expected Setup Time:** 5-10 minutes

---

**Previous:** [Sentry Setup](SENTRY_SETUP_GUIDE.md) | **Next:** Test mobile app on device
