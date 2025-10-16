# Halo Mobile App - Testing Guide

## Prerequisites
- Expo Go app installed on your phone
  - iOS: https://apps.apple.com/app/expo-go/id982107779
  - Android: https://play.google.com/store/apps/details?id=host.exp.exponent
- Backend running at: https://halobackend4k1irws6-halo-backend.functions.fnc.fr-par.scw.cloud

## Starting the App

### 1. Start Metro Bundler
```bash
cd /Users/timothyaikenhead/Desktop/Halo/mobile
npx expo start
```

### 2. Scan QR Code
- iOS: Open Camera app, point at QR code
- Android: Open Expo Go app, tap "Scan QR Code"

---

## Feature Testing Checklist

### ✅ Test 1: App Launch & Initialization
**What to Test:**
- [ ] App loads without crashes
- [ ] Splash screen appears
- [ ] Loading indicator shows
- [ ] Home screen (map) appears

**Expected Results:**
- Map loads showing Stockholm area
- "📴 Offline • X queued" appears if offline mode active
- Bottom navigation bar visible (Map, Report, Profile)

**Console Logs to Verify:**
```
✅ Geofence monitoring started
✅ Offline storage initialized
```

---

### ✅ Test 2: Map View & Navigation
**What to Test:**
- [ ] Map renders correctly
- [ ] Can pan/zoom the map
- [ ] User location appears (blue dot)
- [ ] Hotspot markers visible

**Steps:**
1. Grant location permissions when prompted
2. Pan around the map
3. Zoom in/out with pinch gesture
4. Tap "Center on my location" button

**Expected Results:**
- Smooth map interactions
- Location updates in real-time
- Hotspot markers appear as colored circles/polygons

---

### ✅ Test 3: Predictions/Hotspots Display
**What to Test:**
- [ ] Hotspots load from backend
- [ ] Different risk levels shown with colors
- [ ] Tapping hotspot shows details
- [ ] Neighborhood names displayed

**Steps:**
1. Wait for map to load
2. Look for colored markers/polygons
3. Tap on a hotspot marker
4. View details popup

**Expected Results:**
- Red zones = high risk (>0.7)
- Orange zones = medium risk (0.4-0.7)
- Yellow zones = low risk (<0.4)
- Popup shows: neighborhood name, risk score, prediction

**Test Locations (Stockholm):**
- Rinkeby: 59.386, 17.926 (usually high risk)
- Södermalm: 59.316, 18.070 (usually low risk)

---

### ✅ Test 4: Incident Reporting
**What to Test:**
- [ ] Can navigate to Report tab
- [ ] Report form appears
- [ ] Can select incident type
- [ ] Can add description
- [ ] Can capture photo/video
- [ ] Submit button works

**Steps:**
1. Tap "Report" tab in bottom navigation
2. Select incident type (e.g., "Theft")
3. Enter description
4. Tap "Add Photo" (optional)
5. Grant camera permissions
6. Capture photo or select from library
7. Tap "Submit Report"

**Expected Results:**
- Form validates required fields
- Camera/gallery opens correctly
- "Submitting..." loading indicator
- Success message: "Report submitted successfully"
- Report appears on map (if online)

---

### ✅ Test 5: Sensor Fusion (Advanced Reporting)
**What to Test:**
- [ ] Video capture works
- [ ] Audio recording works
- [ ] GPS coordinates attached
- [ ] Accelerometer data collected
- [ ] All data sent to backend

**Steps:**
1. Go to Report tab
2. Enable "Advanced Mode" (if available)
3. Tap "Record Video"
4. Grant camera and microphone permissions
5. Record 5-10 second video
6. Submit report

**Expected Results:**
- Video uploads to backend
- GPS coordinates attached automatically
- Accelerometer data captured
- Backend processes with AI analysis

**API Endpoint Hit:**
```
POST /api/v1/incidents/sensor-fusion
```

---

### ✅ Test 6: Offline Mode
**What to Test:**
- [ ] Offline indicator appears when disconnected
- [ ] Can create reports offline
- [ ] Reports queue locally
- [ ] Queue count updates
- [ ] Auto-sync when back online

**Steps:**
1. **Go Offline:**
   - Enable Airplane Mode on device
   - Or disable WiFi + Cellular data

2. **Create Report Offline:**
   - Go to Report tab
   - Fill out incident form
   - Submit report

3. **Check Offline Indicator:**
   - Go back to Map tab
   - Look for "📴 Offline • 1 queued" at top

4. **Go Back Online:**
   - Disable Airplane Mode
   - Wait 5-10 seconds

5. **Verify Sync:**
   - Check if "📴 Offline" disappears
   - Queue count should go to 0
   - Report should appear on map

**Expected Results:**
- ✅ Offline indicator visible when disconnected
- ✅ Reports saved to local AsyncStorage
- ✅ Queue persists even if app closes
- ✅ Auto-sync when reconnected
- ✅ Success notification after sync

**Console Logs:**
```
📴 Offline - queueing incident for later sync
✅ Synced 1 queued incidents
```

---

### ✅ Test 7: Geofence Alerts
**What to Test:**
- [ ] Geofence monitoring starts
- [ ] Entering high-risk area triggers alert
- [ ] Notification appears
- [ ] Works in background

**Steps:**
1. **Grant Permissions:**
   - Location: "Always Allow" (not just "While Using")
   - Notifications: "Allow"

2. **Test Geofence:**
   - If in Stockholm: Walk/drive to Rinkeby area (59.386, 17.926)
   - If testing elsewhere: Use simulator to spoof location

3. **Expected Notification:**
   - Title: "⚠️ High Crime Risk Area"
   - Body: "You have entered a high-risk area. Stay alert."

4. **Test Background:**
   - Put app in background (press home button)
   - Enter high-risk area
   - Notification should still appear

**Expected Results:**
- ✅ Alert appears when entering risk_score >= 0.6 areas
- ✅ Works even when app is backgrounded
- ✅ No alert in low-risk areas

**Note:** iOS requires "Always Allow" location permission for background monitoring

---

### ✅ Test 8: User Profile
**What to Test:**
- [ ] Profile tab loads
- [ ] User stats displayed
- [ ] Settings accessible
- [ ] Logout works

**Steps:**
1. Tap "Profile" tab
2. View user statistics
3. Tap "Settings"
4. Try logout

**Expected Stats:**
- Reports submitted
- Incidents near you (last 24h)
- Your safety score

---

## Performance Testing

### Test 9: Map Performance
**What to Test:**
- [ ] Map loads in < 3 seconds
- [ ] Smooth scrolling (60 FPS)
- [ ] No lag when zooming
- [ ] Handles 100+ hotspots

**Steps:**
1. Load map
2. Zoom in/out rapidly
3. Pan across large distances
4. Load area with many hotspots

**Expected:** No frame drops, smooth animations

### Test 10: Memory Usage
**What to Test:**
- [ ] App doesn't crash after 10+ minutes
- [ ] No memory leaks
- [ ] Background location doesn't drain battery

**Steps:**
1. Use app for 10 minutes
2. Create multiple reports
3. Navigate between tabs
4. Put in background for 5 minutes
5. Return to app

**Expected:** App remains responsive, no crashes

---

## Bug Reporting

If you encounter any issues, note:
1. **Device:** iPhone/Android model and OS version
2. **Steps to reproduce:** What did you do?
3. **Expected vs Actual:** What should happen vs what happened?
4. **Screenshots/Videos:** Capture the issue
5. **Console logs:** Copy any error messages

---

## Common Issues & Fixes

### Issue: Map not loading
**Fix:**
- Check internet connection
- Verify backend is running: `curl https://halobackend4k1irws6-halo-backend.functions.fnc.fr-par.scw.cloud/health`
- Restart app

### Issue: Location not working
**Fix:**
- Go to Settings → Halo → Location → "Always Allow"
- Enable location services in phone settings
- Restart app

### Issue: Offline mode not syncing
**Fix:**
- Check internet connection
- Check backend status
- Clear app data and re-login

### Issue: Camera/microphone not working
**Fix:**
- Go to Settings → Halo → Camera/Microphone → Allow
- Grant permissions when prompted
- Restart app

---

## API Endpoints Being Tested

### Map & Predictions
```
GET /api/v1/predictions/hotspots?lat={lat}&lon={lon}&radius_km=5&hours_ahead=0&min_risk=0&limit=50
```

### Incident Reporting
```
POST /api/v1/incidents
POST /api/v1/incidents/sensor-fusion
```

### Clustering
```
POST /api/v1/reports/submit
GET /api/v1/reports/cluster/{id}
```

### Health Check
```
GET /health
```

---

## Success Criteria

**App is considered working if:**
- ✅ All features load without crashes
- ✅ Map displays hotspots from backend
- ✅ Can create and submit incident reports
- ✅ Offline mode queues and syncs reports
- ✅ Geofence alerts trigger in high-risk areas
- ✅ No major performance issues
- ✅ User experience is smooth and intuitive

---

## Next Steps After Testing

1. **Document Issues:** Create list of bugs/improvements
2. **Performance Metrics:** Note load times, response times
3. **User Feedback:** What feels good? What's confusing?
4. **Production Readiness:** Is app ready for real users?

**Current Status:** Code deployed, awaiting device testing

**Contact:** Report issues to development team with details above
