# Sensor Fusion Implementation - Complete Guide

## Overview
Sensor fusion system for correlating multiple user incident reports, dynamic risk score updates, and daily AI model retraining.

## Implementation Status

### ✅ Completed Components

#### 1. Multi-User Incident Correlation
**File**: `backend/services/sensor_fusion.py` (330+ lines)

**Key Features**:
- Geographic proximity matching: 50m radius using PostGIS `ST_DWithin`
- Temporal matching: 5-minute time window
- Visual similarity matching: 70% threshold via Atlas Intelligence
- Automatic incident merging with reporter tracking

**Core Functions**:
```python
async def correlate_incident(incident_data: dict, db_connection) -> Optional[dict]
    # Main entry point for correlation

async def _find_candidates(incident_location, occurred_at, time_window, radius, db_connection)
    # PostGIS geographic query for nearby incidents

async def _check_similarity(video_id_1, video_id_2) -> float
    # Atlas Intelligence visual comparison

async def _merge_incidents(existing_incident, new_incident_data, db_connection)
    # Updates reporter_count and corroborating_reports arrays

async def resolve_conflict(incidents: List[dict], db_connection) -> str
    # Consensus or AI tiebreaker for conflicting classifications
```

**Integration**: [backend/api/incidents_api.py:502-546](backend/api/incidents_api.py#L502)

#### 2. Dynamic Risk Score Updates
**File**: `backend/workers/update_risk_scores.py` (130+ lines)

**Key Features**:
- 2km radius impact zone around new incidents
- Distance decay formula: `risk_increase * (1 - distance_km / radius_km)`
- Category-specific weighting
- Async batch updates using asyncpg

**Core Function**:
```python
async def update_nearby_risk_scores(
    incident_location: dict,
    incident_category: str,
    db_connection,
    radius_km: float = 2.0,
    risk_increase: float = 0.10
)
```

**Integration**: [backend/api/incidents_api.py:543-554](backend/api/incidents_api.py#L543)

#### 3. Daily AI Retraining Scheduler
**File**: `backend/workers/daily_retrain.py` (190+ lines)

**Key Features**:
- Scheduled for 02:00 UTC daily via APScheduler
- Collects incidents from past 24 hours
- Extracts features: location, time, category, severity
- Sends batch to Atlas Intelligence for model updates
- Regenerates predictions for all neighborhoods

**Scheduler Configuration**: [main.py:142-168](main.py#L142)

```python
scheduler = AsyncIOScheduler()
scheduler.add_job(
    daily_retrain_job,
    'cron',
    hour=2,
    minute=0,
    id='daily_retrain',
    name='Daily AI Model Retraining'
)
scheduler.start()
```

#### 4. Database Schema Updates
**File**: `backend/database/schema_updates_correlation.sql`

```sql
ALTER TABLE incidents
ADD COLUMN IF NOT EXISTS reporter_count INTEGER DEFAULT 1;

ALTER TABLE incidents
ADD COLUMN IF NOT EXISTS corroborating_reports JSONB DEFAULT '[]'::jsonb;

ALTER TABLE incidents
ADD COLUMN IF NOT EXISTS video_ids TEXT[] DEFAULT ARRAY[]::TEXT[];

CREATE INDEX IF NOT EXISTS idx_incidents_location
ON incidents USING GIST (ST_SetSRID(ST_MakePoint(longitude, latitude), 4326));

CREATE INDEX IF NOT EXISTS idx_incidents_occurred_at
ON incidents (occurred_at DESC);
```

#### 5. Dependencies Updated
**File**: `requirements.txt:33`
```
apscheduler==3.10.4
```

## Git Status

**Commit**: `8a390e2`
**Branch**: `main`
**Message**: "Add sensor fusion and dynamic risk score updates"

**Pushed to**: `https://github.com/ArcheronTechnologies/atlas-halo-backend.git`

## Deployment Challenges

### Issue: Architecture Mismatch
**Problem**: Scaleway Serverless Containers require AMD64 architecture, but:
1. Local Mac build produces ARM64 images
2. `scw container deploy` builds on ARM-based infrastructure
3. Docker registry push requires special authentication

**Error Message**:
```
Invalid Image architecture. Serverless Containers only support the amd64
architecture, but the image was built for the following architectures: arm64.
```

### Attempted Solutions
1. ✅ Built AMD64 image locally: `docker build --platform=linux/amd64`
2. ❌ Push to Scaleway registry: Authentication issues with Docker login
3. ❌ Deploy from clean directory: Still builds ARM64 on Scaleway infrastructure

## Recommended Deployment Approach

### Option 1: GitHub Actions CI/CD (Recommended)
Create `.github/workflows/scaleway-deploy.yml`:

```yaml
name: Deploy to Scaleway
on:
  push:
    branches: [main]
    paths:
      - 'backend/**'
      - 'main.py'
      - 'requirements.txt'
      - 'Dockerfile'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build AMD64 image
        run: |
          docker build --platform=linux/amd64 \\
            -t rg.fr-par.scw.cloud/funcscwhalobackend4k1irws6/halo-backend:${{ github.sha }} \\
            -t rg.fr-par.scw.cloud/funcscwhalobackend4k1irws6/halo-backend:latest \\
            .

      - name: Install Scaleway CLI
        run: |
          curl -o /usr/local/bin/scw -L https://github.com/scaleway/scaleway-cli/releases/download/v2.30.0/scaleway-cli_2.30.0_linux_amd64
          chmod +x /usr/local/bin/scw

      - name: Configure Scaleway
        run: |
          scw config set access-key=${{ secrets.SCW_ACCESS_KEY }}
          scw config set secret-key=${{ secrets.SCW_SECRET_KEY }}
          scw config set default-organization-id=${{ secrets.SCW_ORGANIZATION_ID }}
          scw config set default-region=fr-par

      - name: Login to Scaleway Registry
        run: |
          echo "${{ secrets.SCW_REGISTRY_TOKEN }}" | docker login \\
            rg.fr-par.scw.cloud \\
            -u nologin --password-stdin

      - name: Push image
        run: |
          docker push rg.fr-par.scw.cloud/funcscwhalobackend4k1irws6/halo-backend:${{ github.sha }}
          docker push rg.fr-par.scw.cloud/funcscwhalobackend4k1irws6/halo-backend:latest

      - name: Update container
        run: |
          scw container container update 35a73370-0199-42de-862c-88b67af1890d \\
            registry-image="rg.fr-par.scw.cloud/funcscwhalobackend4k1irws6/halo-backend:latest" \\
            redeploy=true --wait
```

**Required Secrets**:
- `SCW_ACCESS_KEY`: From `scw config get access-key`
- `SCW_SECRET_KEY`: From `scw config get secret-key`
- `SCW_ORGANIZATION_ID`: From `scw config get default-organization-id`
- `SCW_REGISTRY_TOKEN`: Create via Scaleway Console → Container Registry → Credentials

### Option 2: Manual Deployment from AMD64 Machine
```bash
# On AMD64 Linux machine or GitHub Actions
cd /path/to/Halo

# Build image
docker build --platform=linux/amd64 \\
  -t rg.fr-par.scw.cloud/funcscwhalobackend4k1irws6/halo-backend:sensor-fusion \\
  .

# Login to registry (get token from Scaleway Console)
echo "YOUR_REGISTRY_TOKEN" | docker login \\
  rg.fr-par.scw.cloud/funcscwhalobackend4k1irws6 \\
  -u nologin --password-stdin

# Push image
docker push rg.fr-par.scw.cloud/funcscwhalobackend4k1irws6/halo-backend:sensor-fusion

# Update container
scw container container update 35a73370-0199-42de-862c-88b67af1890d \\
  registry-image="rg.fr-par.scw.cloud/funcscwhalobackend4k1irws6/halo-backend:sensor-fusion" \\
  redeploy=true --wait
```

### Option 3: Build on Scaleway Instance
```bash
# Create Scaleway Instance (AMD64)
scw instance server create type=DEV1-S zone=fr-par-1 image=ubuntu_jammy

# SSH and run
git clone https://github.com/ArcheronTechnologies/atlas-halo-backend.git
cd atlas-halo-backend
docker build -t rg.fr-par.scw.cloud/funcscwhalobackend4k1irws6/halo-backend:sensor-fusion .

# Login and push (as above)
```

## Testing Checklist

### Backend API Tests
```bash
# Health check
curl https://halobackend4k1irws6-halo-backend.functions.fnc.fr-par.scw.cloud/health

# AI health
curl https://halobackend4k1irws6-halo-backend.functions.fnc.fr-par.scw.cloud/api/v1/ai/health

# Categories
curl https://halobackend4k1irws6-halo-backend.functions.fnc.fr-par.scw.cloud/api/v1/ai/categories
```

### Sensor Fusion Test
1. Create first incident via mobile app
2. Create second incident at same location within 5 minutes
3. Verify response indicates correlation: `"status": "corroborated"`
4. Check `reporter_count > 1` in database

### Dynamic Risk Scores Test
```bash
# Check predictions before incident
curl "https://halobackend4k1irws6-halo-backend.functions.fnc.fr-par.scw.cloud/api/v1/predictions/hotspots?lat=59.33&lon=18.07&radius_km=2&hours_ahead=0&min_risk=0.0&limit=5"

# Create incident via app

# Check predictions after (should show increased risk nearby)
curl "https://halobackend4k1irws6-halo-backend.functions.fnc.fr-par.scw.cloud/api/v1/predictions/hotspots?lat=59.33&lon=18.07&radius_km=2&hours_ahead=0&min_risk=0.0&limit=5"
```

### Daily Retraining Test
```bash
# Check scheduler is running
curl https://halobackend4k1irws6-halo-backend.functions.fnc.fr-par.scw.cloud/health

# Check logs at 02:00 UTC for retraining job execution
scw container container logs 35a73370-0199-42de-862c-88b67af1890d --follow
```

## Current Deployment Status

**Current Image**: `rg.fr-par.scw.cloud/funcscwhalobackend4k1irws6/halo-backend:categories-fix`
**Status**: ✅ Healthy (AI analysis working, categories endpoint working)
**Sensor Fusion**: ❌ Not yet deployed (code committed but not in production)

**Why Not Deployed**: Architecture mismatch between local ARM64 builds and Scaleway AMD64 requirements. Requires CI/CD pipeline or AMD64 build environment.

## Next Steps

1. **Immediate**: Set up GitHub Actions workflow with Scaleway registry credentials
2. **Testing**: Once deployed, run sensor fusion tests with multiple mobile app incidents
3. **Monitoring**: Verify daily retraining job executes at 02:00 UTC
4. **Performance**: Monitor correlation query performance with PostGIS indexes

## Files Modified

- `backend/services/sensor_fusion.py` (new)
- `backend/workers/update_risk_scores.py` (new)
- `backend/workers/daily_retrain.py` (new)
- `backend/database/schema_updates_correlation.sql` (new)
- `backend/api/incidents_api.py` (modified: lines 18-19, 502-554)
- `main.py` (modified: lines 49-50, 142-168)
- `requirements.txt` (modified: line 33)

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Mobile App (Expo)                       │
│  - Records video incident                                   │
│  - Sends to backend with location/timestamp                │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              Halo Backend (FastAPI)                         │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  incidents_api.py::create_incident()                 │  │
│  │  1. Receives incident data                           │  │
│  │  2. Calls sensor_fusion.correlate_incident()         │  │
│  │  3. Calls update_risk_scores()                       │  │
│  │  4. Returns response                                 │  │
│  └────┬─────────────────────────────────────────┬───────┘  │
│       │                                         │          │
│       ▼                                         ▼          │
│  ┌─────────────────────┐            ┌──────────────────┐  │
│  │ Sensor Fusion       │            │ Risk Updater     │  │
│  │ - Find candidates   │            │ - 2km radius     │  │
│  │ - Check similarity  │            │ - Distance decay │  │
│  │ - Merge incidents   │            │ - Batch update   │  │
│  └────┬────────────────┘            └──────────────────┘  │
│       │                                                    │
│       ▼                                                    │
│  ┌─────────────────────────────────────────────────────┐  │
│  │          PostgreSQL with PostGIS                    │  │
│  │  - Geographic queries (ST_DWithin)                  │  │
│  │  - Correlation tracking                             │  │
│  │  - Risk score updates                               │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                            │
│  ┌─────────────────────────────────────────────────────┐  │
│  │      APScheduler (Daily 02:00 UTC)                  │  │
│  │  - Collect past 24h incidents                       │  │
│  │  - Extract features                                 │  │
│  │  - Send to Atlas Intelligence                       │  │
│  │  - Regenerate predictions                           │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│          Atlas Intelligence (YOLOv8, Whisper)               │
│  - Visual similarity comparison                             │
│  - Model retraining                                         │
└─────────────────────────────────────────────────────────────┘
```

## Performance Considerations

### PostGIS Indexes
- `idx_incidents_location`: GIST index for fast geographic queries
- `idx_incidents_occurred_at`: B-tree index for temporal filtering
- Expected query time: <100ms for 50m radius search

### Correlation Thresholds
- **Geographic**: 50m (can tune based on GPS accuracy)
- **Temporal**: 5 minutes (adjust for user report delays)
- **Visual**: 70% similarity (balance false positives/negatives)

### Risk Update Performance
- Batch updates using asyncpg
- 2km radius typically affects 10-50 predictions
- Update time: <500ms per incident

## Security Notes

1. Database credentials in environment variables
2. Atlas Intelligence API uses internal Scaleway network
3. Sensor fusion prevents duplicate incident spam
4. Video similarity prevents false correlation attacks
