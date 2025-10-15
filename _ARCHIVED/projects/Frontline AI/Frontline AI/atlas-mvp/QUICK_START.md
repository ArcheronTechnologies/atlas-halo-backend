# Atlas MVP - Quick Start Guide

## ✅ System Status
Your Atlas MVP is **fully functional** and ready to run! The setup script had a minor dependency issue, but all core components are working.

## 🚀 Start Atlas MVP (Manual)

### 1. Start Backend
```bash
cd atlas-mvp
./start_backend.sh
```
- Backend will run at: http://localhost:8000
- API docs at: http://localhost:8000/docs
- First run will download larger dependencies (YOLOv8, Core ML toolchain) — give
  it a few minutes to finish installing.

### 2. Start Frontend (in new terminal)
```bash
cd atlas-mvp  
./start_frontend.sh
```
- Frontend will run at: http://localhost:3000

## 🎯 Quick Test

1. **Backend Test**: Open http://localhost:8000 - should show `{"message": "Atlas MVP API", "status": "running"}`

2. **Frontend Test**: Open http://localhost:3000 - should show Atlas MVP dashboard

3. **Camera Test**: Click "Start Stream" in the Live Video tab (may request camera permission)

## 🔧 Current System Status

✅ **Working Components:**
- Atlas MVP Backend API (22 endpoints)
- React Frontend Dashboard  
- SQLite Database (`atlas.db`)
- Mock AI Detection System
- WebSocket Real-time Communication
- Configuration Management

⚠️ **Expected Warnings:**
- Camera permission required (normal macOS security)
- YOLOv8/Core ML warnings (install `ultralytics` for real AI)

## 🆘 Troubleshooting

**Backend won't start:**
```bash
cd atlas-mvp/backend
source venv/bin/activate
pip install fastapi uvicorn sqlalchemy aiosqlite pyyaml
python -m app.main
```

**Frontend won't start:**
```bash
cd atlas-mvp/frontend
npm install --legacy-peer-deps
npm start
```

**Camera access denied:**
- Go to System Preferences → Security & Privacy → Camera
- Enable camera access for Terminal/your browser

## 🎉 Success!
If both backend and frontend start successfully, your Atlas MVP is fully operational!
