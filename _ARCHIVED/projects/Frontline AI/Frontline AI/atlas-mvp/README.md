# Atlas MVP - Edge AI Perception System

A complete edge AI perception system demonstrating real-time object detection, no-code model training, and offline operation, optimized for MacBook Air M1.

## Features

🎥 **Real-Time Detection**
- Live camera feed with object detection overlay
- Support for person, car, truck, motorcycle, and weapon detection
- Confidence thresholds and customizable alerts
- 20+ FPS performance with sub-100ms latency

🧠 **AutoML Training** 
- Drag-and-drop image upload interface
- Custom model training with transfer learning
- Automatic Core ML conversion for M1 optimization
- Training completed in under 10 minutes

🚨 **Smart Alerts**
- Real-time visual and audio notifications
- Configurable confidence thresholds per object class
- Alert history and detection statistics

⚙️ **Configuration**
- Web-based configuration panel
- Model switching and management
- Performance monitoring and statistics

## System Requirements

- MacBook Air M1 (or compatible Apple Silicon Mac)
- macOS 12.0 or later
- Python 3.9+
- Node.js 18+
- Built-in FaceTime HD camera

## Quick Start

### Backend Setup

1. **Clone and navigate to the project:**
   ```bash
   cd atlas-mvp/backend
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   This installs the full edge inference toolchain, including `ultralytics` for
   YOLOv8 models, `torch`/`torchaudio` for accelerated inference, and
   `coremltools` so Apple Silicon builds can export optimized Core ML bundles.
   Installation can take a few minutes the first time as the larger packages are
   downloaded and built.

4. **Start the backend server:**
   ```bash
   python -m app.main
   ```

   The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory:**
   ```bash
   cd ../frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Start the development server:**
   ```bash
   npm start
   ```

   The web interface will open at `http://localhost:3000`

## Usage

### Live Video Detection

1. Click the **Live Video** tab
2. Click **Start Stream** to begin camera capture
3. Detections will appear as bounding boxes with confidence scores
4. View real-time statistics and detection history in the sidebar

### Training Custom Models

1. Go to the **Training** tab
2. Click **Upload Data** and create a new project
3. Drag and drop 20-100 training images
4. Use the annotation tool to label objects (not implemented in MVP)
5. Click **Start Training** - completion typically takes 5-10 minutes
6. Activate your trained model when ready

### Configuration

1. Open the **Config** tab
2. Adjust confidence thresholds for each object class
3. Enable/disable visual and audio alerts
4. Changes are applied immediately

## API Documentation

### Video Endpoints

- `POST /api/video/start` - Start video streaming
- `POST /api/video/stop` - Stop video streaming  
- `GET /api/video/status` - Get stream status and performance stats
- `GET /api/video/detections?limit=100` - Get recent detections
- `GET /api/video/stats?hours=24` - Get detection statistics

### Training Endpoints

- `GET /api/training/models` - List all trained models
- `POST /api/training/models/{id}/activate` - Activate a model
- `POST /api/training/upload-images` - Upload training images
- `POST /api/training/projects/{name}/train` - Start training

### Configuration Endpoints

- `GET /api/config/` - Get all configuration
- `PUT /api/config/confidence-thresholds` - Update detection thresholds
- `POST /api/config/alerts/enable` - Enable alerts

### WebSocket

Connect to `ws://localhost:8000/ws` for real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  if (data.type === 'frame') {
    // New video frame with detection overlay
    displayFrame(data.data);
  } else if (data.type === 'detection') {
    // New object detection
    handleDetection(data.data);
  } else if (data.type === 'alert') {
    // High-priority alert (e.g., weapon detected)
    showAlert(data.data);
  }
};
```

## Architecture

### Backend (Python FastAPI)
```
backend/
├── app/
│   ├── main.py              # FastAPI application & WebSocket handling
│   ├── database.py          # SQLite database management
│   ├── inference/
│   │   └── engine.py        # Core ML inference engine
│   ├── training/
│   │   └── automl.py        # AutoML training pipeline
│   └── api/
│       ├── video.py         # Video streaming endpoints
│       ├── training.py      # Training API
│       └── config.py        # Configuration API
├── models/                  # Trained model storage
└── uploads/                 # Training data uploads
```

### Frontend (React TypeScript)
```
frontend/src/
├── components/
│   ├── VideoStream.tsx      # Live video component
│   ├── TrainingPanel.tsx    # AutoML interface  
│   ├── AlertsPanel.tsx      # Alerts management
│   └── ConfigPanel.tsx      # Settings panel
├── hooks/
│   └── useWebSocket.ts      # WebSocket connection hook
├── services/
│   └── api.ts              # API client
└── App.tsx                 # Main application
```

## Performance Optimization

### Apple Silicon Optimization
- **Core ML**: Automatic conversion of PyTorch models to Core ML format
- **Neural Engine**: Hardware acceleration for inference
- **Memory Management**: Efficient frame buffering and processing
- **Multi-threading**: Background video processing with async I/O

### Expected Performance
- **Video Processing**: 30 FPS input, 20+ FPS with AI inference
- **Inference Latency**: <50ms per frame on M1 MacBook Air  
- **Training Time**: <10 minutes for 100 training images
- **Memory Usage**: <4GB total system memory
- **Model Size**: <50MB per optimized Core ML model

## Troubleshooting

### Common Issues

**Camera not detected:**
```bash
# Check camera permissions in System Preferences > Security & Privacy > Camera
# Restart the application after granting permissions
```

**YOLOv8 installation issues:**
```bash
# Install with specific PyTorch version for M1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
```

**WebSocket connection failed:**
```bash
# Check if backend is running on port 8000
lsof -i :8000

# Restart backend server
python -m app.main
```

**Training fails:**
```bash
# Check available disk space (training requires ~1GB temporary space)
df -h

# Verify training images are valid
file uploads/project_name/images/*
```

### Logging

Enable debug logging:
```bash
# Backend
export LOG_LEVEL=DEBUG
python -m app.main

# Check logs
tail -f app.log
```

## Development

### Adding New Object Classes

1. Update `detection_classes` in `inference/engine.py`
2. Add class colors in `_get_class_color()` method
3. Update confidence thresholds in database initialization
4. Restart the application

### Custom Model Integration

1. Place model file in `models/` directory
2. Add model entry to database:
   ```python
   await db_manager.add_model("custom_model", "models/custom.pt", 0.85)
   ```
3. Activate via API or web interface

### Extending the API

1. Create new router in `app/api/`
2. Add endpoints with FastAPI decorators
3. Include router in `main.py`
4. Update frontend API client in `services/api.ts`

## Deployment

### Development
```bash
# Backend
cd backend && python -m app.main

# Frontend  
cd frontend && npm start
```

### Production
```bash
# Backend with production ASGI server
cd backend && gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker

# Frontend build
cd frontend && npm run build && serve -s build
```

### Docker (Optional)
```bash
docker-compose up --build
```

## Security Considerations

- **Camera Access**: Requires explicit user permission
- **Local Processing**: All AI inference runs locally, no cloud dependencies
- **Data Privacy**: Training data and models stay on device
- **Network Security**: CORS configured for localhost development
- **File Uploads**: Validated file types and size limits

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable  
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Search existing GitHub issues
3. Create a new issue with detailed error information
4. Include system information (OS, hardware, Python/Node versions)

---

**Atlas MVP** - Built for MacBook Air M1, optimized for edge AI perception tasks.
