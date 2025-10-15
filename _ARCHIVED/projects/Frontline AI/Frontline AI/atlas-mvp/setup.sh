#!/bin/bash

# Atlas MVP Setup Script
# Optimized for MacBook Air M1

set -e

echo "🚀 Setting up Atlas MVP..."

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  This setup script is optimized for macOS. Proceed with caution on other systems."
fi

# Check for required tools
echo "🔍 Checking system requirements..."

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "   Install Python 3.9+ from https://python.org"
    exit 1
fi

if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed."
    echo "   Install Node.js 18+ from https://nodejs.org"
    exit 1
fi

# Check Python version
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
    PYTHON_VERSION=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    echo "❌ Python 3.9+ required. Found: $PYTHON_VERSION"
    exit 1
fi

# Check Node version  
NODE_VERSION=$(node -v | sed 's/v//' | cut -d. -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Node.js 18+ required. Found: $(node -v)"
    exit 1
fi

echo "✅ System requirements satisfied"

# Backend setup
echo "🐍 Setting up Python backend..."

cd backend

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "   Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "   Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
mkdir -p models uploads

echo "✅ Backend setup complete"

# Frontend setup
echo "⚛️  Setting up React frontend..."

cd ../frontend

# Install dependencies
echo "   Installing Node.js dependencies..."
npm install

echo "✅ Frontend setup complete"

cd ..

# Create startup scripts
echo "📝 Creating startup scripts..."

cat > start_backend.sh << 'EOF'
#!/bin/bash
cd backend
source venv/bin/activate
python -m app.main
EOF

cat > start_frontend.sh << 'EOF'
#!/bin/bash
cd frontend
npm start
EOF

cat > start_all.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Atlas MVP..."

# Start backend in background
echo "   Starting backend server..."
./start_backend.sh &
BACKEND_PID=$!

# Wait for backend to start
sleep 5

# Start frontend
echo "   Starting frontend..."
./start_frontend.sh &
FRONTEND_PID=$!

echo "✅ Atlas MVP is starting up!"
echo "   Backend: http://localhost:8000"
echo "   Frontend: http://localhost:3000"
echo ""
echo "   Press Ctrl+C to stop all services"

# Wait for user interrupt
trap 'kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit' INT
wait
EOF

# Make scripts executable
chmod +x start_backend.sh start_frontend.sh start_all.sh

# Camera permission check
echo "📷 Camera Permission Check"
echo "   Atlas requires camera access to function."
echo "   If prompted, please grant camera permissions in System Preferences."

# Final instructions
echo ""
echo "🎉 Setup Complete!"
echo ""
echo "Next steps:"
echo "1. Run './start_all.sh' to start both backend and frontend"
echo "2. Open http://localhost:3000 in your browser"
echo "3. Grant camera permissions when prompted"
echo "4. Click 'Start Stream' to begin real-time detection"
echo ""
echo "For manual startup:"
echo "- Backend only: './start_backend.sh'"
echo "- Frontend only: './start_frontend.sh'"
echo ""
echo "Documentation: https://github.com/your-repo/atlas-mvp"
echo ""
echo "Happy detecting! 🎯"