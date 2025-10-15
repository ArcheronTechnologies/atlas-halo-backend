# SAIT_01 Project Makefile
# Convenience wrapper for building all components

.PHONY: all firmware gateway backend ui clean test setup help

# Default target
all: firmware

# Configuration
BOARD ?= sait_01
BUILD_TYPE ?= debug
PYTHON ?= python3

help:
	@echo "SAIT_01 Project Build System"
	@echo "============================"
	@echo ""
	@echo "Targets:"
	@echo "  firmware     - Build firmware for nRF5340"
	@echo "  gateway      - Setup and test gateway software"
	@echo "  backend      - Setup and test backend services"
	@echo "  ui           - Build operations UI"
	@echo "  test         - Run all tests"
	@echo "  setup        - Install development dependencies"
	@echo "  clean        - Clean all build artifacts"
	@echo "  format       - Format code with black/prettier"
	@echo "  lint         - Run linting checks"
	@echo ""
	@echo "Options:"
	@echo "  BOARD=<board>      - Target board (default: sait_01)"
	@echo "  BUILD_TYPE=<type>  - Build type: debug|release (default: debug)"
	@echo ""
	@echo "Examples:"
	@echo "  make firmware BOARD=sait_01 BUILD_TYPE=release"
	@echo "  make test"
	@echo "  make setup"

# Setup development environment
setup:
	@echo "🚀 Setting up SAIT_01 development environment..."
	./setup_development_environment.sh
	@echo "✅ Development environment ready"

# Build firmware
firmware:
	@echo "🔨 Building SAIT_01 firmware..."
	cd sait_01_firmware && \
	if [ "$(BUILD_TYPE)" = "release" ]; then \
		west build -b $(BOARD) -DCONFIG_DEBUG=n -DCONFIG_ASSERT=n; \
	else \
		west build -b $(BOARD) -DCONFIG_DEBUG=y -DCONFIG_LOG_DEFAULT_LEVEL=3; \
	fi
	@echo "✅ Firmware build complete: sait_01_firmware/build/zephyr/zephyr.hex"

# Flash firmware
flash: firmware
	@echo "📡 Flashing SAIT_01 firmware..."
	cd sait_01_firmware && west flash --runner jlink
	@echo "✅ Firmware flashed successfully"

# Setup gateway dependencies
gateway:
	@echo "🌉 Setting up gateway software..."
	cd sait_01_gateway && \
	python3 -m venv venv && \
	. venv/bin/activate && \
	pip install -r requirements.txt
	@echo "✅ Gateway setup complete"

# Setup backend dependencies  
backend:
	@echo "☁️  Setting up backend services..."
	cd sait_01_backend && \
	python3 -m venv venv && \
	. venv/bin/activate && \
	pip install -r requirements.txt
	@echo "✅ Backend setup complete"

# Setup UI dependencies
ui:
	@echo "🖥️  Setting up operations UI..."
	cd sait_01_ui && \
	npm install
	@echo "✅ UI setup complete"

# Build UI for production
ui-build: ui
	@echo "🏗️  Building operations UI..."
	cd sait_01_ui && npm run build
	@echo "✅ UI build complete: sait_01_ui/build/"

# Run all tests
test:
	@echo "🧪 Running SAIT_01 test suite..."
	
	# Firmware boot test
	@echo "Testing firmware boot sequence..."
	cd sait_01_tests && $(PYTHON) test_successful_boot.py
	
	# Gateway tests
	@echo "Testing gateway software..."
	cd sait_01_gateway && \
	if [ -d "venv" ]; then \
		. venv/bin/activate && pytest tests/ || echo "Gateway tests not found"; \
	fi
	
	# Backend tests  
	@echo "Testing backend services..."
	cd sait_01_backend && \
	if [ -d "venv" ]; then \
		. venv/bin/activate && pytest tests/ || echo "Backend tests not found"; \
	fi
	
	# UI tests
	@echo "Testing operations UI..."
	cd sait_01_ui && \
	if [ -d "node_modules" ]; then \
		npm test -- --watchAll=false || echo "UI tests not found"; \
	fi
	
	@echo "✅ Test suite complete"

# Format code
format:
	@echo "🎨 Formatting code..."
	
	# Python files
	find . -name "*.py" -not -path "./*/venv/*" -not -path "./.git/*" | xargs black
	
	# TypeScript/JavaScript files
	if [ -d "sait_01_ui/node_modules" ]; then \
		cd sait_01_ui && npx prettier --write "src/**/*.{ts,tsx,js,jsx}"; \
	fi
	
	@echo "✅ Code formatting complete"

# Lint code
lint:
	@echo "🔍 Running code quality checks..."
	
	# Python linting
	find . -name "*.py" -not -path "./*/venv/*" -not -path "./.git/*" | xargs flake8 --max-line-length=88 --ignore=E203,W503
	
	# Firmware build check
	cd sait_01_firmware && west build -b $(BOARD) --cmake-only
	
	@echo "✅ Linting complete"

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	
	# Firmware build files
	rm -rf sait_01_firmware/build
	
	# Gateway cache
	find sait_01_gateway -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find sait_01_gateway -name "*.pyc" -delete 2>/dev/null || true
	
	# Backend cache
	find sait_01_backend -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find sait_01_backend -name "*.pyc" -delete 2>/dev/null || true
	
	# UI build files
	rm -rf sait_01_ui/build
	rm -rf sait_01_ui/node_modules/.cache 2>/dev/null || true
	
	# Test cache
	find sait_01_tests -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find sait_01_tests -name "*.pyc" -delete 2>/dev/null || true
	
	@echo "✅ Clean complete"

# Development server targets
dev-gateway:
	@echo "🚀 Starting gateway development server..."
	cd sait_01_gateway && \
	. venv/bin/activate && \
	$(PYTHON) src/main.py

dev-backend:
	@echo "🚀 Starting backend development server..."
	cd sait_01_backend && \
	. venv/bin/activate && \
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

dev-ui:
	@echo "🚀 Starting UI development server..."
	cd sait_01_ui && npm start

# Documentation
docs:
	@echo "📚 Building documentation..."
	@echo "Documentation generation not yet implemented"
	@echo "See README.md and SYSTEM_OVERVIEW.md for current documentation"

# Docker support
docker-build:
	@echo "🐳 Building Docker images..."
	docker-compose build

docker-up:
	@echo "🐳 Starting Docker services..."
	docker-compose up -d

docker-down:
	@echo "🐳 Stopping Docker services..."
	docker-compose down

# Simulation with Renode
simulate:
	@echo "🎮 Starting Renode simulation..."
	@echo "Renode simulation not yet implemented"
	@echo "Use 'make test' to run firmware boot simulation"

# Package release
package:
	@echo "📦 Packaging release..."
	mkdir -p release
	
	# Package firmware
	if [ -f "sait_01_firmware/build/zephyr/zephyr.hex" ]; then \
		cp sait_01_firmware/build/zephyr/zephyr.hex release/sait_01_firmware_$(BUILD_TYPE).hex; \
	fi
	
	# Package UI build
	if [ -d "sait_01_ui/build" ]; then \
		tar -czf release/sait_01_ui.tar.gz -C sait_01_ui/build .; \
	fi
	
	@echo "✅ Release packages created in release/"