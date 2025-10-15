#!/bin/bash

# SAIT_01 Setup Validation Script
# Validates that all development dependencies are properly installed

set -e

echo "🔍 SAIT_01 Development Environment Validation"
echo "============================================="

ERRORS=0
WARNINGS=0

# Function to check command availability
check_command() {
    local cmd=$1
    local description=$2
    local required=${3:-true}
    
    if command -v "$cmd" >/dev/null 2>&1; then
        local version=""
        case $cmd in
            "west") version=$(west --version 2>/dev/null | head -n1 || echo "unknown") ;;
            "cmake") version=$(cmake --version 2>/dev/null | head -n1 || echo "unknown") ;;
            "python3") version=$(python3 --version 2>/dev/null || echo "unknown") ;;
            "node") version=$(node --version 2>/dev/null || echo "unknown") ;;
            "docker") version=$(docker --version 2>/dev/null || echo "unknown") ;;
            *) version="installed" ;;
        esac
        echo "✅ $description: $version"
    else
        if [ "$required" = true ]; then
            echo "❌ $description: NOT FOUND (required)"
            ((ERRORS++))
        else
            echo "⚠️  $description: NOT FOUND (optional)"
            ((WARNINGS++))
        fi
    fi
}

# Function to check directory
check_directory() {
    local dir=$1
    local description=$2
    local required=${3:-true}
    
    if [ -d "$dir" ]; then
        echo "✅ $description: $dir"
    else
        if [ "$required" = true ]; then
            echo "❌ $description: NOT FOUND (required)"
            ((ERRORS++))
        else
            echo "⚠️  $description: NOT FOUND (optional)"
            ((WARNINGS++))
        fi
    fi
}

# Function to check file
check_file() {
    local file=$1
    local description=$2
    local required=${3:-true}
    
    if [ -f "$file" ]; then
        echo "✅ $description: $file"
    else
        if [ "$required" = true ]; then
            echo "❌ $description: NOT FOUND (required)"
            ((ERRORS++))
        else
            echo "⚠️  $description: NOT FOUND (optional)"
            ((WARNINGS++))
        fi
    fi
}

echo ""
echo "🔧 Core Development Tools"
echo "========================="
check_command "git" "Git VCS"
check_command "cmake" "CMake build system"
check_command "ninja" "Ninja build tool"
check_command "python3" "Python 3"
check_command "west" "West meta-tool"

echo ""
echo "🏗️  Build Dependencies"
echo "===================="
check_command "gcc" "GCC compiler"
check_command "gperf" "GNU gperf"
check_command "ccache" "Compiler cache" false
check_command "dfu-util" "DFU utility" false
check_command "dtc" "Device tree compiler"

echo ""
echo "📁 Project Structure"
echo "==================="
check_directory "sait_01_firmware" "Firmware directory"
check_directory "sait_01_gateway" "Gateway directory"  
check_directory "sait_01_backend" "Backend directory"
check_directory "sait_01_ui" "UI directory"
check_directory "sait_01_protocols" "Protocols directory"
check_directory "sait_01_tests" "Tests directory"

echo ""
echo "📄 Configuration Files"
echo "======================"
check_file "sait_01_firmware/west.yml" "West manifest"
check_file "sait_01_firmware/prj.conf" "Firmware project config"
check_file "sait_01_firmware/CMakeLists.txt" "Firmware CMake config"
check_file "requirements-dev.txt" "Development requirements"
check_file "docker-compose.yml" "Docker composition"
check_file "Makefile" "Build system"

echo ""
echo "🛠️  Zephyr SDK"
echo "==============="
if [ -n "$ZEPHYR_SDK_INSTALL_DIR" ] && [ -d "$ZEPHYR_SDK_INSTALL_DIR" ]; then
    echo "✅ Zephyr SDK: $ZEPHYR_SDK_INSTALL_DIR"
else
    echo "⚠️  Zephyr SDK: Environment variable not set or directory missing"
    echo "   Run './setup_development_environment.sh' to install"
    ((WARNINGS++))
fi

if [ -n "$ZEPHYR_BASE" ] && [ -d "$ZEPHYR_BASE" ]; then
    echo "✅ Zephyr base: $ZEPHYR_BASE"
else
    echo "⚠️  Zephyr base: Environment variable not set or directory missing"
    ((WARNINGS++))
fi

echo ""
echo "🐍 Python Environment"
echo "===================="
check_command "pip3" "Python package manager"

# Check Python packages
python_packages=("west" "pyserial" "pyyaml" "intelhex" "cryptography")
for package in "${python_packages[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        echo "✅ Python package '$package': installed"
    else
        echo "⚠️  Python package '$package': not installed"
        ((WARNINGS++))
    fi
done

echo ""
echo "🌐 Web Development"
echo "=================="
check_command "node" "Node.js" false
check_command "npm" "NPM package manager" false

if [ -d "sait_01_ui/node_modules" ]; then
    echo "✅ UI dependencies: installed"
else
    echo "⚠️  UI dependencies: not installed (run 'make ui')"
    ((WARNINGS++))
fi

echo ""
echo "🐳 Container Support"  
echo "==================="
check_command "docker" "Docker" false
check_command "docker-compose" "Docker Compose" false

echo ""
echo "🔬 Testing Framework"
echo "===================="
if [ -f "sait_01_tests/test_successful_boot.py" ]; then
    echo "✅ Boot test: available"
    echo "   Test command: cd sait_01_tests && python3 test_successful_boot.py"
else
    echo "❌ Boot test: missing"
    ((ERRORS++))
fi

echo ""
echo "📋 Build System Validation"
echo "=========================="

# Test make targets
if command -v make >/dev/null 2>&1; then
    echo "✅ Make: available"
    echo "   Available targets: firmware, gateway, backend, ui, test, clean"
else
    echo "❌ Make: not available"
    ((ERRORS++))
fi

# Test firmware build prerequisites  
if [ -f "sait_01_firmware/boards/arm/sait_01/sait_01.dts" ]; then
    echo "✅ Board definition: sait_01"
else
    echo "❌ Board definition: missing"
    ((ERRORS++))
fi

echo ""
echo "🎯 Ready to Build Commands"
echo "=========================="
echo ""
echo "Development setup:"
echo "  ./setup_development_environment.sh  # Install Zephyr SDK"
echo "  source ~/.bashrc                    # Load environment"
echo ""
echo "Build commands:"
echo "  make firmware                       # Build nRF5340 firmware"
echo "  make gateway                        # Setup gateway dependencies"
echo "  make backend                        # Setup backend dependencies"
echo "  make ui                             # Setup UI dependencies"
echo ""
echo "Test commands:"
echo "  make test                          # Run all tests"
echo "  cd sait_01_tests && python3 test_successful_boot.py  # Boot test"
echo ""
echo "Docker deployment:"
echo "  docker-compose up -d               # Start all services"
echo "  docker-compose logs -f             # View logs"
echo ""

echo "========================================="
echo "🏁 Validation Summary"
echo "========================================="

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo "🎉 PERFECT! All dependencies are installed and configured correctly."
    echo "✅ SAIT_01 development environment is ready for use."
elif [ $ERRORS -eq 0 ]; then
    echo "✅ GOOD! Core dependencies are installed."
    echo "⚠️  $WARNINGS optional dependencies are missing but development can proceed."
    echo "💡 Run './setup_development_environment.sh' to install missing components."
else
    echo "❌ ISSUES FOUND! $ERRORS critical dependencies are missing."
    echo "⚠️  $WARNINGS warnings detected."
    echo "🚨 Please install missing dependencies before proceeding."
    echo "🔧 Run './setup_development_environment.sh' to resolve issues."
fi

echo ""
echo "For complete setup, run:"
echo "  ./setup_development_environment.sh"
echo ""

exit $([ $ERRORS -eq 0 ] && echo 0 || echo 1)