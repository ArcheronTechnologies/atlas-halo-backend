#!/bin/bash

# SAIT_01 Development Environment Setup
# Sets up Zephyr SDK, nRF Connect SDK, and all dependencies

set -e

echo "🚀 Setting up SAIT_01 Development Environment"
echo "=============================================="

# Detect OS
OS="Unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    OS="Windows"
fi

echo "🖥️  Detected OS: $OS"

# Function to install dependencies based on OS
install_system_dependencies() {
    echo "📦 Installing system dependencies..."
    
    case $OS in
        "Linux")
            # Ubuntu/Debian
            if command -v apt >/dev/null 2>&1; then
                sudo apt update
                sudo apt install -y --no-install-recommends \
                    git cmake ninja-build gperf ccache dfu-util \
                    device-tree-compiler wget python3-dev python3-pip \
                    python3-setuptools python3-tk python3-wheel xz-utils \
                    file make gcc gcc-multilib g++-multilib libsdl2-dev \
                    libnss3-dev libatk-bridge2.0-dev libdrm-dev \
                    libxcomposite-dev libxdamage-dev libxrandr-dev \
                    libgbm-dev libxss-dev libasound2-dev curl
            
            # Red Hat/CentOS/Fedora
            elif command -v dnf >/dev/null 2>&1; then
                sudo dnf install -y git cmake ninja-build gperf ccache \
                    dfu-util dtc wget python3-devel python3-pip \
                    python3-setuptools python3-tkinter xz file make \
                    gcc gcc-c++ glibc-devel.i686 libgcc.i686 \
                    SDL2-devel curl
            fi
            ;;
            
        "macOS")
            # macOS - use Homebrew
            if ! command -v brew >/dev/null 2>&1; then
                echo "🍺 Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            
            brew install cmake ninja gperf ccache dfu-util dtc python3 wget xz
            ;;
            
        *)
            echo "❌ Unsupported OS: $OS"
            echo "Please install dependencies manually:"
            echo "  - git, cmake, ninja-build, gperf, ccache"
            echo "  - dfu-util, device-tree-compiler, python3"
            echo "  - wget, xz-utils, file, make, gcc"
            exit 1
            ;;
    esac
    
    echo "✅ System dependencies installed"
}

# Install Python packages
install_python_dependencies() {
    echo "🐍 Installing Python dependencies..."
    
    # Upgrade pip
    python3 -m pip install --user --upgrade pip
    
    # Install West (Zephyr meta-tool)
    python3 -m pip install --user west
    
    # Install additional Python packages
    python3 -m pip install --user \
        pyserial \
        pyyaml \
        pykwalify \
        colorama \
        packaging \
        progress \
        psutil \
        kconfiglib \
        intelhex \
        cryptography
    
    echo "✅ Python dependencies installed"
}

# Setup Zephyr SDK
setup_zephyr_sdk() {
    echo "🔧 Setting up Zephyr SDK..."
    
    SDK_VERSION="0.16.4"
    ZEPHYR_SDK_INSTALL_DIR="$HOME/zephyr-sdk-$SDK_VERSION"
    
    # Determine SDK filename based on OS
    case $OS in
        "Linux")
            if [[ $(uname -m) == "x86_64" ]]; then
                SDK_FILE="zephyr-sdk-${SDK_VERSION}_linux-x86_64.tar.xz"
            else
                SDK_FILE="zephyr-sdk-${SDK_VERSION}_linux-aarch64.tar.xz"
            fi
            ;;
        "macOS")
            if [[ $(uname -m) == "arm64" ]]; then
                SDK_FILE="zephyr-sdk-${SDK_VERSION}_macos-aarch64.tar.xz"
            else
                SDK_FILE="zephyr-sdk-${SDK_VERSION}_macos-x86_64.tar.xz"
            fi
            ;;
        *)
            echo "❌ Unsupported OS for automatic SDK download"
            exit 1
            ;;
    esac
    
    # Download and extract SDK if not already present
    if [ ! -d "$ZEPHYR_SDK_INSTALL_DIR" ]; then
        echo "📥 Downloading Zephyr SDK $SDK_VERSION..."
        cd $HOME
        wget -q --show-progress "https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v${SDK_VERSION}/${SDK_FILE}"
        
        echo "📂 Extracting Zephyr SDK..."
        tar xf "$SDK_FILE"
        rm "$SDK_FILE"
        
        echo "🔧 Setting up Zephyr SDK..."
        cd "$ZEPHYR_SDK_INSTALL_DIR"
        ./setup.sh -t all -h -c
    else
        echo "✅ Zephyr SDK already installed at $ZEPHYR_SDK_INSTALL_DIR"
    fi
    
    # Add to environment
    echo "🌍 Setting up environment variables..."
    echo "export ZEPHYR_SDK_INSTALL_DIR=$ZEPHYR_SDK_INSTALL_DIR" >> ~/.bashrc
    echo "export PATH=\$PATH:$ZEPHYR_SDK_INSTALL_DIR" >> ~/.bashrc
    
    export ZEPHYR_SDK_INSTALL_DIR="$ZEPHYR_SDK_INSTALL_DIR"
    export PATH="$PATH:$ZEPHYR_SDK_INSTALL_DIR"
    
    echo "✅ Zephyr SDK setup complete"
}

# Setup nRF Connect SDK
setup_nrf_connect_sdk() {
    echo "📱 Setting up nRF Connect SDK..."
    
    NCS_VERSION="v2.5.0"
    WORKSPACE_DIR="$HOME/ncs"
    
    if [ ! -d "$WORKSPACE_DIR" ]; then
        echo "📥 Initializing nRF Connect SDK workspace..."
        mkdir -p "$WORKSPACE_DIR"
        cd "$WORKSPACE_DIR"
        
        west init -m https://github.com/nrfconnect/sdk-nrf --mr $NCS_VERSION
        west update
        west zephyr-export
        
        echo "🐍 Installing nRF Connect SDK Python requirements..."
        pip3 install --user -r zephyr/scripts/requirements.txt
        pip3 install --user -r nrf/scripts/requirements.txt
        pip3 install --user -r bootloader/mcuboot/scripts/requirements.txt
        
    else
        echo "✅ nRF Connect SDK already present at $WORKSPACE_DIR"
        cd "$WORKSPACE_DIR"
        
        echo "🔄 Updating nRF Connect SDK..."
        west update
    fi
    
    # Set up environment
    echo "🌍 Setting up nRF Connect SDK environment..."
    echo "export ZEPHYR_BASE=$WORKSPACE_DIR/zephyr" >> ~/.bashrc
    echo "export NCS_BASE=$WORKSPACE_DIR" >> ~/.bashrc
    
    export ZEPHYR_BASE="$WORKSPACE_DIR/zephyr"
    export NCS_BASE="$WORKSPACE_DIR"
    
    echo "✅ nRF Connect SDK setup complete"
}

# Setup SAIT_01 project
setup_sait01_project() {
    echo "🏗️  Setting up SAIT_01 project..."
    
    PROJECT_DIR=$(pwd)
    FIRMWARE_DIR="$PROJECT_DIR/sait_01_firmware"
    
    cd "$FIRMWARE_DIR"
    
    # Create west.yml for the project
    cat > west.yml << EOF
manifest:
  remotes:
    - name: nrfconnect
      url-base: https://github.com/nrfconnect
  projects:
    - name: sdk-nrf
      remote: nrfconnect
      path: nrf
      revision: v2.5.0
      import:
        - path: west.yml
        - path-prefix: applications
  self:
    path: sait_01_firmware
EOF

    # Create .west directory if it doesn't exist
    if [ ! -d ".west" ]; then
        echo "🔧 Initializing west workspace for SAIT_01..."
        west init --local .
    fi
    
    echo "✅ SAIT_01 project setup complete"
}

# Create build script
create_build_script() {
    echo "📜 Creating build scripts..."
    
    PROJECT_DIR=$(pwd)
    
    # Build script for firmware
    cat > build_firmware.sh << 'EOF'
#!/bin/bash

# SAIT_01 Firmware Build Script

set -e

BOARD="sait_01"
BUILD_TYPE="debug"
CLEAN_BUILD=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --board)
            BOARD="$2"
            shift 2
            ;;
        --release)
            BUILD_TYPE="release"
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--board BOARD] [--release] [--clean]"
            exit 1
            ;;
    esac
done

echo "🔨 Building SAIT_01 Firmware"
echo "Board: $BOARD"
echo "Build Type: $BUILD_TYPE"

cd sait_01_firmware

# Clean build if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo "🧹 Cleaning build directory..."
    rm -rf build
fi

# Set build configuration
if [ "$BUILD_TYPE" = "release" ]; then
    BUILD_ARGS="-DCONFIG_DEBUG=n -DCONFIG_ASSERT=n"
else
    BUILD_ARGS="-DCONFIG_DEBUG=y -DCONFIG_LOG_DEFAULT_LEVEL=3"
fi

# Build firmware
echo "🔨 Building firmware..."
west build -b $BOARD $BUILD_ARGS

echo "✅ Build complete!"
echo "📁 Firmware binary: sait_01_firmware/build/zephyr/zephyr.hex"
EOF
    
    chmod +x build_firmware.sh
    
    # Flash script
    cat > flash_firmware.sh << 'EOF'
#!/bin/bash

# SAIT_01 Firmware Flash Script

set -e

BOARD="sait_01"
RUNNER="jlink"

echo "📡 Flashing SAIT_01 Firmware"
echo "Board: $BOARD"

cd sait_01_firmware

if [ ! -f "build/zephyr/zephyr.hex" ]; then
    echo "❌ No firmware binary found. Run './build_firmware.sh' first."
    exit 1
fi

echo "🔥 Flashing firmware..."
west flash --runner $RUNNER

echo "✅ Firmware flashed successfully!"
EOF
    
    chmod +x flash_firmware.sh
    
    echo "✅ Build scripts created"
}

# Create development tools setup
setup_development_tools() {
    echo "🛠️  Setting up development tools..."
    
    # Install additional development tools
    python3 -m pip install --user \
        black \
        pylint \
        mypy \
        pytest \
        pre-commit
    
    # Setup pre-commit hooks
    if [ -f ".pre-commit-config.yaml" ]; then
        pre-commit install
        echo "✅ Pre-commit hooks installed"
    fi
    
    echo "✅ Development tools setup complete"
}

# Main installation sequence
main() {
    echo "🚀 Starting SAIT_01 development environment setup..."
    
    # Check if running with necessary permissions
    if [[ $EUID -eq 0 ]]; then
        echo "❌ Don't run this script as root (except for system package installation)"
        exit 1
    fi
    
    # Install system dependencies
    install_system_dependencies
    
    # Install Python dependencies
    install_python_dependencies
    
    # Setup Zephyr SDK
    setup_zephyr_sdk
    
    # Setup nRF Connect SDK
    setup_nrf_connect_sdk
    
    # Setup SAIT_01 project
    setup_sait01_project
    
    # Create build scripts
    create_build_script
    
    # Setup development tools
    setup_development_tools
    
    echo ""
    echo "🎉 SAIT_01 Development Environment Setup Complete!"
    echo "=============================================="
    echo ""
    echo "📋 Next Steps:"
    echo "1. Source your shell profile: source ~/.bashrc"
    echo "2. Build firmware: ./build_firmware.sh"
    echo "3. Flash to hardware: ./flash_firmware.sh"
    echo ""
    echo "📁 Important Directories:"
    echo "  - nRF Connect SDK: $HOME/ncs"
    echo "  - Zephyr SDK: $ZEPHYR_SDK_INSTALL_DIR"
    echo "  - Project Root: $(pwd)"
    echo ""
    echo "🔧 Available Commands:"
    echo "  - west build -b sait_01          # Build firmware"
    echo "  - west flash                     # Flash firmware"
    echo "  - west debug                     # Debug with GDB"
    echo "  - west boards | grep sait        # List available boards"
    echo ""
    echo "📖 Documentation:"
    echo "  - Zephyr: https://docs.zephyrproject.org"
    echo "  - nRF Connect SDK: https://developer.nordicsemi.com"
    echo ""
}

# Run main function
main "$@"