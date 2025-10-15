# SAIT_01 Distributed Mesh - Build Integration Guide

## Overview

This guide provides complete instructions for building and deploying the SAIT_01 distributed autonomous sensor network using Nordic's nRF Connect SDK v3.1.0 on nRF5340 hardware.

## Prerequisites

### Hardware Requirements
- **nRF5340 Development Kit** (nrf5340dk_nrf5340_cpuapp)
- **J-Link Debug Probe** (integrated on DK)
- **USB Cable** for programming and debug
- **Optional**: Custom SAIT_01 hardware with nRF5340

### Software Requirements
- **nRF Connect SDK v3.1.0 or later**
- **Zephyr RTOS** (included with nCS)
- **Nordic Command Line Tools** (nrfjprog, etc.)
- **ARM GNU Toolchain** (included with nCS)
- **Git** for version control

## Build System Setup

### 1. Install nRF Connect SDK

```bash
# Method 1: Using nRF Connect for Desktop (Recommended)
# Download and install nRF Connect for Desktop
# Use Toolchain Manager to install nCS v3.1.0

# Method 2: Manual Installation
cd ~/nrf
git clone https://github.com/nrfconnect/sdk-nrf.git
cd sdk-nrf
git checkout v3.1.0
west init -l .
west update
```

### 2. Environment Setup

```bash
# Set environment variables
export ZEPHYR_BASE=~/nrf/zephyr
export ZEPHYR_TOOLCHAIN_VARIANT=zephyr
export ZEPHYR_SDK_INSTALL_DIR=~/nrf/zephyr-sdk

# Source Zephyr environment
source $ZEPHYR_BASE/zephyr-env.sh

# Verify installation
west --version
zephyr-env.sh --version
```

### 3. Project Structure

```
sait01_distributed_mesh/
├── CMakeLists.txt                 # Main build configuration
├── prj.conf                       # Kconfig project options  
├── sait01_autonomous_main.c       # Main application
├── sait01_distributed_mesh.h      # Protocol definitions
├── sait01_distributed_mesh.c      # Protocol implementation
├── test_distributed_mesh.c        # Test suite
├── boards/                        # Board-specific configs
│   └── nrf5340dk_nrf5340_cpuapp.overlay
├── dts/                          # Device tree overlays
├── include/                      # Additional headers
└── README.md                     # Project documentation
```

## Building the Firmware

### 1. Basic Build

```bash
# Navigate to project directory
cd sait01_distributed_mesh

# Build for nRF5340 DK
west build -b nrf5340dk_nrf5340_cpuapp

# Clean build (if needed)
west build -t clean
west build -b nrf5340dk_nrf5340_cpuapp --pristine
```

### 2. Build Configurations

```bash
# Debug build with full logging
west build -b nrf5340dk_nrf5340_cpuapp -- -DCONFIG_LOG_DEFAULT_LEVEL=4

# Release build with optimizations
west build -b nrf5340dk_nrf5340_cpuapp -- -DCONFIG_SIZE_OPTIMIZATIONS=y

# Test build with Ztest framework
west build -b nrf5340dk_nrf5340_cpuapp -- -DCONFIG_ZTEST=y

# Production build with security features
west build -b nrf5340dk_nrf5340_cpuapp -- -DCONFIG_BT_MESH_USES_MBEDTLS_PSA=y
```

### 3. Custom Board Support

If using custom SAIT_01 hardware, create board definition:

```bash
# Create custom board directory
mkdir -p boards/arm/sait01_nrf5340

# Copy and modify board files from nRF5340 DK
cp -r $ZEPHYR_BASE/../nrf/boards/arm/nrf5340dk_nrf5340_cpuapp/* \
      boards/arm/sait01_nrf5340/

# Edit board configuration files for your hardware
vim boards/arm/sait01_nrf5340/sait01_nrf5340_cpuapp.dts
vim boards/arm/sait01_nrf5340/Kconfig.board

# Build for custom board
west build -b sait01_nrf5340_cpuapp
```

## Flashing and Debugging

### 1. Flash Firmware

```bash
# Flash via J-Link (DK integrated)
west flash

# Flash with specific runner
west flash --runner jlink

# Flash and reset
west flash --runner jlink --reset
```

### 2. Debug Session

```bash
# Start debug session
west debug

# Debug with specific debugger
west debug --runner jlink

# Debug with GDB server
JLinkGDBServer -device nRF5340_XXAA_APP -if SWD -speed 4000 -port 2331

# In another terminal
arm-none-eabi-gdb build/zephyr/zephyr.elf
(gdb) target remote :2331
(gdb) monitor reset
(gdb) load
(gdb) continue
```

### 3. Serial Console

```bash
# Connect to UART console
screen /dev/ttyUSB0 115200

# Or using minicom
minicom -D /dev/ttyUSB0 -b 115200

# Or using nRF Connect Terminal
# Available through nRF Connect for Desktop
```

## Testing and Validation

### 1. Unit Tests

```bash
# Build with test framework
west build -b nrf5340dk_nrf5340_cpuapp -- -DCONFIG_ZTEST=y

# Flash and run tests
west flash
# Tests will run automatically on boot

# Monitor test output
screen /dev/ttyUSB0 115200
```

### 2. Integration Testing

```bash
# Build multiple nodes for mesh testing
for i in {1..3}; do
    west build -b nrf5340dk_nrf5340_cpuapp --build-dir build_node$i \
              -- -DCONFIG_SAIT01_NODE_ID=$i
done

# Flash each node to separate hardware
west flash --build-dir build_node1
west flash --build-dir build_node2  
west flash --build-dir build_node3

# Monitor mesh communication
# Use nRF Connect Bluetooth Low Energy app to observe mesh traffic
```

### 3. Performance Testing

```bash
# Build with profiling enabled
west build -b nrf5340dk_nrf5340_cpuapp -- \
    -DCONFIG_THREAD_ANALYZER=y \
    -DCONFIG_THREAD_NAME=y \
    -DCONFIG_THREAD_RUNTIME_STATS=y

# Enable memory debugging
west build -b nrf5340dk_nrf5340_cpuapp -- \
    -DCONFIG_HEAP_MEM_POOL_SIZE=16384 \
    -DCONFIG_DEBUG_INFO=y

# Monitor performance via shell commands
# Connect to serial console and use:
# > kernel threads
# > kernel stacks  
# > kernel memory
```

## Configuration Options

### Key Kconfig Settings

```kconfig
# Core Bluetooth Mesh
CONFIG_BT_MESH=y
CONFIG_BT_MESH_RELAY=y
CONFIG_BT_MESH_FRIEND=y
CONFIG_BT_MESH_PROXY=y

# Custom Protocol Support
CONFIG_BT_MESH_VENDOR_MODELS=y
CONFIG_BT_MESH_MODEL_EXTENSIONS=y

# Memory and Performance
CONFIG_MAIN_STACK_SIZE=2048
CONFIG_HEAP_MEM_POOL_SIZE=8192
CONFIG_BT_MESH_MSG_CACHE_SIZE=32

# Security Features
CONFIG_BT_MESH_USES_MBEDTLS_PSA=y
CONFIG_TRUSTED_STORAGE=y

# Application Specific
CONFIG_SAIT01_NODE_CAPABILITIES=0x3F
CONFIG_SAIT01_CORRELATION_TIMEOUT_MS=5000
CONFIG_SAIT01_MIN_CONSENSUS_NODES=2
```

### Device Tree Overlays

Create `boards/nrf5340dk_nrf5340_cpuapp.overlay`:

```dts
/ {
    sait01_config {
        compatible = "sait01,sensor-config";
        label = "SAIT01_CONFIG";
        
        audio_pdm: pdm {
            compatible = "nordic,nrf-pdm";
            pinctrl-0 = <&pdm0_default>;
            status = "okay";
        };
        
        uw_spi: uwb_spi {
            compatible = "nordic,nrf-spim";
            status = "okay";
            cs-gpios = <&gpio0 3 GPIO_ACTIVE_LOW>;
        };
    };
};

&pdm0 {
    status = "okay";
    pinctrl-0 = <&pdm0_default>;
    pinctrl-names = "default";
};
```

## Deployment and Production

### 1. Production Build

```bash
# Production configuration
west build -b nrf5340dk_nrf5340_cpuapp -- \
    -DCONFIG_SIZE_OPTIMIZATIONS=y \
    -DCONFIG_COMPILER_OPT="-Os" \
    -DCONFIG_ASSERT=n \
    -DCONFIG_LOG_DEFAULT_LEVEL=2 \
    -DCONFIG_SHELL=n

# Generate production artifacts
west build -t zephyr_final.hex
west build -t zephyr_final.bin
```

### 2. OTA Update Preparation

```bash
# Build with MCUboot support for OTA
west build -b nrf5340dk_nrf5340_cpuapp -- \
    -DCONFIG_BOOTLOADER_MCUBOOT=y \
    -DCONFIG_MCUBOOT_SIGNATURE_KEY_FILE="root-rsa-2048.pem"

# Generate signed update image
imgtool sign --key root-rsa-2048.pem \
             --header-size 0x200 \
             --align 4 \
             --version 1.0.0 \
             --slot-size 0x40000 \
             build/zephyr/zephyr.bin \
             sait01_update_v1.0.0.bin
```

### 3. Factory Programming

```bash
# Create factory programming script
cat > factory_program.sh << 'EOF'
#!/bin/bash
# Factory programming script for SAIT_01 nodes

NODE_ID=$1
if [ -z "$NODE_ID" ]; then
    echo "Usage: $0 <node_id>"
    exit 1
fi

# Build firmware with node-specific configuration
west build -b nrf5340dk_nrf5340_cpuapp --build-dir build_node$NODE_ID \
          -- -DCONFIG_SAIT01_NODE_ID=$NODE_ID

# Flash firmware
west flash --build-dir build_node$NODE_ID

# Verify programming
nrfjprog --verify build_node$NODE_ID/zephyr/zephyr.hex

echo "Node $NODE_ID programmed successfully"
EOF

chmod +x factory_program.sh

# Program multiple nodes
./factory_program.sh 1
./factory_program.sh 2  
./factory_program.sh 3
```

## Troubleshooting

### Common Build Issues

```bash
# West update issues
west update --fetch=always

# CMake cache problems  
rm -rf build/
west build -b nrf5340dk_nrf5340_cpuapp --pristine

# Toolchain issues
west zephyr-export
pip3 install -r scripts/requirements.txt

# Permission issues (Linux/macOS)
sudo usermod -a -G dialout $USER  # Add user to dialout group
# Log out and back in
```

### Runtime Debugging

```bash
# Enable verbose mesh logging
CONFIG_BT_MESH_DEBUG=y
CONFIG_BT_MESH_DEBUG_MODEL=y
CONFIG_BT_MESH_DEBUG_ACCESS=y

# Memory debugging
CONFIG_DEBUG_INFO=y
CONFIG_THREAD_ANALYZER=y
CONFIG_KERNEL_SHELL=y

# Performance profiling
CONFIG_THREAD_RUNTIME_STATS=y
CONFIG_THREAD_NAME=y
```

### Mesh Network Issues

```bash
# Reset mesh provisioning data
shell> mesh reset

# Check mesh status
shell> mesh status

# Manual provisioning
shell> mesh provision <device_uuid> <net_key_idx> <addr> <attention_duration>

# Monitor mesh traffic using nRF Mesh mobile app
# Available on iOS and Android app stores
```

## Integration with nRF Connect SDK

### West Manifest Integration

To integrate into your nRF Connect SDK workspace, add to `west.yml`:

```yaml
manifest:
  projects:
    - name: sait01_distributed_mesh
      url: https://github.com/your_org/sait01_distributed_mesh.git
      revision: main
      path: applications/sait01_distributed_mesh
```

### Module Integration

Create `zephyr/module.yml` in project root:

```yaml
name: sait01_distributed_mesh
build:
  cmake: .
  kconfig: Kconfig
```

This completes the build integration guide. The SAIT_01 distributed mesh system is now ready for hardware deployment and field testing.