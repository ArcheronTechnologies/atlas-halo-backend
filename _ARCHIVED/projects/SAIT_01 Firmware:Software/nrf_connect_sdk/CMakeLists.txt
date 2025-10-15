# SAIT_01 Distributed Mesh CMake Configuration
# For nRF Connect SDK v3.1.0

cmake_minimum_required(VERSION 3.20.0)
find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(sait01_distributed_mesh)

# Source files for distributed autonomous sensor network
target_sources(app PRIVATE
    sait01_autonomous_main.c
    sait01_distributed_mesh.c
)

# Include directories
target_include_directories(app PRIVATE
    .
    ${ZEPHYR_BASE}/include
    ${ZEPHYR_BASE}/subsys/bluetooth/mesh
)

# Test sources (optional - enabled via CONFIG_ZTEST)
if(CONFIG_ZTEST)
    target_sources(app PRIVATE
        test_distributed_mesh.c
    )
endif()

# Additional libraries for enhanced functionality
if(CONFIG_MCUMGR)
    target_sources(app PRIVATE
        mcumgr_integration.c
    )
endif()

# TensorFlow Lite Micro integration (when available)
if(CONFIG_TENSORFLOW_LITE_MICRO)
    target_sources(app PRIVATE
        ml_integration.c
    )
endif()

# LoRa module integration
if(CONFIG_LORA)
    target_sources(app PRIVATE
        lora_fallback.c
    )
endif()

# Debug and profiling support
if(CONFIG_DEBUG)
    target_compile_definitions(app PRIVATE
        -DSAIT01_DEBUG_ENABLED=1
        -DSAIT01_VERBOSE_LOGGING=1
    )
endif()

# Optimization flags for production
if(CONFIG_SIZE_OPTIMIZATIONS)
    target_compile_options(app PRIVATE
        -Os
        -ffunction-sections
        -fdata-sections
    )
    target_link_options(app PRIVATE
        -Wl,--gc-sections
    )
endif()

# Memory layout optimizations
if(CONFIG_FLASH_SIZE_OPTIMIZATION)
    target_compile_definitions(app PRIVATE
        -DSAIT01_COMPACT_MESSAGES=1
        -DSAIT01_REDUCED_LOGGING=1
    )
endif()