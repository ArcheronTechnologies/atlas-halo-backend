# SAIT_01 Defense Sensor Node

Complete firmware + software stack for SAIT_01 defense sensor node product.

## Project Structure

- `sait_01_firmware/` - Zephyr/NCS firmware for nRF5340
- `sait_01_gateway/` - Linux SBC gateway software  
- `sait_01_backend/` - Cloud services and APIs
- `sait_01_ui/` - Operations web interface
- `sait_01_protocols/` - Protocol definitions and schemas
- `sait_01_renode/` - Renode simulation configs
- `sait_01_tests/` - Integration and system tests

## Hardware Components

- MCU: Nordic nRF5340 (dual Cortex-M33)
- Wideband RF: ADRV9002BBCZBC-196-16
- RF Switch: SKY13453-385LF (SP4T)
- LoRa: Murata Type-1SJ
- UWB: NXP Trimension NCJ29D6B
- Crypto: ATECC608A
- Flash: MX25R8035F (QSPI NOR)
- IMU: ADXL362
- Audio: PDM microphone
- Tamper: EVQ-P7C01P switch
- Power: TPS62740 buck, TPS22860 load switch
- Supervisor: APX803

## Key Features

- TinyML audio classification with 64-mel spectrograms
- BLE Mesh with adaptive TX policy
- LoRa fallback communication
- UWB ranging (TWR/TDoA)
- ADRV9002 proxy sensing
- Hardware security with ATECC608A
- Power-optimized operation (<300µA idle)
- Secure OTA updates
- Multi-node fusion and tracking

## Development

See individual README files in each subdirectory for specific build and deployment instructions.