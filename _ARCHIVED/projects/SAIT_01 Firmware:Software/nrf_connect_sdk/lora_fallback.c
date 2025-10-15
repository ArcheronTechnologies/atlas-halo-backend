/*
 * SAIT_01 LoRa Fallback Communication Protocol
 * nRF5340 + SX1276 LoRa Module Integration
 * Long-Range Mesh Backup Communication
 */

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/spi.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/byteorder.h>
#include <zephyr/random/rand32.h>

LOG_MODULE_REGISTER(lora_fallback, LOG_LEVEL_INF);

/* SX1276 LoRa Register Definitions */
#define SX1276_REG_FIFO                 0x00
#define SX1276_REG_OP_MODE             0x01
#define SX1276_REG_FRF_MSB             0x06
#define SX1276_REG_FRF_MID             0x07
#define SX1276_REG_FRF_LSB             0x08
#define SX1276_REG_PA_CONFIG           0x09
#define SX1276_REG_PA_RAMP             0x0A
#define SX1276_REG_LR_OCP              0x0B
#define SX1276_REG_LNA                 0x0C
#define SX1276_REG_LR_FIFO_ADDR_PTR    0x0D
#define SX1276_REG_LR_FIFO_TX_BASE_ADDR 0x0E
#define SX1276_REG_LR_FIFO_RX_BASE_ADDR 0x0F
#define SX1276_REG_LR_FIFO_RX_CURRENT_ADDR 0x10
#define SX1276_REG_LR_IRQ_FLAGS_MASK   0x11
#define SX1276_REG_LR_IRQ_FLAGS        0x12
#define SX1276_REG_LR_RX_NB_BYTES      0x13
#define SX1276_REG_LR_MODEM_CONFIG1    0x1D
#define SX1276_REG_LR_MODEM_CONFIG2    0x1E
#define SX1276_REG_LR_SYMB_TIMEOUT_LSB 0x1F
#define SX1276_REG_LR_PREAMBLE_MSB     0x20
#define SX1276_REG_LR_PREAMBLE_LSB     0x21
#define SX1276_REG_LR_PAYLOAD_LENGTH   0x22
#define SX1276_REG_LR_MODEM_CONFIG3    0x26
#define SX1276_REG_LR_RSSI_VALUE       0x1B
#define SX1276_REG_LR_PKT_SNR_VALUE    0x19
#define SX1276_REG_VERSION             0x42
#define SX1276_REG_DIO_MAPPING1        0x40

/* Operating Modes */
#define SX1276_MODE_SLEEP              0x00
#define SX1276_MODE_STANDBY            0x01
#define SX1276_MODE_TX                 0x03
#define SX1276_MODE_RX_CONTINUOUS      0x05
#define SX1276_MODE_RX_SINGLE          0x06
#define SX1276_LORA_MODE               0x80

/* IRQ Flags */
#define SX1276_IRQ_TX_DONE             0x08
#define SX1276_IRQ_RX_DONE             0x40
#define SX1276_IRQ_RX_TIMEOUT          0x80

/* SAIT_01 LoRa Protocol */
#define SAIT01_LORA_FREQUENCY          868100000  // 868.1 MHz (EU868)
#define SAIT01_LORA_TX_POWER           14         // 14 dBm
#define SAIT01_LORA_SPREADING_FACTOR   7          // SF7 for good range/speed balance
#define SAIT01_LORA_BANDWIDTH          125000     // 125 kHz
#define SAIT01_LORA_CODING_RATE        5          // 4/5 coding rate
#define SAIT01_LORA_PREAMBLE_LENGTH    8          // 8 symbol preamble
#define SAIT01_LORA_MAX_PAYLOAD        128        // Maximum payload size

/* SAIT_01 LoRa Message Types */
#define SAIT01_MSG_EMERGENCY           0x01       // Emergency threat detection
#define SAIT01_MSG_STATUS              0x02       // Node status update
#define SAIT01_MSG_MESH_SYNC           0x03       // Mesh synchronization
#define SAIT01_MSG_COORDINATOR_BEACON  0x04       // Coordinator beacon
#define SAIT01_MSG_ACK                 0x05       // Acknowledgment

/* LoRa Hardware Configuration */
struct sait01_lora_config {
    const struct device *spi_dev;
    const struct device *reset_gpio;
    const struct device *dio0_gpio;
    uint32_t reset_pin;
    uint32_t dio0_pin;
    uint32_t frequency;
    uint8_t tx_power;
    uint8_t spreading_factor;
    uint32_t bandwidth;
    uint8_t coding_rate;
    uint16_t preamble_length;
};

/* LoRa Message Structure */
struct sait01_lora_message {
    uint8_t message_type;
    uint8_t source_node_id[4];
    uint8_t destination_node_id[4];
    uint8_t sequence_number;
    uint8_t ttl;
    uint16_t payload_length;
    uint8_t payload[SAIT01_LORA_MAX_PAYLOAD];
    uint16_t crc;
} __packed;

/* LoRa State Management */
struct sait01_lora_state {
    struct sait01_lora_config config;
    bool initialized;
    bool receiving;
    uint8_t sequence_counter;
    uint8_t node_id[4];
    int16_t last_rssi;
    int8_t last_snr;
    uint32_t messages_sent;
    uint32_t messages_received;
    uint32_t transmission_errors;
};

static struct sait01_lora_state lora_state;

/* SPI Configuration */
static struct spi_config spi_cfg = {
    .frequency = 1000000,  // 1 MHz
    .operation = SPI_WORD_SET(8) | SPI_TRANSFER_MSB,
    .slave = 0,
    .cs = NULL,
};

/* Forward Declarations */
static int sait01_lora_write_reg(uint8_t reg, uint8_t value);
static int sait01_lora_read_reg(uint8_t reg, uint8_t *value);
static int sait01_lora_reset(void);
static int sait01_lora_set_frequency(uint32_t frequency);
static int sait01_lora_set_tx_power(uint8_t power);
static int sait01_lora_set_spreading_factor(uint8_t sf);
static int sait01_lora_set_bandwidth(uint32_t bw);
static int sait01_lora_set_coding_rate(uint8_t cr);

/* Write register to SX1276 */
static int sait01_lora_write_reg(uint8_t reg, uint8_t value)
{
    uint8_t tx_data[2] = {reg | 0x80, value};  // Set MSB for write
    struct spi_buf tx_buf = {.buf = tx_data, .len = 2};
    struct spi_buf_set tx_set = {.buffers = &tx_buf, .count = 1};
    
    return spi_write(lora_state.config.spi_dev, &spi_cfg, &tx_set);
}

/* Read register from SX1276 */
static int sait01_lora_read_reg(uint8_t reg, uint8_t *value)
{
    uint8_t tx_data[2] = {reg & 0x7F, 0x00};  // Clear MSB for read
    uint8_t rx_data[2];
    
    struct spi_buf tx_buf = {.buf = tx_data, .len = 2};
    struct spi_buf rx_buf = {.buf = rx_data, .len = 2};
    struct spi_buf_set tx_set = {.buffers = &tx_buf, .count = 1};
    struct spi_buf_set rx_set = {.buffers = &rx_buf, .count = 1};
    
    int ret = spi_transceive(lora_state.config.spi_dev, &spi_cfg, &tx_set, &rx_set);
    if (ret == 0) {
        *value = rx_data[1];
    }
    return ret;
}

/* Reset LoRa module */
static int sait01_lora_reset(void)
{
    LOG_INF("Resetting LoRa module");
    
    // Pull reset low for 1ms
    gpio_pin_set(lora_state.config.reset_gpio, lora_state.config.reset_pin, 0);
    k_sleep(K_MSEC(1));
    
    // Release reset
    gpio_pin_set(lora_state.config.reset_gpio, lora_state.config.reset_pin, 1);
    k_sleep(K_MSEC(6));  // Wait for module to initialize
    
    // Verify module is responding
    uint8_t version;
    int ret = sait01_lora_read_reg(SX1276_REG_VERSION, &version);
    if (ret != 0 || version != 0x12) {
        LOG_ERR("LoRa module not responding (version: 0x%02x)", version);
        return -1;
    }
    
    LOG_INF("LoRa module reset complete (version: 0x%02x)", version);
    return 0;
}

/* Set LoRa frequency */
static int sait01_lora_set_frequency(uint32_t frequency)
{
    uint64_t frf = ((uint64_t)frequency << 19) / 32000000;
    
    int ret = 0;
    ret |= sait01_lora_write_reg(SX1276_REG_FRF_MSB, (uint8_t)(frf >> 16));
    ret |= sait01_lora_write_reg(SX1276_REG_FRF_MID, (uint8_t)(frf >> 8));
    ret |= sait01_lora_write_reg(SX1276_REG_FRF_LSB, (uint8_t)frf);
    
    LOG_DBG("Set frequency: %u Hz (FRF: 0x%06x)", frequency, (uint32_t)frf);
    return ret;
}

/* Set TX power */
static int sait01_lora_set_tx_power(uint8_t power)
{
    uint8_t pa_config;
    
    if (power > 17) {
        // Use PA_BOOST pin for +20dBm
        pa_config = 0x80 | (power - 5);
    } else if (power > 14) {
        // Use PA_BOOST pin for +17dBm
        pa_config = 0x80 | (power - 2);
    } else {
        // Use RFO pin for +14dBm max
        pa_config = 0x70 | power;
    }
    
    LOG_DBG("Set TX power: %u dBm (PA_CONFIG: 0x%02x)", power, pa_config);
    return sait01_lora_write_reg(SX1276_REG_PA_CONFIG, pa_config);
}

/* Set spreading factor */
static int sait01_lora_set_spreading_factor(uint8_t sf)
{
    if (sf < 6 || sf > 12) {
        LOG_ERR("Invalid spreading factor: %u", sf);
        return -1;
    }
    
    uint8_t modem_config2;
    int ret = sait01_lora_read_reg(SX1276_REG_LR_MODEM_CONFIG2, &modem_config2);
    if (ret != 0) return ret;
    
    modem_config2 = (modem_config2 & 0x0F) | (sf << 4);
    ret = sait01_lora_write_reg(SX1276_REG_LR_MODEM_CONFIG2, modem_config2);
    
    LOG_DBG("Set spreading factor: SF%u", sf);
    return ret;
}

/* Set bandwidth */
static int sait01_lora_set_bandwidth(uint32_t bw)
{
    uint8_t bw_val;
    
    switch (bw) {
        case 7800:   bw_val = 0; break;
        case 10400:  bw_val = 1; break;
        case 15600:  bw_val = 2; break;
        case 20800:  bw_val = 3; break;
        case 31250:  bw_val = 4; break;
        case 41700:  bw_val = 5; break;
        case 62500:  bw_val = 6; break;
        case 125000: bw_val = 7; break;
        case 250000: bw_val = 8; break;
        case 500000: bw_val = 9; break;
        default:
            LOG_ERR("Invalid bandwidth: %u Hz", bw);
            return -1;
    }
    
    uint8_t modem_config1;
    int ret = sait01_lora_read_reg(SX1276_REG_LR_MODEM_CONFIG1, &modem_config1);
    if (ret != 0) return ret;
    
    modem_config1 = (modem_config1 & 0x0F) | (bw_val << 4);
    ret = sait01_lora_write_reg(SX1276_REG_LR_MODEM_CONFIG1, modem_config1);
    
    LOG_DBG("Set bandwidth: %u Hz", bw);
    return ret;
}

/* Set coding rate */
static int sait01_lora_set_coding_rate(uint8_t cr)
{
    if (cr < 5 || cr > 8) {
        LOG_ERR("Invalid coding rate: 4/%u", cr);
        return -1;
    }
    
    uint8_t modem_config1;
    int ret = sait01_lora_read_reg(SX1276_REG_LR_MODEM_CONFIG1, &modem_config1);
    if (ret != 0) return ret;
    
    modem_config1 = (modem_config1 & 0xF1) | ((cr - 4) << 1);
    ret = sait01_lora_write_reg(SX1276_REG_LR_MODEM_CONFIG1, modem_config1);
    
    LOG_DBG("Set coding rate: 4/%u", cr);
    return ret;
}

/* Initialize LoRa module */
int sait01_lora_init(void)
{
    LOG_INF("Initializing SAIT_01 LoRa Fallback Communication");
    
    // Get device references
    lora_state.config.spi_dev = DEVICE_DT_GET(DT_NODELABEL(spi2));
    lora_state.config.reset_gpio = DEVICE_DT_GET(DT_NODELABEL(gpio0));
    lora_state.config.dio0_gpio = DEVICE_DT_GET(DT_NODELABEL(gpio0));
    
    if (!device_is_ready(lora_state.config.spi_dev)) {
        LOG_ERR("SPI device not ready");
        return -1;
    }
    
    if (!device_is_ready(lora_state.config.reset_gpio)) {
        LOG_ERR("Reset GPIO device not ready");
        return -1;
    }
    
    // Configure GPIO pins
    lora_state.config.reset_pin = 15;  // P0.15
    lora_state.config.dio0_pin = 16;   // P0.16
    
    gpio_pin_configure(lora_state.config.reset_gpio, lora_state.config.reset_pin,
                      GPIO_OUTPUT_HIGH);
    gpio_pin_configure(lora_state.config.dio0_gpio, lora_state.config.dio0_pin,
                      GPIO_INPUT);
    
    // Initialize configuration
    lora_state.config.frequency = SAIT01_LORA_FREQUENCY;
    lora_state.config.tx_power = SAIT01_LORA_TX_POWER;
    lora_state.config.spreading_factor = SAIT01_LORA_SPREADING_FACTOR;
    lora_state.config.bandwidth = SAIT01_LORA_BANDWIDTH;
    lora_state.config.coding_rate = SAIT01_LORA_CODING_RATE;
    lora_state.config.preamble_length = SAIT01_LORA_PREAMBLE_LENGTH;
    
    // Generate random node ID
    uint32_t random = sys_rand32_get();
    memcpy(lora_state.node_id, &random, 4);
    
    LOG_INF("Node ID: %02x%02x%02x%02x",
            lora_state.node_id[0], lora_state.node_id[1],
            lora_state.node_id[2], lora_state.node_id[3]);
    
    // Reset and configure LoRa module
    int ret = sait01_lora_reset();
    if (ret != 0) {
        return ret;
    }
    
    // Enter sleep mode
    ret = sait01_lora_write_reg(SX1276_REG_OP_MODE, SX1276_MODE_SLEEP);
    if (ret != 0) return ret;
    
    k_sleep(K_MSEC(10));
    
    // Enable LoRa mode
    ret = sait01_lora_write_reg(SX1276_REG_OP_MODE, SX1276_LORA_MODE | SX1276_MODE_STANDBY);
    if (ret != 0) return ret;
    
    // Configure LoRa parameters
    ret |= sait01_lora_set_frequency(lora_state.config.frequency);
    ret |= sait01_lora_set_tx_power(lora_state.config.tx_power);
    ret |= sait01_lora_set_spreading_factor(lora_state.config.spreading_factor);
    ret |= sait01_lora_set_bandwidth(lora_state.config.bandwidth);
    ret |= sait01_lora_set_coding_rate(lora_state.config.coding_rate);
    
    if (ret != 0) {
        LOG_ERR("Failed to configure LoRa parameters");
        return ret;
    }
    
    // Set preamble length
    ret |= sait01_lora_write_reg(SX1276_REG_LR_PREAMBLE_MSB, 
                                lora_state.config.preamble_length >> 8);
    ret |= sait01_lora_write_reg(SX1276_REG_LR_PREAMBLE_LSB, 
                                lora_state.config.preamble_length & 0xFF);
    
    // Configure FIFO
    ret |= sait01_lora_write_reg(SX1276_REG_LR_FIFO_TX_BASE_ADDR, 0x00);
    ret |= sait01_lora_write_reg(SX1276_REG_LR_FIFO_RX_BASE_ADDR, 0x00);
    
    // Enable all interrupts on DIO0
    ret |= sait01_lora_write_reg(SX1276_REG_DIO_MAPPING1, 0x00);
    
    if (ret != 0) {
        LOG_ERR("Failed to configure LoRa FIFO and interrupts");
        return ret;
    }
    
    lora_state.initialized = true;
    lora_state.receiving = false;
    lora_state.sequence_counter = 0;
    
    LOG_INF("LoRa fallback communication initialized");
    LOG_INF("Frequency: %u Hz, TX Power: %u dBm, SF: %u",
            lora_state.config.frequency,
            lora_state.config.tx_power,
            lora_state.config.spreading_factor);
    
    return 0;
}

/* Transmit LoRa message */
int sait01_lora_transmit(uint8_t msg_type, const uint8_t *dest_id, 
                        const uint8_t *payload, uint16_t payload_len)
{
    if (!lora_state.initialized) {
        LOG_ERR("LoRa not initialized");
        return -1;
    }
    
    if (payload_len > SAIT01_LORA_MAX_PAYLOAD) {
        LOG_ERR("Payload too large: %u bytes", payload_len);
        return -1;
    }
    
    // Prepare message
    struct sait01_lora_message msg = {0};
    msg.message_type = msg_type;
    memcpy(msg.source_node_id, lora_state.node_id, 4);
    if (dest_id) {
        memcpy(msg.destination_node_id, dest_id, 4);
    } else {
        memset(msg.destination_node_id, 0xFF, 4);  // Broadcast
    }
    msg.sequence_number = lora_state.sequence_counter++;
    msg.ttl = 3;  // Maximum 3 hops
    msg.payload_length = payload_len;
    if (payload && payload_len > 0) {
        memcpy(msg.payload, payload, payload_len);
    }
    
    // Calculate CRC (simple checksum for now)
    uint16_t crc = 0;
    uint8_t *msg_bytes = (uint8_t *)&msg;
    for (int i = 0; i < sizeof(msg) - 2; i++) {
        crc += msg_bytes[i];
    }
    msg.crc = crc;
    
    LOG_INF("Transmitting LoRa message type %u (%u bytes)", msg_type, payload_len);
    
    // Enter standby mode
    int ret = sait01_lora_write_reg(SX1276_REG_OP_MODE, SX1276_LORA_MODE | SX1276_MODE_STANDBY);
    if (ret != 0) return ret;
    
    // Reset FIFO pointer
    ret = sait01_lora_write_reg(SX1276_REG_LR_FIFO_ADDR_PTR, 0x00);
    if (ret != 0) return ret;
    
    // Write message to FIFO
    uint16_t msg_len = sizeof(msg) - SAIT01_LORA_MAX_PAYLOAD + payload_len;
    ret = sait01_lora_write_reg(SX1276_REG_LR_PAYLOAD_LENGTH, msg_len);
    if (ret != 0) return ret;
    
    // Write message data
    uint8_t *data = (uint8_t *)&msg;
    for (uint16_t i = 0; i < msg_len; i++) {
        ret = sait01_lora_write_reg(SX1276_REG_FIFO, data[i]);
        if (ret != 0) return ret;
    }
    
    // Start transmission
    ret = sait01_lora_write_reg(SX1276_REG_OP_MODE, SX1276_LORA_MODE | SX1276_MODE_TX);
    if (ret != 0) return ret;
    
    // Wait for transmission complete (poll DIO0 or IRQ register)
    uint8_t irq_flags;
    int timeout_ms = 5000;  // 5 second timeout
    while (timeout_ms > 0) {
        ret = sait01_lora_read_reg(SX1276_REG_LR_IRQ_FLAGS, &irq_flags);
        if (ret != 0) return ret;
        
        if (irq_flags & SX1276_IRQ_TX_DONE) {
            // Clear interrupt flag
            sait01_lora_write_reg(SX1276_REG_LR_IRQ_FLAGS, SX1276_IRQ_TX_DONE);
            break;
        }
        
        k_sleep(K_MSEC(10));
        timeout_ms -= 10;
    }
    
    if (timeout_ms <= 0) {
        LOG_ERR("LoRa transmission timeout");
        lora_state.transmission_errors++;
        return -1;
    }
    
    lora_state.messages_sent++;
    LOG_INF("LoRa message transmitted successfully");
    
    return 0;
}

/* Send emergency threat detection */
int sait01_lora_send_emergency(uint8_t threat_type, float confidence, 
                              int16_t location_x, int16_t location_y)
{
    struct {
        uint8_t threat_type;
        uint8_t confidence_percent;
        int16_t location_x;
        int16_t location_y;
        uint32_t timestamp;
    } __packed emergency_payload;
    
    emergency_payload.threat_type = threat_type;
    emergency_payload.confidence_percent = (uint8_t)(confidence * 100);
    emergency_payload.location_x = location_x;
    emergency_payload.location_y = location_y;
    emergency_payload.timestamp = k_uptime_get_32();
    
    LOG_WRN("EMERGENCY: Threat type %u detected (%.1f%% confidence)",
            threat_type, confidence * 100);
    
    return sait01_lora_transmit(SAIT01_MSG_EMERGENCY, NULL, 
                               (uint8_t *)&emergency_payload, 
                               sizeof(emergency_payload));
}

/* Send status update */
int sait01_lora_send_status(void)
{
    struct {
        uint8_t battery_level;
        uint8_t signal_strength;
        uint16_t uptime_hours;
        uint32_t detections_count;
    } __packed status_payload;
    
    status_payload.battery_level = 85;  // TODO: Get actual battery level
    status_payload.signal_strength = 75;  // TODO: Get actual signal strength
    status_payload.uptime_hours = k_uptime_get_32() / (1000 * 3600);
    status_payload.detections_count = lora_state.messages_sent;
    
    LOG_INF("Sending status update");
    
    return sait01_lora_transmit(SAIT01_MSG_STATUS, NULL,
                               (uint8_t *)&status_payload,
                               sizeof(status_payload));
}

/* Get LoRa statistics */
void sait01_lora_get_stats(uint32_t *msgs_sent, uint32_t *msgs_received, 
                          uint32_t *errors, int16_t *last_rssi, int8_t *last_snr)
{
    if (msgs_sent) *msgs_sent = lora_state.messages_sent;
    if (msgs_received) *msgs_received = lora_state.messages_received;
    if (errors) *errors = lora_state.transmission_errors;
    if (last_rssi) *last_rssi = lora_state.last_rssi;
    if (last_snr) *last_snr = lora_state.last_snr;
}

/* LoRa fallback worker thread */
void sait01_lora_worker_thread(void)
{
    LOG_INF("LoRa fallback worker thread started");
    
    while (1) {
        // Send periodic status every 5 minutes
        sait01_lora_send_status();
        
        // Sleep for 5 minutes
        k_sleep(K_MINUTES(5));
    }
}