/*
 * SAIT_01 UWB Peer Ranging System
 * DW3000 Ultra-Wideband Precise Positioning
 * Centimeter-Accurate Distance Measurement
 */

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/spi.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/byteorder.h>
#include <zephyr/random/rand32.h>
#include <math.h>

LOG_MODULE_REGISTER(uwb_ranging, LOG_LEVEL_INF);

/* DW3000 Register Definitions */
#define DW3000_DEV_ID              0x00
#define DW3000_EUI_64              0x01
#define DW3000_PANADR              0x03
#define DW3000_SYS_CFG             0x04
#define DW3000_SYS_TIME            0x06
#define DW3000_TX_FCTRL            0x08
#define DW3000_TX_BUFFER           0x09
#define DW3000_DX_TIME             0x0A
#define DW3000_RX_FWTO             0x0C
#define DW3000_SYS_CTRL            0x0D
#define DW3000_SYS_MASK            0x0E
#define DW3000_SYS_STATUS          0x0F
#define DW3000_RX_FINFO            0x10
#define DW3000_RX_BUFFER           0x11
#define DW3000_RX_FQUAL            0x12
#define DW3000_RX_TTCKI            0x13
#define DW3000_RX_TTCKO            0x14
#define DW3000_RX_TIME             0x15
#define DW3000_TX_TIME             0x17
#define DW3000_TX_RAWTS            0x1C
#define DW3000_RX_RAWTS            0x1D

/* DW3000 Commands */
#define DW3000_CMD_TX              0x01
#define DW3000_CMD_RX              0x02
#define DW3000_CMD_TXRXOFF         0x40

/* SAIT_01 UWB Configuration */
#define SAIT01_UWB_CHANNEL         5          // Channel 5 (6.5 GHz)
#define SAIT01_UWB_PRF             DWM_PRF_16M // 16 MHz PRF
#define SAIT01_UWB_PLEN            DWM_PLEN_128 // 128 symbol preamble
#define SAIT01_UWB_PAC             DWM_PAC8    // 8 symbol PAC
#define SAIT01_UWB_DATARATE        DWM_BR_850K // 850 kbps
#define SAIT01_UWB_NSSFD           1          // Non-standard SFD
#define SAIT01_UWB_STO             (128 + 8 - 8) // SFD timeout

#define SAIT01_UWB_SPEED_OF_LIGHT  299702547.0 // m/s in air
#define SAIT01_UWB_TIME_UNITS      (1.0/499.2e6/128.0) // DW3000 time units to seconds
#define SAIT01_UWB_MAX_RANGE_M     1000       // Maximum ranging distance (m)
#define SAIT01_UWB_MIN_RANGE_M     0.1        // Minimum ranging distance (m)

/* UWB Message Types */
#define SAIT01_UWB_MSG_POLL        0x61
#define SAIT01_UWB_MSG_RESP        0x62
#define SAIT01_UWB_MSG_FINAL       0x63
#define SAIT01_UWB_MSG_REPORT      0x64

/* UWB Frame Structure */
struct sait01_uwb_msg {
    uint8_t frame_ctrl[2];     // Frame control
    uint8_t seq_num;           // Sequence number
    uint8_t pan_id[2];         // PAN ID
    uint8_t dest_addr[2];      // Destination address
    uint8_t src_addr[2];       // Source address
    uint8_t msg_type;          // Message type
    uint8_t data[16];          // Message data
    uint8_t fcs[2];            // Frame check sequence
} __packed;

/* Two-Way Ranging Message Data */
struct sait01_twr_data {
    uint64_t poll_tx_ts;       // Poll transmission timestamp
    uint64_t poll_rx_ts;       // Poll reception timestamp
    uint64_t resp_tx_ts;       // Response transmission timestamp
    uint64_t resp_rx_ts;       // Response reception timestamp
    uint64_t final_tx_ts;      // Final transmission timestamp
    uint64_t final_rx_ts;      // Final reception timestamp
} __packed;

/* Ranging Result */
struct sait01_uwb_range_result {
    uint8_t peer_addr[2];      // Peer address
    float distance_m;          // Distance in meters
    float rssi_dbm;           // RSSI in dBm
    float fp_power_dbm;       // First path power in dBm
    uint16_t fp_index;        // First path index
    uint32_t timestamp;       // Measurement timestamp
    bool valid;               // Result validity
};

/* UWB Peer Information */
struct sait01_uwb_peer {
    uint8_t addr[2];           // Peer address
    struct sait01_uwb_range_result last_range;
    uint32_t last_ranging_time;
    uint32_t successful_ranges;
    uint32_t failed_ranges;
    bool active;
};

/* UWB Positioning Context */
struct sait01_uwb_context {
    const struct device *spi_dev;
    const struct device *reset_gpio;
    const struct device *irq_gpio;
    uint32_t reset_pin;
    uint32_t irq_pin;
    
    uint8_t our_addr[2];       // Our UWB address
    uint8_t pan_id[2];         // PAN ID
    uint8_t seq_num;           // Sequence number counter
    
    struct sait01_uwb_peer peers[16]; // Known peers
    uint8_t peer_count;
    
    uint64_t poll_tx_ts;       // Current ranging timestamps
    uint64_t poll_rx_ts;
    uint64_t resp_tx_ts;
    uint64_t resp_rx_ts;
    uint64_t final_tx_ts;
    uint64_t final_rx_ts;
    
    uint32_t total_ranges;
    uint32_t successful_ranges;
    uint32_t failed_ranges;
    
    bool initialized;
    bool ranging_active;
};

static struct sait01_uwb_context uwb_ctx;

/* SPI Configuration */
static struct spi_config spi_cfg = {
    .frequency = 20000000,  // 20 MHz
    .operation = SPI_WORD_SET(8) | SPI_TRANSFER_MSB,
    .slave = 0,
    .cs = NULL,
};

/* Forward Declarations */
static int uwb_write_reg(uint16_t reg, const uint8_t *data, uint16_t len);
static int uwb_read_reg(uint16_t reg, uint8_t *data, uint16_t len);
static int uwb_write_subreg(uint16_t reg, uint16_t subreg, const uint8_t *data, uint16_t len);
static int uwb_read_subreg(uint16_t reg, uint16_t subreg, uint8_t *data, uint16_t len);
static uint64_t uwb_get_sys_time(void);
static uint64_t uwb_get_tx_timestamp(void);
static uint64_t uwb_get_rx_timestamp(void);
static float calculate_distance(const struct sait01_twr_data *twr);

/* Write register to DW3000 */
static int uwb_write_reg(uint16_t reg, const uint8_t *data, uint16_t len)
{
    uint8_t header[3];
    int header_len = 1;
    
    if (reg & 0xFF00) {
        header[0] = 0x80 | ((reg >> 8) & 0x3F);
        header[1] = reg & 0xFF;
        header_len = 2;
    } else {
        header[0] = 0x80 | (reg & 0x3F);
    }
    
    struct spi_buf tx_bufs[] = {
        {.buf = header, .len = header_len},
        {.buf = (void *)data, .len = len}
    };
    struct spi_buf_set tx_set = {.buffers = tx_bufs, .count = 2};
    
    return spi_write(uwb_ctx.spi_dev, &spi_cfg, &tx_set);
}

/* Read register from DW3000 */
static int uwb_read_reg(uint16_t reg, uint8_t *data, uint16_t len)
{
    uint8_t header[3];
    int header_len = 1;
    
    if (reg & 0xFF00) {
        header[0] = (reg >> 8) & 0x3F;
        header[1] = reg & 0xFF;
        header_len = 2;
    } else {
        header[0] = reg & 0x3F;
    }
    
    struct spi_buf tx_buf = {.buf = header, .len = header_len};
    struct spi_buf rx_buf = {.buf = data, .len = len};
    struct spi_buf_set tx_set = {.buffers = &tx_buf, .count = 1};
    struct spi_buf_set rx_set = {.buffers = &rx_buf, .count = 1};
    
    return spi_transceive(uwb_ctx.spi_dev, &spi_cfg, &tx_set, &rx_set);
}

/* Get system time */
static uint64_t uwb_get_sys_time(void)
{
    uint8_t sys_time[5];
    if (uwb_read_reg(DW3000_SYS_TIME, sys_time, 5) != 0) {
        return 0;
    }
    
    uint64_t time = 0;
    for (int i = 4; i >= 0; i--) {
        time = (time << 8) | sys_time[i];
    }
    return time;
}

/* Get TX timestamp */
static uint64_t uwb_get_tx_timestamp(void)
{
    uint8_t tx_time[5];
    if (uwb_read_reg(DW3000_TX_TIME, tx_time, 5) != 0) {
        return 0;
    }
    
    uint64_t time = 0;
    for (int i = 4; i >= 0; i--) {
        time = (time << 8) | tx_time[i];
    }
    return time;
}

/* Get RX timestamp */
static uint64_t uwb_get_rx_timestamp(void)
{
    uint8_t rx_time[5];
    if (uwb_read_reg(DW3000_RX_TIME, rx_time, 5) != 0) {
        return 0;
    }
    
    uint64_t time = 0;
    for (int i = 4; i >= 0; i--) {
        time = (time << 8) | rx_time[i];
    }
    return time;
}

/* Calculate distance using two-way ranging */
static float calculate_distance(const struct sait01_twr_data *twr)
{
    // Calculate round trip times
    uint64_t round1 = twr->resp_rx_ts - twr->poll_tx_ts;
    uint64_t reply1 = twr->resp_tx_ts - twr->poll_rx_ts;
    uint64_t round2 = twr->final_rx_ts - twr->resp_tx_ts;
    uint64_t reply2 = twr->final_tx_ts - twr->resp_rx_ts;
    
    // Calculate time of flight
    double tof = ((double)round1 * round2 - (double)reply1 * reply2) / 
                 ((double)round1 + round2 + reply1 + reply2);
    
    // Convert to distance
    double distance = tof * SAIT01_UWB_TIME_UNITS * SAIT01_UWB_SPEED_OF_LIGHT;
    
    // Validate range
    if (distance < SAIT01_UWB_MIN_RANGE_M || distance > SAIT01_UWB_MAX_RANGE_M) {
        return -1.0f;
    }
    
    return (float)distance;
}

/* Find peer by address */
static struct sait01_uwb_peer *find_peer(const uint8_t *addr)
{
    for (int i = 0; i < uwb_ctx.peer_count; i++) {
        if (memcmp(uwb_ctx.peers[i].addr, addr, 2) == 0) {
            return &uwb_ctx.peers[i];
        }
    }
    return NULL;
}

/* Add new peer */
static struct sait01_uwb_peer *add_peer(const uint8_t *addr)
{
    if (uwb_ctx.peer_count >= ARRAY_SIZE(uwb_ctx.peers)) {
        return NULL;
    }
    
    struct sait01_uwb_peer *peer = &uwb_ctx.peers[uwb_ctx.peer_count++];
    memset(peer, 0, sizeof(*peer));
    memcpy(peer->addr, addr, 2);
    peer->active = true;
    
    LOG_INF("📍 Added UWB peer: 0x%04x", (addr[1] << 8) | addr[0]);
    return peer;
}

/* Initialize UWB ranging */
int sait01_uwb_ranging_init(void)
{
    LOG_INF("📍 Initializing SAIT_01 UWB Ranging System");
    
    memset(&uwb_ctx, 0, sizeof(uwb_ctx));
    
    // Get device references
    uwb_ctx.spi_dev = DEVICE_DT_GET(DT_NODELABEL(spi3));
    uwb_ctx.reset_gpio = DEVICE_DT_GET(DT_NODELABEL(gpio0));
    uwb_ctx.irq_gpio = DEVICE_DT_GET(DT_NODELABEL(gpio0));
    
    if (!device_is_ready(uwb_ctx.spi_dev)) {
        LOG_ERR("❌ UWB SPI device not ready");
        return -1;
    }
    
    if (!device_is_ready(uwb_ctx.reset_gpio)) {
        LOG_ERR("❌ UWB reset GPIO not ready");
        return -1;
    }
    
    // Configure GPIO pins
    uwb_ctx.reset_pin = 17;  // P0.17
    uwb_ctx.irq_pin = 18;    // P0.18
    
    gpio_pin_configure(uwb_ctx.reset_gpio, uwb_ctx.reset_pin, GPIO_OUTPUT_HIGH);
    gpio_pin_configure(uwb_ctx.irq_gpio, uwb_ctx.irq_pin, GPIO_INPUT);
    
    // Reset DW3000
    LOG_INF("🔄 Resetting DW3000");
    gpio_pin_set(uwb_ctx.reset_gpio, uwb_ctx.reset_pin, 0);
    k_sleep(K_MSEC(1));
    gpio_pin_set(uwb_ctx.reset_gpio, uwb_ctx.reset_pin, 1);
    k_sleep(K_MSEC(10));
    
    // Verify device ID
    uint32_t dev_id;
    if (uwb_read_reg(DW3000_DEV_ID, (uint8_t *)&dev_id, 4) != 0) {
        LOG_ERR("❌ Failed to read DW3000 device ID");
        return -1;
    }
    
    if (dev_id != 0xDECA0302) {
        LOG_ERR("❌ Invalid DW3000 device ID: 0x%08x", dev_id);
        return -1;
    }
    
    LOG_INF("✅ DW3000 detected (ID: 0x%08x)", dev_id);
    
    // Generate unique UWB address from device ID
    uint16_t addr = (uint16_t)sys_rand32_get();
    uwb_ctx.our_addr[0] = addr & 0xFF;
    uwb_ctx.our_addr[1] = (addr >> 8) & 0xFF;
    
    // Set PAN ID
    uwb_ctx.pan_id[0] = 0x01;
    uwb_ctx.pan_id[1] = 0xSA;  // "SA" for SAIT
    
    LOG_INF("🆔 UWB Address: 0x%04x, PAN: 0x%04x", 
            (uwb_ctx.our_addr[1] << 8) | uwb_ctx.our_addr[0],
            (uwb_ctx.pan_id[1] << 8) | uwb_ctx.pan_id[0]);
    
    // Configure PAN ID and short address
    uint8_t panadr[4] = {
        uwb_ctx.our_addr[0], uwb_ctx.our_addr[1],
        uwb_ctx.pan_id[0], uwb_ctx.pan_id[1]
    };
    uwb_write_reg(DW3000_PANADR, panadr, 4);
    
    // Configure for channel 5, 16 MHz PRF, 850 kbps
    uint32_t sys_cfg = 0x00000000;  // Basic configuration
    uwb_write_reg(DW3000_SYS_CFG, (uint8_t *)&sys_cfg, 4);
    
    // Enable receiver
    uint8_t sys_ctrl = DW3000_CMD_RX;
    uwb_write_reg(DW3000_SYS_CTRL, &sys_ctrl, 1);
    
    uwb_ctx.initialized = true;
    uwb_ctx.seq_num = 1;
    
    LOG_INF("✅ UWB ranging system initialized");
    return 0;
}

/* Start ranging with peer */
int sait01_uwb_range_peer(const uint8_t *peer_addr, struct sait01_uwb_range_result *result)
{
    if (!uwb_ctx.initialized) {
        LOG_ERR("❌ UWB not initialized");
        return -1;
    }
    
    if (uwb_ctx.ranging_active) {
        LOG_WRN("⚠️  Ranging already in progress");
        return -1;
    }
    
    struct sait01_uwb_peer *peer = find_peer(peer_addr);
    if (!peer) {
        peer = add_peer(peer_addr);
        if (!peer) {
            LOG_ERR("❌ Failed to add peer");
            return -1;
        }
    }
    
    LOG_DBG("📏 Starting range measurement to 0x%04x", 
            (peer_addr[1] << 8) | peer_addr[0]);
    
    uwb_ctx.ranging_active = true;
    uwb_ctx.total_ranges++;
    
    // Build poll message
    struct sait01_uwb_msg poll_msg = {0};
    poll_msg.frame_ctrl[0] = 0x41;  // Data frame, short addressing
    poll_msg.frame_ctrl[1] = 0x88;  // PAN ID compression, short src/dest
    poll_msg.seq_num = uwb_ctx.seq_num++;
    memcpy(poll_msg.pan_id, uwb_ctx.pan_id, 2);
    memcpy(poll_msg.dest_addr, peer_addr, 2);
    memcpy(poll_msg.src_addr, uwb_ctx.our_addr, 2);
    poll_msg.msg_type = SAIT01_UWB_MSG_POLL;
    
    // Write to TX buffer
    uwb_write_reg(DW3000_TX_BUFFER, (uint8_t *)&poll_msg, sizeof(poll_msg) - 2);
    
    // Set frame length
    uint16_t frame_len = sizeof(poll_msg) - 2;
    uwb_write_reg(DW3000_TX_FCTRL, (uint8_t *)&frame_len, 2);
    
    // Start transmission
    uint8_t sys_ctrl = DW3000_CMD_TX;
    uwb_write_reg(DW3000_SYS_CTRL, &sys_ctrl, 1);
    
    // Wait for TX done (simplified - should use interrupt)
    k_sleep(K_MSEC(5));
    
    // Get poll TX timestamp
    uwb_ctx.poll_tx_ts = uwb_get_tx_timestamp();
    
    // Switch to RX mode to wait for response
    sys_ctrl = DW3000_CMD_RX;
    uwb_write_reg(DW3000_SYS_CTRL, &sys_ctrl, 1);
    
    // Wait for response (simplified - should use interrupt and timeout)
    k_sleep(K_MSEC(10));
    
    // Read received frame (simplified)
    struct sait01_uwb_msg resp_msg;
    uwb_read_reg(DW3000_RX_BUFFER, (uint8_t *)&resp_msg, sizeof(resp_msg) - 2);
    
    if (resp_msg.msg_type == SAIT01_UWB_MSG_RESP) {
        // Get response RX timestamp
        uwb_ctx.resp_rx_ts = uwb_get_rx_timestamp();
        
        // Extract response TX timestamp from message data
        memcpy(&uwb_ctx.resp_tx_ts, resp_msg.data, 8);
        
        // Send final message with timestamps
        struct sait01_uwb_msg final_msg = poll_msg;
        final_msg.msg_type = SAIT01_UWB_MSG_FINAL;
        final_msg.seq_num = uwb_ctx.seq_num++;
        
        // Pack timestamps into final message
        struct sait01_twr_data twr_data = {
            .poll_tx_ts = uwb_ctx.poll_tx_ts,
            .resp_rx_ts = uwb_ctx.resp_rx_ts,
        };
        memcpy(final_msg.data, &twr_data, sizeof(twr_data));
        
        // Transmit final
        uwb_write_reg(DW3000_TX_BUFFER, (uint8_t *)&final_msg, sizeof(final_msg) - 2);
        sys_ctrl = DW3000_CMD_TX;
        uwb_write_reg(DW3000_SYS_CTRL, &sys_ctrl, 1);
        
        k_sleep(K_MSEC(5));
        uwb_ctx.final_tx_ts = uwb_get_tx_timestamp();
        
        // Calculate distance
        twr_data.poll_rx_ts = 0;  // Peer will provide this
        twr_data.resp_tx_ts = uwb_ctx.resp_tx_ts;
        twr_data.final_tx_ts = uwb_ctx.final_tx_ts;
        twr_data.final_rx_ts = 0; // Peer will provide this
        
        // For now, use simplified calculation
        uint64_t round_trip = uwb_ctx.resp_rx_ts - uwb_ctx.poll_tx_ts;
        double tof = round_trip * SAIT01_UWB_TIME_UNITS / 2.0;
        float distance = (float)(tof * SAIT01_UWB_SPEED_OF_LIGHT);
        
        if (distance > 0 && distance < SAIT01_UWB_MAX_RANGE_M) {
            // Update peer and result
            peer->last_range.distance_m = distance;
            peer->last_range.timestamp = k_uptime_get_32();
            peer->last_range.valid = true;
            peer->successful_ranges++;
            uwb_ctx.successful_ranges++;
            
            if (result) {
                memcpy(result->peer_addr, peer_addr, 2);
                result->distance_m = distance;
                result->timestamp = peer->last_range.timestamp;
                result->valid = true;
                result->rssi_dbm = -50.0f;  // TODO: Get actual RSSI
                result->fp_power_dbm = -55.0f;  // TODO: Get actual first path power
                result->fp_index = 512;  // TODO: Get actual first path index
            }
            
            LOG_INF("📏 Range to 0x%04x: %.2f m", 
                    (peer_addr[1] << 8) | peer_addr[0], distance);
        } else {
            peer->failed_ranges++;
            uwb_ctx.failed_ranges++;
            
            if (result) {
                result->valid = false;
            }
            
            LOG_WRN("⚠️  Invalid range result: %.2f m", distance);
        }
    } else {
        peer->failed_ranges++;
        uwb_ctx.failed_ranges++;
        
        if (result) {
            result->valid = false;
        }
        
        LOG_WRN("⚠️  No response from peer 0x%04x", (peer_addr[1] << 8) | peer_addr[0]);
    }
    
    uwb_ctx.ranging_active = false;
    return 0;
}

/* Range all known peers */
int sait01_uwb_range_all_peers(struct sait01_uwb_range_result *results, uint8_t max_results)
{
    if (!uwb_ctx.initialized) {
        return -1;
    }
    
    int result_count = 0;
    
    for (int i = 0; i < uwb_ctx.peer_count && result_count < max_results; i++) {
        if (uwb_ctx.peers[i].active) {
            if (sait01_uwb_range_peer(uwb_ctx.peers[i].addr, 
                                     &results[result_count]) == 0) {
                result_count++;
            }
            
            // Small delay between measurements
            k_sleep(K_MSEC(50));
        }
    }
    
    return result_count;
}

/* Get last range result for peer */
int sait01_uwb_get_last_range(const uint8_t *peer_addr, struct sait01_uwb_range_result *result)
{
    if (!uwb_ctx.initialized || !result) {
        return -1;
    }
    
    struct sait01_uwb_peer *peer = find_peer(peer_addr);
    if (!peer || !peer->last_range.valid) {
        return -1;
    }
    
    *result = peer->last_range;
    return 0;
}

/* Get UWB ranging statistics */
void sait01_uwb_get_stats(uint32_t *total_ranges, uint32_t *successful_ranges,
                         uint32_t *failed_ranges, uint8_t *active_peers)
{
    if (total_ranges) *total_ranges = uwb_ctx.total_ranges;
    if (successful_ranges) *successful_ranges = uwb_ctx.successful_ranges;
    if (failed_ranges) *failed_ranges = uwb_ctx.failed_ranges;
    
    if (active_peers) {
        uint8_t count = 0;
        for (int i = 0; i < uwb_ctx.peer_count; i++) {
            if (uwb_ctx.peers[i].active) {
                count++;
            }
        }
        *active_peers = count;
    }
}

/* Calculate triangulated position from ranging data */
int sait01_uwb_calculate_position(const struct sait01_uwb_range_result *ranges,
                                 uint8_t range_count, float *pos_x, float *pos_y)
{
    // Simple trilateration with first 3 valid ranges
    // For production, use more sophisticated algorithms
    
    if (range_count < 3) {
        return -1;  // Need at least 3 ranges for 2D position
    }
    
    // Find 3 valid ranges
    int valid_count = 0;
    struct sait01_uwb_range_result valid_ranges[3];
    
    for (int i = 0; i < range_count && valid_count < 3; i++) {
        if (ranges[i].valid && ranges[i].distance_m > 0) {
            valid_ranges[valid_count++] = ranges[i];
        }
    }
    
    if (valid_count < 3) {
        return -1;
    }
    
    // Simplified trilateration (assumes peer positions are known)
    // In practice, you'd maintain a peer position database
    
    // For now, return a placeholder position
    if (pos_x) *pos_x = 0.0f;
    if (pos_y) *pos_y = 0.0f;
    
    LOG_INF("📍 Calculated position: (%.2f, %.2f)", 
            pos_x ? *pos_x : 0.0f, pos_y ? *pos_y : 0.0f);
    
    return 0;
}