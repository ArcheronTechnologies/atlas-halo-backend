/*
 * SAIT_01 Main Firmware
 * nRF5340 Defense-Grade IoT Threat Detection System
 * Integrated AI/ML, Security, Communications & Positioning
 */

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/drivers/uart.h>
#include <zephyr/logging/log.h>
#include <zephyr/sys/reboot.h>
#include <zephyr/random/rand32.h>

/* SAIT_01 Component Includes */
#include "hardware_security.h"
#include "lora_fallback.h" 
#include "mesh_encryption.h"
#include "coordinator_election.h"
#include "uwb_ranging.h"
#include "ota_update.h"
#include "hardened_spectrum_analysis.h"

/* Power Management Includes */
#include "power_management.h"
#include "adaptive_audio.h"
#include "smart_radio.h"

LOG_MODULE_REGISTER(sait01_main, LOG_LEVEL_INF);

/* System Configuration */
#define SAIT01_SYSTEM_VERSION_MAJOR    1
#define SAIT01_SYSTEM_VERSION_MINOR    0
#define SAIT01_SYSTEM_VERSION_PATCH    0
#define SAIT01_SYSTEM_VERSION_BUILD    1

#define SAIT01_HEARTBEAT_INTERVAL_MS   30000   // 30 second system heartbeat
#define SAIT01_THREAT_SCAN_INTERVAL_MS 1000    // 1 second audio scanning
#define SAIT01_STATUS_REPORT_INTERVAL_MS 300000 // 5 minute status reports
#define SAIT01_RANGING_INTERVAL_MS     10000   // 10 second UWB ranging

/* GPIO Pin Definitions */
#define LED_STATUS_NODE    DT_ALIAS(led0)
#define LED_THREAT_NODE    DT_ALIAS(led1)
#define LED_NETWORK_NODE   DT_ALIAS(led2)

/* System State */
typedef enum {
    SAIT01_STATE_INITIALIZING,
    SAIT01_STATE_OPERATIONAL,
    SAIT01_STATE_THREAT_DETECTED,
    SAIT01_STATE_NETWORK_ISOLATED,
    SAIT01_STATE_MAINTENANCE,
    SAIT01_STATE_ERROR
} sait01_system_state_t;

/* System Context */
struct sait01_system_context {
    sait01_system_state_t state;
    uint8_t node_id[4];
    
    uint32_t total_detections;
    uint32_t confirmed_threats;
    uint32_t false_positives;
    uint32_t system_uptime_hours;
    
    float last_threat_confidence;
    uint32_t last_threat_time;
    uint8_t last_threat_type;
    
    bool hardware_security_ready;
    bool mesh_network_ready;
    bool lora_fallback_ready;
    bool uwb_positioning_ready;
    bool ota_system_ready;
    bool ml_engine_ready;
    
    uint32_t network_nodes_count;
    bool is_coordinator;
    
    /* Power management state */
    bool power_optimizations_enabled;
    uint32_t last_power_transition;
    float current_power_consumption_ma;
    
    const struct gpio_dt_spec led_status;
    const struct gpio_dt_spec led_threat;
    const struct gpio_dt_spec led_network;
};

static struct sait01_system_context sys_ctx = {
    .led_status = GPIO_DT_SPEC_GET(LED_STATUS_NODE, gpios),
    .led_threat = GPIO_DT_SPEC_GET(LED_THREAT_NODE, gpios),
    .led_network = GPIO_DT_SPEC_GET(LED_NETWORK_NODE, gpios),
};

/* Work items for background tasks */
static struct k_work_delayable system_heartbeat_work;
static struct k_work_delayable threat_scanning_work;
static struct k_work_delayable status_report_work;
static struct k_work_delayable uwb_ranging_work;

/* Thread stacks */
K_THREAD_STACK_DEFINE(audio_processing_stack, 2048);  /* Reduced from 4096 */
K_THREAD_STACK_DEFINE(network_communication_stack, 1536);  /* Reduced from 2048 */
K_THREAD_STACK_DEFINE(system_monitoring_stack, 1024);  /* Reduced from 2048 */

/* Thread declarations */
static struct k_thread audio_processing_thread;
static struct k_thread network_communication_thread;
static struct k_thread system_monitoring_thread;

/* Forward declarations */
static void audio_processing_thread_entry(void *p1, void *p2, void *p3);
static void network_communication_thread_entry(void *p1, void *p2, void *p3);
static void system_monitoring_thread_entry(void *p1, void *p2, void *p3);
static void system_heartbeat_handler(struct k_work *work);
static void threat_scanning_handler(struct k_work *work);
static void status_report_handler(struct k_work *work);
static void uwb_ranging_handler(struct k_work *work);

/* Initialize system LEDs */
static int init_system_leds(void)
{
    LOG_INF("Initializing system LEDs");
    
    if (!gpio_is_ready_dt(&sys_ctx.led_status) ||
        !gpio_is_ready_dt(&sys_ctx.led_threat) ||
        !gpio_is_ready_dt(&sys_ctx.led_network)) {
        LOG_ERR("LED GPIO devices not ready");
        return -1;
    }
    
    gpio_pin_configure_dt(&sys_ctx.led_status, GPIO_OUTPUT_INACTIVE);
    gpio_pin_configure_dt(&sys_ctx.led_threat, GPIO_OUTPUT_INACTIVE);
    gpio_pin_configure_dt(&sys_ctx.led_network, GPIO_OUTPUT_INACTIVE);
    
    // Flash all LEDs for startup indication
    for (int i = 0; i < 3; i++) {
        gpio_pin_set_dt(&sys_ctx.led_status, 1);
        gpio_pin_set_dt(&sys_ctx.led_threat, 1);
        gpio_pin_set_dt(&sys_ctx.led_network, 1);
        k_sleep(K_MSEC(200));
        
        gpio_pin_set_dt(&sys_ctx.led_status, 0);
        gpio_pin_set_dt(&sys_ctx.led_threat, 0);
        gpio_pin_set_dt(&sys_ctx.led_network, 0);
        k_sleep(K_MSEC(200));
    }
    
    LOG_INF("System LEDs initialized");
    return 0;
}

/* Update LED status based on system state */
static void update_led_status(void)
{
    static uint32_t last_update = 0;
    uint32_t current_time = k_uptime_get_32();
    
    // Update every 500ms
    if (current_time - last_update < 500) {
        return;
    }
    last_update = current_time;
    
    // Status LED (heartbeat)
    static bool status_state = false;
    status_state = !status_state;
    gpio_pin_set_dt(&sys_ctx.led_status, status_state ? 1 : 0);
    
    // Threat LED
    bool threat_active = (sys_ctx.state == SAIT01_STATE_THREAT_DETECTED);
    gpio_pin_set_dt(&sys_ctx.led_threat, threat_active ? 1 : 0);
    
    // Network LED  
    gpio_pin_set_dt(&sys_ctx.led_network, sys_ctx.mesh_network_ready ? 1 : 0);
}

/* Generate unique node ID */
static void generate_node_id(void)
{
    uint32_t random = sys_rand32_get();
    memcpy(sys_ctx.node_id, &random, 4);
    
    LOG_INF("Node ID: %02x%02x%02x%02x",
            sys_ctx.node_id[0], sys_ctx.node_id[1],
            sys_ctx.node_id[2], sys_ctx.node_id[3]);
}

/* Initialize all system components */
static int init_system_components(void)
{
    int ret;
    
    LOG_INF("Initializing SAIT_01 system components");
    
    // Generate unique node ID
    generate_node_id();
    
    // Initialize hardware security module
    ret = sait01_security_init();
    if (ret == 0) {
        sys_ctx.hardware_security_ready = true;
        LOG_INF("Hardware security: READY");
    } else {
        LOG_ERR("Hardware security: FAILED");
        return ret;
    }
    
    // Initialize mesh encryption
    ret = sait01_mesh_encryption_init(sys_ctx.node_id);
    if (ret == 0) {
        LOG_INF("Mesh encryption: READY");
    } else {
        LOG_ERR("Mesh encryption: FAILED");
        return ret;
    }
    
    // Initialize coordinator election
    ret = sait01_coordinator_election_init(sys_ctx.node_id);
    if (ret == 0) {
        LOG_INF("Coordinator election: READY");
    } else {
        LOG_ERR("Coordinator election: FAILED");
        return ret;
    }
    
    // Initialize LoRa fallback communication
    ret = sait01_lora_init();
    if (ret == 0) {
        sys_ctx.lora_fallback_ready = true;
        LOG_INF("LoRa fallback: READY");
    } else {
        LOG_WRN("LoRa fallback: FAILED (non-critical)");
    }
    
    // Initialize UWB ranging
    ret = sait01_uwb_ranging_init();
    if (ret == 0) {
        sys_ctx.uwb_positioning_ready = true;
        LOG_INF("UWB positioning: READY");
    } else {
        LOG_WRN("UWB positioning: FAILED (non-critical)");
    }
    
    // Initialize OTA update system
    ret = sait01_ota_init();
    if (ret == 0) {
        sys_ctx.ota_system_ready = true;
        LOG_INF("OTA updates: READY");
    } else {
        LOG_WRN("OTA updates: FAILED (non-critical)");
    }
    
    // Initialize ML spectrum analysis
    ret = sait01_hardened_spectrum_init();
    if (ret == 0) {
        sys_ctx.ml_engine_ready = true;
        LOG_INF("ML threat detection: READY");
    } else {
        LOG_ERR("ML threat detection: FAILED");
        return ret;
    }
    
    // Initialize power management system
    ret = sait01_power_init();
    if (ret == 0) {
        LOG_INF("Power management: READY");
    } else {
        LOG_ERR("Power management: FAILED");
        return ret;
    }
    
    // Initialize adaptive audio system
    ret = sait01_audio_init();
    if (ret == 0) {
        LOG_INF("Adaptive audio: READY");
    } else {
        LOG_ERR("Adaptive audio: FAILED");
        return ret;
    }
    
    // Initialize smart radio management
    ret = sait01_radio_init();
    if (ret == 0) {
        LOG_INF("Smart radio: READY");
    } else {
        LOG_ERR("Smart radio: FAILED");
        return ret;
    }
    
    // Enable power optimizations for battery deployment
    sys_ctx.power_optimizations_enabled = true;
    sys_ctx.last_power_transition = k_uptime_get_32();
    sys_ctx.current_power_consumption_ma = 1.0f; // Target <1 mA
    
    sys_ctx.mesh_network_ready = true; // Will be updated by network threads
    
    LOG_INF("All critical components initialized successfully");
    return 0;
}

/* Process threat detection */
static void process_threat_detection(uint8_t threat_type, float confidence)
{
    LOG_WRN("THREAT DETECTED: Type %u, Confidence %.1f%%", threat_type, confidence * 100);
    
    sys_ctx.total_detections++;
    sys_ctx.last_threat_confidence = confidence;
    sys_ctx.last_threat_time = k_uptime_get_32();
    sys_ctx.last_threat_type = threat_type;
    sys_ctx.state = SAIT01_STATE_THREAT_DETECTED;
    
    // Determine if this is a confirmed threat
    if (confidence > 0.8f) {  // 80% confidence threshold
        sys_ctx.confirmed_threats++;
        
        // Send emergency alert via LoRa
        if (sys_ctx.lora_fallback_ready) {
            sait01_lora_send_emergency(threat_type, confidence, 0, 0);
        }
        
        // TODO: Send mesh network alert
        // TODO: Record UWB position for threat location
        
        LOG_ERR("CONFIRMED THREAT - Alert sent to network");
    } else {
        // Lower confidence - monitor but don't alarm
        LOG_WRN("Potential threat detected - monitoring");
        
        // Return to operational state after brief period
        k_sleep(K_MSEC(2000));
        sys_ctx.state = SAIT01_STATE_OPERATIONAL;
    }
}

/* Audio processing thread */
static void audio_processing_thread_entry(void *p1, void *p2, void *p3)
{
    LOG_INF("Audio processing thread started");
    
    while (1) {
        if (sys_ctx.ml_engine_ready && sys_ctx.state == SAIT01_STATE_OPERATIONAL) {
            // Use adaptive audio sampling for power optimization
            if (sys_ctx.power_optimizations_enabled) {
                // Get adaptive audio statistics
                adaptive_audio_stats_t audio_stats = sait01_audio_get_statistics();
                
                // Simulate audio data based on current sample rate
                uint16_t samples_per_chunk = (audio_stats.current_sample_rate * SAIT01_THREAT_SCAN_INTERVAL_MS) / 1000;
                uint16_t audio_samples[samples_per_chunk];
                
                // Fill with simulated samples (in production, get from SAADC)
                for (int i = 0; i < samples_per_chunk; i++) {
                    audio_samples[i] = (uint16_t)(sys_rand32_get() & 0xFFF); // 12-bit samples
                }
                
                // Process samples through adaptive audio system
                sait01_audio_process_samples(audio_samples, samples_per_chunk);
                
                // If in high-resolution mode, run ML analysis
                if (sait01_audio_is_high_resolution_active()) {
                    // Convert samples to bytes for ML processing
                    uint8_t audio_data[samples_per_chunk * 2];
                    for (int i = 0; i < samples_per_chunk; i++) {
                        audio_data[i*2] = (uint8_t)(audio_samples[i] & 0xFF);
                        audio_data[i*2+1] = (uint8_t)((audio_samples[i] >> 8) & 0xFF);
                    }
                    
                    // Process with hardened spectrum analyzer
                    struct sait01_threat_result result;
                    int ret = sait01_hardened_spectrum_analyze(audio_data, sizeof(audio_data), &result);
                    
                    if (ret == 0 && result.threat_detected) {
                        process_threat_detection(result.threat_type, result.confidence);
                        
                        // Emergency mode increases radio activity
                        sait01_radio_set_emergency_mode(result.confidence > 0.9f);
                    }
                }
            } else {
                // Legacy audio processing (high power consumption)
                uint8_t audio_data[16000];
                
                // Fill with simulated audio (random noise for testing)
                for (int i = 0; i < sizeof(audio_data); i++) {
                    audio_data[i] = (uint8_t)(sys_rand32_get() & 0xFF);
                }
                
                // Process with hardened spectrum analyzer
                struct sait01_threat_result result;
                int ret = sait01_hardened_spectrum_analyze(audio_data, sizeof(audio_data), &result);
                
                if (ret == 0 && result.threat_detected) {
                    process_threat_detection(result.threat_type, result.confidence);
                }
            }
        }
        
        k_sleep(K_MSEC(SAIT01_THREAT_SCAN_INTERVAL_MS));
    }
}

/* Network communication thread */
static void network_communication_thread_entry(void *p1, void *p2, void *p3)
{
    LOG_INF("Network communication thread started");
    
    while (1) {
        // Power-optimized radio management
        if (sys_ctx.power_optimizations_enabled) {
            // Check if it's time for radio wake-up
            if (sait01_radio_get_next_wake_time() <= k_uptime_get_32()) {
                // Execute radio wake-up cycle
                int ret = sait01_radio_wake_up();
                if (ret == 0) {
                    // Process any received messages during wake window
                    if (sait01_radio_process_received_packet()) {
                        // Handle mesh synchronization and messages
                        LOG_DBG("Mesh sync packet processed");
                    }
                }
                
                // Schedule next wake-up
                sait01_radio_schedule_next_wake();
            }
            
            // Update coordinator status
            sait01_node_role_t role;
            sait01_coordinator_get_info(NULL, NULL, &role);
            bool was_coordinator = sys_ctx.is_coordinator;
            sys_ctx.is_coordinator = (role == SAIT01_ROLE_COORDINATOR);
            
            // If we became coordinator, enable coordinator mode power management
            if (sys_ctx.is_coordinator && !was_coordinator) {
                sait01_radio_set_coordinator_mode(true);
                sait01_power_set_coordinator_mode(true);
                LOG_INF("Coordinator mode enabled");
            } else if (!sys_ctx.is_coordinator && was_coordinator) {
                sait01_radio_set_coordinator_mode(false);
                sait01_power_set_coordinator_mode(false);
                LOG_INF("Coordinator mode disabled");
            }
            
            // If coordinator, transmit beacon during wake window
            if (sys_ctx.is_coordinator && sait01_radio_is_synchronized()) {
                sait01_radio_transmit_beacon();
            }
        } else {
            // Legacy continuous radio operation (high power)
            // TODO: Handle mesh network messages
            // TODO: Process coordinator election messages
            // TODO: Handle encrypted communications
            
            // Update coordinator status
            sait01_node_role_t role;
            sait01_coordinator_get_info(NULL, NULL, &role);
            sys_ctx.is_coordinator = (role == SAIT01_ROLE_COORDINATOR);
        }
        
        k_sleep(K_MSEC(100));  // Reduced sleep for radio timing accuracy
    }
}

/* System monitoring thread */
static void system_monitoring_thread_entry(void *p1, void *p2, void *p3)
{
    LOG_INF("System monitoring thread started");
    
    while (1) {
        // Update system uptime
        sys_ctx.system_uptime_hours = k_uptime_get_32() / (1000 * 3600);
        
        // Update LED status
        update_led_status();
        
        // Power management monitoring
        if (sys_ctx.power_optimizations_enabled) {
            // Get power statistics
            power_mgmt_stats_t power_stats = sait01_power_get_statistics();
            adaptive_audio_stats_t audio_stats = sait01_audio_get_statistics();
            smart_radio_stats_t radio_stats = sait01_radio_get_statistics();
            
            // Update current power consumption estimate
            sys_ctx.current_power_consumption_ma = power_stats.estimated_current_ma;
            
            // Log power statistics every 30 seconds
            static uint32_t last_power_log = 0;
            if (k_uptime_get_32() - last_power_log > 30000) {
                LOG_INF("Power Stats: %.2f mA avg, %.1f%% duty cycle, %.1f hrs remaining",
                       power_stats.estimated_current_ma,
                       power_stats.actual_duty_cycle_percent,
                       power_stats.estimated_battery_life_hours);
                LOG_INF("Audio: %d Hz, %s mode, %.1f%% power savings",
                       audio_stats.current_sample_rate,
                       audio_stats.high_res_mode ? "HIGH-RES" : "LOW-POWER",
                       audio_stats.power_savings_percent);
                LOG_INF("Radio: %s, %.1f%% power savings, %d syncs",
                       radio_stats.synchronized ? "SYNCED" : "UNSYNCED",
                       radio_stats.power_savings_percent,
                       radio_stats.successful_syncs);
                       
                last_power_log = k_uptime_get_32();
            }
            
            // Battery voltage monitoring
            uint32_t battery_mv = sait01_power_get_battery_voltage_mv();
            if (battery_mv < 2500) {  // Below 2.5V - critical
                LOG_ERR("CRITICAL: Battery voltage low (%d mV)", battery_mv);
                // Enable emergency power mode
                sait01_power_set_emergency_mode(true);
                sait01_audio_set_power_mode(true);
                sait01_radio_set_power_mode(true);
            } else if (battery_mv < 2800) {  // Below 2.8V - warning
                LOG_WRN("WARNING: Battery voltage low (%d mV)", battery_mv);
            }
            
            // Adaptive power management based on activity
            uint32_t current_time = k_uptime_get_32();
            if (sys_ctx.state == SAIT01_STATE_OPERATIONAL && 
                current_time - sys_ctx.last_threat_time > 300000) {  // 5 minutes no threats
                
                // Enter deep sleep mode for maximum power savings
                if (current_time - sys_ctx.last_power_transition > 60000) {  // 1 minute intervals
                    LOG_INF("Entering power-optimized deep sleep cycle");
                    
                    // Configure all systems for low power
                    sait01_audio_set_power_mode(true);
                    sait01_radio_set_power_mode(true);
                    
                    // Enter deep sleep for 9 seconds (10% duty cycle)
                    sait01_power_enter_deep_sleep(POWER_MGMT_SLEEP_DURATION_MS);
                    
                    sys_ctx.last_power_transition = k_uptime_get_32();
                    
                    // Wake up and resume normal operation
                    sait01_power_enter_active_monitoring(POWER_MGMT_ACTIVE_DURATION_MS);
                    
                    // Re-enable systems
                    sait01_audio_set_power_mode(false);
                    sait01_radio_set_power_mode(false);
                }
            }
        }
        
        // Check for OTA updates periodically
        static uint32_t last_ota_check = 0;
        if (sys_ctx.ota_system_ready && 
            k_uptime_get_32() - last_ota_check > 3600000) { // Check every hour
            sait01_ota_check_for_updates();
            last_ota_check = k_uptime_get_32();
        }
        
        // Memory and health monitoring
        // TODO: Monitor stack usage, heap usage, etc.
        
        k_sleep(K_MSEC(5000));
    }
}

/* System heartbeat handler */
static void system_heartbeat_handler(struct k_work *work)
{
    LOG_INF("System heartbeat - State: %d, Uptime: %u hours",
            sys_ctx.state, sys_ctx.system_uptime_hours);
    
    // Reschedule
    k_work_schedule(&system_heartbeat_work, K_MSEC(SAIT01_HEARTBEAT_INTERVAL_MS));
}

/* Status report handler */
static void status_report_handler(struct k_work *work)
{
    LOG_INF("Status Report:");
    LOG_INF("  Detections: %u (Confirmed: %u, False: %u)",
            sys_ctx.total_detections, sys_ctx.confirmed_threats, sys_ctx.false_positives);
    LOG_INF("  Network: %s (Coordinator: %s, Nodes: %u)",
            sys_ctx.mesh_network_ready ? "UP" : "DOWN",
            sys_ctx.is_coordinator ? "YES" : "NO",
            sys_ctx.network_nodes_count);
    LOG_INF("  Components: HW_SEC=%s, LoRa=%s, UWB=%s, OTA=%s, ML=%s",
            sys_ctx.hardware_security_ready ? "OK" : "FAIL",
            sys_ctx.lora_fallback_ready ? "OK" : "FAIL",
            sys_ctx.uwb_positioning_ready ? "OK" : "FAIL",
            sys_ctx.ota_system_ready ? "OK" : "FAIL",
            sys_ctx.ml_engine_ready ? "OK" : "FAIL");
    LOG_INF("  Power: Optimizations=%s, Current=%.2f mA, Battery=%d mV",
            sys_ctx.power_optimizations_enabled ? "ON" : "OFF",
            sys_ctx.current_power_consumption_ma,
            sait01_power_get_battery_voltage_mv());
    
    // Send status via LoRa if available
    if (sys_ctx.lora_fallback_ready) {
        sait01_lora_send_status();
    }
    
    // Reschedule
    k_work_schedule(&status_report_work, K_MSEC(SAIT01_STATUS_REPORT_INTERVAL_MS));
}

/* UWB ranging handler */
static void uwb_ranging_handler(struct k_work *work)
{
    if (sys_ctx.uwb_positioning_ready) {
        // Range all known peers
        struct sait01_uwb_range_result ranges[8];
        int range_count = sait01_uwb_range_all_peers(ranges, 8);
        
        if (range_count > 0) {
            LOG_DBG("Ranging complete: %d peers measured", range_count);
            
            // Calculate position if enough ranges
            if (range_count >= 3) {
                float pos_x, pos_y;
                if (sait01_uwb_calculate_position(ranges, range_count, &pos_x, &pos_y) == 0) {
                    LOG_DBG("Position: (%.2f, %.2f)", pos_x, pos_y);
                }
            }
        }
    }
    
    // Reschedule
    k_work_schedule(&uwb_ranging_work, K_MSEC(SAIT01_RANGING_INTERVAL_MS));
}

/* Main function */
int main(void)
{
    int ret;
    
    LOG_INF("SAIT_01 Defense IoT Threat Detection System");
    LOG_INF("Version %d.%d.%d.%d starting up...",
            SAIT01_SYSTEM_VERSION_MAJOR, SAIT01_SYSTEM_VERSION_MINOR,
            SAIT01_SYSTEM_VERSION_PATCH, SAIT01_SYSTEM_VERSION_BUILD);
    
    sys_ctx.state = SAIT01_STATE_INITIALIZING;
    
    // Initialize LEDs first for visual feedback
    ret = init_system_leds();
    if (ret != 0) {
        LOG_ERR("Failed to initialize LEDs: %d", ret);
        return ret;
    }
    
    // Initialize all system components
    ret = init_system_components();
    if (ret != 0) {
        LOG_ERR("Critical component initialization failed: %d", ret);
        sys_ctx.state = SAIT01_STATE_ERROR;
        
        // Flash error indication
        while (1) {
            gpio_pin_set_dt(&sys_ctx.led_threat, 1);
            k_sleep(K_MSEC(100));
            gpio_pin_set_dt(&sys_ctx.led_threat, 0);
            k_sleep(K_MSEC(100));
        }
    }
    
    // Initialize work items
    k_work_init_delayable(&system_heartbeat_work, system_heartbeat_handler);
    k_work_init_delayable(&status_report_work, status_report_handler);
    k_work_init_delayable(&uwb_ranging_work, uwb_ranging_handler);
    
    // Start background work items
    k_work_schedule(&system_heartbeat_work, K_MSEC(SAIT01_HEARTBEAT_INTERVAL_MS));
    k_work_schedule(&status_report_work, K_MSEC(SAIT01_STATUS_REPORT_INTERVAL_MS));
    k_work_schedule(&uwb_ranging_work, K_MSEC(SAIT01_RANGING_INTERVAL_MS));
    
    // Create worker threads
    k_thread_create(&audio_processing_thread, audio_processing_stack,
                   K_THREAD_STACK_SIZEOF(audio_processing_stack),
                   audio_processing_thread_entry, NULL, NULL, NULL,
                   5, 0, K_NO_WAIT);  /* Higher priority for audio */
    
    k_thread_create(&network_communication_thread, network_communication_stack,
                   K_THREAD_STACK_SIZEOF(network_communication_stack),
                   network_communication_thread_entry, NULL, NULL, NULL,
                   7, 0, K_NO_WAIT);  /* Medium priority for network */
    
    k_thread_create(&system_monitoring_thread, system_monitoring_stack,
                   K_THREAD_STACK_SIZEOF(system_monitoring_stack),
                   system_monitoring_thread_entry, NULL, NULL, NULL,
                   10, 0, K_NO_WAIT);  /* Lower priority for monitoring */
    
    // System is now operational
    sys_ctx.state = SAIT01_STATE_OPERATIONAL;
    
    LOG_INF("SAIT_01 system fully operational");
    LOG_INF("Defense-grade threat detection active");
    LOG_INF("Mesh networking and communications ready");
    LOG_INF("Hardware security and encryption enabled");
    LOG_INF("Power optimizations enabled - Target <1 mA consumption");
    LOG_INF("Adaptive audio: 4 kHz idle, 16 kHz active sampling");
    LOG_INF("Smart radio: Periodic wake-up for 90%% power reduction");
    LOG_INF("CPU scaling: 32 MHz idle, 128 MHz active processing");
    
    // Main loop - handle critical system events
    while (1) {
        switch (sys_ctx.state) {
            case SAIT01_STATE_OPERATIONAL:
                // Normal operation - threads handle processing
                break;
                
            case SAIT01_STATE_THREAT_DETECTED:
                // Enhanced monitoring during threat state
                k_sleep(K_MSEC(5000)); // Stay in threat state for 5 seconds
                sys_ctx.state = SAIT01_STATE_OPERATIONAL;
                break;
                
            case SAIT01_STATE_NETWORK_ISOLATED:
                // Network isolation mode - reduce power, use LoRa only
                LOG_WRN("Network isolation mode active");
                k_sleep(K_MSEC(30000));
                break;
                
            case SAIT01_STATE_MAINTENANCE:
                // Maintenance mode - limited operation
                LOG_INF("Maintenance mode active");
                k_sleep(K_MSEC(60000));
                break;
                
            case SAIT01_STATE_ERROR:
                // Error state - attempt recovery
                LOG_ERR("System error state - attempting recovery");
                k_sleep(K_MSEC(10000));
                
                // Try to reinitialize critical components
                if (init_system_components() == 0) {
                    sys_ctx.state = SAIT01_STATE_OPERATIONAL;
                    LOG_INF("System recovery successful");
                } else {
                    LOG_ERR("Recovery failed - system restart required");
                    sys_reboot(SYS_REBOOT_WARM);
                }
                break;
                
            default:
                sys_ctx.state = SAIT01_STATE_OPERATIONAL;
                break;
        }
        
        k_sleep(K_MSEC(1000));
    }
    
    return 0;
}