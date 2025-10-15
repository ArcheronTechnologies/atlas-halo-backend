/*
 * 📡 SAIT_01 Smart Radio Management System
 * ========================================
 * Power-optimized mesh radio with periodic wake-up
 * Target: 90% reduction in radio power consumption
 */

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/net/net_if.h>
#include <zephyr/net/net_core.h>
#include <nrf_radio.h>
#include <nrf_timer.h>
#include <hal/nrf_rtc.h>

#include "smart_radio.h"
#include "power_management.h"

/* Radio Management Configuration */
#define RADIO_WAKE_INTERVAL_MS      500      // Wake every 500ms for sync
#define RADIO_LISTEN_WINDOW_MS      50       // Listen for 50ms per wake
#define RADIO_SYNC_TIMEOUT_MS       5000     // Max time to maintain sync
#define RADIO_EMERGENCY_INTERVAL_MS 100      // Emergency mode: wake every 100ms

/* Mesh Network Parameters */
#define MESH_BEACON_INTERVAL_MS     1000     // Beacon transmission interval
#define MESH_HEARTBEAT_INTERVAL_MS  1000     // Heartbeat to neighbors
#define MESH_SYNC_WINDOW_MS         20       // Synchronization window
#define MESH_MAX_RETRIES            3        // Max retransmission attempts

/* Power Optimization Thresholds */
#define RADIO_IDLE_CURRENT_MA       2.7      // Continuous RX current
#define RADIO_OPTIMIZED_CURRENT_MA  0.3      // Optimized average current
#define RADIO_TX_CURRENT_MA         3.2      // Transmission current

/* Radio power state */
typedef enum {
    RADIO_STATE_DEEP_SLEEP,      // Radio completely off
    RADIO_STATE_SCHEDULED_WAKE,  // Periodic wake-up mode
    RADIO_STATE_ACTIVE_RX,       // Actively receiving
    RADIO_STATE_ACTIVE_TX,       // Actively transmitting
    RADIO_STATE_EMERGENCY        // Emergency high-duty mode
} radio_state_t;

typedef struct {
    radio_state_t current_state;
    uint32_t wake_interval_ms;
    uint32_t listen_window_ms;
    uint32_t last_wake_time;
    uint32_t last_sync_time;
    uint32_t next_wake_time;
    bool synchronized;
    bool emergency_mode;
    bool coordinator_mode;
    uint8_t missed_syncs;
    uint32_t total_wake_count;
    uint32_t successful_syncs;
    uint32_t packets_received;
    uint32_t packets_transmitted;
    float duty_cycle_percent;
    uint32_t power_savings_ms;
} radio_mgmt_context_t;

static radio_mgmt_context_t radio_ctx = {0};

/* Timer for radio wake-up scheduling */
#define RADIO_TIMER_INSTANCE        NRF_TIMER3
#define RADIO_RTC_INSTANCE          NRF_RTC2

/* Mesh synchronization data */
typedef struct {
    uint32_t timestamp;
    uint32_t next_beacon_time;
    uint16_t node_id;
    uint8_t sequence_number;
} mesh_sync_packet_t;

/*
 * Radio Power Management Functions
 */

int sait01_radio_init(void)
{
    // Initialize radio management context
    radio_ctx.current_state = RADIO_STATE_SCHEDULED_WAKE;
    radio_ctx.wake_interval_ms = RADIO_WAKE_INTERVAL_MS;
    radio_ctx.listen_window_ms = RADIO_LISTEN_WINDOW_MS;
    radio_ctx.synchronized = false;
    radio_ctx.emergency_mode = false;
    radio_ctx.coordinator_mode = false;
    radio_ctx.missed_syncs = 0;
    
    // Configure radio timer for wake-up scheduling
    nrf_timer_mode_set(RADIO_TIMER_INSTANCE, NRF_TIMER_MODE_TIMER);
    nrf_timer_bit_width_set(RADIO_TIMER_INSTANCE, NRF_TIMER_BIT_WIDTH_32);
    nrf_timer_prescaler_set(RADIO_TIMER_INSTANCE, NRF_TIMER_FREQ_1MHz);
    
    // Configure RTC for precise timing
    nrf_rtc_prescaler_set(RADIO_RTC_INSTANCE, 0);  // 32.768 kHz
    nrf_rtc_event_clear(RADIO_RTC_INSTANCE, NRF_RTC_EVENT_COMPARE_0);
    nrf_rtc_int_enable(RADIO_RTC_INSTANCE, NRF_RTC_INT_COMPARE0_MASK);
    
    // Initialize radio hardware
    nrf_radio_power_set(false);  // Start with radio off
    
    // Schedule first wake-up
    sait01_radio_schedule_next_wake();
    
    printk("📡 Smart Radio: Initialized (%d ms wake interval, %d ms listen window)\n",
           radio_ctx.wake_interval_ms, radio_ctx.listen_window_ms);
    
    return 0;
}

int sait01_radio_power_on(void)
{
    // Enable radio power
    nrf_radio_power_set(true);
    
    // Wait for radio to stabilize
    k_sleep(K_USEC(100));
    
    // Configure radio for mesh operation
    nrf_radio_mode_set(NRF_RADIO_MODE_BLE_1MBIT);
    nrf_radio_frequency_set(2402);  // Channel 2402 MHz
    nrf_radio_txpower_set(NRF_RADIO_TXPOWER_0DBM);
    
    // Set up packet format for mesh
    nrf_radio_packet_configure(
        NRF_RADIO_PREAMBLE_8BIT,
        NRF_RADIO_HEADER_S1_1_BYTE,
        8,   // Header length
        NRF_RADIO_CRC_24BIT
    );
    
    return 0;
}

int sait01_radio_power_off(void)
{
    // Disable radio to save power
    nrf_radio_power_set(false);
    
    // Record power savings
    radio_ctx.power_savings_ms += k_uptime_get_32() - radio_ctx.last_wake_time;
    
    return 0;
}

int sait01_radio_schedule_next_wake(void)
{
    uint32_t current_time = k_uptime_get_32();
    uint32_t wake_interval = radio_ctx.emergency_mode ? 
        RADIO_EMERGENCY_INTERVAL_MS : radio_ctx.wake_interval_ms;
    
    radio_ctx.next_wake_time = current_time + wake_interval;
    
    // Configure timer for next wake-up
    uint32_t timer_ticks = wake_interval * 1000;  // Convert to microseconds
    nrf_timer_cc_set(RADIO_TIMER_INSTANCE, NRF_TIMER_CC_CHANNEL0, timer_ticks);
    nrf_timer_task_trigger(RADIO_TIMER_INSTANCE, NRF_TIMER_TASK_CLEAR);
    nrf_timer_task_trigger(RADIO_TIMER_INSTANCE, NRF_TIMER_TASK_START);
    
    return 0;
}

int sait01_radio_wake_up(void)
{
    uint32_t wake_start_time = k_uptime_get_32();
    
    printk("📡 Radio wake-up %d (listen for %d ms)\n", 
           radio_ctx.total_wake_count, radio_ctx.listen_window_ms);
    
    radio_ctx.current_state = RADIO_STATE_ACTIVE_RX;
    radio_ctx.last_wake_time = wake_start_time;
    radio_ctx.total_wake_count++;
    
    // Power on radio
    sait01_radio_power_on();
    
    // Listen for mesh synchronization packets
    bool sync_received = sait01_radio_listen_for_sync(radio_ctx.listen_window_ms);
    
    if (sync_received) {
        radio_ctx.successful_syncs++;
        radio_ctx.last_sync_time = wake_start_time;
        radio_ctx.synchronized = true;
        radio_ctx.missed_syncs = 0;
        
        // If we're coordinator, transmit beacon
        if (radio_ctx.coordinator_mode) {
            sait01_radio_transmit_beacon();
        }
    } else {
        radio_ctx.missed_syncs++;
        
        // If we've missed too many syncs, increase wake frequency
        if (radio_ctx.missed_syncs > 5) {
            radio_ctx.wake_interval_ms = RADIO_WAKE_INTERVAL_MS / 2;  // Double frequency
            printk("📡 Increasing wake frequency due to missed syncs\n");
        }
        
        // Lost synchronization after timeout
        if ((wake_start_time - radio_ctx.last_sync_time) > RADIO_SYNC_TIMEOUT_MS) {
            radio_ctx.synchronized = false;
        }
    }
    
    // Power off radio to save power
    sait01_radio_power_off();
    radio_ctx.current_state = RADIO_STATE_SCHEDULED_WAKE;
    
    // Schedule next wake-up
    sait01_radio_schedule_next_wake();
    
    // Calculate actual duty cycle
    uint32_t wake_duration = k_uptime_get_32() - wake_start_time;
    radio_ctx.duty_cycle_percent = (float)wake_duration / radio_ctx.wake_interval_ms * 100.0;
    
    return 0;
}

bool sait01_radio_listen_for_sync(uint32_t listen_duration_ms)
{
    uint32_t listen_start = k_uptime_get_32();
    uint32_t listen_end = listen_start + listen_duration_ms;
    
    // Configure radio for reception
    nrf_radio_shorts_set(NRF_RADIO_SHORT_READY_START_MASK | 
                         NRF_RADIO_SHORT_END_START_MASK);
    
    // Start listening
    nrf_radio_task_trigger(NRF_RADIO_TASK_RXEN);
    
    while (k_uptime_get_32() < listen_end) {
        // Check for received packets
        if (nrf_radio_event_check(NRF_RADIO_EVENT_END)) {
            nrf_radio_event_clear(NRF_RADIO_EVENT_END);
            
            // Check if packet CRC is valid
            if (nrf_radio_crc_status_check()) {
                // Process received packet
                if (sait01_radio_process_received_packet()) {
                    radio_ctx.packets_received++;
                    return true;  // Sync packet received
                }
            }
        }
        
        // Small delay to prevent busy waiting
        k_sleep(K_USEC(100));
    }
    
    // Stop radio
    nrf_radio_task_trigger(NRF_RADIO_TASK_STOP);
    
    return false;  // No sync packet received
}

bool sait01_radio_process_received_packet(void)
{
    // Simplified packet processing
    // In real implementation, this would:
    // 1. Parse packet header and payload
    // 2. Check if it's a mesh sync packet
    // 3. Extract timing information
    // 4. Update synchronization state
    
    mesh_sync_packet_t sync_packet;
    
    // Placeholder: assume packet is valid sync packet
    sync_packet.timestamp = k_uptime_get_32();
    sync_packet.next_beacon_time = sync_packet.timestamp + MESH_BEACON_INTERVAL_MS;
    sync_packet.node_id = 0x1234;  // Sender node ID
    sync_packet.sequence_number = 0;
    
    // Update synchronization timing
    radio_ctx.next_wake_time = sync_packet.next_beacon_time - 10;  // Wake 10ms before beacon
    
    printk("📡 Sync packet received from node 0x%04X\n", sync_packet.node_id);
    
    return true;
}

int sait01_radio_transmit_beacon(void)
{
    radio_ctx.current_state = RADIO_STATE_ACTIVE_TX;
    
    printk("📡 Transmitting mesh beacon\n");
    
    // Prepare beacon packet
    mesh_sync_packet_t beacon;
    beacon.timestamp = k_uptime_get_32();
    beacon.next_beacon_time = beacon.timestamp + MESH_BEACON_INTERVAL_MS;
    beacon.node_id = 0x0001;  // Our node ID
    beacon.sequence_number = radio_ctx.total_wake_count & 0xFF;
    
    // Configure radio for transmission
    nrf_radio_shorts_set(NRF_RADIO_SHORT_READY_START_MASK | 
                         NRF_RADIO_SHORT_END_DISABLE_MASK);
    
    // Set packet pointer (in real implementation, this would point to actual packet buffer)
    // nrf_radio_packetptr_set((uint32_t)&beacon);
    
    // Start transmission
    nrf_radio_task_trigger(NRF_RADIO_TASK_TXEN);
    
    // Wait for transmission to complete
    while (!nrf_radio_event_check(NRF_RADIO_EVENT_END)) {
        k_sleep(K_USEC(10));
    }
    nrf_radio_event_clear(NRF_RADIO_EVENT_END);
    
    radio_ctx.packets_transmitted++;
    
    return 0;
}

int sait01_radio_set_emergency_mode(bool enable)
{
    radio_ctx.emergency_mode = enable;
    
    if (enable) {
        // Increase wake frequency for emergency
        radio_ctx.wake_interval_ms = RADIO_EMERGENCY_INTERVAL_MS;
        printk("📡 Emergency mode ENABLED - %d ms wake interval\n", 
               RADIO_EMERGENCY_INTERVAL_MS);
    } else {
        // Return to normal wake frequency
        radio_ctx.wake_interval_ms = RADIO_WAKE_INTERVAL_MS;
        printk("📡 Emergency mode DISABLED - %d ms wake interval\n", 
               RADIO_WAKE_INTERVAL_MS);
    }
    
    // Reschedule next wake-up with new interval
    sait01_radio_schedule_next_wake();
    
    return 0;
}

int sait01_radio_set_coordinator_mode(bool enable)
{
    radio_ctx.coordinator_mode = enable;
    
    if (enable) {
        // Coordinator needs to transmit beacons regularly
        radio_ctx.listen_window_ms = RADIO_LISTEN_WINDOW_MS + 20;  // Longer listen window
        printk("📡 Coordinator mode ENABLED - extended listen window\n");
    } else {
        // Return to normal listen window
        radio_ctx.listen_window_ms = RADIO_LISTEN_WINDOW_MS;
        printk("📡 Coordinator mode DISABLED - normal listen window\n");
    }
    
    return 0;
}

smart_radio_stats_t sait01_radio_get_statistics(void)
{
    smart_radio_stats_t stats;
    uint32_t current_time = k_uptime_get_32();
    uint32_t total_runtime = current_time;
    
    stats.current_state = radio_ctx.current_state;
    stats.wake_interval_ms = radio_ctx.wake_interval_ms;
    stats.listen_window_ms = radio_ctx.listen_window_ms;
    stats.synchronized = radio_ctx.synchronized;
    stats.emergency_mode = radio_ctx.emergency_mode;
    stats.coordinator_mode = radio_ctx.coordinator_mode;
    stats.total_wake_count = radio_ctx.total_wake_count;
    stats.successful_syncs = radio_ctx.successful_syncs;
    stats.packets_received = radio_ctx.packets_received;
    stats.packets_transmitted = radio_ctx.packets_transmitted;
    stats.missed_syncs = radio_ctx.missed_syncs;
    
    // Calculate sync success rate
    if (radio_ctx.total_wake_count > 0) {
        stats.sync_success_rate = (float)radio_ctx.successful_syncs / radio_ctx.total_wake_count * 100.0;
    } else {
        stats.sync_success_rate = 0.0;
    }
    
    // Calculate actual duty cycle
    if (total_runtime > 0) {
        uint32_t total_active_time = radio_ctx.total_wake_count * radio_ctx.listen_window_ms;
        stats.actual_duty_cycle_percent = (float)total_active_time / total_runtime * 100.0;
    } else {
        stats.actual_duty_cycle_percent = 0.0;
    }
    
    // Calculate power savings
    float continuous_current = RADIO_IDLE_CURRENT_MA;
    float optimized_current = RADIO_OPTIMIZED_CURRENT_MA * stats.actual_duty_cycle_percent / 100.0;
    stats.power_savings_percent = (1.0 - optimized_current / continuous_current) * 100.0;
    
    // Estimate current consumption
    stats.estimated_current_ma = optimized_current;
    
    return stats;
}

/*
 * Radio Timer Interrupt Handler
 */
void radio_timer_isr(void)
{
    if (nrf_timer_event_check(RADIO_TIMER_INSTANCE, NRF_TIMER_EVENT_COMPARE_0)) {
        nrf_timer_event_clear(RADIO_TIMER_INSTANCE, NRF_TIMER_EVENT_COMPARE_0);
        nrf_timer_task_trigger(RADIO_TIMER_INSTANCE, NRF_TIMER_TASK_STOP);
        
        // Trigger radio wake-up
        k_wakeup(&radio_mgmt_thread);
    }
}

/* Register radio timer interrupt */
IRQ_CONNECT(TIMER3_IRQn, 3, radio_timer_isr, NULL, 0);

/*
 * Smart Radio Management Thread
 */
void sait01_radio_management_thread(void *arg1, void *arg2, void *arg3)
{
    ARG_UNUSED(arg1);
    ARG_UNUSED(arg2);
    ARG_UNUSED(arg3);
    
    printk("📡 Smart Radio Management Thread Started\n");
    
    while (1) {
        // Wait for wake-up signal or timeout
        int ret = k_sleep(K_MSEC(radio_ctx.wake_interval_ms));
        
        if (ret == 0 || k_uptime_get_32() >= radio_ctx.next_wake_time) {
            // Time to wake up radio
            sait01_radio_wake_up();
        }
        
        // Print statistics periodically
        static uint32_t stats_counter = 0;
        if (++stats_counter >= 10) {
            smart_radio_stats_t stats = sait01_radio_get_statistics();
            
            printk("📡 Radio Stats: %.1f%% duty, %.1f%% sync success, %.1f%% power savings\n",
                   stats.actual_duty_cycle_percent,
                   stats.sync_success_rate,
                   stats.power_savings_percent);
            
            stats_counter = 0;
        }
    }
}

/* Radio management thread definition */
K_THREAD_DEFINE(radio_mgmt_thread, 1536, sait01_radio_management_thread,
                NULL, NULL, NULL, 6, 0, 0);  /* Optimized stack and priority */

/*
 * Integration with Power Management
 */
int sait01_radio_set_power_mode(bool low_power)
{
    if (low_power) {
        // Enter deep sleep mode - radio completely off
        radio_ctx.current_state = RADIO_STATE_DEEP_SLEEP;
        sait01_radio_power_off();
        
        // Stop wake-up timer
        nrf_timer_task_trigger(RADIO_TIMER_INSTANCE, NRF_TIMER_TASK_STOP);
        
        printk("📡 Entering DEEP SLEEP mode\n");
    } else {
        // Resume scheduled wake-up mode
        radio_ctx.current_state = RADIO_STATE_SCHEDULED_WAKE;
        sait01_radio_schedule_next_wake();
        
        printk("📡 Resuming SCHEDULED WAKE mode\n");
    }
    
    return 0;
}

bool sait01_radio_is_synchronized(void)
{
    return radio_ctx.synchronized;
}

uint32_t sait01_radio_get_next_wake_time(void)
{
    return radio_ctx.next_wake_time;
}