/*
 * SAIT_01 Smart Radio Management System - Header
 * ==================================================
 * Power-optimized mesh radio with periodic wake-up
 */

#ifndef SMART_RADIO_H
#define SMART_RADIO_H

#include <zephyr/types.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Smart Radio Statistics Structure
 */
typedef struct {
    uint8_t current_state;                  // Current radio state
    uint32_t wake_interval_ms;              // Wake-up interval
    uint32_t listen_window_ms;              // Listen window duration
    bool synchronized;                      // Mesh synchronization status
    bool emergency_mode;                    // Emergency mode active
    bool coordinator_mode;                  // Coordinator mode active
    uint32_t total_wake_count;              // Total number of wake-ups
    uint32_t successful_syncs;              // Successful synchronizations
    uint32_t packets_received;              // Packets successfully received
    uint32_t packets_transmitted;           // Packets transmitted
    uint8_t missed_syncs;                   // Consecutive missed syncs
    float sync_success_rate;                // Sync success rate percentage
    float actual_duty_cycle_percent;        // Actual radio duty cycle
    float power_savings_percent;            // Power savings compared to continuous
    float estimated_current_ma;             // Estimated current consumption
} smart_radio_stats_t;

/*
 * Smart Radio API Functions
 */

/**
 * @brief Initialize smart radio management system
 * 
 * Sets up periodic wake-up, mesh synchronization, and power optimization
 * 
 * @return 0 on success, negative error code on failure
 */
int sait01_radio_init(void);

/**
 * @brief Power on radio hardware
 * 
 * Enables radio and configures for mesh operation
 * 
 * @return 0 on success, negative error code on failure
 */
int sait01_radio_power_on(void);

/**
 * @brief Power off radio hardware
 * 
 * Disables radio to save power
 * 
 * @return 0 on success, negative error code on failure
 */
int sait01_radio_power_off(void);

/**
 * @brief Schedule next radio wake-up
 * 
 * Programs timer for next periodic wake-up
 * 
 * @return 0 on success, negative error code on failure
 */
int sait01_radio_schedule_next_wake(void);

/**
 * @brief Execute radio wake-up cycle
 * 
 * Powers on radio, listens for sync packets, transmits if needed, powers off
 * 
 * @return 0 on success, negative error code on failure
 */
int sait01_radio_wake_up(void);

/**
 * @brief Listen for mesh synchronization packets
 * 
 * @param listen_duration_ms Duration to listen in milliseconds
 * @return true if sync packet received, false if timeout
 */
bool sait01_radio_listen_for_sync(uint32_t listen_duration_ms);

/**
 * @brief Process received radio packet
 * 
 * Parses packet and updates synchronization state
 * 
 * @return true if valid sync packet processed
 */
bool sait01_radio_process_received_packet(void);

/**
 * @brief Transmit mesh beacon packet
 * 
 * Sends synchronization beacon (coordinator mode)
 * 
 * @return 0 on success, negative error code on failure
 */
int sait01_radio_transmit_beacon(void);

/**
 * @brief Enable/disable emergency mode
 * 
 * Emergency mode increases wake frequency for rapid response
 * 
 * @param enable true to enable emergency mode, false to disable
 * @return 0 on success, negative error code on failure
 */
int sait01_radio_set_emergency_mode(bool enable);

/**
 * @brief Enable/disable coordinator mode
 * 
 * Coordinator mode enables beacon transmission
 * 
 * @param enable true to enable coordinator mode, false to disable
 * @return 0 on success, negative error code on failure
 */
int sait01_radio_set_coordinator_mode(bool enable);

/**
 * @brief Set power management mode for radio system
 * 
 * @param low_power true to enable deep sleep, false for scheduled wake-up
 * @return 0 on success, negative error code on failure
 */
int sait01_radio_set_power_mode(bool low_power);

/**
 * @brief Check if radio is synchronized with mesh
 * 
 * @return true if synchronized, false if not
 */
bool sait01_radio_is_synchronized(void);

/**
 * @brief Get next scheduled wake-up time
 * 
 * @return Next wake-up time in system ticks
 */
uint32_t sait01_radio_get_next_wake_time(void);

/**
 * @brief Get smart radio statistics
 * 
 * @return Structure containing radio system statistics
 */
smart_radio_stats_t sait01_radio_get_statistics(void);

/*
 * Smart Radio Configuration Macros
 */
#define SMART_RADIO_WAKE_INTERVAL_MS        500     // Normal wake interval
#define SMART_RADIO_LISTEN_WINDOW_MS        50      // Listen window duration
#define SMART_RADIO_EMERGENCY_INTERVAL_MS   100     // Emergency wake interval
#define SMART_RADIO_SYNC_TIMEOUT_MS         5000    // Sync timeout
#define SMART_RADIO_POWER_REDUCTION         90      // Power reduction percentage

#ifdef __cplusplus
}
#endif

#endif /* SMART_RADIO_H */