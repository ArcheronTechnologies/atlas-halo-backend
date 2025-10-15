/*
 * 🔋 SAIT_01 Power Management System - Header
 * ===========================================
 * Critical power optimizations for primary lithium battery deployment
 */

#ifndef POWER_MANAGEMENT_H
#define POWER_MANAGEMENT_H

#include <zephyr/types.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Power Management Statistics Structure
 */
typedef struct {
    uint32_t total_sleep_time_ms;           // Total time spent sleeping
    uint32_t total_active_time_ms;          // Total time spent active
    float actual_duty_cycle_percent;        // Actual measured duty cycle
    uint8_t current_state;                  // Current power state
    uint32_t battery_voltage_mv;            // Battery voltage in millivolts
    bool emergency_mode;                    // Emergency mode active
    bool coordinator_mode;                  // Coordinator mode active
    float estimated_current_ma;             // Estimated average current draw
    float estimated_battery_life_hours;     // Estimated battery life remaining
} power_mgmt_stats_t;

/*
 * Power Management API Functions
 */

/**
 * @brief Initialize power management system
 * 
 * Sets up duty cycling, CPU frequency scaling, and wake-up sources
 * 
 * @return 0 on success, negative error code on failure
 */
int sait01_power_init(void);

/**
 * @brief Set CPU frequency for power optimization
 * 
 * @param frequency_mhz Target frequency (32, 64, or 128 MHz)
 * @return 0 on success, -EINVAL for invalid frequency
 */
int sait01_power_set_cpu_frequency(uint32_t frequency_mhz);

/**
 * @brief Enter deep sleep mode for specified duration
 * 
 * Disables all non-essential peripherals and enters lowest power mode
 * 
 * @param sleep_duration_ms Duration to sleep in milliseconds
 * @return 0 on success, negative error code on failure
 */
int sait01_power_enter_deep_sleep(uint32_t sleep_duration_ms);

/**
 * @brief Enter active monitoring mode
 * 
 * Enables all peripherals and sets CPU to active frequency
 * 
 * @param active_duration_ms Duration to stay active in milliseconds
 * @return 0 on success, negative error code on failure
 */
int sait01_power_enter_active_monitoring(uint32_t active_duration_ms);

/**
 * @brief Enable/disable emergency mode
 * 
 * Emergency mode increases duty cycle to 25% for rapid response
 * 
 * @param enable true to enable emergency mode, false to disable
 * @return 0 on success, negative error code on failure
 */
int sait01_power_set_emergency_mode(bool enable);

/**
 * @brief Enable/disable coordinator mode
 * 
 * Coordinator mode increases duty cycle to 20% for mesh management
 * 
 * @param enable true to enable coordinator mode, false to disable
 * @return 0 on success, negative error code on failure
 */
int sait01_power_set_coordinator_mode(bool enable);

/**
 * @brief Get current battery voltage
 * 
 * @return Battery voltage in millivolts
 */
uint32_t sait01_power_get_battery_voltage_mv(void);

/**
 * @brief Get power management statistics
 * 
 * @return Structure containing power management statistics
 */
power_mgmt_stats_t sait01_power_get_statistics(void);

/**
 * @brief Disable peripherals for low power mode
 * 
 * Internal function to disable high-power peripherals
 * 
 * @return 0 on success, negative error code on failure
 */
int sait01_power_disable_peripherals(void);

/**
 * @brief Enable peripherals for active mode
 * 
 * Internal function to enable required peripherals
 * 
 * @return 0 on success, negative error code on failure
 */
int sait01_power_enable_peripherals(void);

/*
 * Power Management Configuration Macros
 */
#define POWER_MGMT_SLEEP_DURATION_MS    9000    // Default sleep duration
#define POWER_MGMT_ACTIVE_DURATION_MS   1000    // Default active duration
#define POWER_MGMT_DUTY_CYCLE_PERCENT   10      // Default duty cycle
#define POWER_MGMT_TARGET_CURRENT_MA    1.0     // Target average current
#define POWER_MGMT_DEEP_SLEEP_CURRENT_UA 2      // Deep sleep current

#ifdef __cplusplus
}
#endif

#endif /* POWER_MANAGEMENT_H */