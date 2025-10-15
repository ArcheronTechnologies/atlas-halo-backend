/*
 * SAIT_01 Power Management System
 * ==================================
 * Critical power optimizations for primary lithium battery deployment
 * Target: <1 mA average current (from 12.1 mA)
 */

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/pm/pm.h>
#include <zephyr/pm/device.h>
#include <nrfx_clock.h>
#include <nrfx_power.h>
#include <nrf_gpio.h>
#include <nrf_saadc.h>
#include <nrf_radio.h>
#include <hal/nrf_rtc.h>

#include "power_management.h"

/* Power Management Configuration */
#define POWER_SLEEP_DURATION_MS      9000    // 9 seconds sleep
#define POWER_ACTIVE_DURATION_MS     1000    // 1 second active  
#define POWER_DUTY_CYCLE_PERCENT     10      // 10% duty cycle
#define POWER_EMERGENCY_DUTY_CYCLE   25      // 25% during emergencies
#define POWER_DEEP_SLEEP_CURRENT_UA  2       // 2 µA deep sleep target

/* CPU Frequency Management */
#define CPU_FREQ_IDLE_MHZ           32       // Idle frequency
#define CPU_FREQ_ACTIVE_MHZ         128      // Active frequency
#define CPU_FREQ_EMERGENCY_MHZ      128      // Emergency frequency

/* Audio Sampling Configuration */
#define AUDIO_SAMPLE_RATE_IDLE      4000     // 4 kHz idle sampling
#define AUDIO_SAMPLE_RATE_ACTIVE    16000    // 16 kHz active sampling

/* Radio Management */
#define RADIO_WAKE_INTERVAL_MS      500      // Wake every 500ms
#define RADIO_LISTEN_WINDOW_MS      50       // Listen for 50ms
#define RADIO_SYNC_TIMEOUT_MS       5000     // Sync timeout

/* Power State Management */
typedef enum {
    POWER_STATE_DEEP_SLEEP,
    POWER_STATE_IDLE_MONITORING, 
    POWER_STATE_ACTIVE_DETECTION,
    POWER_STATE_EMERGENCY_ALERT,
    POWER_STATE_COORDINATOR_MODE
} power_state_t;

typedef struct {
    power_state_t current_state;
    power_state_t previous_state;
    uint32_t state_entry_time;
    uint32_t sleep_duration_ms;
    uint32_t active_duration_ms;
    uint8_t duty_cycle_percent;
    bool emergency_mode;
    bool coordinator_mode;
    uint32_t battery_voltage_mv;
    uint32_t total_sleep_time_ms;
    uint32_t total_active_time_ms;
} power_mgmt_context_t;

static power_mgmt_context_t power_ctx = {0};

/* RTC for wake-up timing */
#define RTC_INSTANCE         NRF_RTC1
#define RTC_PRESCALER        327    // 32.768 kHz / (327+1) = 100 Hz = 10ms tick

/* GPIO for wake-up sources */
#define MOTION_SENSOR_PIN    DT_GPIO_PIN(DT_ALIAS(sw0), gpios)
#define AUDIO_TRIGGER_PIN    DT_GPIO_PIN(DT_ALIAS(sw1), gpios)

/*
 * Power State Machine Implementation
 */

int sait01_power_init(void)
{
    // Initialize power management context
    power_ctx.current_state = POWER_STATE_IDLE_MONITORING;
    power_ctx.sleep_duration_ms = POWER_SLEEP_DURATION_MS;
    power_ctx.active_duration_ms = POWER_ACTIVE_DURATION_MS;
    power_ctx.duty_cycle_percent = POWER_DUTY_CYCLE_PERCENT;
    power_ctx.emergency_mode = false;
    power_ctx.coordinator_mode = false;
    
    // Configure RTC for wake-up timing
    nrf_rtc_prescaler_set(RTC_INSTANCE, RTC_PRESCALER);
    nrf_rtc_event_clear(RTC_INSTANCE, NRF_RTC_EVENT_COMPARE_0);
    nrf_rtc_int_enable(RTC_INSTANCE, NRF_RTC_INT_COMPARE0_MASK);
    
    // Configure wake-up GPIO pins
    nrf_gpio_cfg_input(MOTION_SENSOR_PIN, NRF_GPIO_PIN_PULLUP);
    nrf_gpio_cfg_input(AUDIO_TRIGGER_PIN, NRF_GPIO_PIN_PULLUP);
    
    // Enable wake-up from GPIO
    nrf_gpio_cfg_sense_input(MOTION_SENSOR_PIN, NRF_GPIO_PIN_PULLUP, NRF_GPIO_PIN_SENSE_LOW);
    nrf_gpio_cfg_sense_input(AUDIO_TRIGGER_PIN, NRF_GPIO_PIN_PULLUP, NRF_GPIO_PIN_SENSE_LOW);
    
    printk("Power Management: Initialized (Target: %d%% duty cycle)\n", 
           power_ctx.duty_cycle_percent);
    
    return 0;
}

int sait01_power_set_cpu_frequency(uint32_t frequency_mhz)
{
    nrfx_clock_hfclk_div_t div;
    
    switch (frequency_mhz) {
        case 32:
            div = NRF_CLOCK_HFCLK_DIV_4;   // 128 MHz / 4 = 32 MHz
            break;
        case 64:
            div = NRF_CLOCK_HFCLK_DIV_2;   // 128 MHz / 2 = 64 MHz
            break;
        case 128:
            div = NRF_CLOCK_HFCLK_DIV_1;   // 128 MHz / 1 = 128 MHz
            break;
        default:
            return -EINVAL;
    }
    
    nrf_clock_hfclk_div_set(div);
    
    // Wait for frequency change to stabilize
    k_sleep(K_MSEC(1));
    
    return 0;
}

int sait01_power_disable_peripherals(void)
{
    // Disable high-power peripherals during sleep
    
    // Disable SAADC (Audio ADC)
    nrf_saadc_disable();
    
    // Disable radio (will be managed by radio wake-up)
    nrf_radio_power_set(false);
    
    // Disable unused GPIO pins to prevent leakage
    for (int pin = 0; pin < 32; pin++) {
        if (pin != MOTION_SENSOR_PIN && pin != AUDIO_TRIGGER_PIN) {
            nrf_gpio_cfg_default(pin);
        }
    }
    
    // Disable unused timers and peripherals
    // (Keep RTC1 for wake-up timing)
    
    return 0;
}

int sait01_power_enable_peripherals(void)
{
    // Re-enable peripherals for active operation
    
    // Enable SAADC for audio sampling
    nrf_saadc_enable();
    
    // Enable radio for mesh communication
    nrf_radio_power_set(true);
    
    // Configure GPIO pins for sensors
    // (Implementation specific to sensor configuration)
    
    return 0;
}

int sait01_power_enter_deep_sleep(uint32_t sleep_duration_ms)
{
    uint32_t sleep_ticks;
    
    printk("Entering deep sleep for %d ms\n", sleep_duration_ms);
    
    // Record sleep entry time
    power_ctx.state_entry_time = k_uptime_get_32();
    power_ctx.current_state = POWER_STATE_DEEP_SLEEP;
    
    // Disable all non-essential peripherals
    sait01_power_disable_peripherals();
    
    // Set CPU to lowest frequency
    sait01_power_set_cpu_frequency(32);
    
    // Configure RTC wake-up
    sleep_ticks = (sleep_duration_ms * 100) / 1000;  // Convert to 10ms ticks
    nrf_rtc_cc_set(RTC_INSTANCE, 0, nrf_rtc_counter_get(RTC_INSTANCE) + sleep_ticks);
    nrf_rtc_task_trigger(RTC_INSTANCE, NRF_RTC_TASK_START);
    
    // Enter System OFF mode (lowest power)
    // NOTE: This will cause a reset wake-up in real implementation
    // For development, we'll use a lighter sleep mode
    
    // Alternative: Use System ON with CPU sleep
    pm_state_force(0U, &(struct pm_state_info){PM_STATE_SOFT_OFF, 0, 0});
    
    // Execution resumes here after wake-up
    
    // Record wake-up time
    uint32_t sleep_time = k_uptime_get_32() - power_ctx.state_entry_time;
    power_ctx.total_sleep_time_ms += sleep_time;
    
    printk("Woke up after %d ms sleep\n", sleep_time);
    
    return 0;
}

int sait01_power_enter_active_monitoring(uint32_t active_duration_ms)
{
    printk("Entering active monitoring for %d ms\n", active_duration_ms);
    
    // Record active entry time
    power_ctx.state_entry_time = k_uptime_get_32();
    power_ctx.current_state = POWER_STATE_ACTIVE_DETECTION;
    
    // Enable all required peripherals
    sait01_power_enable_peripherals();
    
    // Set CPU to active frequency
    sait01_power_set_cpu_frequency(CPU_FREQ_ACTIVE_MHZ);
    
    // Configure audio sampling for active detection
    // (This would integrate with audio processing system)
    
    // Start audio processing and threat detection
    // (Integration point with main detection system)
    
    return 0;
}

int sait01_power_set_emergency_mode(bool enable)
{
    power_ctx.emergency_mode = enable;
    
    if (enable) {
        // Increase duty cycle for emergency response
        power_ctx.duty_cycle_percent = POWER_EMERGENCY_DUTY_CYCLE;
        power_ctx.sleep_duration_ms = 3000;  // 3 seconds sleep
        power_ctx.active_duration_ms = 1000; // 1 second active
        
        // Set maximum CPU frequency
        sait01_power_set_cpu_frequency(CPU_FREQ_EMERGENCY_MHZ);
        
        printk("Emergency mode ENABLED - %d%% duty cycle\n", POWER_EMERGENCY_DUTY_CYCLE);
    } else {
        // Return to normal duty cycle
        power_ctx.duty_cycle_percent = POWER_DUTY_CYCLE_PERCENT;
        power_ctx.sleep_duration_ms = POWER_SLEEP_DURATION_MS;
        power_ctx.active_duration_ms = POWER_ACTIVE_DURATION_MS;
        
        printk("Emergency mode DISABLED - %d%% duty cycle\n", POWER_DUTY_CYCLE_PERCENT);
    }
    
    return 0;
}

int sait01_power_set_coordinator_mode(bool enable)
{
    power_ctx.coordinator_mode = enable;
    
    if (enable) {
        // Coordinator needs higher duty cycle for mesh management
        power_ctx.duty_cycle_percent = 20;  // 20% duty cycle
        power_ctx.sleep_duration_ms = 4000; // 4 seconds sleep
        power_ctx.active_duration_ms = 1000; // 1 second active
        
        printk("Coordinator mode ENABLED - 20%% duty cycle\n");
    } else {
        // Return to normal duty cycle
        power_ctx.duty_cycle_percent = POWER_DUTY_CYCLE_PERCENT;
        power_ctx.sleep_duration_ms = POWER_SLEEP_DURATION_MS;
        power_ctx.active_duration_ms = POWER_ACTIVE_DURATION_MS;
        
        printk("Coordinator mode DISABLED - %d%% duty cycle\n", POWER_DUTY_CYCLE_PERCENT);
    }
    
    return 0;
}

uint32_t sait01_power_get_battery_voltage_mv(void)
{
    // Measure battery voltage using SAADC
    // This is a simplified implementation
    
    uint16_t adc_value;
    uint32_t battery_mv;
    
    // Configure SAADC for battery measurement
    nrf_saadc_channel_config_t channel_config = {
        .resistor_p = NRF_SAADC_RESISTOR_DISABLED,
        .resistor_n = NRF_SAADC_RESISTOR_DISABLED,
        .gain = NRF_SAADC_GAIN1_6,
        .reference = NRF_SAADC_REFERENCE_INTERNAL,
        .acq_time = NRF_SAADC_ACQTIME_10US,
        .mode = NRF_SAADC_MODE_SINGLE_ENDED,
        .burst = NRF_SAADC_BURST_DISABLED,
        .pin_p = NRF_SAADC_INPUT_VDD,
        .pin_n = NRF_SAADC_INPUT_DISABLED
    };
    
    // Simplified ADC reading
    adc_value = 2048;  // Placeholder - real implementation would read ADC
    
    // Convert to millivolts (assuming 3.6V reference and gain of 1/6)
    battery_mv = (adc_value * 3600) / 4096;
    
    power_ctx.battery_voltage_mv = battery_mv;
    
    return battery_mv;
}

power_mgmt_stats_t sait01_power_get_statistics(void)
{
    power_mgmt_stats_t stats;
    uint32_t total_time = power_ctx.total_sleep_time_ms + power_ctx.total_active_time_ms;
    
    stats.total_sleep_time_ms = power_ctx.total_sleep_time_ms;
    stats.total_active_time_ms = power_ctx.total_active_time_ms;
    stats.actual_duty_cycle_percent = total_time > 0 ? 
        (power_ctx.total_active_time_ms * 100) / total_time : 0;
    stats.current_state = power_ctx.current_state;
    stats.battery_voltage_mv = power_ctx.battery_voltage_mv;
    stats.emergency_mode = power_ctx.emergency_mode;
    stats.coordinator_mode = power_ctx.coordinator_mode;
    
    // Calculate estimated current consumption
    if (stats.actual_duty_cycle_percent > 0) {
        // Optimized current: 2 µA sleep + 0.8 mA * duty_cycle
        stats.estimated_current_ma = 0.002 + (0.8 * stats.actual_duty_cycle_percent / 100.0);
    } else {
        stats.estimated_current_ma = 0.002;  // Deep sleep only
    }
    
    // Calculate estimated battery life (3000 mAh AA Lithium, 85% usable)
    float usable_capacity = 3000 * 0.85;
    stats.estimated_battery_life_hours = usable_capacity / stats.estimated_current_ma;
    
    return stats;
}

/*
 * Main Power Management Cycle
 */
void sait01_power_management_thread(void *arg1, void *arg2, void *arg3)
{
    ARG_UNUSED(arg1);
    ARG_UNUSED(arg2); 
    ARG_UNUSED(arg3);
    
    printk("Power Management Thread Started\n");
    
    while (1) {
        // Check for emergency conditions
        if (power_ctx.emergency_mode) {
            // Emergency mode: higher duty cycle
            sait01_power_enter_active_monitoring(power_ctx.active_duration_ms);
            k_sleep(K_MSEC(power_ctx.active_duration_ms));
            
            sait01_power_enter_deep_sleep(power_ctx.sleep_duration_ms);
            k_sleep(K_MSEC(power_ctx.sleep_duration_ms));
        } else {
            // Normal operation: duty cycling
            
            // Active monitoring phase
            uint32_t active_start = k_uptime_get_32();
            sait01_power_enter_active_monitoring(power_ctx.active_duration_ms);
            
            // TODO: Integrate with main threat detection system
            // - Start audio processing
            // - Run ML inference
            // - Check for threats
            // - Handle mesh communication
            
            k_sleep(K_MSEC(power_ctx.active_duration_ms));
            
            // Record actual active time
            uint32_t actual_active_time = k_uptime_get_32() - active_start;
            power_ctx.total_active_time_ms += actual_active_time;
            
            // Deep sleep phase
            sait01_power_enter_deep_sleep(power_ctx.sleep_duration_ms);
            k_sleep(K_MSEC(power_ctx.sleep_duration_ms));
        }
        
        // Update battery voltage periodically
        if ((k_uptime_get_32() % 60000) == 0) {  // Every minute
            sait01_power_get_battery_voltage_mv();
            
            // Print power statistics
            power_mgmt_stats_t stats = sait01_power_get_statistics();
            printk("Power Stats: %.2f mA avg, %.1f%% duty, %.0f hours battery life\n",
                   stats.estimated_current_ma, 
                   stats.actual_duty_cycle_percent,
                   stats.estimated_battery_life_hours);
        }
    }
}

/* Power management thread definition */
K_THREAD_DEFINE(power_mgmt_thread, 1024, sait01_power_management_thread,
                NULL, NULL, NULL, 8, 0, 0);  /* Lower priority background task */

/*
 * RTC Interrupt Handler for Wake-up
 */
void rtc1_isr(void)
{
    if (nrf_rtc_event_check(RTC_INSTANCE, NRF_RTC_EVENT_COMPARE_0)) {
        nrf_rtc_event_clear(RTC_INSTANCE, NRF_RTC_EVENT_COMPARE_0);
        
        // Wake-up from deep sleep
        printk("RTC Wake-up triggered\n");
        
        // Signal main thread to resume operation
        k_wakeup(&power_mgmt_thread);
    }
}

/* Register RTC interrupt */
IRQ_CONNECT(RTC1_IRQn, 1, rtc1_isr, NULL, 0);

/*
 * GPIO Interrupt Handler for Motion/Audio Wake-up
 */
void gpio_isr(const struct device *dev, struct gpio_callback *cb, uint32_t pins)
{
    if (pins & BIT(MOTION_SENSOR_PIN)) {
        printk("Motion sensor wake-up\n");
        
        // Immediately enter active mode for motion detection
        power_ctx.emergency_mode = true;
        k_wakeup(&power_mgmt_thread);
    }
    
    if (pins & BIT(AUDIO_TRIGGER_PIN)) {
        printk("Audio trigger wake-up\n");
        
        // Enter active mode for audio processing
        k_wakeup(&power_mgmt_thread);
    }
}