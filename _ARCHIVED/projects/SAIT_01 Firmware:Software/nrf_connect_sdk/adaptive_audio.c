/*
 * 🎵 SAIT_01 Adaptive Audio Sampling System
 * ==========================================
 * Power-optimized audio processing with dynamic sample rates
 * Target: 4x reduction in ADC power consumption
 */

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/adc.h>
#include <nrf_saadc.h>
#include <nrf_ppi.h>
#include <nrf_timer.h>

#include "adaptive_audio.h"
#include "power_management.h"

/* Audio Sampling Configuration */
#define AUDIO_SAMPLE_RATE_IDLE      4000     // 4 kHz idle sampling (4x power reduction)
#define AUDIO_SAMPLE_RATE_ACTIVE    16000    // 16 kHz active sampling (full resolution)
#define AUDIO_BUFFER_SIZE_SAMPLES   1024     // Buffer size for processing
#define AUDIO_CHUNK_SIZE_MS         32       // 32ms processing chunks

/* Power Optimization Thresholds */
#define AUDIO_ENERGY_THRESHOLD      100      // Energy level to trigger high-res mode
#define AUDIO_SILENCE_TIMEOUT_MS    5000     // Return to low-res after 5s silence
#define AUDIO_PEAK_THRESHOLD        512      // Peak amplitude threshold

/* SAADC Configuration */
#define SAADC_CHANNEL               0
#define SAADC_GAIN                  NRF_SAADC_GAIN1_6
#define SAADC_REFERENCE             NRF_SAADC_REFERENCE_INTERNAL
#define SAADC_RESOLUTION            NRF_SAADC_RESOLUTION_12BIT

/* Audio processing state */
typedef enum {
    AUDIO_STATE_IDLE,           // Low power 4 kHz sampling
    AUDIO_STATE_TRIGGERED,      // Transitioning to high resolution
    AUDIO_STATE_ACTIVE,         // High resolution 16 kHz sampling
    AUDIO_STATE_PROCESSING      // Active threat detection
} audio_state_t;

typedef struct {
    audio_state_t current_state;
    uint32_t current_sample_rate;
    uint32_t state_entry_time;
    uint32_t last_trigger_time;
    uint16_t audio_buffer[AUDIO_BUFFER_SIZE_SAMPLES];
    uint16_t buffer_index;
    uint32_t total_samples_processed;
    uint32_t energy_level;
    uint16_t peak_amplitude;
    bool high_res_mode;
    uint32_t mode_switch_count;
    float power_savings_percent;
} adaptive_audio_context_t;

static adaptive_audio_context_t audio_ctx = {0};

/* Timer for sample rate control */
#define TIMER_INSTANCE              NRF_TIMER2
#define PPI_CHANNEL                 0

/*
 * SAADC Configuration and Control
 */

int sait01_audio_init(void)
{
    // Initialize audio context
    audio_ctx.current_state = AUDIO_STATE_IDLE;
    audio_ctx.current_sample_rate = AUDIO_SAMPLE_RATE_IDLE;
    audio_ctx.buffer_index = 0;
    audio_ctx.high_res_mode = false;
    audio_ctx.mode_switch_count = 0;
    
    // Configure SAADC
    nrf_saadc_resolution_set(SAADC_RESOLUTION);
    nrf_saadc_oversample_set(NRF_SAADC_OVERSAMPLE_DISABLED);
    
    // Configure SAADC channel for microphone input
    nrf_saadc_channel_config_t channel_config = {
        .resistor_p = NRF_SAADC_RESISTOR_DISABLED,
        .resistor_n = NRF_SAADC_RESISTOR_DISABLED,
        .gain = SAADC_GAIN,
        .reference = SAADC_REFERENCE,
        .acq_time = NRF_SAADC_ACQTIME_10US,
        .mode = NRF_SAADC_MODE_DIFFERENTIAL,
        .burst = NRF_SAADC_BURST_DISABLED,
        .pin_p = NRF_SAADC_INPUT_AIN0,  // Microphone positive
        .pin_n = NRF_SAADC_INPUT_AIN1   // Microphone negative
    };
    
    nrf_saadc_channel_init(SAADC_CHANNEL, &channel_config);
    
    // Configure timer for sample rate control
    nrf_timer_mode_set(TIMER_INSTANCE, NRF_TIMER_MODE_TIMER);
    nrf_timer_bit_width_set(TIMER_INSTANCE, NRF_TIMER_BIT_WIDTH_32);
    nrf_timer_prescaler_set(TIMER_INSTANCE, NRF_TIMER_FREQ_1MHz);
    
    // Set initial sample rate
    sait01_audio_set_sample_rate(AUDIO_SAMPLE_RATE_IDLE);
    
    // Enable SAADC
    nrf_saadc_enable();
    
    printk("🎵 Adaptive Audio: Initialized (4 kHz idle, 16 kHz active)\n");
    
    return 0;
}

int sait01_audio_set_sample_rate(uint32_t sample_rate_hz)
{
    uint32_t timer_ticks;
    
    // Calculate timer ticks for desired sample rate
    // Timer frequency is 1 MHz, so ticks = 1,000,000 / sample_rate
    timer_ticks = 1000000 / sample_rate_hz;
    
    // Stop timer
    nrf_timer_task_trigger(TIMER_INSTANCE, NRF_TIMER_TASK_STOP);
    nrf_timer_task_trigger(TIMER_INSTANCE, NRF_TIMER_TASK_CLEAR);
    
    // Set compare value for sample rate
    nrf_timer_cc_set(TIMER_INSTANCE, NRF_TIMER_CC_CHANNEL0, timer_ticks);
    
    // Configure PPI to trigger SAADC sampling
    nrf_ppi_channel_endpoint_setup(
        PPI_CHANNEL,
        nrf_timer_event_address_get(TIMER_INSTANCE, NRF_TIMER_EVENT_COMPARE0),
        nrf_saadc_task_address_get(NRF_SAADC_TASK_SAMPLE)
    );
    nrf_ppi_channel_enable(PPI_CHANNEL);
    
    // Start timer
    nrf_timer_task_trigger(TIMER_INSTANCE, NRF_TIMER_TASK_START);
    
    audio_ctx.current_sample_rate = sample_rate_hz;
    
    printk("🎵 Sample rate set to %d Hz\n", sample_rate_hz);
    
    return 0;
}

int sait01_audio_calculate_energy(uint16_t *samples, uint16_t count)
{
    uint32_t energy = 0;
    uint16_t peak = 0;
    uint16_t baseline = 2048;  // 12-bit ADC midpoint
    
    // Calculate RMS energy and peak amplitude
    for (uint16_t i = 0; i < count; i++) {
        uint16_t sample = samples[i];
        uint16_t amplitude = abs((int16_t)sample - (int16_t)baseline);
        
        energy += amplitude * amplitude;
        if (amplitude > peak) {
            peak = amplitude;
        }
    }
    
    energy = energy / count;  // Average energy
    
    audio_ctx.energy_level = energy;
    audio_ctx.peak_amplitude = peak;
    
    return energy;
}

bool sait01_audio_should_trigger_high_res(void)
{
    // Check if we should switch to high resolution mode
    
    // Energy-based trigger
    if (audio_ctx.energy_level > AUDIO_ENERGY_THRESHOLD) {
        return true;
    }
    
    // Peak amplitude trigger
    if (audio_ctx.peak_amplitude > AUDIO_PEAK_THRESHOLD) {
        return true;
    }
    
    // Pattern-based trigger (simplified)
    // In full implementation, this would use basic signal processing
    // to detect audio patterns that might indicate threats
    
    return false;
}

bool sait01_audio_should_return_to_idle(void)
{
    uint32_t current_time = k_uptime_get_32();
    uint32_t silence_duration = current_time - audio_ctx.last_trigger_time;
    
    // Return to idle mode after silence timeout
    if (silence_duration > AUDIO_SILENCE_TIMEOUT_MS) {
        return true;
    }
    
    // Return to idle if energy level is consistently low
    if (audio_ctx.energy_level < (AUDIO_ENERGY_THRESHOLD / 4)) {
        return true;
    }
    
    return false;
}

int sait01_audio_transition_to_active(void)
{
    if (audio_ctx.current_state == AUDIO_STATE_ACTIVE) {
        return 0;  // Already active
    }
    
    printk("🎵 Transitioning to HIGH RESOLUTION mode (16 kHz)\n");
    
    audio_ctx.current_state = AUDIO_STATE_TRIGGERED;
    audio_ctx.state_entry_time = k_uptime_get_32();
    audio_ctx.last_trigger_time = k_uptime_get_32();
    audio_ctx.mode_switch_count++;
    
    // Switch to high resolution sampling
    sait01_audio_set_sample_rate(AUDIO_SAMPLE_RATE_ACTIVE);
    audio_ctx.high_res_mode = true;
    
    // Notify power management system
    // (This could trigger CPU frequency scaling)
    
    // Small delay for stabilization
    k_sleep(K_MSEC(10));
    
    audio_ctx.current_state = AUDIO_STATE_ACTIVE;
    
    return 0;
}

int sait01_audio_transition_to_idle(void)
{
    if (audio_ctx.current_state == AUDIO_STATE_IDLE) {
        return 0;  // Already idle
    }
    
    printk("🎵 Transitioning to LOW POWER mode (4 kHz)\n");
    
    audio_ctx.current_state = AUDIO_STATE_IDLE;
    audio_ctx.state_entry_time = k_uptime_get_32();
    
    // Switch to low resolution sampling
    sait01_audio_set_sample_rate(AUDIO_SAMPLE_RATE_IDLE);
    audio_ctx.high_res_mode = false;
    
    // Notify power management system
    // (This could trigger CPU frequency scaling down)
    
    return 0;
}

void sait01_audio_process_samples(uint16_t *samples, uint16_t count)
{
    // Update total sample count
    audio_ctx.total_samples_processed += count;
    
    // Calculate audio energy and characteristics
    sait01_audio_calculate_energy(samples, count);
    
    // State machine for adaptive sampling
    switch (audio_ctx.current_state) {
        case AUDIO_STATE_IDLE:
            // Check if we should trigger high resolution mode
            if (sait01_audio_should_trigger_high_res()) {
                sait01_audio_transition_to_active();
            }
            break;
            
        case AUDIO_STATE_ACTIVE:
        case AUDIO_STATE_PROCESSING:
            // Check if we should return to idle mode
            if (sait01_audio_should_return_to_idle()) {
                sait01_audio_transition_to_idle();
            } else {
                // Continue high resolution processing
                // This is where full ML inference would occur
                audio_ctx.current_state = AUDIO_STATE_PROCESSING;
                
                // TODO: Integrate with main threat detection system
                // - Run TensorFlow Lite inference
                // - Analyze spectral features
                // - Detect threat signatures
            }
            break;
            
        default:
            audio_ctx.current_state = AUDIO_STATE_IDLE;
            break;
    }
}

adaptive_audio_stats_t sait01_audio_get_statistics(void)
{
    adaptive_audio_stats_t stats;
    uint32_t current_time = k_uptime_get_32();
    uint32_t total_runtime = current_time;
    
    stats.current_sample_rate = audio_ctx.current_sample_rate;
    stats.current_state = audio_ctx.current_state;
    stats.total_samples_processed = audio_ctx.total_samples_processed;
    stats.energy_level = audio_ctx.energy_level;
    stats.peak_amplitude = audio_ctx.peak_amplitude;
    stats.high_res_mode = audio_ctx.high_res_mode;
    stats.mode_switch_count = audio_ctx.mode_switch_count;
    
    // Calculate power savings
    // Assuming 4x power reduction when in idle mode
    uint32_t idle_time = 0;  // Simplified calculation
    if (total_runtime > 0) {
        float idle_ratio = (float)idle_time / total_runtime;
        stats.power_savings_percent = idle_ratio * 75.0;  // 75% power reduction in idle
    } else {
        stats.power_savings_percent = 0.0;
    }
    
    // Calculate effective sample rate (average over time)
    // This accounts for time spent in each mode
    stats.effective_sample_rate = audio_ctx.current_sample_rate;  // Simplified
    
    return stats;
}

/*
 * SAADC Interrupt Handler
 */
void saadc_isr(void)
{
    if (nrf_saadc_event_check(NRF_SAADC_EVENT_DONE)) {
        nrf_saadc_event_clear(NRF_SAADC_EVENT_DONE);
        
        // Read sample from SAADC result buffer
        nrf_saadc_value_t sample_value = nrf_saadc_result_get(SAADC_CHANNEL);
        
        // Store sample in buffer
        if (audio_ctx.buffer_index < AUDIO_BUFFER_SIZE_SAMPLES) {
            audio_ctx.audio_buffer[audio_ctx.buffer_index] = (uint16_t)sample_value;
            audio_ctx.buffer_index++;
        }
        
        // Process buffer when full or at chunk boundaries
        uint16_t chunk_size = (audio_ctx.current_sample_rate * AUDIO_CHUNK_SIZE_MS) / 1000;
        
        if (audio_ctx.buffer_index >= chunk_size || 
            audio_ctx.buffer_index >= AUDIO_BUFFER_SIZE_SAMPLES) {
            
            // Process accumulated samples
            sait01_audio_process_samples(audio_ctx.audio_buffer, audio_ctx.buffer_index);
            
            // Reset buffer
            audio_ctx.buffer_index = 0;
        }
    }
}

/* Register SAADC interrupt */
IRQ_CONNECT(SAADC_IRQn, 2, saadc_isr, NULL, 0);

/*
 * Audio Management Thread
 */
void sait01_audio_thread(void *arg1, void *arg2, void *arg3)
{
    ARG_UNUSED(arg1);
    ARG_UNUSED(arg2);
    ARG_UNUSED(arg3);
    
    printk("🎵 Adaptive Audio Thread Started\n");
    
    while (1) {
        // Periodic statistics and maintenance
        k_sleep(K_MSEC(1000));
        
        // Print statistics every 10 seconds
        static uint32_t stats_counter = 0;
        if (++stats_counter >= 10) {
            adaptive_audio_stats_t stats = sait01_audio_get_statistics();
            
            printk("🎵 Audio Stats: %d Hz, %s mode, %d switches, %.1f%% power savings\n",
                   stats.current_sample_rate,
                   stats.high_res_mode ? "HIGH-RES" : "LOW-POWER",
                   stats.mode_switch_count,
                   stats.power_savings_percent);
            
            stats_counter = 0;
        }
        
        // Auto-transition logic for power management
        if (audio_ctx.current_state == AUDIO_STATE_ACTIVE) {
            // Check for extended silence
            if (sait01_audio_should_return_to_idle()) {
                sait01_audio_transition_to_idle();
            }
        }
    }
}

/* Audio management thread definition */
K_THREAD_DEFINE(audio_thread, 1536, sait01_audio_thread,
                NULL, NULL, NULL, 4, 0, 0);  /* High priority for audio processing */

/*
 * Integration with Power Management
 */
int sait01_audio_set_power_mode(bool low_power)
{
    if (low_power) {
        // Force transition to idle mode for power saving
        sait01_audio_transition_to_idle();
        
        // Disable SAADC during deep sleep
        nrf_saadc_disable();
    } else {
        // Re-enable SAADC for active operation
        nrf_saadc_enable();
        
        // May stay in idle mode but ready to trigger
    }
    
    return 0;
}

bool sait01_audio_is_high_resolution_active(void)
{
    return audio_ctx.high_res_mode;
}