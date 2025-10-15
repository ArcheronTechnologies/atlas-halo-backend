/*
 * SAIT_01 Adaptive Audio Sampling System - Header
 * ===================================================
 * Power-optimized audio processing with dynamic sample rates
 */

#ifndef ADAPTIVE_AUDIO_H
#define ADAPTIVE_AUDIO_H

#include <zephyr/types.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Adaptive Audio Statistics Structure
 */
typedef struct {
    uint32_t current_sample_rate;       // Current sampling rate (Hz)
    uint8_t current_state;              // Current audio processing state
    uint32_t total_samples_processed;   // Total samples processed
    uint32_t energy_level;              // Current audio energy level
    uint16_t peak_amplitude;            // Peak amplitude in current window
    bool high_res_mode;                 // High resolution mode active
    uint32_t mode_switch_count;         // Number of mode switches
    float power_savings_percent;        // Estimated power savings
    uint32_t effective_sample_rate;     // Time-averaged sample rate
} adaptive_audio_stats_t;

/*
 * Adaptive Audio API Functions
 */

/**
 * @brief Initialize adaptive audio sampling system
 * 
 * Sets up SAADC, timers, and adaptive sampling logic
 * 
 * @return 0 on success, negative error code on failure
 */
int sait01_audio_init(void);

/**
 * @brief Set audio sampling rate
 * 
 * @param sample_rate_hz Target sampling rate in Hz
 * @return 0 on success, negative error code on failure
 */
int sait01_audio_set_sample_rate(uint32_t sample_rate_hz);

/**
 * @brief Transition to active high-resolution mode
 * 
 * Switches from 4 kHz to 16 kHz sampling for threat detection
 * 
 * @return 0 on success, negative error code on failure
 */
int sait01_audio_transition_to_active(void);

/**
 * @brief Transition to idle low-power mode
 * 
 * Switches from 16 kHz to 4 kHz sampling for power saving
 * 
 * @return 0 on success, negative error code on failure
 */
int sait01_audio_transition_to_idle(void);

/**
 * @brief Process audio samples with adaptive logic
 * 
 * Analyzes samples and determines if mode switching is needed
 * 
 * @param samples Pointer to audio sample buffer
 * @param count Number of samples to process
 */
void sait01_audio_process_samples(uint16_t *samples, uint16_t count);

/**
 * @brief Calculate audio energy level
 * 
 * Computes RMS energy and peak amplitude for trigger detection
 * 
 * @param samples Pointer to audio sample buffer
 * @param count Number of samples
 * @return Energy level (arbitrary units)
 */
int sait01_audio_calculate_energy(uint16_t *samples, uint16_t count);

/**
 * @brief Check if high resolution mode should be triggered
 * 
 * @return true if high resolution mode should be activated
 */
bool sait01_audio_should_trigger_high_res(void);

/**
 * @brief Check if system should return to idle mode
 * 
 * @return true if system should return to low power mode
 */
bool sait01_audio_should_return_to_idle(void);

/**
 * @brief Set power management mode for audio system
 * 
 * @param low_power true to enable low power mode, false for normal operation
 * @return 0 on success, negative error code on failure
 */
int sait01_audio_set_power_mode(bool low_power);

/**
 * @brief Check if high resolution mode is currently active
 * 
 * @return true if high resolution mode is active
 */
bool sait01_audio_is_high_resolution_active(void);

/**
 * @brief Get adaptive audio statistics
 * 
 * @return Structure containing audio system statistics
 */
adaptive_audio_stats_t sait01_audio_get_statistics(void);

/*
 * Adaptive Audio Configuration Macros
 */
#define ADAPTIVE_AUDIO_SAMPLE_RATE_IDLE     4000    // Low power sampling rate
#define ADAPTIVE_AUDIO_SAMPLE_RATE_ACTIVE   16000   // High resolution sampling rate
#define ADAPTIVE_AUDIO_ENERGY_THRESHOLD     100     // Energy trigger threshold
#define ADAPTIVE_AUDIO_SILENCE_TIMEOUT_MS   5000    // Silence timeout for idle return
#define ADAPTIVE_AUDIO_PEAK_THRESHOLD       512     // Peak amplitude threshold
#define ADAPTIVE_AUDIO_POWER_REDUCTION      75      // Power reduction percentage in idle

#ifdef __cplusplus
}
#endif

#endif /* ADAPTIVE_AUDIO_H */