/*
 * SAIT_01 TinyML Integration - Live Audio Classification
 * TensorFlow Lite Micro integration for nRF5340 with distributed mesh
 */

#ifndef SAIT01_TINYML_INTEGRATION_H
#define SAIT01_TINYML_INTEGRATION_H

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/pdm.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/logging/log.h>

/* TensorFlow Lite Micro includes */
#ifdef CONFIG_TENSORFLOW_LITE_MICRO
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_log.h>
#include <tensorflow/lite/micro/system_setup.h>
#include <tensorflow/lite/schema/schema_generated.h>
#endif

/* CMSIS-DSP for signal processing */
#ifdef CONFIG_CMSIS_DSP
#include <arm_math.h>
#include <arm_const_structs.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* =============================================================================
 * AUDIO CONFIGURATION
 * =============================================================================
 */

/* Audio sampling parameters */
#define SAIT01_AUDIO_SAMPLE_RATE_HZ     16000   /* 16 kHz sampling rate */
#define SAIT01_AUDIO_CHANNELS           1       /* Mono audio */
#define SAIT01_AUDIO_SAMPLE_SIZE_BITS   16      /* 16-bit samples */
#define SAIT01_AUDIO_WINDOW_MS          250     /* 0.25 second analysis window - further optimized */
#define SAIT01_AUDIO_OVERLAP_MS         500     /* 50% overlap between windows */
#define SAIT01_AUDIO_SAMPLES_PER_WINDOW (SAIT01_AUDIO_SAMPLE_RATE_HZ * SAIT01_AUDIO_WINDOW_MS / 1000)
#define SAIT01_AUDIO_BUFFER_SIZE        (SAIT01_AUDIO_SAMPLES_PER_WINDOW + SAIT01_AUDIO_SAMPLES_PER_WINDOW/2) /* 1.5x buffer - memory optimized */

/* Mel spectrogram parameters */
#define SAIT01_MEL_BINS                 16      /* 16 mel frequency bins - further optimized */
#define SAIT01_FFT_SIZE                 128     /* 128 FFT window size - further optimized */
#define SAIT01_HOP_LENGTH               256     /* Hop length for STFT */
#define SAIT01_N_FRAMES                 ((SAIT01_AUDIO_SAMPLES_PER_WINDOW - SAIT01_FFT_SIZE) / SAIT01_HOP_LENGTH + 1)

/* TinyML model parameters */
#define SAIT01_MODEL_INPUT_SIZE         (SAIT01_MEL_BINS * SAIT01_N_FRAMES)  /* 64x63 = 4032 */
#define SAIT01_MODEL_OUTPUT_CLASSES     8       /* Number of classification classes */
#define SAIT01_MODEL_TENSOR_ARENA_SIZE  (24 * 1024)  /* 24KB tensor arena - nRF5340 optimized */

/* Detection classes matching protocol definitions */
typedef enum {
    SAIT01_ML_CLASS_UNKNOWN = 0,
    SAIT01_ML_CLASS_VEHICLE = 1,
    SAIT01_ML_CLASS_FOOTSTEPS = 2,
    SAIT01_ML_CLASS_VOICES = 3,
    SAIT01_ML_CLASS_AIRCRAFT = 4,
    SAIT01_ML_CLASS_MACHINERY = 5,
    SAIT01_ML_CLASS_GUNSHOT = 6,
    SAIT01_ML_CLASS_EXPLOSION = 7
} sait01_ml_class_t;

/* =============================================================================
 * DATA STRUCTURES
 * =============================================================================
 */

/* Memory-efficient audio buffer management */
typedef struct {
    int16_t buffer[SAIT01_AUDIO_BUFFER_SIZE];
    uint16_t write_index;  /* Reduced from uint32_t */
    uint16_t read_index;   /* Reduced from uint32_t */
    uint8_t buffer_full;   /* Reduced from bool */
    struct k_mutex mutex;
} sait01_audio_buffer_t;

/* Memory-efficient feature extraction state */
typedef struct {
    union {
        float window_buffer[SAIT01_AUDIO_SAMPLES_PER_WINDOW];
        float fft_workspace[SAIT01_FFT_SIZE * 2];  /* Shared workspace */
    };
    float power_spectrum[SAIT01_FFT_SIZE / 2 + 1];
    /* mel_filter_bank moved to ROM - see mel_filter_bank_rom[] */
    float mel_spectrum[SAIT01_MEL_BINS * SAIT01_N_FRAMES];
    uint8_t mel_filters_initialized;  /* Reduced from bool */
#ifdef CONFIG_CMSIS_DSP
    arm_rfft_fast_instance_f32 rfft_instance;
#endif
} sait01_feature_extractor_t;

/* Memory-efficient ML inference state */
typedef struct {
#ifdef CONFIG_TENSORFLOW_LITE_MICRO
    const tflite::Model* model;
    tflite::MicroInterpreter* interpreter;
    tflite::AllOpsResolver* resolver;
    TfLiteTensor* input_tensor;
    TfLiteTensor* output_tensor;
#endif
    uint8_t tensor_arena[SAIT01_MODEL_TENSOR_ARENA_SIZE];
    union {
        float model_input[SAIT01_MODEL_INPUT_SIZE];
        float preprocessing_buffer[SAIT01_MODEL_INPUT_SIZE];  /* Shared buffer */
    };
    float model_output[SAIT01_MODEL_OUTPUT_CLASSES];
    uint8_t model_loaded;  /* Reduced from bool */
} sait01_ml_inference_t;

/* Complete TinyML system state */
typedef struct {
    const struct device* pdm_dev;
    sait01_audio_buffer_t audio_buffer;
    sait01_feature_extractor_t feature_extractor;
    sait01_ml_inference_t ml_inference;
    struct k_work_delayable audio_processing_work;
    struct k_work_delayable ml_inference_work;
    uint32_t inference_count;
    uint32_t detection_count;
    bool system_active;
} sait01_tinyml_system_t;

/* ML detection result */
typedef struct {
    sait01_ml_class_t detected_class;
    float confidence;
    float class_probabilities[SAIT01_MODEL_OUTPUT_CLASSES];
    uint32_t inference_time_us;
    uint32_t timestamp;
    float embedding[16];  /* Compressed feature representation */
} sait01_ml_detection_t;

/* =============================================================================
 * CALLBACK TYPES
 * =============================================================================
 */

/* Callback for ML detection results */
typedef void (*sait01_detection_callback_t)(const sait01_ml_detection_t* detection, void* user_data);

/* Callback for audio data ready */
typedef void (*sait01_audio_ready_callback_t)(const int16_t* audio_data, size_t sample_count, void* user_data);

/* =============================================================================
 * PUBLIC API FUNCTIONS
 * =============================================================================
 */

/**
 * Initialize the TinyML system
 * @param system Pointer to TinyML system structure
 * @param model_data Pointer to TensorFlow Lite model data
 * @param model_size Size of model data in bytes
 * @param detection_callback Callback for detection results
 * @param user_data User data passed to callbacks
 * @return 0 on success, negative error code on failure
 */
int sait01_tinyml_init(sait01_tinyml_system_t* system,
                       const uint8_t* model_data,
                       size_t model_size,
                       sait01_detection_callback_t detection_callback,
                       void* user_data);

/**
 * Start audio capture and ML processing
 * @param system Pointer to TinyML system
 * @return 0 on success, negative error code on failure
 */
int sait01_tinyml_start(sait01_tinyml_system_t* system);

/**
 * Stop audio capture and ML processing
 * @param system Pointer to TinyML system
 * @return 0 on success, negative error code on failure
 */
int sait01_tinyml_stop(sait01_tinyml_system_t* system);

/**
 * Process audio data manually (for testing)
 * @param system Pointer to TinyML system
 * @param audio_data Raw audio samples
 * @param sample_count Number of samples
 * @param result Output detection result
 * @return 0 on success, negative error code on failure
 */
int sait01_tinyml_process_audio(sait01_tinyml_system_t* system,
                               const int16_t* audio_data,
                               size_t sample_count,
                               sait01_ml_detection_t* result);

/**
 * Get system statistics
 * @param system Pointer to TinyML system
 * @param inference_count Output: total inferences performed
 * @param detection_count Output: total detections found
 * @param avg_inference_time_us Output: average inference time
 * @return 0 on success, negative error code on failure
 */
int sait01_tinyml_get_stats(sait01_tinyml_system_t* system,
                           uint32_t* inference_count,
                           uint32_t* detection_count,
                           uint32_t* avg_inference_time_us);

/**
 * Update ML model at runtime
 * @param system Pointer to TinyML system
 * @param model_data New model data
 * @param model_size Size of new model
 * @return 0 on success, negative error code on failure
 */
int sait01_tinyml_update_model(sait01_tinyml_system_t* system,
                              const uint8_t* model_data,
                              size_t model_size);

/* =============================================================================
 * FEATURE EXTRACTION FUNCTIONS
 * =============================================================================
 */

/**
 * Initialize mel filter bank
 * @param extractor Pointer to feature extractor
 * @return 0 on success, negative error code on failure
 */
int sait01_mel_filters_init(sait01_feature_extractor_t* extractor);

/**
 * Extract mel spectrogram from audio window
 * @param extractor Pointer to feature extractor
 * @param audio_window Input audio samples
 * @param mel_spectrogram Output mel spectrogram
 * @return 0 on success, negative error code on failure
 */
int sait01_extract_mel_spectrogram(sait01_feature_extractor_t* extractor,
                                  const int16_t* audio_window,
                                  float* mel_spectrogram);

/**
 * Normalize mel spectrogram for ML input
 * @param mel_spectrogram Input/output mel spectrogram
 * @param size Number of elements
 */
void sait01_normalize_mel_spectrogram(float* mel_spectrogram, size_t size);

/**
 * Generate embedding from mel spectrogram
 * @param mel_spectrogram Input mel spectrogram
 * @param embedding Output compressed embedding (16 floats)
 */
void sait01_generate_embedding(const float* mel_spectrogram, float* embedding);

/* =============================================================================
 * AUDIO PROCESSING FUNCTIONS  
 * =============================================================================
 */

/**
 * Initialize PDM audio capture
 * @param system Pointer to TinyML system
 * @return 0 on success, negative error code on failure
 */
int sait01_audio_init(sait01_tinyml_system_t* system);

/**
 * Start PDM audio capture
 * @param system Pointer to TinyML system
 * @return 0 on success, negative error code on failure
 */
int sait01_audio_start(sait01_tinyml_system_t* system);

/**
 * Stop PDM audio capture
 * @param system Pointer to TinyML system
 * @return 0 on success, negative error code on failure
 */
int sait01_audio_stop(sait01_tinyml_system_t* system);

/**
 * Get audio samples from buffer
 * @param system Pointer to TinyML system
 * @param buffer Output buffer for audio samples
 * @param requested_samples Number of samples requested
 * @param actual_samples Output: actual samples copied
 * @return 0 on success, negative error code on failure
 */
int sait01_audio_get_samples(sait01_tinyml_system_t* system,
                            int16_t* buffer,
                            size_t requested_samples,
                            size_t* actual_samples);

/* =============================================================================
 * UTILITY FUNCTIONS
 * =============================================================================
 */

/**
 * Convert detection class to string
 * @param class_id Detection class ID
 * @return String representation of class
 */
const char* sait01_class_to_string(sait01_ml_class_t class_id);

/**
 * Calculate confidence threshold for alert generation
 * @param class_id Detection class
 * @return Recommended confidence threshold (0.0-1.0)
 */
float sait01_get_confidence_threshold(sait01_ml_class_t class_id);

/**
 * Check if detection should trigger alert
 * @param detection ML detection result
 * @return true if alert should be generated
 */
bool sait01_should_generate_alert(const sait01_ml_detection_t* detection);

/* =============================================================================
 * TENSORFLOW LITE PRODUCTION MODEL FUNCTIONS
 * =============================================================================
 */

/**
 * Initialize production TensorFlow Lite model for nRF5340
 * @param system TinyML system instance
 * @return 0 on success, negative error code on failure
 */
int sait01_init_tflite_model(sait01_tinyml_system_t* system);

/**
 * Process audio using production TensorFlow Lite model
 * @param system TinyML system instance
 * @param audio_data Input audio samples (16-bit, 16kHz)
 * @param sample_count Number of audio samples
 * @param result Output detection result
 * @return 0 on success, negative error code on failure
 */
int sait01_tflite_process_audio(sait01_tinyml_system_t* system,
                               const int16_t* audio_data,
                               size_t sample_count,
                               sait01_ml_detection_t* result);

/**
 * Test TensorFlow Lite model accuracy with synthetic data
 * @return 0 on success, negative error code on failure
 */
int sait01_test_tflite_model_accuracy(void);

/* =============================================================================
 * MOCK MODEL FUNCTIONS (FALLBACK)
 * =============================================================================
 */

/**
 * Initialize mock model for testing and fallback
 * @param system TinyML system instance
 * @return 0 on success, negative error code on failure
 */
int sait01_init_mock_model(sait01_tinyml_system_t* system);

/**
 * Process audio using mock model (pattern-based classification)
 * @param system TinyML system instance
 * @param audio_data Input audio samples
 * @param sample_count Number of audio samples
 * @param result Output detection result
 * @return 0 on success, negative error code on failure
 */
int sait01_mock_tinyml_process_audio(sait01_tinyml_system_t* system,
                                    const int16_t* audio_data,
                                    size_t sample_count,
                                    sait01_ml_detection_t* result);

#ifdef __cplusplus
}
#endif

#endif /* SAIT01_TINYML_INTEGRATION_H */