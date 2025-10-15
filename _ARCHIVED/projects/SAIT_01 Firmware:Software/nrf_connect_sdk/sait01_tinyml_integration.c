/*
 * SAIT_01 TinyML Integration - Live Audio Classification Implementation
 * TensorFlow Lite Micro with nRF5340 PDM audio processing
 */

#include "sait01_tinyml_integration.h"
#include <zephyr/kernel.h>
#include <zephyr/drivers/pdm.h>
#include <zephyr/logging/log.h>
#include <string.h>
#include <math.h>

LOG_MODULE_REGISTER(sait01_tinyml, CONFIG_LOG_DEFAULT_LEVEL);

/* Global callback storage */
static sait01_detection_callback_t g_detection_callback = NULL;
static void* g_user_data = NULL;

/* Detection confidence thresholds per class */
static const float confidence_thresholds[] = {
    0.5f,  /* UNKNOWN - low threshold */
    0.7f,  /* VEHICLE - medium threshold */
    0.6f,  /* FOOTSTEPS - medium threshold */
    0.6f,  /* VOICES - medium threshold */
    0.8f,  /* AIRCRAFT - high threshold */
    0.7f,  /* MACHINERY - medium threshold */ 
    0.9f,  /* GUNSHOT - very high threshold */
    0.95f  /* EXPLOSION - critical threshold */
};

/* Class names for debugging */
static const char* class_names[] = {
    "UNK", "VEH", "FOOT", "VOICE", 
    "AIR", "MACH", "GUN", "EXP"
};

/* Pre-computed mel filter bank in ROM (16 bins x 65 FFT bins) */
static const float mel_filter_bank_rom[SAIT01_MEL_BINS][SAIT01_FFT_SIZE / 2 + 1] = {
    /* Simplified triangular filters - in production, use proper calculated values */
    {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, 
    {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}
};

/* =============================================================================
 * MEL FILTER BANK INITIALIZATION
 * =============================================================================
 */

/* Optimized mel scale conversion using approximations */
static float hz_to_mel(float hz) {
    /* Fast approximation: avoids log10f for speed */
    return hz < 1000.0f ? hz : 1127.0f * (hz / 1000.0f + 0.5f);
}

static float mel_to_hz(float mel) {
    /* Fast approximation: avoids powf for speed */
    return mel < 1127.0f ? mel : 1000.0f * (mel / 1127.0f - 0.5f);
}

int sait01_mel_filters_init(sait01_feature_extractor_t* extractor)
{
    if (!extractor) {
        return -EINVAL;
    }

    if (extractor->mel_filters_initialized) {
        return 0; /* Already initialized */
    }

    /* Calculate mel filter bank parameters */
    float sample_rate = (float)SAIT01_AUDIO_SAMPLE_RATE_HZ;
    float nyquist = sample_rate / 2.0f;
    int fft_bins = SAIT01_FFT_SIZE / 2 + 1;
    
    /* Mel frequency range */
    float mel_min = hz_to_mel(0.0f);
    float mel_max = hz_to_mel(nyquist);
    float mel_step = (mel_max - mel_min) / (SAIT01_MEL_BINS + 1);
    
    /* Create mel frequency points */
    float mel_points[SAIT01_MEL_BINS + 2];
    for (int i = 0; i < SAIT01_MEL_BINS + 2; i++) {
        mel_points[i] = mel_to_hz(mel_min + i * mel_step);
    }
    
    /* Convert to FFT bin indices */
    float bin_points[SAIT01_MEL_BINS + 2];
    for (int i = 0; i < SAIT01_MEL_BINS + 2; i++) {
        bin_points[i] = floorf(mel_points[i] * SAIT01_FFT_SIZE / sample_rate);
    }
    
    /* Initialize filter bank matrix */
    memset(extractor->mel_filter_bank, 0, sizeof(extractor->mel_filter_bank));
    
    /* Create triangular filters */
    for (int m = 0; m < SAIT01_MEL_BINS; m++) {
        int left_bin = (int)bin_points[m];
        int center_bin = (int)bin_points[m + 1];
        int right_bin = (int)bin_points[m + 2];
        
        /* Left slope */
        for (int k = left_bin; k <= center_bin && k < fft_bins; k++) {
            if (center_bin != left_bin) {
                extractor->mel_filter_bank[m][k] = (float)(k - left_bin) / (center_bin - left_bin);
            }
        }
        
        /* Right slope */
        for (int k = center_bin; k <= right_bin && k < fft_bins; k++) {
            if (right_bin != center_bin) {
                extractor->mel_filter_bank[m][k] = (float)(right_bin - k) / (right_bin - center_bin);
            }
        }
    }
    
    extractor->mel_filters_initialized = true;
    LOG_INF("Mel filter bank initialized: %d filters, %d FFT bins", SAIT01_MEL_BINS, fft_bins);
    
    return 0;
}

/* =============================================================================
 * FEATURE EXTRACTION PIPELINE
 * =============================================================================
 */

int sait01_extract_mel_spectrogram(sait01_feature_extractor_t* extractor,
                                  const int16_t* audio_window,
                                  float* mel_spectrogram)
{
    if (!extractor || !audio_window || !mel_spectrogram) {
        return -EINVAL;
    }
    
    /* Initialize mel filters if needed */
    int ret = sait01_mel_filters_init(extractor);
    if (ret < 0) {
        return ret;
    }
    
    /* Convert int16 audio to float */
    for (int i = 0; i < SAIT01_AUDIO_SAMPLES_PER_WINDOW; i++) {
        extractor->window_buffer[i] = (float)audio_window[i] / 32768.0f;
    }
    
    /* Process audio in overlapping frames */
    int frame_idx = 0;
    for (int start = 0; start < SAIT01_AUDIO_SAMPLES_PER_WINDOW - SAIT01_FFT_SIZE; 
         start += SAIT01_HOP_LENGTH) {
        
        if (frame_idx >= SAIT01_N_FRAMES) break;
        
        /* Apply Hamming window */
        for (int i = 0; i < SAIT01_FFT_SIZE; i++) {
            float hamming_coeff = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (SAIT01_FFT_SIZE - 1));
            extractor->fft_input[i * 2] = extractor->window_buffer[start + i] * hamming_coeff;
            extractor->fft_input[i * 2 + 1] = 0.0f; /* Imaginary part */
        }
        
#ifdef CONFIG_CMSIS_DSP
        /* Use CMSIS-DSP FFT if available */
        arm_cfft_f32(&arm_cfft_sR_f32_len512, extractor->fft_input, 0, 1);
        memcpy(extractor->fft_output, extractor->fft_input, sizeof(float) * SAIT01_FFT_SIZE * 2);
#else
        /* Simple DFT implementation (fallback) */
        for (int k = 0; k < SAIT01_FFT_SIZE; k++) {
            float real = 0.0f, imag = 0.0f;
            for (int n = 0; n < SAIT01_FFT_SIZE; n++) {
                float angle = -2.0f * M_PI * k * n / SAIT01_FFT_SIZE;
                real += extractor->fft_input[n * 2] * cosf(angle);
                imag += extractor->fft_input[n * 2] * sinf(angle);
            }
            extractor->fft_output[k * 2] = real;
            extractor->fft_output[k * 2 + 1] = imag;
        }
#endif
        
        /* Calculate power spectrum */
        int power_bins = SAIT01_FFT_SIZE / 2 + 1;
        for (int i = 0; i < power_bins; i++) {
            float real = extractor->fft_output[i * 2];
            float imag = extractor->fft_output[i * 2 + 1];
            extractor->power_spectrum[i] = real * real + imag * imag;
        }
        
        /* Apply mel filter bank */
        for (int m = 0; m < SAIT01_MEL_BINS; m++) {
            float mel_energy = 0.0f;
            for (int k = 0; k < power_bins; k++) {
                mel_energy += extractor->mel_filter_bank[m][k] * extractor->power_spectrum[k];
            }
            /* Convert to log scale (add small epsilon to avoid log(0)) */
            extractor->mel_spectrum[frame_idx * SAIT01_MEL_BINS + m] = 
                logf(fmaxf(mel_energy, 1e-8f));
        }
        
        frame_idx++;
    }
    
    /* Copy to output buffer */
    memcpy(mel_spectrogram, extractor->mel_spectrum, 
           sizeof(float) * SAIT01_MEL_BINS * frame_idx);
    
    LOG_DBG("Extracted mel spectrogram: %d frames, %d mel bins", frame_idx, SAIT01_MEL_BINS);
    return 0;
}

void sait01_normalize_mel_spectrogram(float* mel_spectrogram, size_t size)
{
    if (!mel_spectrogram || size == 0) return;
    
    /* Calculate mean and standard deviation */
    float mean = 0.0f;
    for (size_t i = 0; i < size; i++) {
        mean += mel_spectrogram[i];
    }
    mean /= size;
    
    float std = 0.0f;
    for (size_t i = 0; i < size; i++) {
        float diff = mel_spectrogram[i] - mean;
        std += diff * diff;
    }
    std = sqrtf(std / size);
    
    /* Normalize to zero mean, unit variance */
    if (std > 1e-6f) {
        for (size_t i = 0; i < size; i++) {
            mel_spectrogram[i] = (mel_spectrogram[i] - mean) / std;
        }
    }
}

void sait01_generate_embedding(const float* mel_spectrogram, float* embedding)
{
    if (!mel_spectrogram || !embedding) return;
    
    /* Generate compressed 16-dimensional embedding using PCA-like reduction */
    /* This is a simplified version - in practice, you'd use learned embeddings */
    
    int input_size = SAIT01_MEL_BINS * SAIT01_N_FRAMES;
    int reduction_factor = input_size / 16;
    
    for (int i = 0; i < 16; i++) {
        float sum = 0.0f;
        int start_idx = i * reduction_factor;
        int end_idx = (i + 1) * reduction_factor;
        
        for (int j = start_idx; j < end_idx && j < input_size; j++) {
            sum += mel_spectrogram[j];
        }
        
        embedding[i] = sum / reduction_factor;
        
        /* Clamp to reasonable range */
        embedding[i] = fmaxf(-3.0f, fminf(3.0f, embedding[i]));
    }
}

/* =============================================================================
 * AUDIO PROCESSING
 * =============================================================================
 */

static void pdm_data_ready_handler(const struct device* dev, struct pdm_pcm_buf* buf)
{
    /* This callback is called from interrupt context */
    /* Copy data to circular buffer for processing */
    
    /* Note: Implementation would depend on the specific PDM driver */
    /* For now, this is a placeholder showing the structure */
    LOG_DBG("PDM data ready: %d samples", buf->size / sizeof(int16_t));
}

int sait01_audio_init(sait01_tinyml_system_t* system)
{
    if (!system) {
        return -EINVAL;
    }
    
    /* Find PDM device */
    system->pdm_dev = DEVICE_DT_GET_OR_NULL(DT_NODELABEL(pdm0));
    if (!system->pdm_dev) {
        LOG_ERR("PDM device not found");
        return -ENODEV;
    }
    
    /* Initialize audio buffer */
    memset(&system->audio_buffer, 0, sizeof(system->audio_buffer));
    k_mutex_init(&system->audio_buffer.mutex);
    
    /* Configure PDM */
    struct pdm_config config = {
        .operation_mode = PDM_OP_MODE_PDM,
        .sample_rate = SAIT01_AUDIO_SAMPLE_RATE_HZ,
        .channels = SAIT01_AUDIO_CHANNELS,
        .bit_width = SAIT01_AUDIO_SAMPLE_SIZE_BITS,
        .block_size = 1024, /* Block size in samples */
    };
    
    int ret = pdm_configure(system->pdm_dev, &config);
    if (ret < 0) {
        LOG_ERR("Failed to configure PDM: %d", ret);
        return ret;
    }
    
    LOG_INF("Audio system initialized: %d Hz, %d channels, %d bits",
            config.sample_rate, config.channels, config.bit_width);
    
    return 0;
}

int sait01_audio_start(sait01_tinyml_system_t* system)
{
    if (!system || !system->pdm_dev) {
        return -EINVAL;
    }
    
    int ret = pdm_start(system->pdm_dev);
    if (ret < 0) {
        LOG_ERR("Failed to start PDM: %d", ret);
        return ret;
    }
    
    LOG_INF("Audio capture started");
    return 0;
}

int sait01_audio_stop(sait01_tinyml_system_t* system)
{
    if (!system || !system->pdm_dev) {
        return -EINVAL;
    }
    
    int ret = pdm_stop(system->pdm_dev);
    if (ret < 0) {
        LOG_ERR("Failed to stop PDM: %d", ret);
        return ret;
    }
    
    LOG_INF("Audio capture stopped");
    return 0;
}

int sait01_audio_get_samples(sait01_tinyml_system_t* system,
                            int16_t* buffer,
                            size_t requested_samples,
                            size_t* actual_samples)
{
    if (!system || !buffer || !actual_samples) {
        return -EINVAL;
    }
    
    k_mutex_lock(&system->audio_buffer.mutex, K_FOREVER);
    
    size_t available = 0;
    if (system->audio_buffer.buffer_full) {
        available = SAIT01_AUDIO_BUFFER_SIZE;
    } else {
        available = system->audio_buffer.write_index - system->audio_buffer.read_index;
    }
    
    size_t to_copy = MIN(requested_samples, available);
    *actual_samples = to_copy;
    
    /* Copy samples from circular buffer */
    for (size_t i = 0; i < to_copy; i++) {
        buffer[i] = system->audio_buffer.buffer[
            (system->audio_buffer.read_index + i) % SAIT01_AUDIO_BUFFER_SIZE];
    }
    
    system->audio_buffer.read_index = 
        (system->audio_buffer.read_index + to_copy) % SAIT01_AUDIO_BUFFER_SIZE;
    
    if (system->audio_buffer.buffer_full && to_copy > 0) {
        system->audio_buffer.buffer_full = false;
    }
    
    k_mutex_unlock(&system->audio_buffer.mutex);
    
    return 0;
}

/* =============================================================================
 * ML INFERENCE
 * =============================================================================
 */

#ifdef CONFIG_TENSORFLOW_LITE_MICRO

static int sait01_ml_init(sait01_tinyml_system_t* system,
                         const uint8_t* model_data,
                         size_t model_size)
{
    if (!system || !model_data) {
        return -EINVAL;
    }
    
    sait01_ml_inference_t* ml = &system->ml_inference;
    
    /* Load model */
    ml->model = tflite::GetModel(model_data);
    if (ml->model->version() != TFLITE_SCHEMA_VERSION) {
        LOG_ERR("Model schema version mismatch: expected %d, got %d",
                TFLITE_SCHEMA_VERSION, ml->model->version());
        return -EINVAL;
    }
    
    /* Create resolver */
    static tflite::AllOpsResolver resolver;
    ml->resolver = &resolver;
    
    /* Create interpreter */
    static tflite::MicroInterpreter static_interpreter(
        ml->model, *ml->resolver, ml->tensor_arena, 
        SAIT01_MODEL_TENSOR_ARENA_SIZE, nullptr);
    ml->interpreter = &static_interpreter;
    
    /* Allocate tensors */
    TfLiteStatus allocate_status = ml->interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        LOG_ERR("Failed to allocate tensors");
        return -ENOMEM;
    }
    
    /* Get input and output tensors */
    ml->input_tensor = ml->interpreter->input(0);
    ml->output_tensor = ml->interpreter->output(0);
    
    /* Verify tensor dimensions */
    if (ml->input_tensor->dims->size != 3 ||
        ml->input_tensor->dims->data[1] * ml->input_tensor->dims->data[2] != SAIT01_MODEL_INPUT_SIZE) {
        LOG_ERR("Input tensor dimension mismatch");
        return -EINVAL;
    }
    
    if (ml->output_tensor->dims->size != 2 ||
        ml->output_tensor->dims->data[1] != SAIT01_MODEL_OUTPUT_CLASSES) {
        LOG_ERR("Output tensor dimension mismatch");
        return -EINVAL;
    }
    
    ml->model_loaded = true;
    
    LOG_INF("TensorFlow Lite model loaded successfully");
    LOG_INF("Input shape: [%d, %d, %d]", ml->input_tensor->dims->data[0],
            ml->input_tensor->dims->data[1], ml->input_tensor->dims->data[2]);
    LOG_INF("Output shape: [%d, %d]", ml->output_tensor->dims->data[0],
            ml->output_tensor->dims->data[1]);
    
    return 0;
}

static int sait01_ml_infer(sait01_tinyml_system_t* system,
                          const float* input_data,
                          sait01_ml_detection_t* result)
{
    if (!system || !input_data || !result) {
        return -EINVAL;
    }
    
    sait01_ml_inference_t* ml = &system->ml_inference;
    if (!ml->model_loaded) {
        return -ENOENT;
    }
    
    uint64_t start_time = k_uptime_get();
    
    /* Copy input data to tensor */
    float* input_tensor_data = tflite::GetTensorData<float>(ml->input_tensor);
    memcpy(input_tensor_data, input_data, sizeof(float) * SAIT01_MODEL_INPUT_SIZE);
    
    /* Run inference */
    TfLiteStatus invoke_status = ml->interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        LOG_ERR("Failed to invoke model");
        return -EIO;
    }
    
    /* Get output data */
    float* output_tensor_data = tflite::GetTensorData<float>(ml->output_tensor);
    memcpy(ml->model_output, output_tensor_data, sizeof(float) * SAIT01_MODEL_OUTPUT_CLASSES);
    
    /* Find class with highest confidence */
    float max_confidence = 0.0f;
    sait01_ml_class_t detected_class = SAIT01_ML_CLASS_UNKNOWN;
    
    for (int i = 0; i < SAIT01_MODEL_OUTPUT_CLASSES; i++) {
        ml->model_output[i] = fmaxf(0.0f, fminf(1.0f, ml->model_output[i])); /* Clamp 0-1 */
        if (ml->model_output[i] > max_confidence) {
            max_confidence = ml->model_output[i];
            detected_class = (sait01_ml_class_t)i;
        }
    }
    
    uint64_t end_time = k_uptime_get();
    
    /* Fill result structure */
    result->detected_class = detected_class;
    result->confidence = max_confidence;
    memcpy(result->class_probabilities, ml->model_output, sizeof(ml->model_output));
    result->inference_time_us = (end_time - start_time) * 1000; /* Convert to microseconds */
    result->timestamp = k_uptime_get_32();
    
    /* Generate embedding from input features */
    sait01_generate_embedding(input_data, result->embedding);
    
    system->inference_count++;
    if (max_confidence > sait01_get_confidence_threshold(detected_class)) {
        system->detection_count++;
    }
    
    LOG_DBG("ML inference: class=%s, confidence=%.3f, time=%dus",
            sait01_class_to_string(detected_class), max_confidence, result->inference_time_us);
    
    return 0;
}

#else /* CONFIG_TENSORFLOW_LITE_MICRO */

static int sait01_ml_init(sait01_tinyml_system_t* system,
                         const uint8_t* model_data,
                         size_t model_size)
{
    LOG_WRN("TensorFlow Lite Micro not enabled - using mock ML inference");
    system->ml_inference.model_loaded = true;
    return 0;
}

static int sait01_ml_infer(sait01_tinyml_system_t* system,
                          const float* input_data,
                          sait01_ml_detection_t* result)
{
    /* Mock inference for testing when TFLite is not available */
    static uint32_t inference_counter = 0;
    inference_counter++;
    
    uint64_t start_time = k_uptime_get();
    
    /* Simulate processing delay */
    k_busy_wait(15000); /* 15ms */
    
    /* Mock classification based on input characteristics */
    float energy = 0.0f;
    for (int i = 0; i < SAIT01_MODEL_INPUT_SIZE; i++) {
        energy += fabsf(input_data[i]);
    }
    energy /= SAIT01_MODEL_INPUT_SIZE;
    
    sait01_ml_class_t detected_class = SAIT01_ML_CLASS_UNKNOWN;
    float confidence = 0.0f;
    
    /* Simple heuristic classification */
    if (energy > 0.8f) {
        detected_class = SAIT01_ML_CLASS_EXPLOSION;
        confidence = 0.95f;
    } else if (energy > 0.6f) {
        detected_class = SAIT01_ML_CLASS_GUNSHOT;
        confidence = 0.90f;
    } else if (energy > 0.4f) {
        detected_class = SAIT01_ML_CLASS_VEHICLE;
        confidence = 0.75f + (inference_counter % 10) * 0.02f;
    } else if (energy > 0.2f) {
        detected_class = SAIT01_ML_CLASS_FOOTSTEPS;
        confidence = 0.65f + (inference_counter % 8) * 0.03f;
    } else {
        detected_class = SAIT01_ML_CLASS_UNKNOWN;
        confidence = 0.3f;
    }
    
    uint64_t end_time = k_uptime_get();
    
    /* Fill result structure */
    result->detected_class = detected_class;
    result->confidence = confidence;
    memset(result->class_probabilities, 0, sizeof(result->class_probabilities));
    result->class_probabilities[detected_class] = confidence;
    result->inference_time_us = (end_time - start_time) * 1000;
    result->timestamp = k_uptime_get_32();
    
    /* Generate mock embedding */
    for (int i = 0; i < 16; i++) {
        result->embedding[i] = (float)(sys_rand32_get() % 200 - 100) / 100.0f;
    }
    
    system->inference_count++;
    if (confidence > sait01_get_confidence_threshold(detected_class)) {
        system->detection_count++;
    }
    
    LOG_DBG("Mock ML inference: class=%s, confidence=%.3f, time=%dus",
            sait01_class_to_string(detected_class), confidence, result->inference_time_us);
    
    return 0;
}

#endif /* CONFIG_TENSORFLOW_LITE_MICRO */

/* =============================================================================
 * MAIN PROCESSING PIPELINE
 * =============================================================================
 */

static void audio_processing_work_handler(struct k_work* work)
{
    struct k_work_delayable* dwork = k_work_delayable_from_work(work);
    sait01_tinyml_system_t* system = CONTAINER_OF(dwork, sait01_tinyml_system_t, 
                                                  audio_processing_work);
    
    if (!system->system_active) {
        return;
    }
    
    /* Get audio samples */
    int16_t audio_window[SAIT01_AUDIO_SAMPLES_PER_WINDOW];
    size_t samples_obtained;
    
    int ret = sait01_audio_get_samples(system, audio_window, 
                                      SAIT01_AUDIO_SAMPLES_PER_WINDOW, 
                                      &samples_obtained);
    
    if (ret == 0 && samples_obtained >= SAIT01_AUDIO_SAMPLES_PER_WINDOW) {
        /* Extract features */
        float mel_spectrogram[SAIT01_MODEL_INPUT_SIZE];
        ret = sait01_extract_mel_spectrogram(&system->feature_extractor, 
                                           audio_window, mel_spectrogram);
        
        if (ret == 0) {
            /* Normalize features */
            sait01_normalize_mel_spectrogram(mel_spectrogram, SAIT01_MODEL_INPUT_SIZE);
            
            /* Run ML inference */
            sait01_ml_detection_t detection;
            ret = sait01_ml_infer(system, mel_spectrogram, &detection);
            
            if (ret == 0) {
                /* Check if detection should trigger alert */
                if (sait01_should_generate_alert(&detection)) {
                    LOG_INF("DETECTION: %s (%.1f%% confidence)",
                            sait01_class_to_string(detection.detected_class),
                            detection.confidence * 100.0f);
                    
                    /* Call user callback */
                    if (g_detection_callback) {
                        g_detection_callback(&detection, g_user_data);
                    }
                }
            } else {
                LOG_ERR("ML inference failed: %d", ret);
            }
        } else {
            LOG_ERR("Feature extraction failed: %d", ret);
        }
    }
    
    /* Schedule next processing cycle */
    if (system->system_active) {
        k_work_reschedule(&system->audio_processing_work, 
                         K_MSEC(SAIT01_AUDIO_OVERLAP_MS));
    }
}

/* =============================================================================
 * PUBLIC API IMPLEMENTATION
 * =============================================================================
 */

int sait01_tinyml_init(sait01_tinyml_system_t* system,
                       const uint8_t* model_data,
                       size_t model_size,
                       sait01_detection_callback_t detection_callback,
                       void* user_data)
{
    if (!system) {
        return -EINVAL;
    }
    
    memset(system, 0, sizeof(*system));
    
    /* Store callbacks */
    g_detection_callback = detection_callback;
    g_user_data = user_data;
    
    /* Initialize audio subsystem */
    int ret = sait01_audio_init(system);
    if (ret < 0) {
        LOG_ERR("Failed to initialize audio: %d", ret);
        return ret;
    }
    
    /* Initialize feature extractor */
#ifdef CONFIG_CMSIS_DSP
    arm_rfft_fast_init_f32(&system->feature_extractor.rfft_instance, SAIT01_FFT_SIZE);
#endif
    
    /* Initialize ML inference */
    ret = sait01_ml_init(system, model_data, model_size);
    if (ret < 0) {
        LOG_ERR("Failed to initialize ML: %d", ret);
        return ret;
    }
    
    /* Initialize work queue */
    k_work_init_delayable(&system->audio_processing_work, audio_processing_work_handler);
    
    LOG_INF("SAIT_01 TinyML system initialized successfully");
    return 0;
}

int sait01_tinyml_start(sait01_tinyml_system_t* system)
{
    if (!system) {
        return -EINVAL;
    }
    
    if (system->system_active) {
        return -EALREADY;
    }
    
    /* Start audio capture */
    int ret = sait01_audio_start(system);
    if (ret < 0) {
        return ret;
    }
    
    /* Start processing pipeline */
    system->system_active = true;
    k_work_schedule(&system->audio_processing_work, K_MSEC(100));
    
    LOG_INF("SAIT_01 TinyML system started");
    return 0;
}

int sait01_tinyml_stop(sait01_tinyml_system_t* system)
{
    if (!system) {
        return -EINVAL;
    }
    
    if (!system->system_active) {
        return -EALREADY;
    }
    
    /* Stop processing */
    system->system_active = false;
    k_work_cancel_delayable(&system->audio_processing_work);
    
    /* Stop audio capture */
    int ret = sait01_audio_stop(system);
    if (ret < 0) {
        LOG_ERR("Failed to stop audio: %d", ret);
    }
    
    LOG_INF("SAIT_01 TinyML system stopped");
    return 0;
}

int sait01_tinyml_get_stats(sait01_tinyml_system_t* system,
                           uint32_t* inference_count,
                           uint32_t* detection_count,
                           uint32_t* avg_inference_time_us)
{
    if (!system || !inference_count || !detection_count || !avg_inference_time_us) {
        return -EINVAL;
    }
    
    *inference_count = system->inference_count;
    *detection_count = system->detection_count;
    
    /* Average inference time calculation would need additional tracking */
    *avg_inference_time_us = 15000; /* Mock value - 15ms */
    
    return 0;
}

/* =============================================================================
 * UTILITY FUNCTIONS
 * =============================================================================
 */

const char* sait01_class_to_string(sait01_ml_class_t class_id)
{
    if (class_id >= 0 && class_id < ARRAY_SIZE(class_names)) {
        return class_names[class_id];
    }
    return "Invalid";
}

float sait01_get_confidence_threshold(sait01_ml_class_t class_id)
{
    if (class_id >= 0 && class_id < ARRAY_SIZE(confidence_thresholds)) {
        return confidence_thresholds[class_id];
    }
    return 0.5f; /* Default threshold */
}

bool sait01_should_generate_alert(const sait01_ml_detection_t* detection)
{
    if (!detection) {
        return false;
    }
    
    float threshold = sait01_get_confidence_threshold(detection->detected_class);
    return detection->confidence >= threshold && 
           detection->detected_class != SAIT01_ML_CLASS_UNKNOWN;
}