/*
 * Mock TinyML Model for Immediate Deployment Testing
 * Simulates trained model behavior for system integration
 */

#include "sait01_tinyml_integration.h"
#include "optimized_preprocessing.h"
#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <math.h>

LOG_MODULE_REGISTER(mock_model, CONFIG_LOG_DEFAULT_LEVEL);

/* Mock model weights and biases (simplified neural network) */
static const float mock_weights_layer1[64][16] = {
    /* Simplified weight matrix for first layer */
    /* In production, this would be the actual trained model weights */
    {0.1f, -0.2f, 0.3f, 0.1f, -0.1f, 0.2f, 0.0f, 0.1f, 
     0.2f, -0.1f, 0.1f, 0.0f, 0.3f, -0.2f, 0.1f, 0.2f}
    /* ... additional weight rows would be here */
};

static const float mock_weights_layer2[16][8] = {
    /* Simplified weight matrix for output layer */
    {0.5f, -0.3f, 0.2f, 0.1f, 0.4f, -0.2f, 0.3f, 0.1f}
    /* ... additional weight rows would be here */
};

static const float mock_bias_layer1[16] = {
    0.1f, -0.05f, 0.2f, 0.0f, 0.15f, -0.1f, 0.05f, 0.1f,
    0.0f, 0.2f, -0.1f, 0.1f, 0.05f, 0.0f, -0.05f, 0.1f
};

static const float mock_bias_layer2[8] = {
    0.1f, 0.0f, -0.1f, 0.2f, 0.0f, 0.1f, -0.05f, 0.15f
};

/* Pattern-based detection thresholds per class */
static const float class_thresholds[SAIT01_MODEL_OUTPUT_CLASSES] = {
    0.3f,  /* UNKNOWN */
    0.6f,  /* VEHICLE */
    0.5f,  /* FOOTSTEPS */
    0.5f,  /* VOICES */
    0.7f,  /* AIRCRAFT */
    0.6f,  /* MACHINERY */
    0.8f,  /* GUNSHOT */
    0.9f   /* EXPLOSION */
};

/* =============================================================================
 * MOCK NEURAL NETWORK INFERENCE
 * =============================================================================
 */

static float relu_activation(float x)
{
    return fmaxf(0.0f, x);
}

static void softmax_activation(float* input, int size)
{
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        input[i] = expf(input[i] - max_val);
        sum += input[i];
    }
    
    for (int i = 0; i < size; i++) {
        input[i] /= sum;
    }
}

static int mock_neural_network_inference(const float* input_features, 
                                        float* output_probabilities)
{
    /* Simplified 2-layer neural network */
    static float hidden_layer[16];
    
    /* Layer 1: Linear transformation + ReLU */
    for (int i = 0; i < 16; i++) {
        hidden_layer[i] = mock_bias_layer1[i];
        
        /* Simplified weight computation (use first 16 features) */
        for (int j = 0; j < 16; j++) {
            hidden_layer[i] += input_features[j] * mock_weights_layer1[i][j];
        }
        
        hidden_layer[i] = relu_activation(hidden_layer[i]);
    }
    
    /* Layer 2: Linear transformation + Softmax */
    for (int i = 0; i < SAIT01_MODEL_OUTPUT_CLASSES; i++) {
        output_probabilities[i] = mock_bias_layer2[i];
        
        for (int j = 0; j < 16; j++) {
            output_probabilities[i] += hidden_layer[j] * mock_weights_layer2[j][i];
        }
    }
    
    /* Apply softmax */
    softmax_activation(output_probabilities, SAIT01_MODEL_OUTPUT_CLASSES);
    
    return 0;
}

/* =============================================================================
 * PATTERN-BASED CLASSIFICATION
 * =============================================================================
 */

static sait01_ml_class_t classify_audio_patterns(const float* mel_spectrogram)
{
    /* Simple pattern-based classification for realistic behavior */
    
    /* Compute basic audio features */
    float energy = 0.0f;
    float spectral_centroid = 0.0f;
    float spectral_rolloff = 0.0f;
    float zero_crossing_rate = 0.0f;
    
    /* Energy calculation */
    for (int i = 0; i < SAIT01_MODEL_INPUT_SIZE; i++) {
        energy += mel_spectrogram[i] * mel_spectrogram[i];
    }
    energy = sqrtf(energy / SAIT01_MODEL_INPUT_SIZE);
    
    /* Spectral centroid (frequency content center) */
    float weighted_sum = 0.0f;
    float magnitude_sum = 0.0f;
    for (int i = 0; i < SAIT01_MEL_BINS; i++) {
        float magnitude = 0.0f;
        for (int j = 0; j < SAIT01_N_FRAMES; j++) {
            magnitude += fabsf(mel_spectrogram[j * SAIT01_MEL_BINS + i]);
        }
        weighted_sum += i * magnitude;
        magnitude_sum += magnitude;
    }
    spectral_centroid = (magnitude_sum > 0) ? (weighted_sum / magnitude_sum) : 0.0f;
    
    /* Pattern matching based on audio characteristics */
    if (energy < 0.1f) {
        return SAIT01_ML_CLASS_UNKNOWN; /* Low energy = background/unknown */
    }
    
    if (spectral_centroid > 40.0f && energy > 0.5f) {
        return SAIT01_ML_CLASS_AIRCRAFT; /* High frequency, high energy = aircraft */
    }
    
    if (spectral_centroid > 30.0f && energy > 0.3f && energy < 0.7f) {
        return SAIT01_ML_CLASS_VEHICLE; /* Mid-high frequency, medium energy = vehicle */
    }
    
    if (spectral_centroid < 20.0f && energy > 0.2f) {
        return SAIT01_ML_CLASS_MACHINERY; /* Low frequency, medium energy = machinery */
    }
    
    if (energy > 0.8f) {
        return SAIT01_ML_CLASS_GUNSHOT; /* Very high energy = gunshot */
    }
    
    if (spectral_centroid > 15.0f && spectral_centroid < 35.0f && energy > 0.2f) {
        return SAIT01_ML_CLASS_VOICES; /* Mid frequency, medium energy = voices */
    }
    
    return SAIT01_ML_CLASS_FOOTSTEPS; /* Default fallback */
}

static float calculate_confidence(sait01_ml_class_t detected_class, 
                                const float* mel_spectrogram)
{
    /* Calculate confidence based on pattern strength */
    float energy = 0.0f;
    for (int i = 0; i < SAIT01_MODEL_INPUT_SIZE; i++) {
        energy += mel_spectrogram[i] * mel_spectrogram[i];
    }
    energy = sqrtf(energy / SAIT01_MODEL_INPUT_SIZE);
    
    /* Base confidence from energy level */
    float confidence = fminf(energy * 2.0f, 1.0f);
    
    /* Adjust based on class-specific patterns */
    switch (detected_class) {
        case SAIT01_ML_CLASS_AIRCRAFT:
            confidence *= 0.85f; /* High confidence for aircraft detection */
            break;
        case SAIT01_ML_CLASS_GUNSHOT:
            confidence *= 0.95f; /* Very high confidence for gunshots */
            break;
        case SAIT01_ML_CLASS_EXPLOSION:
            confidence *= 0.98f; /* Extremely high confidence for explosions */
            break;
        case SAIT01_ML_CLASS_VEHICLE:
            confidence *= 0.75f; /* Medium confidence for vehicles */
            break;
        case SAIT01_ML_CLASS_UNKNOWN:
            confidence *= 0.3f; /* Low confidence for unknown */
            break;
        default:
            confidence *= 0.6f; /* Default medium confidence */
            break;
    }
    
    /* Add some randomness for realistic behavior */
    confidence += (sys_rand32_get() % 100) / 1000.0f; /* ±0.1 variation */
    
    return fminf(fmaxf(confidence, 0.0f), 1.0f);
}

/* =============================================================================
 * MOCK TINYML IMPLEMENTATION
 * =============================================================================
 */

int sait01_mock_tinyml_process_audio(sait01_tinyml_system_t* system,
                                    const int16_t* audio_data,
                                    size_t sample_count,
                                    sait01_ml_detection_t* result)
{
    if (!system || !audio_data || !result || sample_count == 0) {
        return -EINVAL;
    }
    
    uint32_t start_time = k_uptime_get_32();
    
    /* Extract mel spectrogram features */
    float mel_spectrogram[SAIT01_MODEL_INPUT_SIZE];
    int ret = sait01_optimized_extract_features(&system->feature_extractor,
                                               audio_data, mel_spectrogram);
    if (ret < 0) {
        LOG_ERR("Feature extraction failed: %d", ret);
        return ret;
    }
    
    /* Normalize features */
    ret = sait01_optimized_normalize_features(mel_spectrogram, SAIT01_MODEL_INPUT_SIZE);
    if (ret < 0) {
        LOG_ERR("Feature normalization failed: %d", ret);
        return ret;
    }
    
    /* Run mock inference */
    float output_probabilities[SAIT01_MODEL_OUTPUT_CLASSES];
    
    /* Use both neural network and pattern-based classification */
    ret = mock_neural_network_inference(mel_spectrogram, output_probabilities);
    if (ret < 0) {
        LOG_ERR("Neural network inference failed: %d", ret);
        return ret;
    }
    
    /* Pattern-based classification for backup */
    sait01_ml_class_t pattern_class = classify_audio_patterns(mel_spectrogram);
    
    /* Find highest probability class */
    int max_class = 0;
    float max_prob = output_probabilities[0];
    for (int i = 1; i < SAIT01_MODEL_OUTPUT_CLASSES; i++) {
        if (output_probabilities[i] > max_prob) {
            max_prob = output_probabilities[i];
            max_class = i;
        }
    }
    
    /* Use pattern-based result if neural network confidence is low */
    if (max_prob < 0.6f) {
        max_class = pattern_class;
        max_prob = calculate_confidence(pattern_class, mel_spectrogram);
    }
    
    /* Fill result structure */
    result->detected_class = (sait01_ml_class_t)max_class;
    result->confidence = max_prob;
    result->inference_time_us = (k_uptime_get_32() - start_time) * 1000;
    result->timestamp = k_uptime_get_32();
    
    /* Copy class probabilities */
    for (int i = 0; i < SAIT01_MODEL_OUTPUT_CLASSES; i++) {
        result->class_probabilities[i] = output_probabilities[i];
    }
    
    /* Generate embedding */
    ret = sait01_generate_optimized_embedding(mel_spectrogram, result->embedding);
    if (ret < 0) {
        LOG_WRN("Embedding generation failed: %d", ret);
        /* Continue without embedding */
    }
    
    LOG_DBG("Mock inference: class=%s, confidence=%.2f, time=%d μs",
            sait01_class_to_string(result->detected_class),
            result->confidence,
            result->inference_time_us);
    
    return 0;
}

int sait01_init_mock_model(sait01_tinyml_system_t* system)
{
    if (!system) {
        return -EINVAL;
    }
    
    LOG_INF("Initializing mock TinyML model");
    
    /* Initialize optimized preprocessing */
    int ret = sait01_init_optimized_preprocessing();
    if (ret < 0) {
        LOG_ERR("Failed to initialize preprocessing: %d", ret);
        return ret;
    }
    
    /* Initialize mel filters */
    ret = sait01_mel_filters_init(&system->feature_extractor);
    if (ret < 0) {
        LOG_ERR("Failed to initialize mel filters: %d", ret);
        return ret;
    }
    
    /* Mark model as loaded */
    system->ml_inference.model_loaded = true;
    
    LOG_INF("Mock TinyML model initialized successfully");
    LOG_INF("Model capabilities:");
    LOG_INF("  - Pattern-based classification");
    LOG_INF("  - Optimized preprocessing");
    LOG_INF("  - Real-time inference");
    LOG_INF("  - 8-class detection");
    
    return 0;
}

/* =============================================================================
 * TESTING AND VALIDATION
 * =============================================================================
 */

int sait01_test_mock_model_accuracy(void)
{
    LOG_INF("Testing mock model accuracy with synthetic data");
    
    sait01_tinyml_system_t test_system = {0};
    int ret = sait01_init_mock_model(&test_system);
    if (ret < 0) {
        LOG_ERR("Failed to initialize mock model: %d", ret);
        return ret;
    }
    
    /* Test with different synthetic audio patterns */
    struct {
        const char* name;
        float frequency;
        float amplitude;
        sait01_ml_class_t expected_class;
    } test_cases[] = {
        {"High frequency aircraft", 2000.0f, 0.8f, SAIT01_ML_CLASS_AIRCRAFT},
        {"Low frequency machinery", 100.0f, 0.6f, SAIT01_ML_CLASS_MACHINERY},
        {"Medium frequency vehicle", 800.0f, 0.5f, SAIT01_ML_CLASS_VEHICLE},
        {"Low amplitude background", 440.0f, 0.1f, SAIT01_ML_CLASS_UNKNOWN},
        {"High amplitude gunshot", 1000.0f, 0.9f, SAIT01_ML_CLASS_GUNSHOT}
    };
    
    int correct_predictions = 0;
    int total_tests = sizeof(test_cases) / sizeof(test_cases[0]);
    
    for (int i = 0; i < total_tests; i++) {
        /* Generate synthetic audio */
        int16_t test_audio[SAIT01_AUDIO_SAMPLES_PER_WINDOW];
        for (int j = 0; j < SAIT01_AUDIO_SAMPLES_PER_WINDOW; j++) {
            float t = (float)j / SAIT01_AUDIO_SAMPLE_RATE_HZ;
            test_audio[j] = (int16_t)(sinf(2.0f * M_PI * test_cases[i].frequency * t) * 
                                     test_cases[i].amplitude * 16000.0f);
        }
        
        /* Run inference */
        sait01_ml_detection_t result;
        ret = sait01_mock_tinyml_process_audio(&test_system, test_audio,
                                             SAIT01_AUDIO_SAMPLES_PER_WINDOW, &result);
        
        if (ret == 0) {
            LOG_INF("Test %d (%s): detected=%s, confidence=%.2f, time=%d μs",
                    i + 1, test_cases[i].name,
                    sait01_class_to_string(result.detected_class),
                    result.confidence,
                    result.inference_time_us);
            
            if (result.detected_class == test_cases[i].expected_class) {
                correct_predictions++;
            }
        } else {
            LOG_ERR("Test %d failed: %d", i + 1, ret);
        }
    }
    
    float accuracy = (float)correct_predictions / total_tests;
    LOG_INF("Mock model test results:");
    LOG_INF("  Correct predictions: %d/%d", correct_predictions, total_tests);
    LOG_INF("  Accuracy: %.1f%%", accuracy * 100.0f);
    
    if (accuracy >= 0.6f) {
        LOG_INF("Mock model performance acceptable (≥60%)");
    } else {
        LOG_WRN("Mock model performance below target (<60%)");
    }
    
    return 0;
}