/*
 * SAIT_01 Software-based Stress Test Runner
 * Comprehensive stress testing without hardware simulation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/time.h>

/* Mock the SAIT_01 distributed mesh protocols for testing */
typedef enum {
    SAIT01_CLASS_UNKNOWN = 0,
    SAIT01_CLASS_VEHICLE,
    SAIT01_CLASS_FOOTSTEPS,
    SAIT01_CLASS_VOICES,
    SAIT01_CLASS_AIRCRAFT,
    SAIT01_CLASS_MACHINERY,
    SAIT01_CLASS_GUNSHOT,
    SAIT01_CLASS_EXPLOSION
} sait01_detection_class_t;

typedef enum {
    SAIT01_ALERT_INFO = 0,
    SAIT01_ALERT_LOW = 1,
    SAIT01_ALERT_MEDIUM = 2,
    SAIT01_ALERT_HIGH = 3,
    SAIT01_ALERT_CRITICAL = 4
} sait01_alert_level_t;

typedef struct {
    uint32_t timestamp;
    uint16_t sequence_id;
    uint8_t class_id;
    uint8_t confidence;
    uint8_t battery_level;
    int8_t rssi;
    uint8_t flags;
    int8_t embedding[16];
    uint32_t location_hash;
} __attribute__((packed)) sait01_detection_msg_t;

typedef struct {
    uint32_t correlation_window;
    uint8_t min_nodes;
    uint8_t detection_class;
    uint8_t confidence_thresh;
    uint32_t area_mask;
} __attribute__((packed)) sait01_fusion_request_t;

typedef struct {
    uint32_t correlation_id;
    uint8_t participating_nodes;
    uint8_t consensus_class;
    uint8_t consensus_confidence;
    uint8_t alert_level;
    uint32_t fused_location;
    uint16_t time_to_live;
} __attribute__((packed)) sait01_fusion_response_t;

/* Test configuration */
#define MAX_NODES 20
#define TEST_DURATION_SECONDS 60
#define MAX_DETECTIONS_PER_SECOND 50
#define MAX_CONCURRENT_FUSIONS 10
#define CORRELATION_TIMEOUT_MS 5000

/* Test statistics */
typedef struct {
    uint64_t detections_generated;
    uint64_t detections_processed;
    uint64_t fusion_requests_sent;
    uint64_t fusion_responses_received;
    uint64_t alerts_generated;
    uint64_t packets_dropped;
    uint64_t processing_time_total_us;
    uint64_t processing_time_max_us;
    uint64_t processing_time_min_us;
    uint32_t memory_allocations;
    uint32_t coordinator_elections;
    time_t start_time;
    bool test_running;
} stress_test_stats_t;

static stress_test_stats_t test_stats;
static pthread_mutex_t stats_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Node simulation */
typedef struct {
    uint16_t node_id;
    bool active;
    uint8_t battery_level;
    int8_t rssi;
    uint32_t detection_count;
    time_t last_detection_time;
    bool is_coordinator;
    uint8_t capabilities;
} simulated_node_t;

static simulated_node_t nodes[MAX_NODES];
static int active_node_count = 0;

/* Detection storage for correlation */
typedef struct {
    sait01_detection_msg_t detection;
    uint16_t source_node;
    struct timeval receive_time;
    bool used_in_fusion;
} detection_record_t;

#define MAX_DETECTION_RECORDS 1000
static detection_record_t detection_records[MAX_DETECTION_RECORDS];
static int detection_record_count = 0;
static pthread_mutex_t detection_mutex = PTHREAD_MUTEX_INITIALIZER;

/* =============================================================================
 * UTILITY FUNCTIONS
 * =============================================================================
 */

static uint64_t get_time_us(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000ULL + tv.tv_usec;
}

static uint32_t get_time_ms(void)
{
    return get_time_us() / 1000;
}

static uint8_t random_detection_class(void)
{
    uint8_t classes[] = {
        SAIT01_CLASS_VEHICLE, SAIT01_CLASS_FOOTSTEPS, SAIT01_CLASS_VOICES,
        SAIT01_CLASS_AIRCRAFT, SAIT01_CLASS_MACHINERY, SAIT01_CLASS_GUNSHOT,
        SAIT01_CLASS_EXPLOSION
    };
    return classes[rand() % (sizeof(classes) / sizeof(classes[0]))];
}

static void update_stats_processing_time(uint64_t time_us)
{
    pthread_mutex_lock(&stats_mutex);
    test_stats.processing_time_total_us += time_us;
    if (time_us > test_stats.processing_time_max_us) {
        test_stats.processing_time_max_us = time_us;
    }
    if (time_us < test_stats.processing_time_min_us || test_stats.processing_time_min_us == 0) {
        test_stats.processing_time_min_us = time_us;
    }
    pthread_mutex_unlock(&stats_mutex);
}

/* =============================================================================
 * NODE SIMULATION
 * =============================================================================
 */

static void initialize_nodes(int count)
{
    active_node_count = count;
    
    for (int i = 0; i < count; i++) {
        nodes[i].node_id = 0x1000 + i;
        nodes[i].active = true;
        nodes[i].battery_level = 70 + (rand() % 30); // 70-99%
        nodes[i].rssi = -30 - (rand() % 40); // -30 to -70 dBm
        nodes[i].detection_count = 0;
        nodes[i].last_detection_time = 0;
        nodes[i].is_coordinator = (i == 0); // First node starts as coordinator
        nodes[i].capabilities = 0x3F; // All capabilities
    }
    
    printf("Initialized %d simulated nodes\n", count);
}

static simulated_node_t* get_random_active_node(void)
{
    if (active_node_count == 0) return NULL;
    
    int attempts = 0;
    while (attempts < 10) {
        int idx = rand() % active_node_count;
        if (nodes[idx].active) {
            return &nodes[idx];
        }
        attempts++;
    }
    return NULL;
}

/* =============================================================================
 * DETECTION PROCESSING
 * =============================================================================
 */

static void process_detection(const sait01_detection_msg_t* detection, uint16_t source_node)
{
    uint64_t start_time = get_time_us();
    
    pthread_mutex_lock(&detection_mutex);
    
    // Store detection for correlation
    if (detection_record_count < MAX_DETECTION_RECORDS) {
        detection_records[detection_record_count].detection = *detection;
        detection_records[detection_record_count].source_node = source_node;
        gettimeofday(&detection_records[detection_record_count].receive_time, NULL);
        detection_records[detection_record_count].used_in_fusion = false;
        detection_record_count++;
    }
    
    pthread_mutex_unlock(&detection_mutex);
    
    // Update statistics
    pthread_mutex_lock(&stats_mutex);
    test_stats.detections_processed++;
    pthread_mutex_unlock(&stats_mutex);
    
    uint64_t processing_time = get_time_us() - start_time;
    update_stats_processing_time(processing_time);
}

static int correlate_detections(uint8_t target_class, sait01_fusion_response_t* response)
{
    if (!response) return 0;
    
    struct timeval now;
    gettimeofday(&now, NULL);
    
    int correlated_count = 0;
    uint8_t class_votes[8] = {0};
    uint16_t total_confidence = 0;
    
    pthread_mutex_lock(&detection_mutex);
    
    // Find recent detections within correlation window
    for (int i = 0; i < detection_record_count; i++) {
        detection_record_t* rec = &detection_records[i];
        
        // Check if within correlation window (5 seconds)
        long time_diff = (now.tv_sec - rec->receive_time.tv_sec) * 1000 +
                        (now.tv_usec - rec->receive_time.tv_usec) / 1000;
        
        if (time_diff > CORRELATION_TIMEOUT_MS) {
            continue; // Too old
        }
        
        if (!rec->used_in_fusion) {
            if (rec->detection.class_id < 8) {
                class_votes[rec->detection.class_id]++;
            }
            total_confidence += rec->detection.confidence;
            correlated_count++;
            rec->used_in_fusion = true;
        }
    }
    
    pthread_mutex_unlock(&detection_mutex);
    
    if (correlated_count == 0) {
        return 0;
    }
    
    // Find consensus class (most votes)
    uint8_t consensus_class = 0;
    uint8_t max_votes = 0;
    for (int i = 0; i < 8; i++) {
        if (class_votes[i] > max_votes) {
            max_votes = i;
            consensus_class = i;
        }
    }
    
    // Fill response
    response->correlation_id = rand();
    response->participating_nodes = correlated_count;
    response->consensus_class = consensus_class;
    response->consensus_confidence = total_confidence / correlated_count;
    
    // Determine alert level
    if (max_votes >= 3 && response->consensus_confidence >= 12) {
        response->alert_level = SAIT01_ALERT_HIGH;
    } else if (max_votes >= 2 && response->consensus_confidence >= 8) {
        response->alert_level = SAIT01_ALERT_MEDIUM;
    } else {
        response->alert_level = SAIT01_ALERT_LOW;
    }
    
    response->time_to_live = 300; // 5 minutes
    response->fused_location = rand();
    
    return correlated_count;
}

/* =============================================================================
 * STRESS TEST THREADS
 * =============================================================================
 */

static void* detection_generator_thread(void* arg)
{
    printf("Detection generator thread started\n");
    
    while (test_stats.test_running) {
        // Generate burst of detections
        int burst_size = 1 + (rand() % 10); // 1-10 detections per burst
        
        for (int i = 0; i < burst_size; i++) {
            simulated_node_t* node = get_random_active_node();
            if (!node) break;
            
            sait01_detection_msg_t detection = {
                .timestamp = get_time_ms(),
                .sequence_id = node->detection_count++,
                .class_id = random_detection_class(),
                .confidence = 8 + (rand() % 8), // 8-15
                .battery_level = node->battery_level,
                .rssi = node->rssi,
                .flags = (rand() % 20 == 0) ? 1 : 0, // 5% tamper events
                .location_hash = 0x10000000 + (node->node_id & 0xFFFF)
            };
            
            // Generate random embedding
            for (int j = 0; j < 16; j++) {
                detection.embedding[j] = (rand() % 256) - 128;
            }
            
            // Process detection
            process_detection(&detection, node->node_id);
            
            pthread_mutex_lock(&stats_mutex);
            test_stats.detections_generated++;
            pthread_mutex_unlock(&stats_mutex);
            
            node->last_detection_time = time(NULL);
        }
        
        // Variable delay between bursts
        usleep(50000 + (rand() % 200000)); // 50-250ms
    }
    
    printf("Detection generator thread finished\n");
    return NULL;
}

static void* fusion_processor_thread(void* arg)
{
    printf("Fusion processor thread started\n");
    
    while (test_stats.test_running) {
        // Generate fusion requests
        sait01_fusion_request_t request = {
            .correlation_window = 3000 + (rand() % 5000), // 3-8 seconds
            .min_nodes = 2 + (rand() % 3), // 2-4 nodes minimum
            .detection_class = random_detection_class(),
            .confidence_thresh = 8 + (rand() % 4), // 8-11
            .area_mask = rand()
        };
        
        pthread_mutex_lock(&stats_mutex);
        test_stats.fusion_requests_sent++;
        pthread_mutex_unlock(&stats_mutex);
        
        // Process fusion
        sait01_fusion_response_t response;
        int correlated = correlate_detections(request.detection_class, &response);
        
        if (correlated >= request.min_nodes) {
            pthread_mutex_lock(&stats_mutex);
            test_stats.fusion_responses_received++;
            
            // Check if alert should be generated
            if (response.consensus_confidence >= 10 && response.alert_level >= SAIT01_ALERT_MEDIUM) {
                test_stats.alerts_generated++;
                printf("ALERT: Class=%d, Level=%d, Confidence=%d, Nodes=%d\n",
                       response.consensus_class, response.alert_level,
                       response.consensus_confidence, response.participating_nodes);
            }
            
            pthread_mutex_unlock(&stats_mutex);
        }
        
        // Delay between fusion cycles
        usleep(1000000 + (rand() % 2000000)); // 1-3 seconds
    }
    
    printf("Fusion processor thread finished\n");
    return NULL;
}

static void* coordinator_election_thread(void* arg)
{
    printf("Coordinator thread started\n");
    
    while (test_stats.test_running) {
        // Simulate coordinator elections periodically
        if (rand() % 100 < 5) { // 5% chance per cycle
            printf("Coordinator election triggered\n");
            
            // Select new coordinator based on priority
            int best_node = 0;
            uint8_t best_priority = 0;
            
            for (int i = 0; i < active_node_count; i++) {
                if (!nodes[i].active) continue;
                
                uint8_t priority = nodes[i].battery_level + 
                                 (nodes[i].capabilities * 10) + 
                                 (rand() % 20); // Random component
                
                if (priority > best_priority) {
                    best_priority = priority;
                    best_node = i;
                }
            }
            
            // Update coordinator status
            for (int i = 0; i < active_node_count; i++) {
                nodes[i].is_coordinator = (i == best_node);
            }
            
            pthread_mutex_lock(&stats_mutex);
            test_stats.coordinator_elections++;
            pthread_mutex_unlock(&stats_mutex);
            
            printf("New coordinator: Node 0x%04x (priority: %d)\n", 
                   nodes[best_node].node_id, best_priority);
        }
        
        sleep(5); // Check every 5 seconds
    }
    
    printf("Coordinator thread finished\n");
    return NULL;
}

/* =============================================================================
 * MAIN STRESS TEST CONTROLLER
 * =============================================================================
 */

static void print_statistics(void)
{
    pthread_mutex_lock(&stats_mutex);
    
    time_t elapsed = time(NULL) - test_stats.start_time;
    double detection_rate = elapsed > 0 ? (double)test_stats.detections_processed / elapsed : 0;
    double fusion_rate = elapsed > 0 ? (double)test_stats.fusion_responses_received / elapsed : 0;
    double alert_rate = elapsed > 0 ? (double)test_stats.alerts_generated / elapsed : 0;
    double avg_processing_time = test_stats.detections_processed > 0 ? 
        (double)test_stats.processing_time_total_us / test_stats.detections_processed : 0;
    
    printf("\n=== STRESS TEST STATISTICS ===\n");
    printf("Runtime: %ld seconds\n", elapsed);
    printf("Detections: %llu generated, %llu processed (%.1f/sec)\n", 
           test_stats.detections_generated, test_stats.detections_processed, detection_rate);
    printf("Fusion: %llu requests, %llu responses (%.1f/sec)\n",
           test_stats.fusion_requests_sent, test_stats.fusion_responses_received, fusion_rate);
    printf("Alerts: %llu generated (%.1f/sec)\n", test_stats.alerts_generated, alert_rate);
    printf("Elections: %u coordinator changes\n", test_stats.coordinator_elections);
    printf("Processing time: %.1f μs avg, %llu μs max, %llu μs min\n",
           avg_processing_time, test_stats.processing_time_max_us, test_stats.processing_time_min_us);
    printf("Active nodes: %d\n", active_node_count);
    printf("Detection records: %d stored\n", detection_record_count);
    
    pthread_mutex_unlock(&stats_mutex);
}

static void run_stress_test(int duration_seconds, int node_count)
{
    printf("\n=== SAIT_01 DISTRIBUTED MESH STRESS TEST ===\n");
    printf("Duration: %d seconds\n", duration_seconds);
    printf("Nodes: %d\n", node_count);
    printf("Starting test...\n\n");
    
    // Initialize test
    memset(&test_stats, 0, sizeof(test_stats));
    test_stats.start_time = time(NULL);
    test_stats.test_running = true;
    test_stats.processing_time_min_us = UINT64_MAX;
    
    initialize_nodes(node_count);
    
    // Create test threads
    pthread_t detection_thread, fusion_thread, coordinator_thread;
    
    pthread_create(&detection_thread, NULL, detection_generator_thread, NULL);
    pthread_create(&fusion_thread, NULL, fusion_processor_thread, NULL);
    pthread_create(&coordinator_thread, NULL, coordinator_election_thread, NULL);
    
    // Run test with periodic status updates
    for (int i = 0; i < duration_seconds; i += 10) {
        sleep(10);
        printf("Test running... %d/%d seconds\n", i + 10, duration_seconds);
        print_statistics();
    }
    
    // Stop test
    test_stats.test_running = false;
    
    // Wait for threads to finish
    pthread_join(detection_thread, NULL);
    pthread_join(fusion_thread, NULL);
    pthread_join(coordinator_thread, NULL);
    
    // Final statistics
    printf("\n=== FINAL RESULTS ===\n");
    print_statistics();
    
    // Performance analysis
    printf("\n=== PERFORMANCE ANALYSIS ===\n");
    double throughput = test_stats.detections_processed / (double)duration_seconds;
    printf("Overall throughput: %.1f detections/second\n", throughput);
    
    if (test_stats.processing_time_max_us > 10000) {
        printf("WARNING: Maximum processing time exceeded 10ms (%llu μs)\n", 
               test_stats.processing_time_max_us);
    }
    
    if (throughput < 10.0) {
        printf("WARNING: Low throughput detected (< 10 det/sec)\n");
    } else {
        printf("PASS: Throughput within acceptable range\n");
    }
    
    if (test_stats.alerts_generated > 0) {
        double alert_ratio = (double)test_stats.alerts_generated / test_stats.detections_processed;
        printf("Alert generation ratio: %.2f%% (%s)\n", alert_ratio * 100,
               alert_ratio > 0.1 ? "HIGH" : alert_ratio > 0.01 ? "NORMAL" : "LOW");
    }
    
    printf("\nStress test completed successfully!\n");
}

int main(int argc, char* argv[])
{
    srand(time(NULL));
    
    int duration = TEST_DURATION_SECONDS;
    int nodes = 10;
    
    // Parse command line arguments
    if (argc > 1) {
        duration = atoi(argv[1]);
        if (duration <= 0) duration = TEST_DURATION_SECONDS;
    }
    
    if (argc > 2) {
        nodes = atoi(argv[2]);
        if (nodes <= 0 || nodes > MAX_NODES) nodes = 10;
    }
    
    run_stress_test(duration, nodes);
    return 0;
}

/* Build instructions:
 * gcc -o sait01_stress_test sait01_software_stress_test.c -lpthread
 * ./sait01_stress_test [duration_seconds] [node_count]
 */