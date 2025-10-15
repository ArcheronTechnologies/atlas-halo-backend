/*
 * SAIT_01 Distributed Mesh - Comprehensive Stress Test Suite
 * Hardware-in-the-loop testing with Renode simulation support
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/random/random.h>
#include <zephyr/sys/printk.h>
#include <zephyr/shell/shell.h>
#include "sait01_distributed_mesh.h"

LOG_MODULE_REGISTER(stress_test, CONFIG_LOG_DEFAULT_LEVEL);

/* =============================================================================
 * STRESS TEST CONFIGURATION
 * =============================================================================
 */

#define STRESS_TEST_DURATION_MS        300000  // 5 minutes
#define MAX_CONCURRENT_DETECTIONS      100
#define MAX_FUSION_REQUESTS            50
#define HIGH_FREQUENCY_DETECTION_HZ    20      // 20 detections per second
#define BURST_DETECTION_COUNT          200
#define MESH_FAILURE_SIMULATION_RATE   5       // % of messages to drop
#define MEMORY_STRESS_ALLOCATIONS      1000
#define COORDINATOR_ELECTION_CYCLES    20

/* Test statistics */
static struct {
    uint32_t detections_generated;
    uint32_t detections_processed;
    uint32_t fusion_requests_sent;
    uint32_t fusion_responses_received;
    uint32_t alerts_generated;
    uint32_t mesh_failures_simulated;
    uint32_t coordinator_elections;
    uint32_t memory_allocations;
    uint32_t test_start_time;
    uint32_t total_processing_time;
    uint32_t max_processing_time;
    uint32_t min_processing_time;
    uint32_t packet_loss_count;
    uint32_t false_positive_count;
    uint32_t false_negative_count;
} stress_stats = {
    .min_processing_time = UINT32_MAX,
};

/* Test control flags */
static volatile bool stress_test_running = false;
static volatile bool high_frequency_mode = false;
static volatile bool mesh_failure_simulation = false;
static volatile bool memory_stress_mode = false;

/* Work queues for stress testing */
static struct k_work_delayable stress_detection_work;
static struct k_work_delayable stress_fusion_work;
static struct k_work_delayable stress_coordinator_work;
static struct k_work_delayable stress_memory_work;
static struct k_work_delayable stress_monitor_work;

/* Mock node instances for multi-node simulation */
#define MAX_SIMULATED_NODES 10
static struct {
    uint16_t node_id;
    bool active;
    uint32_t last_detection_time;
    uint8_t battery_level;
    int8_t rssi;
    uint32_t detection_count;
} simulated_nodes[MAX_SIMULATED_NODES];

/* =============================================================================
 * STRESS TEST UTILITIES
 * =============================================================================
 */

static uint32_t get_test_time_ms(void)
{
    return k_uptime_get_32() - stress_stats.test_start_time;
}

static void update_processing_time(uint32_t processing_time)
{
    stress_stats.total_processing_time += processing_time;
    if (processing_time > stress_stats.max_processing_time) {
        stress_stats.max_processing_time = processing_time;
    }
    if (processing_time < stress_stats.min_processing_time) {
        stress_stats.min_processing_time = processing_time;
    }
}

static bool should_simulate_failure(void)
{
    if (!mesh_failure_simulation) {
        return false;
    }
    return (sys_rand32_get() % 100) < MESH_FAILURE_SIMULATION_RATE;
}

static uint8_t generate_random_class(void)
{
    uint8_t classes[] = {
        SAIT01_CLASS_VEHICLE,
        SAIT01_CLASS_FOOTSTEPS, 
        SAIT01_CLASS_VOICES,
        SAIT01_CLASS_AIRCRAFT,
        SAIT01_CLASS_MACHINERY,
        SAIT01_CLASS_GUNSHOT,
        SAIT01_CLASS_EXPLOSION
    };
    return classes[sys_rand32_get() % ARRAY_SIZE(classes)];
}

static void initialize_simulated_nodes(void)
{
    for (int i = 0; i < MAX_SIMULATED_NODES; i++) {
        simulated_nodes[i].node_id = 0x1000 + i;
        simulated_nodes[i].active = (i < 5); // Start with 5 active nodes
        simulated_nodes[i].last_detection_time = 0;
        simulated_nodes[i].battery_level = 85 + (sys_rand32_get() % 15); // 85-99%
        simulated_nodes[i].rssi = -40 - (sys_rand32_get() % 30); // -40 to -70 dBm
        simulated_nodes[i].detection_count = 0;
    }
}

/* =============================================================================
 * HIGH FREQUENCY DETECTION STRESS TEST
 * =============================================================================
 */

static void stress_detection_handler(struct k_work *work)
{
    if (!stress_test_running) {
        return;
    }

    uint32_t start_time = k_uptime_get_32();
    
    /* Generate burst of detections if in high frequency mode */
    int detection_count = high_frequency_mode ? 5 : 1;
    
    for (int i = 0; i < detection_count; i++) {
        /* Select random active node */
        int node_idx = sys_rand32_get() % MAX_SIMULATED_NODES;
        if (!simulated_nodes[node_idx].active) {
            continue;
        }
        
        /* Generate detection message */
        struct sait01_detection_msg detection = {
            .timestamp = k_uptime_get_32(),
            .sequence_id = simulated_nodes[node_idx].detection_count++,
            .class_id = generate_random_class(),
            .confidence = 8 + (sys_rand32_get() % 8), // 8-15
            .battery_level = simulated_nodes[node_idx].battery_level,
            .rssi = simulated_nodes[node_idx].rssi,
            .flags = (sys_rand32_get() % 10 == 0) ? BIT(0) : 0, // 10% tamper events
            .location_hash = 0x10000000 + node_idx
        };
        
        /* Generate mock embedding */
        for (int j = 0; j < 16; j++) {
            detection.embedding[j] = (int8_t)(sys_rand32_get() % 256 - 128);
        }
        
        /* Simulate mesh failure */
        if (should_simulate_failure()) {
            stress_stats.mesh_failures_simulated++;
            stress_stats.packet_loss_count++;
            continue;
        }
        
        /* Process detection (simulate mesh reception) */
        LOG_DBG("Stress test detection: node=0x%04x class=%d confidence=%d",
                simulated_nodes[node_idx].node_id, detection.class_id, detection.confidence);
        
        stress_stats.detections_generated++;
        stress_stats.detections_processed++;
        simulated_nodes[node_idx].last_detection_time = k_uptime_get_32();
    }
    
    uint32_t processing_time = k_uptime_get_32() - start_time;
    update_processing_time(processing_time);
    
    /* Schedule next detection burst */
    uint32_t delay = high_frequency_mode ? 
                    K_MSEC(1000 / HIGH_FREQUENCY_DETECTION_HZ) : 
                    K_MSEC(500 + sys_rand32_get() % 1500); // 0.5-2s random
    
    if (get_test_time_ms() < STRESS_TEST_DURATION_MS) {
        k_work_reschedule(&stress_detection_work, delay);
    }
}

/* =============================================================================
 * FUSION PROTOCOL STRESS TEST
 * =============================================================================
 */

static void stress_fusion_handler(struct k_work *work)
{
    if (!stress_test_running) {
        return;
    }
    
    uint32_t start_time = k_uptime_get_32();
    
    /* Generate multiple concurrent fusion requests */
    int fusion_count = 1 + (sys_rand32_get() % 5); // 1-5 concurrent requests
    
    for (int i = 0; i < fusion_count; i++) {
        struct sait01_fusion_request request = {
            .correlation_window = 3000 + (sys_rand32_get() % 7000), // 3-10s window
            .min_nodes = 2 + (sys_rand32_get() % 4), // 2-5 nodes minimum  
            .detection_class = generate_random_class(),
            .confidence_thresh = 8 + (sys_rand32_get() % 4), // 8-11 threshold
            .area_mask = sys_rand32_get()
        };
        
        /* Count fusion request attempt */
        stress_stats.fusion_requests_sent++;
        
        /* Simulate fusion response */
        if (!should_simulate_failure()) {
            struct sait01_fusion_response response = {
                .correlation_id = sys_rand32_get(),
                .participating_nodes = request.min_nodes + (sys_rand32_get() % 3),
                .consensus_class = request.detection_class,
                .consensus_confidence = 10 + (sys_rand32_get() % 6), // 10-15
                .alert_level = SAIT01_ALERT_LOW + (sys_rand32_get() % 4),
                .fused_location = sys_rand32_get(),
                .time_to_live = 300 + (sys_rand32_get() % 300)
            };
            
            stress_stats.fusion_responses_received++;
            
            /* Check if alert should be generated */
            if (response.consensus_confidence >= 12 && response.alert_level >= SAIT01_ALERT_MEDIUM) {
                stress_stats.alerts_generated++;
                LOG_WRN("Stress test alert: class=%d level=%d confidence=%d nodes=%d",
                        response.consensus_class, response.alert_level,
                        response.consensus_confidence, response.participating_nodes);
            }
        } else {
            stress_stats.mesh_failures_simulated++;
            stress_stats.packet_loss_count++;
        }
    }
    
    uint32_t processing_time = k_uptime_get_32() - start_time;
    update_processing_time(processing_time);
    
    /* Schedule next fusion cycle */
    if (get_test_time_ms() < STRESS_TEST_DURATION_MS) {
        k_work_reschedule(&stress_fusion_work, K_MSEC(2000 + sys_rand32_get() % 3000));
    }
}

/* =============================================================================
 * COORDINATOR ELECTION STRESS TEST  
 * =============================================================================
 */

static void stress_coordinator_handler(struct k_work *work)
{
    if (!stress_test_running) {
        return;
    }
    
    uint32_t start_time = k_uptime_get_32();
    
    /* Simulate coordinator failure and election */
    LOG_INF("Stress test: Simulating coordinator election #%d", 
            stress_stats.coordinator_elections + 1);
    
    /* Generate multiple election messages from different nodes */
    for (int i = 0; i < 3; i++) {
        struct sait01_coord_election election = {
            .node_priority = 50 + (sys_rand32_get() % 50), // 50-99 priority
            .capabilities = SAIT01_CAP_AUDIO_ML | SAIT01_CAP_COORDINATOR |
                          ((sys_rand32_get() % 2) ? SAIT01_CAP_UWB_RANGING : 0) |
                          ((sys_rand32_get() % 2) ? SAIT01_CAP_RF_PROXY : 0),
            .neighbor_count = 3 + (sys_rand32_get() % 5), // 3-7 neighbors
            .uptime_seconds = 1800 + (sys_rand32_get() % 7200), // 30min-2hr
            .battery_level = 70 + (sys_rand32_get() % 30), // 70-99%
            .compute_capacity = 5 + (sys_rand32_get() % 10) // 5-14
        };
        
        if (!should_simulate_failure()) {
            LOG_DBG("Election candidate: priority=%d caps=0x%02x neighbors=%d",
                    election.node_priority, election.capabilities, election.neighbor_count);
        }
    }
    
    stress_stats.coordinator_elections++;
    
    uint32_t processing_time = k_uptime_get_32() - start_time;
    update_processing_time(processing_time);
    
    /* Schedule next election cycle */
    if (get_test_time_ms() < STRESS_TEST_DURATION_MS && 
        stress_stats.coordinator_elections < COORDINATOR_ELECTION_CYCLES) {
        k_work_reschedule(&stress_coordinator_work, K_SECONDS(10 + sys_rand32_get() % 20));
    }
}

/* =============================================================================
 * MEMORY STRESS TEST
 * =============================================================================
 */

static void stress_memory_handler(struct k_work *work)
{
    if (!stress_test_running || !memory_stress_mode) {
        return;
    }
    
    uint32_t start_time = k_uptime_get_32();
    
    /* Allocate and free memory blocks to test heap management */
    void *ptrs[20];
    int alloc_count = 0;
    
    for (int i = 0; i < 20; i++) {
        size_t size = 64 + (sys_rand32_get() % 512); // 64-576 bytes
        ptrs[i] = k_malloc(size);
        if (ptrs[i] != NULL) {
            memset(ptrs[i], 0xAA, size); // Fill with pattern
            alloc_count++;
            stress_stats.memory_allocations++;
        } else {
            LOG_WRN("Memory allocation failed: size=%d", size);
            break;
        }
    }
    
    /* Free allocated memory */
    for (int i = 0; i < alloc_count; i++) {
        if (ptrs[i] != NULL) {
            k_free(ptrs[i]);
        }
    }
    
    uint32_t processing_time = k_uptime_get_32() - start_time;
    update_processing_time(processing_time);
    
    /* Schedule next memory stress cycle */
    if (get_test_time_ms() < STRESS_TEST_DURATION_MS) {
        k_work_reschedule(&stress_memory_work, K_MSEC(1000));
    }
}

/* =============================================================================
 * MONITORING AND STATISTICS
 * =============================================================================
 */

static void stress_monitor_handler(struct k_work *work)
{
    if (!stress_test_running) {
        return;
    }
    
    uint32_t test_time = get_test_time_ms();
    uint32_t avg_processing_time = stress_stats.detections_processed > 0 ?
                                  stress_stats.total_processing_time / stress_stats.detections_processed : 0;
    
    /* Calculate rates */
    uint32_t detection_rate = (stress_stats.detections_processed * 1000) / (test_time + 1);
    uint32_t fusion_rate = (stress_stats.fusion_responses_received * 1000) / (test_time + 1);
    uint32_t alert_rate = (stress_stats.alerts_generated * 1000) / (test_time + 1);
    uint32_t packet_loss_rate = stress_stats.detections_generated > 0 ?
                               (stress_stats.packet_loss_count * 100) / stress_stats.detections_generated : 0;
    
    LOG_INF("=== Stress Test Status ===");
    LOG_INF("Runtime: %d.%ds", test_time / 1000, (test_time % 1000) / 100);
    LOG_INF("Detections: %d generated, %d processed (%.1f/s)",
            stress_stats.detections_generated, stress_stats.detections_processed, 
            (float)detection_rate);
    LOG_INF("Fusion: %d requests, %d responses (%.1f/s)",
            stress_stats.fusion_requests_sent, stress_stats.fusion_responses_received,
            (float)fusion_rate);
    LOG_INF("Alerts: %d generated (%.1f/s)", stress_stats.alerts_generated, (float)alert_rate);
    LOG_INF("Elections: %d completed", stress_stats.coordinator_elections);
    LOG_INF("Failures: %d simulated, %d%% packet loss", 
            stress_stats.mesh_failures_simulated, packet_loss_rate);
    LOG_INF("Processing: avg=%dms, min=%dms, max=%dms",
            avg_processing_time, stress_stats.min_processing_time, stress_stats.max_processing_time);
    LOG_INF("Memory: %d allocations", stress_stats.memory_allocations);
    
    /* Check for anomalies */
    if (avg_processing_time > 100) {
        LOG_WRN("High average processing time: %dms", avg_processing_time);
    }
    if (packet_loss_rate > 20) {
        LOG_WRN("High packet loss rate: %d%%", packet_loss_rate);
    }
    
    /* Schedule next monitoring cycle */
    if (test_time < STRESS_TEST_DURATION_MS) {
        k_work_reschedule(&stress_monitor_work, K_SECONDS(10));
    } else {
        /* Test completed */
        stress_test_running = false;
        LOG_INF("=== Stress Test Completed ===");
        LOG_INF("Total runtime: %d seconds", STRESS_TEST_DURATION_MS / 1000);
        LOG_INF("Performance summary:");
        LOG_INF("  - Detection throughput: %.1f detections/second", (float)detection_rate);
        LOG_INF("  - Fusion throughput: %.1f fusions/second", (float)fusion_rate);
        LOG_INF("  - Alert generation: %.1f alerts/second", (float)alert_rate);
        LOG_INF("  - Network reliability: %d%% success rate", 100 - packet_loss_rate);
        LOG_INF("  - Processing latency: %dms average", avg_processing_time);
    }
}

/* =============================================================================
 * SHELL COMMANDS FOR STRESS TEST CONTROL
 * =============================================================================
 */

static int cmd_stress_start(const struct shell *shell, size_t argc, char **argv)
{
    if (stress_test_running) {
        shell_print(shell, "Stress test already running");
        return -EBUSY;
    }
    
    /* Parse options */
    high_frequency_mode = false;
    mesh_failure_simulation = false;
    memory_stress_mode = false;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--high-freq") == 0) {
            high_frequency_mode = true;
        } else if (strcmp(argv[i], "--mesh-failures") == 0) {
            mesh_failure_simulation = true;
        } else if (strcmp(argv[i], "--memory-stress") == 0) {
            memory_stress_mode = true;
        }
    }
    
    /* Initialize test */
    memset(&stress_stats, 0, sizeof(stress_stats));
    stress_stats.min_processing_time = UINT32_MAX;
    stress_stats.test_start_time = k_uptime_get_32();
    stress_test_running = true;
    
    initialize_simulated_nodes();
    
    /* Start stress test work queues */
    k_work_init_delayable(&stress_detection_work, stress_detection_handler);
    k_work_init_delayable(&stress_fusion_work, stress_fusion_handler);
    k_work_init_delayable(&stress_coordinator_work, stress_coordinator_handler);
    k_work_init_delayable(&stress_memory_work, stress_memory_handler);
    k_work_init_delayable(&stress_monitor_work, stress_monitor_handler);
    
    k_work_schedule(&stress_detection_work, K_SECONDS(1));
    k_work_schedule(&stress_fusion_work, K_SECONDS(3));
    k_work_schedule(&stress_coordinator_work, K_SECONDS(15));
    if (memory_stress_mode) {
        k_work_schedule(&stress_memory_work, K_SECONDS(5));
    }
    k_work_schedule(&stress_monitor_work, K_SECONDS(10));
    
    shell_print(shell, "Stress test started:");
    shell_print(shell, "  Duration: %d seconds", STRESS_TEST_DURATION_MS / 1000);
    shell_print(shell, "  High frequency: %s", high_frequency_mode ? "enabled" : "disabled");
    shell_print(shell, "  Mesh failures: %s", mesh_failure_simulation ? "enabled" : "disabled");
    shell_print(shell, "  Memory stress: %s", memory_stress_mode ? "enabled" : "disabled");
    
    return 0;
}

static int cmd_stress_stop(const struct shell *shell, size_t argc, char **argv)
{
    if (!stress_test_running) {
        shell_print(shell, "No stress test running");
        return -EINVAL;
    }
    
    stress_test_running = false;
    
    /* Cancel all work */
    k_work_cancel_delayable(&stress_detection_work);
    k_work_cancel_delayable(&stress_fusion_work);
    k_work_cancel_delayable(&stress_coordinator_work);
    k_work_cancel_delayable(&stress_memory_work);
    k_work_cancel_delayable(&stress_monitor_work);
    
    shell_print(shell, "Stress test stopped");
    return 0;
}

static int cmd_stress_status(const struct shell *shell, size_t argc, char **argv)
{
    if (!stress_test_running) {
        shell_print(shell, "No stress test running");
        return 0;
    }
    
    uint32_t test_time = get_test_time_ms();
    uint32_t avg_processing_time = stress_stats.detections_processed > 0 ?
                                  stress_stats.total_processing_time / stress_stats.detections_processed : 0;
    
    shell_print(shell, "Stress Test Status:");
    shell_print(shell, "  Runtime: %d.%ds", test_time / 1000, (test_time % 1000) / 100);
    shell_print(shell, "  Detections: %d processed", stress_stats.detections_processed);
    shell_print(shell, "  Fusion ops: %d completed", stress_stats.fusion_responses_received);
    shell_print(shell, "  Alerts: %d generated", stress_stats.alerts_generated);
    shell_print(shell, "  Elections: %d completed", stress_stats.coordinator_elections);
    shell_print(shell, "  Failures: %d simulated", stress_stats.mesh_failures_simulated);
    shell_print(shell, "  Processing time: %dms avg", avg_processing_time);
    
    return 0;
}

static int cmd_stress_burst(const struct shell *shell, size_t argc, char **argv)
{
    int count = (argc > 1) ? atoi(argv[1]) : BURST_DETECTION_COUNT;
    
    if (count <= 0 || count > 1000) {
        shell_print(shell, "Invalid burst count (1-1000)");
        return -EINVAL;
    }
    
    shell_print(shell, "Starting detection burst: %d detections", count);
    
    uint32_t start_time = k_uptime_get_32();
    
    for (int i = 0; i < count; i++) {
        struct sait01_detection_msg detection = {
            .timestamp = k_uptime_get_32(),
            .sequence_id = i,
            .class_id = generate_random_class(),
            .confidence = 8 + (sys_rand32_get() % 8),
            .battery_level = 85,
            .rssi = -45,
            .flags = 0,
            .location_hash = 0x12345678
        };
        
        /* Simulate processing delay */
        k_busy_wait(100); // 100 microseconds
    }
    
    uint32_t total_time = k_uptime_get_32() - start_time;
    float throughput = (float)count * 1000.0f / total_time;
    
    shell_print(shell, "Burst completed: %d detections in %dms (%.1f det/s)",
                count, total_time, throughput);
    
    return 0;
}

SHELL_STATIC_SUBCMD_SET_CREATE(stress_commands,
    SHELL_CMD_ARG(start, NULL, "Start stress test [--high-freq] [--mesh-failures] [--memory-stress]", 
                  cmd_stress_start, 1, 3),
    SHELL_CMD(stop, NULL, "Stop stress test", cmd_stress_stop),
    SHELL_CMD(status, NULL, "Show stress test status", cmd_stress_status),
    SHELL_CMD_ARG(burst, NULL, "Run detection burst test [count]", cmd_stress_burst, 1, 1),
    SHELL_SUBCMD_SET_END
);

SHELL_CMD_REGISTER(stress, &stress_commands, "SAIT_01 stress test commands", NULL);

/* =============================================================================
 * INITIALIZATION  
 * =============================================================================
 */

void sait01_stress_test_init(void)
{
    LOG_INF("SAIT_01 stress test framework initialized");
    LOG_INF("Available commands:");
    LOG_INF("  stress start [options] - Start comprehensive stress test");
    LOG_INF("  stress stop            - Stop running stress test");
    LOG_INF("  stress status          - Show current test status");
    LOG_INF("  stress burst [count]   - Run detection burst test");
}