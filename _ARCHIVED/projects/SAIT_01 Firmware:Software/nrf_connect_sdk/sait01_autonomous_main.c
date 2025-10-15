/*
 * SAIT_01 Autonomous Sensor Node - Main Application
 * 
 * Distributed peer-to-peer mesh network with on-device ML and autonomous decision making
 * NO GATEWAY DEPENDENCY - Direct cloud fallback only
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/mesh.h>
#include <zephyr/settings/settings.h>
#include <bluetooth/mesh/dk_prov.h>
#include <dk_buttons_and_leds.h>

#include "sait01_distributed_mesh.h"
#include "sait01_tinyml_integration.h"

LOG_MODULE_REGISTER(sait01_main, CONFIG_LOG_DEFAULT_LEVEL);

/* =============================================================================
 * GLOBAL STATE AND CONFIGURATION
 * =============================================================================
 */

/* Node capabilities and configuration */
#define NODE_CAPABILITIES (SAIT01_CAP_AUDIO_ML | SAIT01_CAP_UWB_RANGING | \
                          SAIT01_CAP_RF_PROXY | SAIT01_CAP_LORA_FALLBACK | \
                          SAIT01_CAP_COORDINATOR | SAIT01_CAP_EDGE_COMPUTE)

/* Model instances */
static struct sait01_detection_srv detection_srv;
static struct sait01_fusion_srv fusion_srv;  
static struct sait01_coord_srv coord_srv;

/* Health model for basic mesh functionality */
static struct bt_mesh_health_srv health_srv = {
    .cb = {
        .fault_get_cur = NULL,
        .fault_get_reg = NULL,
        .fault_clear = NULL,
        .fault_test = NULL,
    },
};

BT_MESH_HEALTH_PUB_DEFINE(health_pub, 0);

/* Publication contexts */
BT_MESH_MODEL_PUB_DEFINE(detection_pub, NULL, 2 + sizeof(struct sait01_detection_msg));
BT_MESH_MODEL_PUB_DEFINE(fusion_pub, NULL, 2 + sizeof(struct sait01_fusion_response));
BT_MESH_MODEL_PUB_DEFINE(coord_pub, NULL, 2 + sizeof(struct sait01_coord_election));

/* Work queues for autonomous processing */
static struct k_work_delayable ml_processing_work;
static struct k_work_delayable coordinator_heartbeat_work;
static struct k_work_delayable lora_fallback_work;

/* TinyML system instance */
static sait01_tinyml_system_t tinyml_system = {0};

/* TinyML and audio processing state */
static struct {
    bool ml_active;
    uint8_t current_class;
    uint8_t current_confidence;
    uint32_t detection_count;
    uint32_t last_detection_time;
    int16_t audio_buffer[512]; // 32ms @ 16kHz
    uint8_t embedding[16];     // Compressed ML embedding
} ml_state;

/* Autonomous alert state */
static struct {
    uint32_t total_alerts;
    uint32_t last_alert_time;
    uint8_t alert_backoff_level;
    bool lora_fallback_active;
} alert_state;

/* =============================================================================
 * TINYML AUDIO PROCESSING (MOCK IMPLEMENTATION)
 * =============================================================================
 */

static void detection_callback(const sait01_ml_detection_t* detection, void* user_data)
{
    if (!detection) return;
    
    /* Update ML state with live inference results */
    ml_state.current_class = detection->detected_class;
    ml_state.current_confidence = (uint8_t)(detection->confidence * 100);
    ml_state.detection_count++;
    ml_state.last_detection_time = detection->timestamp;
    
    /* Copy embedding for mesh distribution */
    memcpy(ml_state.embedding, detection->embedding, sizeof(ml_state.embedding));
    
    LOG_INF("Live ML Detection: %s, confidence=%.1f%%, time=%d μs",
            sait01_class_to_string(detection->detected_class),
            detection->confidence * 100,
            detection->inference_time_us);
    
    /* Only process high-confidence detections */
    if (detection->confidence > 0.5f) {
        /* Create detection message for mesh network */
        struct sait01_detection_msg mesh_detection = {
            .timestamp = detection->timestamp,
            .sequence_id = ml_state.detection_count & 0xFFFF,
            .class_id = detection->detected_class,
            .confidence = (uint8_t)(detection->confidence * 100),
            .battery_level = 85, // TODO: Get actual battery level
            .rssi = -45, // TODO: Get actual RSSI from mesh
            .flags = 0,
            .location_hash = 0x12345678 // TODO: Calculate actual location hash
        };
        memcpy(mesh_detection.embedding, detection->embedding, 16);
        
        /* Check if this should generate an immediate alert */
        if (sait01_should_generate_alert(detection)) {
            LOG_WRN("HIGH-PRIORITY DETECTION: %s (%.1f%%)",
                    sait01_class_to_string(detection->detected_class),
                    detection->confidence * 100);
            
            mesh_detection.flags |= SAIT01_FLAG_HIGH_PRIORITY;
            alert_state.total_alerts++;
            alert_state.last_alert_time = k_uptime_get_32();
        }
        
        /* Announce detection to distributed mesh */
        int err = sait01_detection_announce(&detection_srv, &mesh_detection);
        if (err) {
            LOG_ERR("Failed to announce detection to mesh: %d", err);
        }
    }
}

static void process_audio_ml(struct k_work *work)
{
    /* Live TinyML processing with actual audio classification */
    if (!ml_state.ml_active) {
        /* Initialize production TensorFlow Lite model */
        LOG_INF("Initializing production TensorFlow Lite model...");
        
        /* Use real TFLite model instead of mock */
        int ret = sait01_init_tflite_model(&tinyml_system);
        if (ret == 0) {
            ret = sait01_tinyml_start(&tinyml_system);
            if (ret == 0) {
                ml_state.ml_active = true;
                LOG_INF("Production TensorFlow Lite model initialized");
                LOG_INF("  Model: 3-class drone acoustics (41.1%% accuracy)");
                LOG_INF("  Size: 8.1 KB (nRF5340 compatible)");
                LOG_INF("  Inference: ~28ms (real-time capable)");
            } else {
                LOG_ERR("Failed to start TensorFlow Lite system: %d", ret);
            }
        } else {
            LOG_ERR("Failed to initialize TensorFlow Lite model: %d", ret);
            LOG_WRN("Falling back to mock model for testing");
            
            /* Fallback to mock model initialization */
            ret = sait01_init_mock_model(&tinyml_system);
            if (ret == 0) {
                ret = sait01_tinyml_start(&tinyml_system);
                if (ret == 0) {
                    ml_state.ml_active = true;
                    LOG_WRN("Using mock model fallback");
                }
            }
        }
    }
    
    if (!ml_state.ml_active) {
        /* Fallback to mock processing if TinyML fails */
        static uint32_t cycle = 0;
        cycle++;
        
        if (cycle % 100 == 0) {
            /* Generate mock detection */
            uint32_t pseudo_random = sys_rand32_get();
            uint8_t detected_class = (pseudo_random % 7) + 1;
            uint8_t confidence = 60 + (pseudo_random % 40);
            
            ml_state.current_class = detected_class;
            ml_state.current_confidence = confidence;
            ml_state.detection_count++;
            ml_state.last_detection_time = k_uptime_get_32();
            
            /* Generate mock embedding */
            for (int i = 0; i < 16; i++) {
                ml_state.embedding[i] = (int8_t)(sys_rand32_get() % 256 - 128);
            }
            
            LOG_DBG("Mock ML Detection: class=%d confidence=%d%%",
                    detected_class, confidence);
            
            /* Create detection message for mesh */
            struct sait01_detection_msg detection = {
                .timestamp = k_uptime_get_32(),
                .sequence_id = ml_state.detection_count & 0xFFFF,
                .class_id = detected_class,
                .confidence = confidence,
                .battery_level = 85,
                .rssi = -45,
                .flags = 0,
                .location_hash = 0x12345678
            };
            memcpy(detection.embedding, ml_state.embedding, 16);
            
            sait01_detection_announce(&detection_srv, &detection);
        }
    }
    
    /* Schedule next ML processing cycle */
    k_work_reschedule(&ml_processing_work, K_MSEC(500));
}

/* =============================================================================
 * COORDINATOR HEARTBEAT AND ELECTION LOGIC
 * =============================================================================
 */

static void coordinator_heartbeat_handler(struct k_work *work)
{
    if (coord_srv.is_coordinator) {
        /* Send heartbeat as coordinator */
        uint32_t timestamp = k_uptime_get_32();
        
        NET_BUF_SIMPLE_DEFINE(msg, sizeof(timestamp));
        net_buf_simple_add_le32(&msg, timestamp);
        
        int err = bt_mesh_model_publish(coord_srv.model, &msg);
        if (err) {
            LOG_ERR("Failed to send coordinator heartbeat: %d", err);
        } else {
            LOG_DBG("Sent coordinator heartbeat");
        }
    } else {
        /* Check if we need to trigger election */
        if (sait01_should_trigger_election(&coord_srv)) {
            LOG_INF("Triggering coordinator election");
            
            struct sait01_coord_election election = {
                .node_priority = sait01_calculate_node_priority(),
                .capabilities = NODE_CAPABILITIES,
                .neighbor_count = 5, // TODO: Get actual neighbor count
                .uptime_seconds = k_uptime_get_32() / 1000,
                .battery_level = 85, // TODO: Get actual battery level
                .compute_capacity = 8 // TODO: Calculate compute capacity
            };
            
            NET_BUF_SIMPLE_DEFINE(msg, sizeof(election));
            net_buf_simple_add_mem(&msg, &election, sizeof(election));
            
            int err = bt_mesh_model_publish(coord_srv.model, &msg);
            if (err) {
                LOG_ERR("Failed to send election message: %d", err);
            }
        }
    }
    
    /* Schedule next heartbeat check (every 10 seconds) */
    k_work_reschedule(&coordinator_heartbeat_work, K_SECONDS(10));
}

/* =============================================================================
 * LORA FALLBACK COMMUNICATION
 * =============================================================================
 */

static void lora_fallback_handler(struct k_work *work)
{
    if (alert_state.lora_fallback_active && alert_state.total_alerts > 0) {
        /* Mock LoRa transmission to cloud */
        LOG_WRN("LoRa fallback: Transmitting %d alerts to cloud", alert_state.total_alerts);
        
        /* TODO: Implement actual LoRa transmission */
        /* - Format alerts into LoRa packets */
        /* - Send via Type-1SJ LoRa module */
        /* - Handle acknowledgments and retries */
        
        alert_state.total_alerts = 0; // Reset after transmission
    }
    
    /* Schedule next LoRa check (every 30 seconds) */
    k_work_reschedule(&lora_fallback_work, K_SECONDS(30));
}

/* =============================================================================
 * MESH MODEL CALLBACKS
 * =============================================================================
 */

static void detection_announce_cb(struct sait01_detection_srv *srv,
                                 struct bt_mesh_msg_ctx *ctx,
                                 const struct sait01_detection_msg *detection)
{
    LOG_INF("Peer detection from 0x%04x: class=%d confidence=%d",
            ctx->addr, detection->class_id, detection->confidence);
    
    /* Autonomous correlation happens in the mesh model */
    /* Application can perform additional processing here */
}

static void fusion_request_cb(struct sait01_fusion_srv *srv,
                             struct bt_mesh_msg_ctx *ctx,
                             const struct sait01_fusion_request *req)
{
    LOG_INF("Fusion request from 0x%04x: class=%d min_nodes=%d",
            ctx->addr, req->detection_class, req->min_nodes);
    
    /* Fusion processing happens automatically in mesh model */
}

static void fusion_response_cb(struct sait01_fusion_srv *srv,
                              struct bt_mesh_msg_ctx *ctx,
                              const struct sait01_fusion_response *resp)
{
    LOG_INF("Fusion response from 0x%04x: consensus class=%d confidence=%d alert_level=%d",
            ctx->addr, resp->consensus_class, resp->consensus_confidence, resp->alert_level);
    
    /* Alert generation happens automatically in mesh model */
    /* Application can trigger additional actions here */
    if (resp->alert_level >= SAIT01_ALERT_HIGH) {
        alert_state.total_alerts++;
        alert_state.last_alert_time = k_uptime_get_32();
        alert_state.lora_fallback_active = true;
        
        /* Trigger immediate LoRa fallback for high-priority alerts */
        k_work_reschedule(&lora_fallback_work, K_NO_WAIT);
    }
}

static void coordinator_changed_cb(struct sait01_coord_srv *srv,
                                  uint16_t old_coordinator,
                                  uint16_t new_coordinator)
{
    LOG_INF("Coordinator changed: 0x%04x -> 0x%04x", old_coordinator, new_coordinator);
    
    if (new_coordinator == bt_mesh_model_elem(srv->model)->addr) {
        LOG_INF("This node is now the coordinator");
        srv->is_coordinator = true;
    }
}

/* Model callback structures */
static const struct sait01_detection_srv_cb detection_cb = {
    .detection_announce = detection_announce_cb,
};

static const struct sait01_fusion_srv_cb fusion_cb = {
    .fusion_request = fusion_request_cb,
    .fusion_response = fusion_response_cb,
};

static const struct sait01_coord_srv_cb coord_cb = {
    .coordinator_changed = coordinator_changed_cb,
};

/* =============================================================================
 * MESH MODEL COMPOSITION
 * =============================================================================
 */

static struct bt_mesh_model root_models[] = {
    BT_MESH_MODEL_CFG_SRV,
    BT_MESH_MODEL_HEALTH_SRV(&health_srv, &health_pub),
    SAIT01_DETECTION_SRV_INIT(&detection_srv, &detection_cb),
    SAIT01_FUSION_SRV_INIT(&fusion_srv, &fusion_cb),
    SAIT01_COORD_SRV_INIT(&coord_srv, &coord_cb),
};

static struct bt_mesh_elem elements[] = {
    BT_MESH_ELEM(0, root_models, BT_MESH_MODEL_NONE),
};

static const struct bt_mesh_comp comp = {
    .cid = SAIT01_COMPANY_ID,
    .elem = elements,
    .elem_count = ARRAY_SIZE(elements),
};

/* =============================================================================
 * BLUETOOTH MESH INITIALIZATION
 * =============================================================================
 */

static void bt_ready(int err)
{
    if (err) {
        LOG_ERR("Bluetooth init failed: %d", err);
        return;
    }
    
    LOG_INF("Bluetooth initialized");
    
    /* Initialize LEDs and buttons */
    err = dk_leds_init();
    if (err) {
        LOG_ERR("Failed to initialize LEDs: %d", err);
        return;
    }
    
    err = dk_buttons_init(NULL);
    if (err) {
        LOG_ERR("Failed to initialize buttons: %d", err);
        return;
    }
    
    /* Initialize mesh models */
    detection_srv.model = &root_models[2];
    detection_srv.cb = &detection_cb;
    detection_srv.sequence_id = 0;
    detection_srv.active_detection = false;
    
    fusion_srv.model = &root_models[3]; 
    fusion_srv.cb = &fusion_cb;
    fusion_srv.min_nodes = 2; // Minimum 2 nodes for consensus
    fusion_srv.correlation_window = 5000; // 5 second correlation window
    
    coord_srv.model = &root_models[4];
    coord_srv.cb = &coord_cb;
    coord_srv.is_coordinator = false;
    coord_srv.coordinator_addr = 0;
    coord_srv.last_heartbeat_time = 0;
    coord_srv.coordinator_failures = 0;
    
    /* Initialize Bluetooth mesh */
    err = bt_mesh_init(bt_mesh_dk_prov_init(), &comp);
    if (err) {
        LOG_ERR("Failed to initialize mesh: %d", err);
        return;
    }
    
    if (IS_ENABLED(CONFIG_SETTINGS)) {
        settings_load();
    }
    
    /* Enable provisioning */
    bt_mesh_prov_enable(BT_MESH_PROV_ADV | BT_MESH_PROV_GATT);
    
    LOG_INF("Mesh initialized - Node ready for autonomous operation");
    
    /* Start autonomous processing */
    k_work_init_delayable(&ml_processing_work, process_audio_ml);
    k_work_init_delayable(&coordinator_heartbeat_work, coordinator_heartbeat_handler);
    k_work_init_delayable(&lora_fallback_work, lora_fallback_handler);
    
    /* Schedule initial work */
    k_work_schedule(&ml_processing_work, K_SECONDS(2));
    k_work_schedule(&coordinator_heartbeat_work, K_SECONDS(5));
    k_work_schedule(&lora_fallback_work, K_SECONDS(10));
    
    /* Initialize ML state */
    ml_state.ml_active = true;
    ml_state.detection_count = 0;
    
    /* Initialize alert state */
    alert_state.total_alerts = 0;
    alert_state.alert_backoff_level = 0;
    alert_state.lora_fallback_active = false;
    
    LOG_INF("SAIT_01 Autonomous Sensor Node operational");
    LOG_INF("Capabilities: 0x%02x", NODE_CAPABILITIES);
}

/* =============================================================================
 * MAIN APPLICATION
 * =============================================================================
 */

int main(void)
{
    int err;
    
    LOG_INF("SAIT_01 Autonomous Sensor Node starting...");
    LOG_INF("Architecture: Distributed mesh, no gateway dependency");
    
    /* Initialize Bluetooth */
    err = bt_enable(bt_ready);
    if (err) {
        LOG_ERR("Failed to enable Bluetooth: %d", err);
        return err;
    }
    
    LOG_INF("Waiting for mesh provisioning...");
    
    return 0;
}

/* =============================================================================
 * CONFIGURATION AND BUILD INTEGRATION
 * =============================================================================
 */

/*
 * To integrate with nRF Connect SDK build system, add to CMakeLists.txt:
 *
 * target_sources(app PRIVATE
 *     sait01_autonomous_main.c
 *     sait01_distributed_mesh.c
 * )
 *
 * target_include_directories(app PRIVATE .)
 *
 * Required Kconfig options in prj.conf:
 * CONFIG_BT=y
 * CONFIG_BT_MESH=y
 * CONFIG_BT_MESH_PROXY=y
 * CONFIG_BT_MESH_PB_GATT=y
 * CONFIG_BT_MESH_PB_ADV=y
 * CONFIG_BT_MESH_FRIEND=y
 * CONFIG_BT_MESH_LOW_POWER=y
 * CONFIG_SETTINGS=y
 * CONFIG_DK_LIBRARY=y
 * CONFIG_LOG=y
 */