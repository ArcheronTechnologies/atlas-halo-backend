/*
 * SAIT_01 Distributed Mesh Protocol
 * Custom BLE Mesh models for autonomous sensor network operation
 * 
 * Architecture: Peer-to-peer mesh with distributed fusion and autonomous alerts
 * - No gateway dependency
 * - On-device ML processing and correlation
 * - Dynamic coordinator election
 * - Direct cloud fallback via LoRa
 */

#ifndef SAIT01_DISTRIBUTED_MESH_H
#define SAIT01_DISTRIBUTED_MESH_H

#include <zephyr/bluetooth/mesh.h>
#include <zephyr/sys/util.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Custom Company ID for SAIT_01 vendor models */
#define SAIT01_COMPANY_ID           0xFFFF  // Placeholder - needs official assignment

/* SAIT_01 Custom Model IDs */
#define SAIT01_DETECTION_CLI_MODEL_ID    0x0001
#define SAIT01_DETECTION_SRV_MODEL_ID    0x0002
#define SAIT01_FUSION_CLI_MODEL_ID       0x0003  
#define SAIT01_FUSION_SRV_MODEL_ID       0x0004
#define SAIT01_COORD_CLI_MODEL_ID        0x0005
#define SAIT01_COORD_SRV_MODEL_ID        0x0006

/* Message Opcodes */
#define SAIT01_DETECTION_ANNOUNCE    BT_MESH_MODEL_OP_3(0x80, SAIT01_COMPANY_ID)
#define SAIT01_DETECTION_STATUS      BT_MESH_MODEL_OP_3(0x81, SAIT01_COMPANY_ID)
#define SAIT01_FUSION_REQUEST        BT_MESH_MODEL_OP_3(0x82, SAIT01_COMPANY_ID)
#define SAIT01_FUSION_RESPONSE       BT_MESH_MODEL_OP_3(0x83, SAIT01_COMPANY_ID)
#define SAIT01_COORD_ELECTION        BT_MESH_MODEL_OP_3(0x84, SAIT01_COMPANY_ID)
#define SAIT01_COORD_HEARTBEAT       BT_MESH_MODEL_OP_3(0x85, SAIT01_COMPANY_ID)
#define SAIT01_ALERT_BROADCAST       BT_MESH_MODEL_OP_3(0x86, SAIT01_COMPANY_ID)

/* Detection Classes - TinyML output classifications */
enum sait01_detection_class {
    SAIT01_CLASS_UNKNOWN = 0,
    SAIT01_CLASS_VEHICLE,
    SAIT01_CLASS_FOOTSTEPS,
    SAIT01_CLASS_VOICES,
    SAIT01_CLASS_AIRCRAFT,
    SAIT01_CLASS_MACHINERY,
    SAIT01_CLASS_GUNSHOT,
    SAIT01_CLASS_EXPLOSION,
    SAIT01_CLASS_MAX = 15  // 4-bit field
};

/* Alert Levels */
enum sait01_alert_level {
    SAIT01_ALERT_INFO = 0,
    SAIT01_ALERT_LOW = 1, 
    SAIT01_ALERT_MEDIUM = 2,
    SAIT01_ALERT_HIGH = 3,
    SAIT01_ALERT_CRITICAL = 4
};

/* Node Capabilities */
#define SAIT01_CAP_AUDIO_ML         BIT(0)
#define SAIT01_CAP_UWB_RANGING      BIT(1) 
#define SAIT01_CAP_RF_PROXY         BIT(2)
#define SAIT01_CAP_LORA_FALLBACK    BIT(3)
#define SAIT01_CAP_COORDINATOR      BIT(4)
#define SAIT01_CAP_EDGE_COMPUTE     BIT(5)

/* Detection Announcement Message */
struct sait01_detection_msg {
    uint32_t timestamp;         // Unix timestamp (ms)
    uint16_t sequence_id;       // Message sequence
    uint8_t class_id:4;         // Detection class (0-15)
    uint8_t confidence:4;       // Confidence scaled 0-15
    uint8_t battery_level;      // Battery percentage
    int8_t rssi;               // BLE RSSI
    uint8_t flags;             // Status flags (tamper, motion, etc.)
    uint8_t embedding[16];      // Compressed ML embedding (int8)
    uint32_t location_hash;     // Obfuscated location identifier
} __packed;

/* Fusion Request Message */
struct sait01_fusion_request {
    uint32_t correlation_window;  // Time window for correlation (ms)
    uint8_t min_nodes;           // Minimum nodes required for consensus
    uint8_t detection_class;     // Target detection class
    uint8_t confidence_thresh;   // Minimum confidence threshold
    uint32_t area_mask;          // Geographic area mask
} __packed;

/* Fusion Response Message */
struct sait01_fusion_response {
    uint32_t correlation_id;     // Request correlation ID
    uint8_t participating_nodes; // Number of correlated nodes
    uint8_t consensus_class;     // Agreed detection class
    uint8_t consensus_confidence; // Weighted confidence
    uint8_t alert_level;         // Recommended alert level
    uint32_t fused_location;     // Triangulated position
    uint16_t time_to_live;       // Alert TTL (seconds)
} __packed;

/* Coordinator Election Message */
struct sait01_coord_election {
    uint16_t node_priority;      // Node priority score
    uint8_t capabilities;        // Node capability flags
    uint8_t neighbor_count;      // Number of mesh neighbors
    uint32_t uptime_seconds;     // Node uptime
    uint8_t battery_level;       // Current battery level
    uint16_t compute_capacity;   // Available CPU/memory score
} __packed;

/* Alert Broadcast Message */
struct sait01_alert_broadcast {
    uint32_t alert_id;          // Unique alert identifier
    uint32_t timestamp;         // Alert generation time
    uint8_t alert_level;        // Alert severity level
    uint8_t detection_class;    // Primary detection class
    uint8_t source_nodes;       // Number of contributing nodes
    uint8_t confidence;         // Final fused confidence
    uint32_t location;          // Alert location (if available)
    uint16_t radius_meters;     // Approximate radius of event
    uint8_t metadata[8];        // Additional alert context
} __packed;

/* Model Instance Structure */
struct sait01_detection_srv {
    struct bt_mesh_model *model;
    uint16_t sequence_id;
    uint32_t last_detection_time;
    uint8_t current_class;
    uint8_t current_confidence;
    bool active_detection;
};

struct sait01_fusion_srv {
    struct bt_mesh_model *model;
    uint32_t correlation_window;
    uint8_t min_nodes;
    uint8_t active_correlations;
    uint32_t last_fusion_time;
};

struct sait01_coord_srv {
    struct bt_mesh_model *model;
    bool is_coordinator;
    uint16_t coordinator_addr;
    uint32_t last_election_time;
    uint32_t last_heartbeat_time;
    uint8_t coordinator_failures;
};

/* Detection Model Callbacks */
struct sait01_detection_srv_cb {
    void (*detection_announce)(struct sait01_detection_srv *srv,
                              struct bt_mesh_msg_ctx *ctx,
                              const struct sait01_detection_msg *detection);
                              
    void (*detection_status_request)(struct sait01_detection_srv *srv,
                                    struct bt_mesh_msg_ctx *ctx);
};

/* Fusion Model Callbacks */  
struct sait01_fusion_srv_cb {
    void (*fusion_request)(struct sait01_fusion_srv *srv,
                          struct bt_mesh_msg_ctx *ctx,
                          const struct sait01_fusion_request *req);
                          
    void (*fusion_response)(struct sait01_fusion_srv *srv,
                           struct bt_mesh_msg_ctx *ctx,
                           const struct sait01_fusion_response *resp);
};

/* Coordinator Model Callbacks */
struct sait01_coord_srv_cb {
    void (*election_announce)(struct sait01_coord_srv *srv,
                             struct bt_mesh_msg_ctx *ctx,
                             const struct sait01_coord_election *election);
                             
    void (*heartbeat_received)(struct sait01_coord_srv *srv,
                              struct bt_mesh_msg_ctx *ctx,
                              uint32_t timestamp);
                              
    void (*coordinator_changed)(struct sait01_coord_srv *srv,
                               uint16_t old_coordinator,
                               uint16_t new_coordinator);
};

/* Model Initialization Functions */
int sait01_detection_srv_init(struct sait01_detection_srv *srv,
                             const struct sait01_detection_srv_cb *cb);

int sait01_fusion_srv_init(struct sait01_fusion_srv *srv,
                          const struct sait01_fusion_srv_cb *cb);

int sait01_coord_srv_init(struct sait01_coord_srv *srv,
                         const struct sait01_coord_srv_cb *cb);

/* Message Publishing Functions */
int sait01_detection_announce(struct sait01_detection_srv *srv,
                             const struct sait01_detection_msg *detection);

int sait01_fusion_request_send(struct sait01_fusion_srv *srv,
                              uint16_t addr,
                              const struct sait01_fusion_request *req);

int sait01_alert_broadcast_send(struct bt_mesh_model *model,
                               const struct sait01_alert_broadcast *alert);

/* Utility Functions */
uint8_t sait01_calculate_node_priority(void);
bool sait01_should_trigger_election(struct sait01_coord_srv *srv);
uint8_t sait01_correlate_detections(const struct sait01_detection_msg *detections,
                                   size_t count,
                                   struct sait01_fusion_response *response);

/* Model Composition */
#define SAIT01_DETECTION_SRV_INIT(_srv, _cb)                     \
    BT_MESH_MODEL_VND_CB(SAIT01_COMPANY_ID,                      \
                         SAIT01_DETECTION_SRV_MODEL_ID,          \
                         _sait01_detection_srv_op,               \
                         &(_srv)->pub,                           \
                         _srv,                                   \
                         _cb)

#define SAIT01_FUSION_SRV_INIT(_srv, _cb)                        \
    BT_MESH_MODEL_VND_CB(SAIT01_COMPANY_ID,                      \
                         SAIT01_FUSION_SRV_MODEL_ID,             \
                         _sait01_fusion_srv_op,                  \
                         &(_srv)->pub,                           \
                         _srv,                                   \
                         _cb)

#define SAIT01_COORD_SRV_INIT(_srv, _cb)                         \
    BT_MESH_MODEL_VND_CB(SAIT01_COMPANY_ID,                      \
                         SAIT01_COORD_SRV_MODEL_ID,              \
                         _sait01_coord_srv_op,                   \
                         &(_srv)->pub,                           \
                         _srv,                                   \
                         _cb)

/* Internal operation arrays (defined in implementation) */
extern const struct bt_mesh_model_op _sait01_detection_srv_op[];
extern const struct bt_mesh_model_op _sait01_fusion_srv_op[];
extern const struct bt_mesh_model_op _sait01_coord_srv_op[];

#ifdef __cplusplus
}
#endif

#endif /* SAIT01_DISTRIBUTED_MESH_H */