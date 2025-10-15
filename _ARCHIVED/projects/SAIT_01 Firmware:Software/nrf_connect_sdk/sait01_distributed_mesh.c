/*
 * SAIT_01 Distributed Mesh Protocol Implementation
 * Peer-to-peer autonomous sensor network with distributed decision making
 */

#include "sait01_distributed_mesh.h"
#include <zephyr/bluetooth/mesh.h>
#include <zephyr/logging/log.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/byteorder.h>
#include <zephyr/random/random.h>

LOG_MODULE_REGISTER(sait01_mesh, CONFIG_LOG_DEFAULT_LEVEL);

/* Detection correlation storage */
#define MAX_RECENT_DETECTIONS 32
#define CORRELATION_TIMEOUT_MS 5000

struct detection_record {
    struct sait01_detection_msg detection;
    uint16_t source_addr;
    int64_t receive_time;
    bool used_in_fusion;
};

static struct detection_record recent_detections[MAX_RECENT_DETECTIONS];
static size_t detection_count = 0;
static K_MUTEX_DEFINE(detection_mutex);

/* Coordinator election state */
static struct {
    bool election_active;
    uint16_t candidate_addr;
    uint16_t candidate_priority;
    int64_t election_start_time;
    int64_t last_heartbeat_time;
} coordinator_state;

/* =============================================================================
 * DETECTION SERVER MODEL IMPLEMENTATION
 * =============================================================================
 */

static void handle_detection_announce(struct bt_mesh_model *model,
                                     struct bt_mesh_msg_ctx *ctx,
                                     struct net_buf_simple *buf)
{
    struct sait01_detection_srv *srv = model->user_data;
    struct sait01_detection_msg detection;
    
    if (buf->len != sizeof(detection)) {
        LOG_ERR("Invalid detection message length: %d", buf->len);
        return;
    }
    
    memcpy(&detection, buf->data, sizeof(detection));
    
    LOG_INF("Detection from 0x%04x: class=%d, confidence=%d, seq=%d",
            ctx->addr, detection.class_id, detection.confidence, detection.sequence_id);
    
    /* Store detection for correlation */
    k_mutex_lock(&detection_mutex, K_FOREVER);
    
    size_t idx = detection_count % MAX_RECENT_DETECTIONS;
    recent_detections[idx].detection = detection;
    recent_detections[idx].source_addr = ctx->addr;
    recent_detections[idx].receive_time = k_uptime_get();
    recent_detections[idx].used_in_fusion = false;
    detection_count++;
    
    k_mutex_unlock(&detection_mutex);
    
    /* Trigger autonomous correlation if we have enough recent detections */
    if (detection_count >= 2) {
        LOG_INF("Triggering autonomous fusion with %d detections", detection_count);
        /* Schedule fusion processing - would be implemented as work queue */
        /* k_work_submit(&srv->fusion_work); */
    }
    
    /* Callback to application */
    if (srv->cb && srv->cb->detection_announce) {
        srv->cb->detection_announce(srv, ctx, &detection);
    }
}

static void handle_detection_status_request(struct bt_mesh_model *model,
                                           struct bt_mesh_msg_ctx *ctx,
                                           struct net_buf_simple *buf)
{
    struct sait01_detection_srv *srv = model->user_data;
    
    /* Send current detection status */
    struct sait01_detection_msg status;
    memset(&status, 0, sizeof(status));
    status.timestamp = k_uptime_get_32();
    status.sequence_id = srv->sequence_id;
    status.class_id = srv->current_class;
    status.confidence = srv->current_confidence;
    status.battery_level = 85; // TODO: Get actual battery level
    status.flags = srv->active_detection ? BIT(0) : 0;
    
    NET_BUF_SIMPLE_DEFINE(msg, sizeof(status));
    net_buf_simple_add_mem(&msg, &status, sizeof(status));
    
    int err = bt_mesh_model_send(model, ctx, &msg, NULL, NULL);
    if (err) {
        LOG_ERR("Failed to send detection status: %d", err);
    }
    
    LOG_DBG("Sent detection status to 0x%04x", ctx->addr);
}

const struct bt_mesh_model_op _sait01_detection_srv_op[] = {
    { SAIT01_DETECTION_ANNOUNCE, sizeof(struct sait01_detection_msg), handle_detection_announce },
    { SAIT01_DETECTION_STATUS, 0, handle_detection_status_request },
    BT_MESH_MODEL_OP_END,
};

/* =============================================================================
 * FUSION SERVER MODEL IMPLEMENTATION  
 * =============================================================================
 */

static void handle_fusion_request(struct bt_mesh_model *model,
                                 struct bt_mesh_msg_ctx *ctx,
                                 struct net_buf_simple *buf)
{
    struct sait01_fusion_srv *srv = model->user_data;
    struct sait01_fusion_request req;
    
    if (buf->len != sizeof(req)) {
        LOG_ERR("Invalid fusion request length: %d", buf->len);
        return;
    }
    
    memcpy(&req, buf->data, sizeof(req));
    
    LOG_INF("Fusion request from 0x%04x: class=%d, min_nodes=%d",
            ctx->addr, req.detection_class, req.min_nodes);
    
    /* Process fusion request with recent detections */
    struct sait01_fusion_response resp;
    memset(&resp, 0, sizeof(resp));
    
    uint8_t correlated = sait01_correlate_detections(NULL, 0, &resp);
    
    if (correlated >= req.min_nodes) {
        /* Send positive fusion response */
        resp.correlation_id = req.correlation_window; // Use as correlation ID
        resp.participating_nodes = correlated;
        resp.consensus_class = req.detection_class;
        resp.consensus_confidence = 12; // TODO: Calculate weighted confidence
        resp.alert_level = SAIT01_ALERT_MEDIUM;
        resp.time_to_live = 300; // 5 minutes
        
        NET_BUF_SIMPLE_DEFINE(msg, sizeof(resp));
        net_buf_simple_add_mem(&msg, &resp, sizeof(resp));
        
        int err = bt_mesh_model_send(model, ctx, &msg, NULL, NULL);
        if (err) {
            LOG_ERR("Failed to send fusion response: %d", err);
        }
        
        LOG_INF("Sent fusion response: consensus=%d nodes=%d confidence=%d",
                resp.consensus_class, resp.participating_nodes, resp.consensus_confidence);
    }
    
    /* Callback to application */
    if (srv->cb && srv->cb->fusion_request) {
        srv->cb->fusion_request(srv, ctx, &req);
    }
}

static void handle_fusion_response(struct bt_mesh_model *model,
                                  struct bt_mesh_msg_ctx *ctx,
                                  struct net_buf_simple *buf)
{
    struct sait01_fusion_srv *srv = model->user_data;
    struct sait01_fusion_response resp;
    
    if (buf->len != sizeof(resp)) {
        LOG_ERR("Invalid fusion response length: %d", buf->len);
        return;
    }
    
    memcpy(&resp, buf->data, sizeof(resp));
    
    LOG_INF("Fusion response from 0x%04x: class=%d, confidence=%d, alert_level=%d",
            ctx->addr, resp.consensus_class, resp.consensus_confidence, resp.alert_level);
    
    /* Check if this response warrants autonomous alert generation */
    if (resp.participating_nodes >= srv->min_nodes && 
        resp.consensus_confidence >= 10 &&
        resp.alert_level >= SAIT01_ALERT_MEDIUM) {
        
        /* Generate autonomous alert */
        struct sait01_alert_broadcast alert;
        memset(&alert, 0, sizeof(alert));
        
        alert.alert_id = sys_rand32_get();
        alert.timestamp = k_uptime_get_32();
        alert.alert_level = resp.alert_level;
        alert.detection_class = resp.consensus_class;
        alert.source_nodes = resp.participating_nodes;
        alert.confidence = resp.consensus_confidence;
        alert.radius_meters = 50; // Default 50m radius
        
        /* Broadcast alert to mesh network */
        sait01_alert_broadcast_send(model, &alert);
        
        LOG_WRN("AUTONOMOUS ALERT GENERATED: class=%d level=%d nodes=%d",
                alert.detection_class, alert.alert_level, alert.source_nodes);
    }
    
    /* Callback to application */
    if (srv->cb && srv->cb->fusion_response) {
        srv->cb->fusion_response(srv, ctx, &resp);
    }
}

const struct bt_mesh_model_op _sait01_fusion_srv_op[] = {
    { SAIT01_FUSION_REQUEST, sizeof(struct sait01_fusion_request), handle_fusion_request },
    { SAIT01_FUSION_RESPONSE, sizeof(struct sait01_fusion_response), handle_fusion_response },
    BT_MESH_MODEL_OP_END,
};

/* =============================================================================
 * COORDINATOR SERVER MODEL IMPLEMENTATION
 * =============================================================================
 */

static void handle_coord_election(struct bt_mesh_model *model,
                                 struct bt_mesh_msg_ctx *ctx,
                                 struct net_buf_simple *buf)
{
    struct sait01_coord_srv *srv = model->user_data;
    struct sait01_coord_election election;
    
    if (buf->len != sizeof(election)) {
        LOG_ERR("Invalid election message length: %d", buf->len);
        return;
    }
    
    memcpy(&election, buf->data, sizeof(election));
    
    LOG_INF("Election from 0x%04x: priority=%d, caps=0x%02x, neighbors=%d",
            ctx->addr, election.node_priority, election.capabilities, election.neighbor_count);
    
    /* Compare with our own priority */
    uint8_t our_priority = sait01_calculate_node_priority();
    
    if (election.node_priority > our_priority) {
        /* Accept this node as coordinator */
        srv->coordinator_addr = ctx->addr;
        srv->is_coordinator = false;
        coordinator_state.election_active = false;
        
        LOG_INF("Accepted 0x%04x as coordinator (priority %d > %d)",
                ctx->addr, election.node_priority, our_priority);
        
        /* Callback for coordinator change */
        if (srv->cb && srv->cb->coordinator_changed) {
            srv->cb->coordinator_changed(srv, 0, ctx->addr);
        }
    } else if (election.node_priority < our_priority) {
        /* Send our own election message */
        struct sait01_coord_election our_election;
        memset(&our_election, 0, sizeof(our_election));
        
        our_election.node_priority = our_priority;
        our_election.capabilities = SAIT01_CAP_AUDIO_ML | SAIT01_CAP_COORDINATOR;
        our_election.neighbor_count = 5; // TODO: Get actual neighbor count
        our_election.uptime_seconds = k_uptime_get_32() / 1000;
        our_election.battery_level = 85; // TODO: Get actual battery level
        our_election.compute_capacity = 8; // TODO: Calculate compute capacity
        
        NET_BUF_SIMPLE_DEFINE(msg, sizeof(our_election));
        net_buf_simple_add_mem(&msg, &our_election, sizeof(our_election));
        
        int err = bt_mesh_model_publish(model, &msg);
        if (err) {
            LOG_ERR("Failed to publish counter-election: %d", err);
        }
        
        LOG_INF("Sent counter-election: priority=%d", our_priority);
    }
    
    /* Callback to application */
    if (srv->cb && srv->cb->election_announce) {
        srv->cb->election_announce(srv, ctx, &election);
    }
}

static void handle_coord_heartbeat(struct bt_mesh_model *model,
                                  struct bt_mesh_msg_ctx *ctx,
                                  struct net_buf_simple *buf)
{
    struct sait01_coord_srv *srv = model->user_data;
    uint32_t timestamp;
    
    if (buf->len != sizeof(timestamp)) {
        LOG_ERR("Invalid heartbeat length: %d", buf->len);
        return;
    }
    
    timestamp = net_buf_simple_pull_le32(buf);
    
    if (ctx->addr == srv->coordinator_addr) {
        srv->last_heartbeat_time = k_uptime_get();
        srv->coordinator_failures = 0;
        
        LOG_DBG("Heartbeat from coordinator 0x%04x", ctx->addr);
    }
    
    /* Callback to application */
    if (srv->cb && srv->cb->heartbeat_received) {
        srv->cb->heartbeat_received(srv, ctx, timestamp);
    }
}

static void handle_alert_broadcast(struct bt_mesh_model *model,
                                  struct bt_mesh_msg_ctx *ctx,
                                  struct net_buf_simple *buf)
{
    struct sait01_alert_broadcast alert;
    
    if (buf->len != sizeof(alert)) {
        LOG_ERR("Invalid alert broadcast length: %d", buf->len);
        return;
    }
    
    memcpy(&alert, buf->data, sizeof(alert));
    
    LOG_WRN("ALERT RECEIVED: ID=0x%08x class=%d level=%d confidence=%d nodes=%d",
            alert.alert_id, alert.detection_class, alert.alert_level, 
            alert.confidence, alert.source_nodes);
    
    /* TODO: Forward to LoRa fallback if configured */
    /* TODO: Store in local alert log */
    /* TODO: Trigger local response actions */
}

const struct bt_mesh_model_op _sait01_coord_srv_op[] = {
    { SAIT01_COORD_ELECTION, sizeof(struct sait01_coord_election), handle_coord_election },
    { SAIT01_COORD_HEARTBEAT, sizeof(uint32_t), handle_coord_heartbeat },
    { SAIT01_ALERT_BROADCAST, sizeof(struct sait01_alert_broadcast), handle_alert_broadcast },
    BT_MESH_MODEL_OP_END,
};

/* =============================================================================
 * PUBLIC API IMPLEMENTATION
 * =============================================================================
 */

int sait01_detection_announce(struct sait01_detection_srv *srv,
                             const struct sait01_detection_msg *detection)
{
    if (!srv || !detection) {
        return -EINVAL;
    }
    
    NET_BUF_SIMPLE_DEFINE(msg, sizeof(*detection));
    net_buf_simple_add_mem(&msg, detection, sizeof(*detection));
    
    int err = bt_mesh_model_publish(srv->model, &msg);
    if (err) {
        LOG_ERR("Failed to publish detection: %d", err);
        return err;
    }
    
    srv->sequence_id++;
    srv->last_detection_time = k_uptime_get_32();
    srv->current_class = detection->class_id;
    srv->current_confidence = detection->confidence;
    srv->active_detection = true;
    
    LOG_INF("Published detection: class=%d confidence=%d seq=%d",
            detection->class_id, detection->confidence, srv->sequence_id);
    
    return 0;
}

int sait01_fusion_request_send(struct sait01_fusion_srv *srv,
                              uint16_t addr,
                              const struct sait01_fusion_request *req)
{
    if (!srv || !req) {
        return -EINVAL;
    }
    
    NET_BUF_SIMPLE_DEFINE(msg, sizeof(*req));
    net_buf_simple_add_mem(&msg, req, sizeof(*req));
    
    struct bt_mesh_msg_ctx ctx = {
        .addr = addr,
        .app_idx = srv->model->keys[0],
        .net_idx = srv->model->keys[0], 
    };
    
    int err = bt_mesh_model_send(srv->model, &ctx, &msg, NULL, NULL);
    if (err) {
        LOG_ERR("Failed to send fusion request: %d", err);
        return err;
    }
    
    LOG_INF("Sent fusion request to 0x%04x: class=%d min_nodes=%d",
            addr, req->detection_class, req->min_nodes);
    
    return 0;
}

int sait01_alert_broadcast_send(struct bt_mesh_model *model,
                               const struct sait01_alert_broadcast *alert)
{
    if (!model || !alert) {
        return -EINVAL;
    }
    
    NET_BUF_SIMPLE_DEFINE(msg, sizeof(*alert));
    net_buf_simple_add_mem(&msg, alert, sizeof(*alert));
    
    int err = bt_mesh_model_publish(model, &msg);
    if (err) {
        LOG_ERR("Failed to broadcast alert: %d", err);
        return err;
    }
    
    LOG_WRN("Broadcasted alert: ID=0x%08x level=%d class=%d",
            alert->alert_id, alert->alert_level, alert->detection_class);
    
    return 0;
}

uint8_t sait01_calculate_node_priority(void)
{
    /* Calculate node priority based on multiple factors */
    uint8_t priority = 0;
    
    /* Base priority from capabilities */
    priority += 20; // Base audio ML capability
    
    /* Battery level contribution (0-30 points) */
    uint8_t battery = 85; // TODO: Get actual battery level
    priority += (battery * 30) / 100;
    
    /* Uptime contribution (0-20 points) */
    uint32_t uptime = k_uptime_get_32() / 1000;
    if (uptime > 3600) priority += 20; // >1 hour uptime
    else if (uptime > 1800) priority += 15; // >30 min uptime
    else if (uptime > 600) priority += 10; // >10 min uptime
    else priority += 5;
    
    /* Neighbor count contribution (0-15 points) */
    uint8_t neighbors = 5; // TODO: Get actual neighbor count
    priority += MIN(neighbors * 3, 15);
    
    /* Random tie-breaker (0-15 points) */
    priority += sys_rand32_get() % 16;
    
    return priority;
}

uint8_t sait01_correlate_detections(const struct sait01_detection_msg *detections,
                                   size_t count,
                                   struct sait01_fusion_response *response)
{
    if (!response) {
        return 0;
    }
    
    k_mutex_lock(&detection_mutex, K_FOREVER);
    
    /* Find recent detections within correlation window */
    int64_t now = k_uptime_get();
    uint8_t correlated_count = 0;
    uint8_t class_votes[16] = {0}; // Vote count per class
    uint16_t total_confidence = 0;
    
    for (size_t i = 0; i < MIN(detection_count, MAX_RECENT_DETECTIONS); i++) {
        size_t idx = (detection_count - 1 - i) % MAX_RECENT_DETECTIONS;
        struct detection_record *rec = &recent_detections[idx];
        
        /* Check if within correlation window */
        if (now - rec->receive_time > CORRELATION_TIMEOUT_MS) {
            break; // Older detections are too old
        }
        
        if (!rec->used_in_fusion) {
            class_votes[rec->detection.class_id]++;
            total_confidence += rec->detection.confidence;
            correlated_count++;
            rec->used_in_fusion = true;
        }
    }
    
    k_mutex_unlock(&detection_mutex);
    
    if (correlated_count == 0) {
        return 0;
    }
    
    /* Find consensus class (most votes) */
    uint8_t consensus_class = 0;
    uint8_t max_votes = 0;
    for (int i = 0; i < 16; i++) {
        if (class_votes[i] > max_votes) {
            max_votes = class_votes[i];
            consensus_class = i;
        }
    }
    
    /* Fill response */
    response->participating_nodes = correlated_count;
    response->consensus_class = consensus_class;
    response->consensus_confidence = MIN(15, total_confidence / correlated_count);
    
    /* Determine alert level based on consensus */
    if (max_votes >= 3 && response->consensus_confidence >= 12) {
        response->alert_level = SAIT01_ALERT_HIGH;
    } else if (max_votes >= 2 && response->consensus_confidence >= 8) {
        response->alert_level = SAIT01_ALERT_MEDIUM;
    } else {
        response->alert_level = SAIT01_ALERT_LOW;
    }
    
    response->time_to_live = 300; // 5 minutes default TTL
    
    LOG_INF("Correlated %d detections: class=%d confidence=%d alert_level=%d",
            correlated_count, consensus_class, response->consensus_confidence, 
            response->alert_level);
    
    return correlated_count;
}

bool sait01_should_trigger_election(struct sait01_coord_srv *srv)
{
    if (!srv) {
        return false;
    }
    
    /* Trigger election if:
     * 1. No coordinator assigned
     * 2. Coordinator hasn't sent heartbeat in >30 seconds
     * 3. Multiple coordinator failures detected
     */
    
    if (srv->coordinator_addr == 0) {
        return true; // No coordinator
    }
    
    int64_t now = k_uptime_get();
    if (now - srv->last_heartbeat_time > 30000) { // 30 seconds
        srv->coordinator_failures++;
        if (srv->coordinator_failures >= 3) {
            LOG_WRN("Coordinator 0x%04x failed - triggering election", srv->coordinator_addr);
            return true;
        }
    }
    
    return false;
}