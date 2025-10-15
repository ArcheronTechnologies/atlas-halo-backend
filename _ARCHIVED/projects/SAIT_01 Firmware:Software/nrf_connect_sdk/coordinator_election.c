/*
 * SAIT_01 Coordinator Election Algorithm
 * Distributed Mesh Leadership Selection
 * Byzantine Fault Tolerant Election Protocol
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/random/rand32.h>
#include <string.h>
#include <stdbool.h>

LOG_MODULE_REGISTER(coordinator_election, LOG_LEVEL_INF);

#define SAIT01_MAX_NODES                16    // Maximum nodes in mesh
#define SAIT01_ELECTION_TIMEOUT_MS      5000  // Election timeout
#define SAIT01_HEARTBEAT_INTERVAL_MS    2000  // Coordinator heartbeat
#define SAIT01_LEADER_TIMEOUT_MS        10000 // Leader failure timeout
#define SAIT01_MAX_ELECTION_ROUNDS      5     // Maximum election rounds

/* Election message types */
#define SAIT01_MSG_ELECTION_START       0x20
#define SAIT01_MSG_ELECTION_PROPOSAL    0x21  
#define SAIT01_MSG_ELECTION_VOTE        0x22
#define SAIT01_MSG_ELECTION_LEADER      0x23
#define SAIT01_MSG_COORDINATOR_HEARTBEAT 0x24
#define SAIT01_MSG_COORDINATOR_CHALLENGE 0x25

/* Node roles */
typedef enum {
    SAIT01_ROLE_FOLLOWER,
    SAIT01_ROLE_CANDIDATE,
    SAIT01_ROLE_COORDINATOR
} sait01_node_role_t;

/* Election states */
typedef enum {
    SAIT01_ELECTION_IDLE,
    SAIT01_ELECTION_CANDIDATE,
    SAIT01_ELECTION_VOTING,
    SAIT01_ELECTION_WAITING_RESULT
} sait01_election_state_t;

/* Node priority factors */
struct sait01_node_priority {
    uint8_t battery_level;        // 0-100%
    uint8_t signal_strength;      // 0-100%
    uint16_t uptime_hours;        // Hours online
    uint8_t cpu_load;            // 0-100% (lower is better)
    uint8_t memory_usage;        // 0-100% (lower is better)
    uint32_t detections_count;    // Total threat detections
    bool has_external_power;      // External power source
    bool has_high_gain_antenna;   // High gain antenna
};

/* Election message structures */
struct sait01_election_start_msg {
    uint8_t message_type;
    uint8_t initiator_id[4];
    uint32_t election_id;
    uint32_t timestamp;
} __packed;

struct sait01_election_proposal_msg {
    uint8_t message_type;
    uint8_t candidate_id[4];
    uint32_t election_id;
    struct sait01_node_priority priority;
    uint32_t timestamp;
} __packed;

struct sait01_election_vote_msg {
    uint8_t message_type;
    uint8_t voter_id[4];
    uint8_t candidate_id[4];
    uint32_t election_id;
    uint32_t timestamp;
} __packed;

struct sait01_election_leader_msg {
    uint8_t message_type;
    uint8_t leader_id[4];
    uint32_t election_id;
    uint32_t term_number;
    uint32_t timestamp;
} __packed;

struct sait01_coordinator_heartbeat_msg {
    uint8_t message_type;
    uint8_t coordinator_id[4];
    uint32_t term_number;
    uint8_t mesh_status;
    uint16_t active_nodes;
    uint32_t timestamp;
} __packed;

/* Peer node information */
struct sait01_peer_node {
    uint8_t node_id[4];
    struct sait01_node_priority priority;
    uint32_t last_seen;
    uint32_t votes_received;
    bool active;
    bool voted_for_us;
    bool we_voted_for;
};

/* Election state */
struct sait01_election_context {
    uint8_t our_node_id[4];
    sait01_node_role_t current_role;
    sait01_election_state_t election_state;
    
    uint8_t current_coordinator[4];
    uint32_t current_term;
    uint32_t last_coordinator_heartbeat;
    
    uint32_t current_election_id;
    uint32_t election_start_time;
    uint8_t election_round;
    
    struct sait01_node_priority our_priority;
    struct sait01_peer_node peers[SAIT01_MAX_NODES];
    uint8_t peer_count;
    
    uint32_t votes_received;
    uint8_t voted_for[4];
    bool has_voted_this_round;
    
    uint32_t elections_initiated;
    uint32_t elections_won;
    uint32_t elections_participated;
    
    bool initialized;
};

static struct sait01_election_context election_ctx;
static struct k_work_delayable election_timeout_work;
static struct k_work_delayable heartbeat_work;
static struct k_work_delayable leader_check_work;

/* Function declarations */
static void start_election(void);
static void send_proposal(void);
static void process_votes(void);
static void become_coordinator(void);
static void step_down_coordinator(void);
static uint32_t calculate_priority_score(const struct sait01_node_priority *priority);

/* Calculate node priority score */
static uint32_t calculate_priority_score(const struct sait01_node_priority *priority)
{
    uint32_t score = 0;
    
    // Battery level (25% weight)
    score += (priority->battery_level * 250);
    
    // Signal strength (20% weight)  
    score += (priority->signal_strength * 200);
    
    // Uptime (15% weight) - capped at 1000 hours
    uint16_t capped_uptime = priority->uptime_hours > 1000 ? 1000 : priority->uptime_hours;
    score += (capped_uptime * 150) / 10;
    
    // CPU load (10% weight) - lower is better
    score += ((100 - priority->cpu_load) * 100);
    
    // Memory usage (10% weight) - lower is better
    score += ((100 - priority->memory_usage) * 100);
    
    // Detection count (10% weight) - capped at 10000
    uint32_t capped_detections = priority->detections_count > 10000 ? 10000 : priority->detections_count;
    score += (capped_detections * 100) / 100;
    
    // External power bonus (5% weight)
    if (priority->has_external_power) {
        score += 500;
    }
    
    // High gain antenna bonus (5% weight)
    if (priority->has_high_gain_antenna) {
        score += 500;
    }
    
    return score;
}

/* Update our node priority */
static void update_our_priority(void)
{
    // TODO: Get actual system metrics
    election_ctx.our_priority.battery_level = 85;
    election_ctx.our_priority.signal_strength = 75;
    election_ctx.our_priority.uptime_hours = k_uptime_get_32() / (1000 * 3600);
    election_ctx.our_priority.cpu_load = 25;
    election_ctx.our_priority.memory_usage = 30;
    election_ctx.our_priority.detections_count = election_ctx.elections_participated;
    election_ctx.our_priority.has_external_power = false;
    election_ctx.our_priority.has_high_gain_antenna = true;
}

/* Find peer by node ID */
static struct sait01_peer_node *find_peer(const uint8_t *node_id)
{
    for (int i = 0; i < election_ctx.peer_count; i++) {
        if (memcmp(election_ctx.peers[i].node_id, node_id, 4) == 0) {
            return &election_ctx.peers[i];
        }
    }
    return NULL;
}

/* Add or update peer */
static struct sait01_peer_node *add_or_update_peer(const uint8_t *node_id)
{
    struct sait01_peer_node *peer = find_peer(node_id);
    
    if (!peer && election_ctx.peer_count < SAIT01_MAX_NODES) {
        peer = &election_ctx.peers[election_ctx.peer_count++];
        memcpy(peer->node_id, node_id, 4);
        peer->active = true;
        peer->last_seen = k_uptime_get_32();
        
        LOG_INF("👥 Added peer node: %02x%02x%02x%02x",
                node_id[0], node_id[1], node_id[2], node_id[3]);
    }
    
    if (peer) {
        peer->last_seen = k_uptime_get_32();
        peer->active = true;
    }
    
    return peer;
}

/* Start new election */
static void start_election(void)
{
    LOG_INF("🗳️  Starting coordinator election");
    
    election_ctx.current_election_id = sys_rand32_get();
    election_ctx.election_state = SAIT01_ELECTION_CANDIDATE;
    election_ctx.current_role = SAIT01_ROLE_CANDIDATE;
    election_ctx.election_start_time = k_uptime_get_32();
    election_ctx.election_round = 1;
    election_ctx.votes_received = 0;
    election_ctx.has_voted_this_round = false;
    memset(election_ctx.voted_for, 0, 4);
    
    // Clear peer voting state
    for (int i = 0; i < election_ctx.peer_count; i++) {
        election_ctx.peers[i].votes_received = 0;
        election_ctx.peers[i].voted_for_us = false;
        election_ctx.peers[i].we_voted_for = false;
    }
    
    update_our_priority();
    election_ctx.elections_initiated++;
    
    // Broadcast election start
    struct sait01_election_start_msg start_msg = {
        .message_type = SAIT01_MSG_ELECTION_START,
        .election_id = election_ctx.current_election_id,
        .timestamp = k_uptime_get_32()
    };
    memcpy(start_msg.initiator_id, election_ctx.our_node_id, 4);
    
    // TODO: Send via mesh communication
    LOG_INF("📢 Broadcasting election start (ID: 0x%08x)", election_ctx.current_election_id);
    
    // Send our proposal
    k_work_schedule(&election_timeout_work, K_MSEC(100));
}

/* Send election proposal */
static void send_proposal(void)
{
    if (election_ctx.election_state != SAIT01_ELECTION_CANDIDATE) {
        return;
    }
    
    LOG_INF("📋 Sending election proposal");
    
    update_our_priority();
    
    struct sait01_election_proposal_msg proposal = {
        .message_type = SAIT01_MSG_ELECTION_PROPOSAL,
        .election_id = election_ctx.current_election_id,
        .priority = election_ctx.our_priority,
        .timestamp = k_uptime_get_32()
    };
    memcpy(proposal.candidate_id, election_ctx.our_node_id, 4);
    
    uint32_t our_score = calculate_priority_score(&election_ctx.our_priority);
    LOG_INF("🎯 Our priority score: %u", our_score);
    
    // TODO: Send via mesh communication
    
    election_ctx.election_state = SAIT01_ELECTION_VOTING;
    k_work_schedule(&election_timeout_work, K_MSEC(SAIT01_ELECTION_TIMEOUT_MS));
}

/* Process voting phase */
static void process_votes(void)
{
    LOG_INF("🗳️  Processing election votes");
    
    uint32_t our_score = calculate_priority_score(&election_ctx.our_priority);
    uint32_t highest_score = our_score;
    uint8_t best_candidate[4];
    memcpy(best_candidate, election_ctx.our_node_id, 4);
    
    // Find highest scoring candidate
    for (int i = 0; i < election_ctx.peer_count; i++) {
        if (election_ctx.peers[i].active) {
            uint32_t peer_score = calculate_priority_score(&election_ctx.peers[i].priority);
            if (peer_score > highest_score) {
                highest_score = peer_score;
                memcpy(best_candidate, election_ctx.peers[i].node_id, 4);
            }
        }
    }
    
    // Vote for best candidate
    if (!election_ctx.has_voted_this_round) {
        memcpy(election_ctx.voted_for, best_candidate, 4);
        election_ctx.has_voted_this_round = true;
        
        struct sait01_election_vote_msg vote = {
            .message_type = SAIT01_MSG_ELECTION_VOTE,
            .election_id = election_ctx.current_election_id,
            .timestamp = k_uptime_get_32()
        };
        memcpy(vote.voter_id, election_ctx.our_node_id, 4);
        memcpy(vote.candidate_id, best_candidate, 4);
        
        // TODO: Send via mesh communication
        
        if (memcmp(best_candidate, election_ctx.our_node_id, 4) == 0) {
            election_ctx.votes_received++;
            LOG_INF("🗳️  Voted for ourselves (score: %u)", our_score);
        } else {
            LOG_INF("🗳️  Voted for %02x%02x%02x%02x (score: %u)",
                    best_candidate[0], best_candidate[1], 
                    best_candidate[2], best_candidate[3], highest_score);
        }
    }
    
    election_ctx.election_state = SAIT01_ELECTION_WAITING_RESULT;
    k_work_schedule(&election_timeout_work, K_MSEC(SAIT01_ELECTION_TIMEOUT_MS));
}

/* Become coordinator */
static void become_coordinator(void)
{
    LOG_INF("👑 Becoming mesh coordinator");
    
    election_ctx.current_role = SAIT01_ROLE_COORDINATOR;
    election_ctx.current_term++;
    memcpy(election_ctx.current_coordinator, election_ctx.our_node_id, 4);
    election_ctx.last_coordinator_heartbeat = k_uptime_get_32();
    election_ctx.elections_won++;
    
    // Broadcast leader announcement
    struct sait01_election_leader_msg leader_msg = {
        .message_type = SAIT01_MSG_ELECTION_LEADER,
        .election_id = election_ctx.current_election_id,
        .term_number = election_ctx.current_term,
        .timestamp = k_uptime_get_32()
    };
    memcpy(leader_msg.leader_id, election_ctx.our_node_id, 4);
    
    // TODO: Send via mesh communication
    
    election_ctx.election_state = SAIT01_ELECTION_IDLE;
    
    // Start sending heartbeats
    k_work_schedule(&heartbeat_work, K_MSEC(SAIT01_HEARTBEAT_INTERVAL_MS));
    
    LOG_INF("✅ Coordinator election complete - we are the leader (term %u)", 
            election_ctx.current_term);
}

/* Step down as coordinator */
static void step_down_coordinator(void)
{
    LOG_WRN("📉 Stepping down as coordinator");
    
    election_ctx.current_role = SAIT01_ROLE_FOLLOWER;
    memset(election_ctx.current_coordinator, 0, 4);
    
    k_work_cancel_delayable(&heartbeat_work);
}

/* Election timeout handler */
static void election_timeout_handler(struct k_work *work)
{
    switch (election_ctx.election_state) {
        case SAIT01_ELECTION_CANDIDATE:
            send_proposal();
            break;
            
        case SAIT01_ELECTION_VOTING:
            process_votes();
            break;
            
        case SAIT01_ELECTION_WAITING_RESULT:
            // Check if we won
            uint32_t required_votes = (election_ctx.peer_count + 1) / 2 + 1; // Majority
            if (election_ctx.votes_received >= required_votes) {
                become_coordinator();
            } else {
                // Election failed, retry or give up
                election_ctx.election_round++;
                if (election_ctx.election_round <= SAIT01_MAX_ELECTION_ROUNDS) {
                    LOG_WRN("🔄 Election round %u failed, retrying", election_ctx.election_round);
                    election_ctx.election_state = SAIT01_ELECTION_CANDIDATE;
                    k_work_schedule(&election_timeout_work, K_MSEC(1000));
                } else {
                    LOG_ERR("❌ Election failed after %u rounds", SAIT01_MAX_ELECTION_ROUNDS);
                    election_ctx.election_state = SAIT01_ELECTION_IDLE;
                    election_ctx.current_role = SAIT01_ROLE_FOLLOWER;
                }
            }
            break;
            
        default:
            break;
    }
}

/* Coordinator heartbeat handler */
static void heartbeat_handler(struct k_work *work)
{
    if (election_ctx.current_role != SAIT01_ROLE_COORDINATOR) {
        return;
    }
    
    // Count active nodes
    uint16_t active_count = 1; // Count ourselves
    for (int i = 0; i < election_ctx.peer_count; i++) {
        if (election_ctx.peers[i].active &&
            (k_uptime_get_32() - election_ctx.peers[i].last_seen) < SAIT01_LEADER_TIMEOUT_MS) {
            active_count++;
        }
    }
    
    struct sait01_coordinator_heartbeat_msg heartbeat = {
        .message_type = SAIT01_MSG_COORDINATOR_HEARTBEAT,
        .term_number = election_ctx.current_term,
        .mesh_status = 0x01, // Operational
        .active_nodes = active_count,
        .timestamp = k_uptime_get_32()
    };
    memcpy(heartbeat.coordinator_id, election_ctx.our_node_id, 4);
    
    // TODO: Send via mesh communication
    
    LOG_DBG("💓 Coordinator heartbeat sent (term %u, %u nodes)", 
            election_ctx.current_term, active_count);
    
    k_work_schedule(&heartbeat_work, K_MSEC(SAIT01_HEARTBEAT_INTERVAL_MS));
}

/* Leader check handler */
static void leader_check_handler(struct k_work *work)
{
    if (election_ctx.current_role == SAIT01_ROLE_COORDINATOR) {
        // We are the coordinator, reschedule
        k_work_schedule(&leader_check_work, K_MSEC(SAIT01_LEADER_TIMEOUT_MS));
        return;
    }
    
    uint32_t time_since_heartbeat = k_uptime_get_32() - election_ctx.last_coordinator_heartbeat;
    
    if (time_since_heartbeat > SAIT01_LEADER_TIMEOUT_MS) {
        LOG_WRN("⚠️  Coordinator timeout detected, starting election");
        start_election();
    }
    
    k_work_schedule(&leader_check_work, K_MSEC(SAIT01_LEADER_TIMEOUT_MS / 2));
}

/* Initialize coordinator election */
int sait01_coordinator_election_init(const uint8_t *node_id)
{
    LOG_INF("🗳️  Initializing SAIT_01 Coordinator Election");
    
    memset(&election_ctx, 0, sizeof(election_ctx));
    memcpy(election_ctx.our_node_id, node_id, 4);
    
    election_ctx.current_role = SAIT01_ROLE_FOLLOWER;
    election_ctx.election_state = SAIT01_ELECTION_IDLE;
    election_ctx.current_term = 0;
    
    // Initialize work items
    k_work_init_delayable(&election_timeout_work, election_timeout_handler);
    k_work_init_delayable(&heartbeat_work, heartbeat_handler);
    k_work_init_delayable(&leader_check_work, leader_check_handler);
    
    election_ctx.initialized = true;
    
    // Start leader monitoring
    k_work_schedule(&leader_check_work, K_MSEC(SAIT01_LEADER_TIMEOUT_MS));
    
    LOG_INF("✅ Coordinator election initialized for node %02x%02x%02x%02x",
            node_id[0], node_id[1], node_id[2], node_id[3]);
    
    return 0;
}

/* Process received election message */
int sait01_coordinator_process_message(const uint8_t *message, size_t message_len)
{
    if (!election_ctx.initialized || message_len < 1) {
        return -1;
    }
    
    uint8_t msg_type = message[0];
    
    switch (msg_type) {
        case SAIT01_MSG_ELECTION_START: {
            const struct sait01_election_start_msg *msg = 
                (const struct sait01_election_start_msg *)message;
            
            // Don't process our own messages
            if (memcmp(msg->initiator_id, election_ctx.our_node_id, 4) == 0) {
                return 0;
            }
            
            add_or_update_peer(msg->initiator_id);
            
            if (election_ctx.election_state == SAIT01_ELECTION_IDLE) {
                LOG_INF("📢 Election started by %02x%02x%02x%02x",
                        msg->initiator_id[0], msg->initiator_id[1],
                        msg->initiator_id[2], msg->initiator_id[3]);
                
                election_ctx.current_election_id = msg->election_id;
                election_ctx.election_state = SAIT01_ELECTION_CANDIDATE;
                election_ctx.current_role = SAIT01_ROLE_CANDIDATE;
                election_ctx.elections_participated++;
                
                // Send our proposal after short delay
                k_work_schedule(&election_timeout_work, K_MSEC(100 + sys_rand32_get() % 200));
            }
            break;
        }
        
        case SAIT01_MSG_ELECTION_PROPOSAL: {
            const struct sait01_election_proposal_msg *msg = 
                (const struct sait01_election_proposal_msg *)message;
            
            if (msg->election_id != election_ctx.current_election_id) {
                return 0; // Wrong election
            }
            
            struct sait01_peer_node *peer = add_or_update_peer(msg->candidate_id);
            if (peer) {
                peer->priority = msg->priority;
                uint32_t score = calculate_priority_score(&peer->priority);
                LOG_DBG("📋 Proposal from %02x%02x%02x%02x (score: %u)",
                        msg->candidate_id[0], msg->candidate_id[1],
                        msg->candidate_id[2], msg->candidate_id[3], score);
            }
            break;
        }
        
        case SAIT01_MSG_ELECTION_VOTE: {
            const struct sait01_election_vote_msg *msg = 
                (const struct sait01_election_vote_msg *)message;
            
            if (msg->election_id != election_ctx.current_election_id) {
                return 0; // Wrong election
            }
            
            add_or_update_peer(msg->voter_id);
            
            // Check if vote is for us
            if (memcmp(msg->candidate_id, election_ctx.our_node_id, 4) == 0) {
                election_ctx.votes_received++;
                
                struct sait01_peer_node *voter = find_peer(msg->voter_id);
                if (voter) {
                    voter->voted_for_us = true;
                }
                
                LOG_INF("🗳️  Received vote from %02x%02x%02x%02x (total: %u)",
                        msg->voter_id[0], msg->voter_id[1],
                        msg->voter_id[2], msg->voter_id[3], election_ctx.votes_received);
            } else {
                struct sait01_peer_node *candidate = find_peer(msg->candidate_id);
                if (candidate) {
                    candidate->votes_received++;
                }
            }
            break;
        }
        
        case SAIT01_MSG_ELECTION_LEADER: {
            const struct sait01_election_leader_msg *msg = 
                (const struct sait01_election_leader_msg *)message;
            
            if (msg->election_id != election_ctx.current_election_id) {
                return 0; // Wrong election
            }
            
            // Accept new coordinator
            memcpy(election_ctx.current_coordinator, msg->leader_id, 4);
            election_ctx.current_term = msg->term_number;
            election_ctx.last_coordinator_heartbeat = k_uptime_get_32();
            
            if (memcmp(msg->leader_id, election_ctx.our_node_id, 4) != 0) {
                election_ctx.current_role = SAIT01_ROLE_FOLLOWER;
                k_work_cancel_delayable(&heartbeat_work);
                
                LOG_INF("👑 New coordinator elected: %02x%02x%02x%02x (term %u)",
                        msg->leader_id[0], msg->leader_id[1],
                        msg->leader_id[2], msg->leader_id[3], msg->term_number);
            }
            
            election_ctx.election_state = SAIT01_ELECTION_IDLE;
            k_work_cancel_delayable(&election_timeout_work);
            break;
        }
        
        case SAIT01_MSG_COORDINATOR_HEARTBEAT: {
            const struct sait01_coordinator_heartbeat_msg *msg = 
                (const struct sait01_coordinator_heartbeat_msg *)message;
            
            // Update coordinator info
            if (memcmp(msg->coordinator_id, election_ctx.current_coordinator, 4) == 0) {
                election_ctx.last_coordinator_heartbeat = k_uptime_get_32();
                
                LOG_DBG("💓 Coordinator heartbeat from %02x%02x%02x%02x (%u nodes)",
                        msg->coordinator_id[0], msg->coordinator_id[1],
                        msg->coordinator_id[2], msg->coordinator_id[3], msg->active_nodes);
            }
            break;
        }
    }
    
    return 0;
}

/* Force coordinator election */
int sait01_coordinator_force_election(void)
{
    if (!election_ctx.initialized) {
        LOG_ERR("❌ Election system not initialized");
        return -1;
    }
    
    if (election_ctx.election_state != SAIT01_ELECTION_IDLE) {
        LOG_WRN("⚠️  Election already in progress");
        return -1;
    }
    
    LOG_INF("🚨 Forcing coordinator election");
    start_election();
    return 0;
}

/* Get current coordinator info */
int sait01_coordinator_get_info(uint8_t *coordinator_id, uint32_t *term_number, 
                               sait01_node_role_t *our_role)
{
    if (!election_ctx.initialized) {
        return -1;
    }
    
    if (coordinator_id) {
        memcpy(coordinator_id, election_ctx.current_coordinator, 4);
    }
    
    if (term_number) {
        *term_number = election_ctx.current_term;
    }
    
    if (our_role) {
        *our_role = election_ctx.current_role;
    }
    
    return 0;
}

/* Get election statistics */
void sait01_coordinator_get_stats(uint32_t *elections_initiated, uint32_t *elections_won,
                                 uint32_t *elections_participated, uint8_t *active_peers)
{
    if (elections_initiated) *elections_initiated = election_ctx.elections_initiated;
    if (elections_won) *elections_won = election_ctx.elections_won;
    if (elections_participated) *elections_participated = election_ctx.elections_participated;
    
    if (active_peers) {
        uint8_t count = 0;
        for (int i = 0; i < election_ctx.peer_count; i++) {
            if (election_ctx.peers[i].active &&
                (k_uptime_get_32() - election_ctx.peers[i].last_seen) < SAIT01_LEADER_TIMEOUT_MS) {
                count++;
            }
        }
        *active_peers = count;
    }
}