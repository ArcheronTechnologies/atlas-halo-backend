/*
 * SAIT_01 End-to-End Mesh Encryption
 * AES-256-GCM + ECDH Key Exchange
 * Defense-Grade Secure Mesh Communications
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/random/rand32.h>
#include <psa/crypto.h>
#include <string.h>

LOG_MODULE_REGISTER(mesh_encryption, LOG_LEVEL_INF);

#define SAIT01_MESH_KEY_SIZE           32    // AES-256 key size
#define SAIT01_MESH_NONCE_SIZE         12    // GCM nonce size  
#define SAIT01_MESH_TAG_SIZE           16    // GCM authentication tag size
#define SAIT01_MESH_MAX_PAYLOAD        256   // Maximum encrypted payload
#define SAIT01_MESH_ECDH_KEY_SIZE      32    // ECDH private key size
#define SAIT01_MESH_ECDH_PUBLIC_SIZE   64    // ECDH public key size
#define SAIT01_MESH_SESSION_TIMEOUT    3600  // Session timeout (1 hour)

/* Mesh encryption message types */
#define SAIT01_MSG_KEY_EXCHANGE        0x10
#define SAIT01_MSG_ENCRYPTED_DATA      0x11
#define SAIT01_MSG_KEY_RENEWAL         0x12

/* Encryption context for each peer */
struct sait01_mesh_peer {
    uint8_t peer_id[4];
    uint8_t session_key[SAIT01_MESH_KEY_SIZE];
    uint8_t our_private_key[SAIT01_MESH_ECDH_KEY_SIZE];
    uint8_t our_public_key[SAIT01_MESH_ECDH_PUBLIC_SIZE];
    uint8_t peer_public_key[SAIT01_MESH_ECDH_PUBLIC_SIZE];
    uint32_t last_activity;
    uint32_t messages_encrypted;
    uint32_t messages_decrypted;
    bool key_established;
    bool active;
};

/* Key exchange message structure */
struct sait01_key_exchange_msg {
    uint8_t message_type;
    uint8_t sender_id[4];
    uint8_t public_key[SAIT01_MESH_ECDH_PUBLIC_SIZE];
    uint32_t timestamp;
    uint8_t signature[64];  // ECDSA signature
} __packed;

/* Encrypted message structure */
struct sait01_encrypted_msg {
    uint8_t message_type;
    uint8_t sender_id[4];
    uint8_t recipient_id[4];
    uint8_t nonce[SAIT01_MESH_NONCE_SIZE];
    uint16_t encrypted_length;
    uint8_t encrypted_payload[SAIT01_MESH_MAX_PAYLOAD];
    uint8_t auth_tag[SAIT01_MESH_TAG_SIZE];
} __packed;

/* Global encryption state */
struct sait01_mesh_encryption_state {
    struct sait01_mesh_peer peers[16];  // Support up to 16 peers
    uint8_t our_node_id[4];
    psa_key_id_t master_key_id;
    psa_key_id_t signing_key_id;
    uint32_t total_encrypted;
    uint32_t total_decrypted;
    uint32_t encryption_errors;
    bool initialized;
};

static struct sait01_mesh_encryption_state enc_state;

/* Generate ECDH key pair */
static int generate_ecdh_keypair(uint8_t *private_key, uint8_t *public_key)
{
    psa_key_attributes_t key_attr = PSA_KEY_ATTRIBUTES_INIT;
    psa_key_id_t key_id;
    psa_status_t status;
    
    // Set up key attributes for ECDH
    psa_set_key_usage_flags(&key_attr, PSA_KEY_USAGE_DERIVE);
    psa_set_key_algorithm(&key_attr, PSA_ALG_ECDH);
    psa_set_key_type(&key_attr, PSA_KEY_TYPE_ECC_KEY_PAIR(PSA_ECC_FAMILY_SECP_R1));
    psa_set_key_bits(&key_attr, 256);
    psa_set_key_lifetime(&key_attr, PSA_KEY_LIFETIME_VOLATILE);
    
    // Generate key pair
    status = psa_generate_key(&key_attr, &key_id);
    if (status != PSA_SUCCESS) {
        LOG_ERR("❌ Failed to generate ECDH key pair: %d", status);
        return -1;
    }
    
    // Export private key
    size_t private_key_len;
    status = psa_export_key(key_id, private_key, SAIT01_MESH_ECDH_KEY_SIZE, &private_key_len);
    if (status != PSA_SUCCESS) {
        psa_destroy_key(key_id);
        LOG_ERR("❌ Failed to export private key: %d", status);
        return -1;
    }
    
    // Export public key
    size_t public_key_len;
    status = psa_export_public_key(key_id, public_key, SAIT01_MESH_ECDH_PUBLIC_SIZE, &public_key_len);
    if (status != PSA_SUCCESS) {
        psa_destroy_key(key_id);
        LOG_ERR("❌ Failed to export public key: %d", status);
        return -1;
    }
    
    psa_destroy_key(key_id);
    
    LOG_DBG("✅ Generated ECDH key pair");
    return 0;
}

/* Derive shared secret using ECDH */
static int derive_shared_secret(const uint8_t *our_private, const uint8_t *peer_public,
                              uint8_t *shared_secret)
{
    psa_key_attributes_t key_attr = PSA_KEY_ATTRIBUTES_INIT;
    psa_key_id_t private_key_id;
    psa_key_derivation_operation_t derivation = PSA_KEY_DERIVATION_OPERATION_INIT;
    psa_status_t status;
    
    // Import our private key
    psa_set_key_usage_flags(&key_attr, PSA_KEY_USAGE_DERIVE);
    psa_set_key_algorithm(&key_attr, PSA_ALG_ECDH);
    psa_set_key_type(&key_attr, PSA_KEY_TYPE_ECC_KEY_PAIR(PSA_ECC_FAMILY_SECP_R1));
    psa_set_key_bits(&key_attr, 256);
    psa_set_key_lifetime(&key_attr, PSA_KEY_LIFETIME_VOLATILE);
    
    status = psa_import_key(&key_attr, our_private, SAIT01_MESH_ECDH_KEY_SIZE, &private_key_id);
    if (status != PSA_SUCCESS) {
        LOG_ERR("❌ Failed to import private key: %d", status);
        return -1;
    }
    
    // Set up key derivation
    status = psa_key_derivation_setup(&derivation, PSA_ALG_HKDF(PSA_ALG_SHA_256));
    if (status != PSA_SUCCESS) {
        psa_destroy_key(private_key_id);
        LOG_ERR("❌ Failed to setup key derivation: %d", status);
        return -1;
    }
    
    // Perform ECDH
    status = psa_key_derivation_key_agreement(&derivation, PSA_KEY_DERIVATION_INPUT_SECRET,
                                            private_key_id, peer_public, SAIT01_MESH_ECDH_PUBLIC_SIZE);
    if (status != PSA_SUCCESS) {
        psa_key_derivation_abort(&derivation);
        psa_destroy_key(private_key_id);
        LOG_ERR("❌ Failed ECDH key agreement: %d", status);
        return -1;
    }
    
    // Add salt and info for HKDF
    const uint8_t salt[] = "SAIT01_MESH_SALT";
    const uint8_t info[] = "SAIT01_SESSION_KEY";
    
    status = psa_key_derivation_input_bytes(&derivation, PSA_KEY_DERIVATION_INPUT_SALT,
                                          salt, sizeof(salt) - 1);
    status |= psa_key_derivation_input_bytes(&derivation, PSA_KEY_DERIVATION_INPUT_INFO,
                                           info, sizeof(info) - 1);
    
    if (status != PSA_SUCCESS) {
        psa_key_derivation_abort(&derivation);
        psa_destroy_key(private_key_id);
        LOG_ERR("❌ Failed to add HKDF inputs: %d", status);
        return -1;
    }
    
    // Derive session key
    size_t key_len;
    status = psa_key_derivation_output_bytes(&derivation, shared_secret, 
                                           SAIT01_MESH_KEY_SIZE, &key_len);
    
    psa_key_derivation_abort(&derivation);
    psa_destroy_key(private_key_id);
    
    if (status == PSA_SUCCESS) {
        LOG_DBG("✅ Derived shared secret (%zu bytes)", key_len);
        return 0;
    } else {
        LOG_ERR("❌ Failed to derive shared secret: %d", status);
        return -1;
    }
}

/* Find peer by ID */
static struct sait01_mesh_peer *find_peer(const uint8_t *peer_id)
{
    for (int i = 0; i < ARRAY_SIZE(enc_state.peers); i++) {
        if (enc_state.peers[i].active && 
            memcmp(enc_state.peers[i].peer_id, peer_id, 4) == 0) {
            return &enc_state.peers[i];
        }
    }
    return NULL;
}

/* Add new peer */
static struct sait01_mesh_peer *add_peer(const uint8_t *peer_id)
{
    // Find free slot
    for (int i = 0; i < ARRAY_SIZE(enc_state.peers); i++) {
        if (!enc_state.peers[i].active) {
            memset(&enc_state.peers[i], 0, sizeof(struct sait01_mesh_peer));
            memcpy(enc_state.peers[i].peer_id, peer_id, 4);
            enc_state.peers[i].active = true;
            enc_state.peers[i].last_activity = k_uptime_get_32();
            
            // Generate our key pair for this peer
            if (generate_ecdh_keypair(enc_state.peers[i].our_private_key,
                                    enc_state.peers[i].our_public_key) != 0) {
                enc_state.peers[i].active = false;
                return NULL;
            }
            
            LOG_INF("👥 Added new peer: %02x%02x%02x%02x",
                    peer_id[0], peer_id[1], peer_id[2], peer_id[3]);
            return &enc_state.peers[i];
        }
    }
    
    LOG_WRN("⚠️  No free peer slots available");
    return NULL;
}

/* Initialize mesh encryption */
int sait01_mesh_encryption_init(const uint8_t *node_id)
{
    LOG_INF("🔐 Initializing SAIT_01 Mesh Encryption");
    
    memset(&enc_state, 0, sizeof(enc_state));
    memcpy(enc_state.our_node_id, node_id, 4);
    
    // Initialize PSA Crypto if not already done
    psa_status_t status = psa_crypto_init();
    if (status != PSA_SUCCESS && status != PSA_ERROR_ALREADY_EXISTS) {
        LOG_ERR("❌ PSA crypto initialization failed: %d", status);
        return -1;
    }
    
    enc_state.initialized = true;
    
    LOG_INF("✅ Mesh encryption initialized for node %02x%02x%02x%02x",
            node_id[0], node_id[1], node_id[2], node_id[3]);
    
    return 0;
}

/* Initiate key exchange with peer */
int sait01_mesh_initiate_key_exchange(const uint8_t *peer_id, uint8_t *exchange_msg, 
                                     size_t *msg_len)
{
    if (!enc_state.initialized) {
        LOG_ERR("❌ Mesh encryption not initialized");
        return -1;
    }
    
    struct sait01_mesh_peer *peer = find_peer(peer_id);
    if (!peer) {
        peer = add_peer(peer_id);
        if (!peer) {
            return -1;
        }
    }
    
    // Create key exchange message
    struct sait01_key_exchange_msg *msg = (struct sait01_key_exchange_msg *)exchange_msg;
    msg->message_type = SAIT01_MSG_KEY_EXCHANGE;
    memcpy(msg->sender_id, enc_state.our_node_id, 4);
    memcpy(msg->public_key, peer->our_public_key, SAIT01_MESH_ECDH_PUBLIC_SIZE);
    msg->timestamp = k_uptime_get_32();
    
    // TODO: Add ECDSA signature for authentication
    memset(msg->signature, 0, sizeof(msg->signature));
    
    *msg_len = sizeof(struct sait01_key_exchange_msg);
    
    LOG_INF("🤝 Initiating key exchange with %02x%02x%02x%02x",
            peer_id[0], peer_id[1], peer_id[2], peer_id[3]);
    
    return 0;
}

/* Process received key exchange */
int sait01_mesh_process_key_exchange(const uint8_t *exchange_msg, size_t msg_len,
                                    uint8_t *response_msg, size_t *response_len)
{
    if (!enc_state.initialized) {
        LOG_ERR("❌ Mesh encryption not initialized");
        return -1;
    }
    
    if (msg_len < sizeof(struct sait01_key_exchange_msg)) {
        LOG_ERR("❌ Invalid key exchange message size");
        return -1;
    }
    
    const struct sait01_key_exchange_msg *msg = 
        (const struct sait01_key_exchange_msg *)exchange_msg;
    
    if (msg->message_type != SAIT01_MSG_KEY_EXCHANGE) {
        LOG_ERR("❌ Invalid message type for key exchange");
        return -1;
    }
    
    // Don't process our own messages
    if (memcmp(msg->sender_id, enc_state.our_node_id, 4) == 0) {
        return -1;
    }
    
    struct sait01_mesh_peer *peer = find_peer(msg->sender_id);
    if (!peer) {
        peer = add_peer(msg->sender_id);
        if (!peer) {
            return -1;
        }
    }
    
    // Store peer's public key
    memcpy(peer->peer_public_key, msg->public_key, SAIT01_MESH_ECDH_PUBLIC_SIZE);
    
    // Derive shared session key
    if (derive_shared_secret(peer->our_private_key, peer->peer_public_key,
                           peer->session_key) != 0) {
        LOG_ERR("❌ Failed to derive session key");
        return -1;
    }
    
    peer->key_established = true;
    peer->last_activity = k_uptime_get_32();
    
    LOG_INF("🔑 Established session key with %02x%02x%02x%02x",
            msg->sender_id[0], msg->sender_id[1], 
            msg->sender_id[2], msg->sender_id[3]);
    
    // Create response message
    struct sait01_key_exchange_msg *response = (struct sait01_key_exchange_msg *)response_msg;
    response->message_type = SAIT01_MSG_KEY_EXCHANGE;
    memcpy(response->sender_id, enc_state.our_node_id, 4);
    memcpy(response->public_key, peer->our_public_key, SAIT01_MESH_ECDH_PUBLIC_SIZE);
    response->timestamp = k_uptime_get_32();
    memset(response->signature, 0, sizeof(response->signature));
    
    *response_len = sizeof(struct sait01_key_exchange_msg);
    
    return 0;
}

/* Encrypt message for peer */
int sait01_mesh_encrypt_message(const uint8_t *peer_id, const uint8_t *plaintext,
                               size_t plaintext_len, uint8_t *encrypted_msg,
                               size_t *encrypted_len)
{
    if (!enc_state.initialized) {
        LOG_ERR("❌ Mesh encryption not initialized");
        return -1;
    }
    
    if (plaintext_len > SAIT01_MESH_MAX_PAYLOAD) {
        LOG_ERR("❌ Message too large for encryption: %zu bytes", plaintext_len);
        return -1;
    }
    
    struct sait01_mesh_peer *peer = find_peer(peer_id);
    if (!peer || !peer->key_established) {
        LOG_ERR("❌ No established session key for peer");
        return -1;
    }
    
    struct sait01_encrypted_msg *msg = (struct sait01_encrypted_msg *)encrypted_msg;
    msg->message_type = SAIT01_MSG_ENCRYPTED_DATA;
    memcpy(msg->sender_id, enc_state.our_node_id, 4);
    memcpy(msg->recipient_id, peer_id, 4);
    
    // Generate random nonce
    psa_status_t status = psa_generate_random(msg->nonce, SAIT01_MESH_NONCE_SIZE);
    if (status != PSA_SUCCESS) {
        LOG_ERR("❌ Failed to generate nonce: %d", status);
        return -1;
    }
    
    // Import session key for encryption
    psa_key_attributes_t key_attr = PSA_KEY_ATTRIBUTES_INIT;
    psa_key_id_t key_id;
    
    psa_set_key_usage_flags(&key_attr, PSA_KEY_USAGE_ENCRYPT);
    psa_set_key_algorithm(&key_attr, PSA_ALG_GCM);
    psa_set_key_type(&key_attr, PSA_KEY_TYPE_AES);
    psa_set_key_bits(&key_attr, 256);
    psa_set_key_lifetime(&key_attr, PSA_KEY_LIFETIME_VOLATILE);
    
    status = psa_import_key(&key_attr, peer->session_key, SAIT01_MESH_KEY_SIZE, &key_id);
    if (status != PSA_SUCCESS) {
        LOG_ERR("❌ Failed to import session key: %d", status);
        return -1;
    }
    
    // Encrypt with AES-GCM
    size_t output_len;
    status = psa_aead_encrypt(key_id, PSA_ALG_GCM,
                             msg->nonce, SAIT01_MESH_NONCE_SIZE,
                             NULL, 0,  // No additional data
                             plaintext, plaintext_len,
                             msg->encrypted_payload, 
                             SAIT01_MESH_MAX_PAYLOAD + SAIT01_MESH_TAG_SIZE,
                             &output_len);
    
    psa_destroy_key(key_id);
    
    if (status != PSA_SUCCESS) {
        LOG_ERR("❌ Encryption failed: %d", status);
        enc_state.encryption_errors++;
        return -1;
    }
    
    // Split encrypted data and authentication tag
    msg->encrypted_length = output_len - SAIT01_MESH_TAG_SIZE;
    memcpy(msg->auth_tag, 
           msg->encrypted_payload + msg->encrypted_length, 
           SAIT01_MESH_TAG_SIZE);
    
    *encrypted_len = sizeof(struct sait01_encrypted_msg) - SAIT01_MESH_MAX_PAYLOAD + 
                    msg->encrypted_length;
    
    peer->messages_encrypted++;
    peer->last_activity = k_uptime_get_32();
    enc_state.total_encrypted++;
    
    LOG_DBG("🔒 Encrypted %zu bytes for peer %02x%02x%02x%02x",
            plaintext_len, peer_id[0], peer_id[1], peer_id[2], peer_id[3]);
    
    return 0;
}

/* Decrypt message from peer */
int sait01_mesh_decrypt_message(const uint8_t *encrypted_msg, size_t encrypted_len,
                               uint8_t *plaintext, size_t *plaintext_len)
{
    if (!enc_state.initialized) {
        LOG_ERR("❌ Mesh encryption not initialized");
        return -1;
    }
    
    if (encrypted_len < sizeof(struct sait01_encrypted_msg) - SAIT01_MESH_MAX_PAYLOAD) {
        LOG_ERR("❌ Invalid encrypted message size");
        return -1;
    }
    
    const struct sait01_encrypted_msg *msg = 
        (const struct sait01_encrypted_msg *)encrypted_msg;
    
    if (msg->message_type != SAIT01_MSG_ENCRYPTED_DATA) {
        LOG_ERR("❌ Invalid message type for decryption");
        return -1;
    }
    
    // Check if message is for us
    if (memcmp(msg->recipient_id, enc_state.our_node_id, 4) != 0) {
        LOG_DBG("📤 Message not for us, forwarding");
        return -2;  // Not for us, but not an error
    }
    
    struct sait01_mesh_peer *peer = find_peer(msg->sender_id);
    if (!peer || !peer->key_established) {
        LOG_ERR("❌ No established session key for sender");
        return -1;
    }
    
    // Import session key for decryption
    psa_key_attributes_t key_attr = PSA_KEY_ATTRIBUTES_INIT;
    psa_key_id_t key_id;
    psa_status_t status;
    
    psa_set_key_usage_flags(&key_attr, PSA_KEY_USAGE_DECRYPT);
    psa_set_key_algorithm(&key_attr, PSA_ALG_GCM);
    psa_set_key_type(&key_attr, PSA_KEY_TYPE_AES);
    psa_set_key_bits(&key_attr, 256);
    psa_set_key_lifetime(&key_attr, PSA_KEY_LIFETIME_VOLATILE);
    
    status = psa_import_key(&key_attr, peer->session_key, SAIT01_MESH_KEY_SIZE, &key_id);
    if (status != PSA_SUCCESS) {
        LOG_ERR("❌ Failed to import session key: %d", status);
        return -1;
    }
    
    // Reconstruct encrypted data with authentication tag
    uint8_t encrypted_with_tag[SAIT01_MESH_MAX_PAYLOAD + SAIT01_MESH_TAG_SIZE];
    memcpy(encrypted_with_tag, msg->encrypted_payload, msg->encrypted_length);
    memcpy(encrypted_with_tag + msg->encrypted_length, msg->auth_tag, SAIT01_MESH_TAG_SIZE);
    
    // Decrypt with AES-GCM
    size_t output_len;
    status = psa_aead_decrypt(key_id, PSA_ALG_GCM,
                             msg->nonce, SAIT01_MESH_NONCE_SIZE,
                             NULL, 0,  // No additional data
                             encrypted_with_tag, msg->encrypted_length + SAIT01_MESH_TAG_SIZE,
                             plaintext, SAIT01_MESH_MAX_PAYLOAD,
                             &output_len);
    
    psa_destroy_key(key_id);
    
    if (status != PSA_SUCCESS) {
        LOG_ERR("❌ Decryption failed: %d", status);
        enc_state.encryption_errors++;
        return -1;
    }
    
    *plaintext_len = output_len;
    
    peer->messages_decrypted++;
    peer->last_activity = k_uptime_get_32();
    enc_state.total_decrypted++;
    
    LOG_DBG("🔓 Decrypted %zu bytes from peer %02x%02x%02x%02x",
            output_len, msg->sender_id[0], msg->sender_id[1], 
            msg->sender_id[2], msg->sender_id[3]);
    
    return 0;
}

/* Clean up expired sessions */
void sait01_mesh_cleanup_sessions(void)
{
    uint32_t current_time = k_uptime_get_32();
    int cleaned = 0;
    
    for (int i = 0; i < ARRAY_SIZE(enc_state.peers); i++) {
        if (enc_state.peers[i].active &&
            (current_time - enc_state.peers[i].last_activity) > 
            (SAIT01_MESH_SESSION_TIMEOUT * 1000)) {
            
            LOG_INF("🧹 Cleaning up expired session for %02x%02x%02x%02x",
                    enc_state.peers[i].peer_id[0], enc_state.peers[i].peer_id[1],
                    enc_state.peers[i].peer_id[2], enc_state.peers[i].peer_id[3]);
            
            memset(&enc_state.peers[i], 0, sizeof(struct sait01_mesh_peer));
            cleaned++;
        }
    }
    
    if (cleaned > 0) {
        LOG_INF("🧹 Cleaned up %d expired sessions", cleaned);
    }
}

/* Get encryption statistics */
void sait01_mesh_get_encryption_stats(uint32_t *total_encrypted, uint32_t *total_decrypted,
                                     uint32_t *active_sessions, uint32_t *errors)
{
    if (total_encrypted) *total_encrypted = enc_state.total_encrypted;
    if (total_decrypted) *total_decrypted = enc_state.total_decrypted;
    if (errors) *errors = enc_state.encryption_errors;
    
    if (active_sessions) {
        uint32_t count = 0;
        for (int i = 0; i < ARRAY_SIZE(enc_state.peers); i++) {
            if (enc_state.peers[i].active && enc_state.peers[i].key_established) {
                count++;
            }
        }
        *active_sessions = count;
    }
}