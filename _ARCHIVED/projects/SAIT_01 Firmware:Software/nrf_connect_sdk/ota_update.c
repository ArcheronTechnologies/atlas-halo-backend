/*
 * SAIT_01 OTA Update System  
 * Secure Over-The-Air Firmware Updates
 * Defense-Grade Security with Rollback Protection
 */

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/storage/flash_map.h>
#include <zephyr/dfu/mcuboot.h>
#include <zephyr/sys/reboot.h>
#include <zephyr/fs/fs.h>
#include <zephyr/logging/log.h>
#include <psa/crypto.h>
#include <bootutil/bootutil.h>
#include <string.h>

LOG_MODULE_REGISTER(ota_update, LOG_LEVEL_INF);

#define SAIT01_OTA_MAX_CHUNK_SIZE      1024   // Maximum chunk size for transfers
#define SAIT01_OTA_VERSION_SIZE        16     // Version string size
#define SAIT01_OTA_SIGNATURE_SIZE      64     // ECDSA P-256 signature size
#define SAIT01_OTA_HASH_SIZE           32     // SHA-256 hash size
#define SAIT01_OTA_MAX_IMAGE_SIZE      (512 * 1024) // 512KB max firmware
#define SAIT01_OTA_MAGIC               0x5A170001    // SAIT01 OTA magic

/* OTA Update States */
typedef enum {
    SAIT01_OTA_IDLE,
    SAIT01_OTA_CHECKING_VERSION,
    SAIT01_OTA_DOWNLOADING,
    SAIT01_OTA_VERIFYING,
    SAIT01_OTA_INSTALLING,
    SAIT01_OTA_COMPLETE,
    SAIT01_OTA_ERROR
} sait01_ota_state_t;

/* OTA Error Codes */
typedef enum {
    SAIT01_OTA_ERROR_NONE = 0,
    SAIT01_OTA_ERROR_INVALID_VERSION,
    SAIT01_OTA_ERROR_INVALID_SIGNATURE,
    SAIT01_OTA_ERROR_INVALID_HASH,
    SAIT01_OTA_ERROR_FLASH_WRITE,
    SAIT01_OTA_ERROR_NETWORK,
    SAIT01_OTA_ERROR_INSUFFICIENT_SPACE,
    SAIT01_OTA_ERROR_ROLLBACK_DETECTED,
    SAIT01_OTA_ERROR_CORRUPTED_IMAGE
} sait01_ota_error_t;

/* OTA Message Types */
#define SAIT01_OTA_MSG_VERSION_CHECK   0x30
#define SAIT01_OTA_MSG_VERSION_RESP    0x31
#define SAIT01_OTA_MSG_UPDATE_START    0x32
#define SAIT01_OTA_MSG_UPDATE_CHUNK    0x33
#define SAIT01_OTA_MSG_UPDATE_COMPLETE 0x34
#define SAIT01_OTA_MSG_UPDATE_STATUS   0x35

/* Firmware Version Structure */
struct sait01_firmware_version {
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
    uint32_t build;
    char variant[8];           // e.g., "release", "debug"
} __packed;

/* OTA Update Header */
struct sait01_ota_header {
    uint32_t magic;            // SAIT01_OTA_MAGIC
    struct sait01_firmware_version version;
    uint32_t image_size;       // Total firmware image size
    uint32_t chunk_count;      // Number of chunks
    uint32_t chunk_size;       // Size of each chunk (except last)
    uint8_t image_hash[SAIT01_OTA_HASH_SIZE];      // SHA-256 of firmware
    uint8_t signature[SAIT01_OTA_SIGNATURE_SIZE];  // ECDSA signature
    uint32_t timestamp;        // Build timestamp
    uint32_t min_version_major; // Minimum compatible version
    uint32_t min_version_minor;
    uint32_t crc32;            // Header CRC32
} __packed;

/* OTA Chunk Message */
struct sait01_ota_chunk_msg {
    uint8_t message_type;
    uint32_t chunk_number;
    uint16_t chunk_size;
    uint8_t chunk_data[SAIT01_OTA_MAX_CHUNK_SIZE];
    uint32_t chunk_crc32;
} __packed;

/* OTA Status Message */
struct sait01_ota_status_msg {
    uint8_t message_type;
    uint8_t node_id[4];
    sait01_ota_state_t state;
    sait01_ota_error_t error;
    uint32_t chunks_received;
    uint32_t total_chunks;
    uint8_t progress_percent;
} __packed;

/* OTA Update Context */
struct sait01_ota_context {
    sait01_ota_state_t state;
    sait01_ota_error_t last_error;
    
    struct sait01_firmware_version current_version;
    struct sait01_ota_header update_header;
    
    const struct flash_area *update_area;
    size_t bytes_written;
    uint32_t chunks_received;
    uint32_t expected_chunks;
    
    uint8_t *chunk_buffer;
    bool *chunk_received_flags;
    
    psa_hash_operation_t hash_op;
    uint8_t calculated_hash[SAIT01_OTA_HASH_SIZE];
    
    uint32_t update_start_time;
    uint32_t total_updates_attempted;
    uint32_t successful_updates;
    uint32_t failed_updates;
    
    bool initialized;
    bool update_in_progress;
};

static struct sait01_ota_context ota_ctx;

/* Function Declarations */
static int verify_firmware_signature(const struct sait01_ota_header *header);
static int verify_rollback_protection(const struct sait01_firmware_version *new_version);
static int erase_update_partition(void);
static int write_chunk_to_flash(uint32_t chunk_num, const uint8_t *data, uint16_t size);
static int finalize_update(void);

/* Get current firmware version */
static void get_current_version(struct sait01_firmware_version *version)
{
    // In production, this would read from build-time constants or MCUBOOT
    version->major = 1;
    version->minor = 0;
    version->patch = 0;
    version->build = 1;
    strcpy(version->variant, "release");
}

/* Compare firmware versions */
static int compare_versions(const struct sait01_firmware_version *v1,
                          const struct sait01_firmware_version *v2)
{
    if (v1->major != v2->major) {
        return (v1->major > v2->major) ? 1 : -1;
    }
    if (v1->minor != v2->minor) {
        return (v1->minor > v2->minor) ? 1 : -1;
    }
    if (v1->patch != v2->patch) {
        return (v1->patch > v2->patch) ? 1 : -1;
    }
    if (v1->build != v2->build) {
        return (v1->build > v2->build) ? 1 : -1;
    }
    return 0;
}

/* Verify firmware signature */
static int verify_firmware_signature(const struct sait01_ota_header *header)
{
    psa_status_t status;
    psa_key_id_t public_key_id;
    psa_key_attributes_t key_attr = PSA_KEY_ATTRIBUTES_INIT;
    
    LOG_INF("Verifying firmware signature");
    
    // Import OTA public key (would be stored securely in production)
    static const uint8_t ota_public_key[64] = {
        // X coordinate (32 bytes)
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10,
        0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
        0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20,
        // Y coordinate (32 bytes)
        0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F, 0x30,
        0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38,
        0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F, 0x40
    };
    
    psa_set_key_usage_flags(&key_attr, PSA_KEY_USAGE_VERIFY_HASH);
    psa_set_key_algorithm(&key_attr, PSA_ALG_ECDSA(PSA_ALG_SHA_256));
    psa_set_key_type(&key_attr, PSA_KEY_TYPE_ECC_PUBLIC_KEY(PSA_ECC_FAMILY_SECP_R1));
    psa_set_key_bits(&key_attr, 256);
    psa_set_key_lifetime(&key_attr, PSA_KEY_LIFETIME_VOLATILE);
    
    status = psa_import_key(&key_attr, ota_public_key, sizeof(ota_public_key), &public_key_id);
    if (status != PSA_SUCCESS) {
        LOG_ERR("Failed to import OTA public key: %d", status);
        return -1;
    }
    
    // Create hash of header (excluding signature)
    uint8_t header_hash[SAIT01_OTA_HASH_SIZE];
    size_t header_size = sizeof(struct sait01_ota_header) - SAIT01_OTA_SIGNATURE_SIZE - 4; // Exclude signature and CRC
    
    status = psa_hash_compute(PSA_ALG_SHA_256,
                             (const uint8_t *)header, header_size,
                             header_hash, sizeof(header_hash),
                             &header_size);
    
    if (status != PSA_SUCCESS) {
        psa_destroy_key(public_key_id);
        LOG_ERR("Failed to compute header hash: %d", status);
        return -1;
    }
    
    // Verify signature
    status = psa_verify_hash(public_key_id,
                            PSA_ALG_ECDSA(PSA_ALG_SHA_256),
                            header_hash, sizeof(header_hash),
                            header->signature, SAIT01_OTA_SIGNATURE_SIZE);
    
    psa_destroy_key(public_key_id);
    
    if (status == PSA_SUCCESS) {
        LOG_INF("Firmware signature verified");
        return 0;
    } else {
        LOG_ERR("Firmware signature verification failed: %d", status);
        return -1;
    }
}

/* Verify rollback protection */
static int verify_rollback_protection(const struct sait01_firmware_version *new_version)
{
    // Prevent rollback to older firmware versions
    int comparison = compare_versions(new_version, &ota_ctx.current_version);
    
    if (comparison <= 0) {
        LOG_ERR("Rollback detected: new %u.%u.%u.%u <= current %u.%u.%u.%u",
                new_version->major, new_version->minor, new_version->patch, new_version->build,
                ota_ctx.current_version.major, ota_ctx.current_version.minor,
                ota_ctx.current_version.patch, ota_ctx.current_version.build);
        return -1;
    }
    
    LOG_INF("Version upgrade verified: %u.%u.%u.%u -> %u.%u.%u.%u",
            ota_ctx.current_version.major, ota_ctx.current_version.minor,
            ota_ctx.current_version.patch, ota_ctx.current_version.build,
            new_version->major, new_version->minor, new_version->patch, new_version->build);
    
    return 0;
}

/* Erase update partition */
static int erase_update_partition(void)
{
    LOG_INF("Erasing update partition");
    
    int ret = flash_area_erase(ota_ctx.update_area, 0, ota_ctx.update_area->fa_size);
    if (ret != 0) {
        LOG_ERR("Failed to erase update partition: %d", ret);
        return ret;
    }
    
    LOG_INF("Update partition erased");
    return 0;
}

/* Write chunk to flash */
static int write_chunk_to_flash(uint32_t chunk_num, const uint8_t *data, uint16_t size)
{
    size_t offset = chunk_num * SAIT01_OTA_MAX_CHUNK_SIZE;
    
    if (offset + size > ota_ctx.update_area->fa_size) {
        LOG_ERR("Chunk would exceed partition size");
        return -1;
    }
    
    int ret = flash_area_write(ota_ctx.update_area, offset, data, size);
    if (ret != 0) {
        LOG_ERR("Failed to write chunk %u: %d", chunk_num, ret);
        return ret;
    }
    
    // Update hash calculation
    psa_status_t status = psa_hash_update(&ota_ctx.hash_op, data, size);
    if (status != PSA_SUCCESS) {
        LOG_ERR("Failed to update hash: %d", status);
        return -1;
    }
    
    ota_ctx.bytes_written += size;
    
    LOG_DBG("Wrote chunk %u (%u bytes) to offset 0x%x", chunk_num, size, offset);
    return 0;
}

/* Finalize update */
static int finalize_update(void)
{
    LOG_INF("Finalizing OTA update");
    
    // Finalize hash calculation
    size_t hash_length;
    psa_status_t status = psa_hash_finish(&ota_ctx.hash_op,
                                         ota_ctx.calculated_hash,
                                         sizeof(ota_ctx.calculated_hash),
                                         &hash_length);
    
    if (status != PSA_SUCCESS) {
        LOG_ERR("Failed to finalize hash: %d", status);
        return -1;
    }
    
    // Verify image hash
    if (memcmp(ota_ctx.calculated_hash, ota_ctx.update_header.image_hash, SAIT01_OTA_HASH_SIZE) != 0) {
        LOG_ERR("Image hash mismatch");
        return -1;
    }
    
    LOG_INF("Image hash verified");
    
    // Mark image as ready for boot
    int ret = boot_request_upgrade(BOOT_UPGRADE_TEST);
    if (ret != 0) {
        LOG_ERR("Failed to request upgrade: %d", ret);
        return ret;
    }
    
    LOG_INF("OTA update ready - will apply on next reboot");
    return 0;
}

/* Initialize OTA update system */
int sait01_ota_init(void)
{
    LOG_INF("Initializing SAIT_01 OTA Update System");
    
    memset(&ota_ctx, 0, sizeof(ota_ctx));
    
    // Get current firmware version
    get_current_version(&ota_ctx.current_version);
    
    LOG_INF("Current firmware: %u.%u.%u.%u (%s)",
            ota_ctx.current_version.major, ota_ctx.current_version.minor,
            ota_ctx.current_version.patch, ota_ctx.current_version.build,
            ota_ctx.current_version.variant);
    
    // Open update partition
    int ret = flash_area_open(FLASH_AREA_ID(image_1), &ota_ctx.update_area);
    if (ret != 0) {
        LOG_ERR("Failed to open update partition: %d", ret);
        return ret;
    }
    
    LOG_INF("Update partition: 0x%x bytes at 0x%x",
            ota_ctx.update_area->fa_size, ota_ctx.update_area->fa_off);
    
    // Initialize PSA crypto for signature verification
    psa_status_t status = psa_crypto_init();
    if (status != PSA_SUCCESS && status != PSA_ERROR_ALREADY_EXISTS) {
        LOG_ERR("PSA crypto initialization failed: %d", status);
        flash_area_close(ota_ctx.update_area);
        return -1;
    }
    
    ota_ctx.state = SAIT01_OTA_IDLE;
    ota_ctx.initialized = true;
    
    // Check if we just booted from an update
    if (boot_is_img_confirmed()) {
        LOG_INF("Previous OTA update confirmed");
        ota_ctx.successful_updates++;
    } else {
        // Confirm current image to prevent revert
        ret = boot_write_img_confirmed();
        if (ret != 0) {
            LOG_WRN("Failed to confirm current image: %d", ret);
        }
    }
    
    LOG_INF("OTA update system initialized");
    return 0;
}

/* Start OTA update */
int sait01_ota_start_update(const struct sait01_ota_header *header)
{
    if (!ota_ctx.initialized) {
        LOG_ERR("OTA not initialized");
        return -1;
    }
    
    if (ota_ctx.update_in_progress) {
        LOG_WRN("Update already in progress");
        return -1;
    }
    
    LOG_INF("Starting OTA update");
    
    // Validate header magic
    if (header->magic != SAIT01_OTA_MAGIC) {
        LOG_ERR("Invalid OTA header magic: 0x%08x", header->magic);
        ota_ctx.last_error = SAIT01_OTA_ERROR_CORRUPTED_IMAGE;
        return -1;
    }
    
    // Copy header
    ota_ctx.update_header = *header;
    
    // Verify signature
    if (verify_firmware_signature(&ota_ctx.update_header) != 0) {
        ota_ctx.last_error = SAIT01_OTA_ERROR_INVALID_SIGNATURE;
        return -1;
    }
    
    // Verify rollback protection
    if (verify_rollback_protection(&ota_ctx.update_header.version) != 0) {
        ota_ctx.last_error = SAIT01_OTA_ERROR_ROLLBACK_DETECTED;
        return -1;
    }
    
    // Check if image fits in partition
    if (ota_ctx.update_header.image_size > ota_ctx.update_area->fa_size) {
        LOG_ERR("Image too large: %u bytes (max %u)",
                ota_ctx.update_header.image_size, ota_ctx.update_area->fa_size);
        ota_ctx.last_error = SAIT01_OTA_ERROR_INSUFFICIENT_SPACE;
        return -1;
    }
    
    // Allocate chunk tracking
    ota_ctx.expected_chunks = ota_ctx.update_header.chunk_count;
    ota_ctx.chunk_received_flags = k_calloc(ota_ctx.expected_chunks, sizeof(bool));
    
    if (!ota_ctx.chunk_received_flags) {
        LOG_ERR("Failed to allocate chunk tracking");
        return -1;
    }
    
    // Erase update partition
    if (erase_update_partition() != 0) {
        k_free(ota_ctx.chunk_received_flags);
        ota_ctx.last_error = SAIT01_OTA_ERROR_FLASH_WRITE;
        return -1;
    }
    
    // Initialize hash operation
    psa_status_t status = psa_hash_setup(&ota_ctx.hash_op, PSA_ALG_SHA_256);
    if (status != PSA_SUCCESS) {
        LOG_ERR("Failed to setup hash: %d", status);
        k_free(ota_ctx.chunk_received_flags);
        return -1;
    }
    
    ota_ctx.state = SAIT01_OTA_DOWNLOADING;
    ota_ctx.update_in_progress = true;
    ota_ctx.bytes_written = 0;
    ota_ctx.chunks_received = 0;
    ota_ctx.update_start_time = k_uptime_get_32();
    ota_ctx.total_updates_attempted++;
    
    LOG_INF("OTA update started: %u.%u.%u.%u (%u bytes, %u chunks)",
            ota_ctx.update_header.version.major, ota_ctx.update_header.version.minor,
            ota_ctx.update_header.version.patch, ota_ctx.update_header.version.build,
            ota_ctx.update_header.image_size, ota_ctx.update_header.chunk_count);
    
    return 0;
}

/* Process received chunk */
int sait01_ota_process_chunk(const struct sait01_ota_chunk_msg *chunk_msg)
{
    if (!ota_ctx.initialized || !ota_ctx.update_in_progress) {
        LOG_ERR("No update in progress");
        return -1;
    }
    
    if (ota_ctx.state != SAIT01_OTA_DOWNLOADING) {
        LOG_ERR("Not in downloading state");
        return -1;
    }
    
    if (chunk_msg->chunk_number >= ota_ctx.expected_chunks) {
        LOG_ERR("Invalid chunk number: %u (max %u)",
                chunk_msg->chunk_number, ota_ctx.expected_chunks - 1);
        return -1;
    }
    
    // Check if we already received this chunk
    if (ota_ctx.chunk_received_flags[chunk_msg->chunk_number]) {
        LOG_DBG("Duplicate chunk %u, ignoring", chunk_msg->chunk_number);
        return 0;
    }
    
    // Verify chunk CRC (simplified)
    // In production, calculate actual CRC32 of chunk data
    
    // Write chunk to flash
    if (write_chunk_to_flash(chunk_msg->chunk_number, 
                           chunk_msg->chunk_data, 
                           chunk_msg->chunk_size) != 0) {
        ota_ctx.last_error = SAIT01_OTA_ERROR_FLASH_WRITE;
        return -1;
    }
    
    // Mark chunk as received
    ota_ctx.chunk_received_flags[chunk_msg->chunk_number] = true;
    ota_ctx.chunks_received++;
    
    uint8_t progress = (ota_ctx.chunks_received * 100) / ota_ctx.expected_chunks;
    
    LOG_INF("Received chunk %u/%u (%u%% complete)",
            ota_ctx.chunks_received, ota_ctx.expected_chunks, progress);
    
    // Check if all chunks received
    if (ota_ctx.chunks_received == ota_ctx.expected_chunks) {
        LOG_INF("All chunks received, finalizing update");
        ota_ctx.state = SAIT01_OTA_VERIFYING;
        
        if (finalize_update() == 0) {
            ota_ctx.state = SAIT01_OTA_COMPLETE;
            ota_ctx.successful_updates++;
            LOG_INF("OTA update completed successfully");
        } else {
            ota_ctx.state = SAIT01_OTA_ERROR;
            ota_ctx.failed_updates++;
            LOG_ERR("OTA update failed during finalization");
        }
        
        // Cleanup
        k_free(ota_ctx.chunk_received_flags);
        ota_ctx.chunk_received_flags = NULL;
        ota_ctx.update_in_progress = false;
    }
    
    return 0;
}

/* Abort current update */
int sait01_ota_abort_update(void)
{
    if (!ota_ctx.update_in_progress) {
        return 0;
    }
    
    LOG_WRN("Aborting OTA update");
    
    // Cleanup resources
    if (ota_ctx.chunk_received_flags) {
        k_free(ota_ctx.chunk_received_flags);
        ota_ctx.chunk_received_flags = NULL;
    }
    
    psa_hash_abort(&ota_ctx.hash_op);
    
    ota_ctx.state = SAIT01_OTA_IDLE;
    ota_ctx.update_in_progress = false;
    ota_ctx.failed_updates++;
    
    return 0;
}

/* Apply update and reboot */
int sait01_ota_apply_update(void)
{
    if (ota_ctx.state != SAIT01_OTA_COMPLETE) {
        LOG_ERR("No completed update to apply");
        return -1;
    }
    
    LOG_INF("Applying OTA update - system will reboot");
    
    // Allow some time for log messages
    k_sleep(K_MSEC(100));
    
    // Reboot to apply update
    sys_reboot(SYS_REBOOT_WARM);
    
    // Should not reach here
    return 0;
}

/* Get OTA status */
int sait01_ota_get_status(struct sait01_ota_status_msg *status)
{
    if (!ota_ctx.initialized || !status) {
        return -1;
    }
    
    status->message_type = SAIT01_OTA_MSG_UPDATE_STATUS;
    status->state = ota_ctx.state;
    status->error = ota_ctx.last_error;
    status->chunks_received = ota_ctx.chunks_received;
    status->total_chunks = ota_ctx.expected_chunks;
    
    if (ota_ctx.expected_chunks > 0) {
        status->progress_percent = (ota_ctx.chunks_received * 100) / ota_ctx.expected_chunks;
    } else {
        status->progress_percent = 0;
    }
    
    return 0;
}

/* Get OTA statistics */
void sait01_ota_get_stats(uint32_t *total_updates, uint32_t *successful_updates,
                         uint32_t *failed_updates, bool *update_in_progress)
{
    if (total_updates) *total_updates = ota_ctx.total_updates_attempted;
    if (successful_updates) *successful_updates = ota_ctx.successful_updates;
    if (failed_updates) *failed_updates = ota_ctx.failed_updates;
    if (update_in_progress) *update_in_progress = ota_ctx.update_in_progress;
}

/* Check for available updates (stub for network integration) */
int sait01_ota_check_for_updates(void)
{
    if (!ota_ctx.initialized) {
        return -1;
    }
    
    ota_ctx.state = SAIT01_OTA_CHECKING_VERSION;
    
    LOG_INF("Checking for firmware updates");
    
    // In production, this would contact update server
    // For now, just return to idle
    ota_ctx.state = SAIT01_OTA_IDLE;
    
    LOG_INF("No updates available");
    return 0;
}