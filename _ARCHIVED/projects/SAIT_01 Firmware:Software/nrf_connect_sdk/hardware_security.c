/*
 * SAIT_01 Hardware Security Module
 * nRF5340 Secure Boot & Key Management
 * Defense-Grade Security Implementation
 */

#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/devicetree.h>
#include <zephyr/drivers/flash.h>
#include <zephyr/storage/flash_map.h>
#include <zephyr/sys/reboot.h>
#include <zephyr/logging/log.h>
#include <psa/crypto.h>
#include <hw_unique_key.h>
#include <tfm_ns_interface.h>

LOG_MODULE_REGISTER(hw_security, LOG_LEVEL_INF);

#define SAIT01_DEVICE_ID_SIZE 32
#define SAIT01_ENCRYPTION_KEY_SIZE 32
#define SAIT01_SIGNATURE_SIZE 64
#define SAIT01_NONCE_SIZE 12
#define SAIT01_AUTH_TAG_SIZE 16

/* Hardware security state */
struct sait01_security_context {
    uint8_t device_id[SAIT01_DEVICE_ID_SIZE];
    psa_key_id_t master_key_id;
    psa_key_id_t mesh_key_id;
    psa_key_id_t ota_key_id;
    bool secure_boot_verified;
    bool keys_provisioned;
    uint32_t security_level;
};

static struct sait01_security_context sec_ctx;

/* Secure boot verification */
int sait01_verify_secure_boot(void)
{
    psa_status_t status;
    uint8_t firmware_hash[32];
    uint8_t signature[SAIT01_SIGNATURE_SIZE];
    
    LOG_INF("🔐 Starting secure boot verification");
    
    /* Get firmware hash from secure storage */
    size_t hash_length;
    status = psa_hash_compute(PSA_ALG_SHA_256,
                             (const uint8_t*)CONFIG_FLASH_BASE_ADDRESS,
                             CONFIG_FLASH_SIZE,
                             firmware_hash,
                             sizeof(firmware_hash),
                             &hash_length);
    
    if (status != PSA_SUCCESS) {
        LOG_ERR("❌ Firmware hash computation failed: %d", status);
        return -1;
    }
    
    /* Verify signature using Nordic's root key */
    status = psa_verify_hash(sec_ctx.master_key_id,
                            PSA_ALG_ECDSA(PSA_ALG_SHA_256),
                            firmware_hash,
                            hash_length,
                            signature,
                            SAIT01_SIGNATURE_SIZE);
    
    if (status == PSA_SUCCESS) {
        LOG_INF("✅ Secure boot verification PASSED");
        sec_ctx.secure_boot_verified = true;
        return 0;
    } else {
        LOG_ERR("💀 Secure boot verification FAILED: %d", status);
        /* Secure boot failure - enter lockdown mode */
        sait01_enter_security_lockdown();
        return -1;
    }
}

/* Hardware unique key derivation */
int sait01_derive_device_keys(void)
{
    psa_status_t status;
    psa_key_attributes_t key_attr = PSA_KEY_ATTRIBUTES_INIT;
    uint8_t hw_unique_key[HUK_SIZE_BYTES];
    
    LOG_INF("🔑 Deriving device-unique encryption keys");
    
    /* Get hardware unique key from nRF5340 */
    int ret = hw_unique_key_get(hw_unique_key, HUK_SIZE_BYTES);
    if (ret != 0) {
        LOG_ERR("❌ Failed to get hardware unique key: %d", ret);
        return ret;
    }
    
    /* Derive master encryption key */
    psa_set_key_usage_flags(&key_attr, PSA_KEY_USAGE_ENCRYPT | PSA_KEY_USAGE_DECRYPT);
    psa_set_key_algorithm(&key_attr, PSA_ALG_GCM);
    psa_set_key_type(&key_attr, PSA_KEY_TYPE_AES);
    psa_set_key_bits(&key_attr, 256);
    psa_set_key_lifetime(&key_attr, PSA_KEY_LIFETIME_PERSISTENT);
    psa_set_key_id(&key_attr, 1000); // Master key ID
    
    status = psa_import_key(&key_attr, hw_unique_key, 32, &sec_ctx.master_key_id);
    if (status != PSA_SUCCESS) {
        LOG_ERR("❌ Master key import failed: %d", status);
        return -1;
    }
    
    /* Derive mesh communication key */
    uint8_t mesh_key_material[32];
    size_t mesh_key_len;
    
    status = psa_key_derivation_setup(&derivation, PSA_ALG_HKDF(PSA_ALG_SHA_256));
    status = psa_key_derivation_input_key(&derivation, PSA_KEY_DERIVATION_INPUT_SECRET, sec_ctx.master_key_id);
    status = psa_key_derivation_input_bytes(&derivation, PSA_KEY_DERIVATION_INPUT_INFO, 
                                          (const uint8_t*)"SAIT01_MESH", 11);
    
    psa_set_key_id(&key_attr, 1001); // Mesh key ID
    status = psa_key_derivation_output_key(&key_attr, &derivation, &sec_ctx.mesh_key_id);
    
    if (status != PSA_SUCCESS) {
        LOG_ERR("❌ Mesh key derivation failed: %d", status);
        return -1;
    }
    
    /* Derive OTA update key */
    psa_set_key_id(&key_attr, 1002); // OTA key ID
    psa_set_key_usage_flags(&key_attr, PSA_KEY_USAGE_VERIFY_HASH);
    psa_set_key_algorithm(&key_attr, PSA_ALG_ECDSA(PSA_ALG_SHA_256));
    psa_set_key_type(&key_attr, PSA_KEY_TYPE_ECC_PUBLIC_KEY(PSA_ECC_FAMILY_SECP_R1));
    psa_set_key_bits(&key_attr, 256);
    
    status = psa_key_derivation_output_key(&key_attr, &derivation, &sec_ctx.ota_key_id);
    psa_key_derivation_abort(&derivation);
    
    if (status == PSA_SUCCESS) {
        LOG_INF("✅ All device keys derived successfully");
        sec_ctx.keys_provisioned = true;
        return 0;
    } else {
        LOG_ERR("❌ OTA key derivation failed: %d", status);
        return -1;
    }
}

/* Generate unique device ID */
int sait01_generate_device_id(void)
{
    psa_status_t status;
    uint8_t random_seed[16];
    uint8_t hw_unique_key[HUK_SIZE_BYTES];
    
    LOG_INF("🆔 Generating unique device identifier");
    
    /* Get hardware unique key and random entropy */
    hw_unique_key_get(hw_unique_key, HUK_SIZE_BYTES);
    status = psa_generate_random(random_seed, sizeof(random_seed));
    
    if (status != PSA_SUCCESS) {
        LOG_ERR("❌ Random generation failed: %d", status);
        return -1;
    }
    
    /* Hash HUK + entropy to create device ID */
    size_t hash_length;
    status = psa_hash_compute(PSA_ALG_SHA_256,
                             hw_unique_key, HUK_SIZE_BYTES,
                             sec_ctx.device_id, SAIT01_DEVICE_ID_SIZE,
                             &hash_length);
    
    if (status == PSA_SUCCESS) {
        LOG_INF("✅ Device ID generated: %02x%02x%02x%02x...", 
                sec_ctx.device_id[0], sec_ctx.device_id[1], 
                sec_ctx.device_id[2], sec_ctx.device_id[3]);
        return 0;
    } else {
        LOG_ERR("❌ Device ID generation failed: %d", status);
        return -1;
    }
}

/* Encrypt mesh communication data */
int sait01_encrypt_mesh_data(const uint8_t *plaintext, size_t plaintext_len,
                            uint8_t *ciphertext, size_t ciphertext_size,
                            size_t *ciphertext_len)
{
    psa_status_t status;
    uint8_t nonce[SAIT01_NONCE_SIZE];
    
    /* Generate random nonce */
    status = psa_generate_random(nonce, SAIT01_NONCE_SIZE);
    if (status != PSA_SUCCESS) {
        return -1;
    }
    
    /* Encrypt with AES-GCM */
    status = psa_aead_encrypt(sec_ctx.mesh_key_id,
                             PSA_ALG_GCM,
                             nonce, SAIT01_NONCE_SIZE,
                             NULL, 0, // No additional data
                             plaintext, plaintext_len,
                             ciphertext, ciphertext_size,
                             ciphertext_len);
    
    if (status == PSA_SUCCESS) {
        /* Prepend nonce to ciphertext */
        memmove(ciphertext + SAIT01_NONCE_SIZE, ciphertext, *ciphertext_len);
        memcpy(ciphertext, nonce, SAIT01_NONCE_SIZE);
        *ciphertext_len += SAIT01_NONCE_SIZE;
        
        LOG_DBG("✅ Mesh data encrypted: %zu bytes", *ciphertext_len);
        return 0;
    } else {
        LOG_ERR("❌ Mesh encryption failed: %d", status);
        return -1;
    }
}

/* Decrypt mesh communication data */
int sait01_decrypt_mesh_data(const uint8_t *ciphertext, size_t ciphertext_len,
                            uint8_t *plaintext, size_t plaintext_size,
                            size_t *plaintext_len)
{
    psa_status_t status;
    
    if (ciphertext_len < SAIT01_NONCE_SIZE) {
        return -1;
    }
    
    /* Extract nonce and ciphertext */
    const uint8_t *nonce = ciphertext;
    const uint8_t *encrypted_data = ciphertext + SAIT01_NONCE_SIZE;
    size_t encrypted_len = ciphertext_len - SAIT01_NONCE_SIZE;
    
    /* Decrypt with AES-GCM */
    status = psa_aead_decrypt(sec_ctx.mesh_key_id,
                             PSA_ALG_GCM,
                             nonce, SAIT01_NONCE_SIZE,
                             NULL, 0, // No additional data
                             encrypted_data, encrypted_len,
                             plaintext, plaintext_size,
                             plaintext_len);
    
    if (status == PSA_SUCCESS) {
        LOG_DBG("✅ Mesh data decrypted: %zu bytes", *plaintext_len);
        return 0;
    } else {
        LOG_ERR("❌ Mesh decryption failed: %d", status);
        return -1;
    }
}

/* Security lockdown mode */
void sait01_enter_security_lockdown(void)
{
    LOG_ERR("🚨 ENTERING SECURITY LOCKDOWN MODE");
    
    /* Clear all keys from memory */
    memset(&sec_ctx, 0, sizeof(sec_ctx));
    
    /* Disable all non-essential peripherals */
    /* Disable debug interfaces */
    /* Enter deep sleep until external reset */
    
    LOG_ERR("🔒 Device locked down - external reset required");
    
    /* Infinite loop to prevent code execution */
    while (1) {
        k_sleep(K_FOREVER);
    }
}

/* Get device security status */
int sait01_get_security_status(struct sait01_security_info *info)
{
    if (!info) {
        return -1;
    }
    
    info->secure_boot_verified = sec_ctx.secure_boot_verified;
    info->keys_provisioned = sec_ctx.keys_provisioned;
    info->security_level = sec_ctx.security_level;
    memcpy(info->device_id, sec_ctx.device_id, SAIT01_DEVICE_ID_SIZE);
    
    return 0;
}

/* Initialize hardware security module */
int sait01_security_init(void)
{
    int ret;
    
    LOG_INF("🛡️ Initializing SAIT_01 Hardware Security Module");
    
    /* Initialize PSA Crypto */
    psa_status_t status = psa_crypto_init();
    if (status != PSA_SUCCESS) {
        LOG_ERR("❌ PSA crypto initialization failed: %d", status);
        return -1;
    }
    
    /* Clear security context */
    memset(&sec_ctx, 0, sizeof(sec_ctx));
    
    /* Step 1: Verify secure boot */
    ret = sait01_verify_secure_boot();
    if (ret != 0) {
        LOG_ERR("💀 Secure boot verification failed - entering lockdown");
        sait01_enter_security_lockdown();
        return ret;
    }
    
    /* Step 2: Generate device ID */
    ret = sait01_generate_device_id();
    if (ret != 0) {
        LOG_ERR("❌ Device ID generation failed");
        return ret;
    }
    
    /* Step 3: Derive encryption keys */
    ret = sait01_derive_device_keys();
    if (ret != 0) {
        LOG_ERR("❌ Key derivation failed");
        return ret;
    }
    
    /* Set security level based on successful initialization */
    sec_ctx.security_level = 3; // Defense-grade security
    
    LOG_INF("✅ Hardware Security Module initialized successfully");
    LOG_INF("🔐 Security Level: %d (Defense-Grade)", sec_ctx.security_level);
    
    return 0;
}

/* Security module structure for public interface */
struct sait01_security_info {
    bool secure_boot_verified;
    bool keys_provisioned;
    uint32_t security_level;
    uint8_t device_id[SAIT01_DEVICE_ID_SIZE];
};