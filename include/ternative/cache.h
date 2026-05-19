#pragma once

#include "model.h"

#include <string>
#include <vector>
#include <cstdint>

namespace ternative {

// ---------------------------------------------------------------------------
// .tvcache — Ternative merged-weight cache
//
// Stores the F16 result of (I2_S base + LoRA delta) so subsequent server
// starts skip the 8-minute LoRA merge.
//
// Cache key: SHA-256(base GGUF bytes) || SHA-256(LoRA GGUF bytes) || alpha
// Invalidated on any content change, not just path change.
// ---------------------------------------------------------------------------

static constexpr uint32_t TVCACHE_VERSION = 1;

// Use packed structs so the binary layout is identical on all compilers
#ifdef _MSC_VER
#  pragma pack(push, 1)
#endif

struct
#ifndef _MSC_VER
__attribute__((packed))
#endif
TvCacheHeader {
    char     magic[8];               // "TVCACHE\0"
    uint32_t version;                // 1
    uint32_t dtype;                  // 1 = F16
    uint64_t base_size;              // base GGUF file size in bytes
    uint8_t  base_sha256[32];        // SHA-256 of base GGUF
    uint64_t lora_size;              // LoRA GGUF size (0 = no LoRA)
    uint8_t  lora_sha256[32];        // SHA-256 of LoRA GGUF (zeros if no LoRA)
    float    lora_alpha;
    uint32_t lora_rank;
    uint32_t num_tensors;
    uint64_t tensor_table_offset;    // byte offset from file start
    uint64_t data_offset;            // byte offset from file start (4096-aligned)
    uint64_t total_file_size;
    uint8_t  reserved[124];          // pad to 256 bytes total
    // 8+4+4+8+32+8+32+4+4+4+8+8+8 = 132 + 124 = 256
};

struct
#ifndef _MSC_VER
__attribute__((packed))
#endif
TvCacheTensor {
    char     name[128];
    uint32_t dtype;                  // 1 = F16
    uint32_t n_dims;
    uint64_t shape[4];               // shape[0..n_dims-1], rest = 0
    uint64_t data_offset;            // offset from TvCacheHeader::data_offset
    uint64_t nbytes;
    uint8_t  _pad[8];                // pad to 192 bytes: 128+4+4+32+8+8 = 184 + 8 = 192
};

#ifdef _MSC_VER
#  pragma pack(pop)
#endif

static_assert(sizeof(TvCacheHeader) == 256, "TvCacheHeader must be 256 bytes");
static_assert(sizeof(TvCacheTensor) == 192, "TvCacheTensor must be 192 bytes");

// ---------------------------------------------------------------------------

struct CacheKey {
    std::string base_sha256_hex;
    std::string lora_sha256_hex;   // empty string means no LoRA
    float       lora_alpha  = 0.f;
    uint32_t    lora_rank   = 0;
};

class ModelCache {
public:
    // Compute cache key from file content (hashes the full files).
    static CacheKey compute_key(const std::string& base_path,
                                const std::string& lora_path = "");

    // Return the .tvcache path for a given key under cache_dir.
    // cache_dir defaults to platform cache dir if empty.
    static std::string cache_path(const CacheKey& key,
                                  const std::string& cache_dir = "");

    // Try loading a cached merged model.
    // Returns true and populates model.layers on success.
    // Returns false on miss, format mismatch, or I/O error.
    static bool try_load(Model& model, const std::string& path);

    // Save the current merged F16 weights to a .tvcache file.
    // Called once after a successful LoRA merge.
    static void save(const Model& model,
                     const std::string& path,
                     const CacheKey& key);

    // Compute SHA-256 of a file, returned as 32-byte array.
    static bool sha256_file(const std::string& path, uint8_t out[32]);

    // Format a 32-byte hash as a 64-char hex string.
    static std::string hex(const uint8_t bytes[32]);

    // Ensure a directory exists (create if necessary).
    static bool mkdir_p(const std::string& dir);

    // Default OS cache directory.
    static std::string default_cache_dir();
};

} // namespace ternative
