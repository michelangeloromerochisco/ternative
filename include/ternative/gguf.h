#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <memory>

namespace ternative {

// GGUF format version 3
constexpr uint32_t GGUF_MAGIC = 0x46554747;  // "GGUF" little-endian
constexpr uint32_t GGUF_VERSION = 3;

// GGML tensor types
enum class ggml_type : uint32_t {
    F32  = 0,
    F16  = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    BF16 = 30,   // bfloat16
    I2_S = 36,  // BitNet-specific ternary
};

// GGUF metadata value types
enum class gguf_metadata_type : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

// Metadata value container
struct MetadataValue {
    gguf_metadata_type type;
    std::vector<uint8_t> raw;
    std::string str_val;
    std::vector<MetadataValue> array_val;

    uint32_t as_u32() const;
    int32_t as_i32() const;
    float as_f32() const;
    uint64_t as_u64() const;
    int64_t as_i64() const;
    double as_f64() const;
    bool as_bool() const;
    const std::string& as_string() const;
};

// Tensor info (no data, just metadata)
struct TensorInfo {
    std::string name;
    std::vector<uint64_t> shape;
    ggml_type type;
    uint64_t offset;  // Offset from start of tensor data section
};

// Parsed GGUF file
struct GGUFile {
    std::map<std::string, MetadataValue> metadata;
    std::vector<TensorInfo> tensors;
    std::vector<uint8_t> tensor_data;  // Raw tensor bytes
    uint64_t tensor_data_offset = 0;    // Offset in original file

    bool has_metadata(const std::string& key) const;
    MetadataValue get_metadata(const std::string& key) const;
    const TensorInfo* find_tensor(const std::string& name) const;
};

// Load a GGUF file from disk (full: reads all tensor data)
std::unique_ptr<GGUFile> gguf_load(const std::string& path);

// Load only metadata + tensor info from a GGUF file.
// Reads at most ~32 MB instead of the full file — suitable for tokenizer init.
// Returned GGUFile has empty tensor_data; do not use for weight loading.
std::unique_ptr<GGUFile> gguf_load_metadata(const std::string& path);

// Pretty-print GGUF info
void gguf_print_info(const GGUFile& gguf);

// Get element size for a GGML type
size_t ggml_type_size(ggml_type type);

// Get block size (number of elements per block) for quantized types
size_t ggml_block_size(ggml_type type);

} // namespace ternative
