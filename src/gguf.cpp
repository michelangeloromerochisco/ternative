#include "ternative/gguf.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>

namespace ternative {

// === MetadataValue accessors ===

uint32_t MetadataValue::as_u32() const {
    if (type != gguf_metadata_type::UINT32) return 0;
    uint32_t val;
    std::memcpy(&val, raw.data(), sizeof(val));
    return val;
}

int32_t MetadataValue::as_i32() const {
    if (type != gguf_metadata_type::INT32) return 0;
    int32_t val;
    std::memcpy(&val, raw.data(), sizeof(val));
    return val;
}

float MetadataValue::as_f32() const {
    if (type != gguf_metadata_type::FLOAT32) return 0.0f;
    float val;
    std::memcpy(&val, raw.data(), sizeof(val));
    return val;
}

uint64_t MetadataValue::as_u64() const {
    if (type != gguf_metadata_type::UINT64) return 0;
    uint64_t val;
    std::memcpy(&val, raw.data(), sizeof(val));
    return val;
}

int64_t MetadataValue::as_i64() const {
    if (type != gguf_metadata_type::INT64) return 0;
    int64_t val;
    std::memcpy(&val, raw.data(), sizeof(val));
    return val;
}

double MetadataValue::as_f64() const {
    if (type != gguf_metadata_type::FLOAT64) return 0.0;
    double val;
    std::memcpy(&val, raw.data(), sizeof(val));
    return val;
}

bool MetadataValue::as_bool() const {
    if (type != gguf_metadata_type::BOOL) return false;
    return raw[0] != 0;
}

const std::string& MetadataValue::as_string() const {
    return str_val;
}

// === GGUFile helpers ===

bool GGUFile::has_metadata(const std::string& key) const {
    return metadata.find(key) != metadata.end();
}

MetadataValue GGUFile::get_metadata(const std::string& key) const {
    auto it = metadata.find(key);
    if (it != metadata.end()) return it->second;
    return MetadataValue{};
}

const TensorInfo* GGUFile::find_tensor(const std::string& name) const {
    for (const auto& t : tensors) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

// === GGUF parsing ===

static uint32_t read_u32_le(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t read_u64_le(const uint8_t* p) {
    uint64_t val = 0;
    for (int i = 0; i < 8; ++i) val |= ((uint64_t)p[i]) << (8 * i);
    return val;
}

static int64_t read_i64_le(const uint8_t* p) {
    return (int64_t)read_u64_le(p);
}

static float read_f32_le(const uint8_t* p) {
    float val;
    std::memcpy(&val, p, sizeof(val));
    return val;
}

static double read_f64_le(const uint8_t* p) {
    double val;
    std::memcpy(&val, p, sizeof(val));
    return val;
}

// All read_* functions return false / empty on buffer overrun.
// buf_end is always data.data() + buf_size — the first byte past the valid buffer.

static bool read_string(const uint8_t*& ptr, const uint8_t* buf_end, std::string& out) {
    if (buf_end - ptr < 8) return false;
    uint64_t len = read_u64_le(ptr); ptr += 8;
    if (static_cast<uint64_t>(buf_end - ptr) < len) return false;
    out.assign(reinterpret_cast<const char*>(ptr), len);
    ptr += len;
    return true;
}

static MetadataValue read_metadata_value(const uint8_t*& ptr, const uint8_t* buf_end);

static MetadataValue read_metadata_array(const uint8_t*& ptr, const uint8_t* buf_end) {
    MetadataValue arr;
    arr.type = gguf_metadata_type::ARRAY;
    if (buf_end - ptr < 12) return arr;
    uint32_t item_type_val = read_u32_le(ptr); ptr += 4;
    gguf_metadata_type item_type = (gguf_metadata_type)item_type_val;
    uint64_t n_items = read_u64_le(ptr); ptr += 8;
    arr.array_val.reserve(static_cast<size_t>(std::min(n_items, uint64_t(10000000))));
    for (uint64_t i = 0; i < n_items; ++i) {
        if (ptr >= buf_end) break;
        MetadataValue item;
        item.type = item_type;
        switch (item_type) {
            case gguf_metadata_type::UINT8:
            case gguf_metadata_type::INT8:
            case gguf_metadata_type::BOOL:
                if (buf_end - ptr < 1) goto array_done;
                item.raw.assign(ptr, ptr + 1); ptr += 1; break;
            case gguf_metadata_type::UINT16:
            case gguf_metadata_type::INT16:
                if (buf_end - ptr < 2) goto array_done;
                item.raw.assign(ptr, ptr + 2); ptr += 2; break;
            case gguf_metadata_type::UINT32:
            case gguf_metadata_type::INT32:
            case gguf_metadata_type::FLOAT32:
                if (buf_end - ptr < 4) goto array_done;
                item.raw.assign(ptr, ptr + 4); ptr += 4; break;
            case gguf_metadata_type::UINT64:
            case gguf_metadata_type::INT64:
            case gguf_metadata_type::FLOAT64:
                if (buf_end - ptr < 8) goto array_done;
                item.raw.assign(ptr, ptr + 8); ptr += 8; break;
            case gguf_metadata_type::STRING: {
                std::string s;
                if (!read_string(ptr, buf_end, s)) goto array_done;
                item.str_val = std::move(s);
                item.raw.assign(item.str_val.begin(), item.str_val.end());
                break;
            }
            case gguf_metadata_type::ARRAY:
                item = read_metadata_array(ptr, buf_end);
                break;
        }
        arr.array_val.push_back(std::move(item));
    }
    array_done:
    return arr;
}

static MetadataValue read_metadata_value(const uint8_t*& ptr, const uint8_t* buf_end) {
    MetadataValue val;
    if (buf_end - ptr < 4) return val;
    uint32_t type_val = read_u32_le(ptr); ptr += 4;
    val.type = (gguf_metadata_type)type_val;
    switch (val.type) {
        case gguf_metadata_type::UINT8:
        case gguf_metadata_type::INT8:
        case gguf_metadata_type::BOOL:
            if (buf_end - ptr < 1) break;
            val.raw.assign(ptr, ptr + 1); ptr += 1; break;
        case gguf_metadata_type::UINT16:
        case gguf_metadata_type::INT16:
            if (buf_end - ptr < 2) break;
            val.raw.assign(ptr, ptr + 2); ptr += 2; break;
        case gguf_metadata_type::UINT32:
        case gguf_metadata_type::INT32:
        case gguf_metadata_type::FLOAT32:
            if (buf_end - ptr < 4) break;
            val.raw.assign(ptr, ptr + 4); ptr += 4; break;
        case gguf_metadata_type::UINT64:
        case gguf_metadata_type::INT64:
        case gguf_metadata_type::FLOAT64:
            if (buf_end - ptr < 8) break;
            val.raw.assign(ptr, ptr + 8); ptr += 8; break;
        case gguf_metadata_type::STRING: {
            std::string s;
            read_string(ptr, buf_end, s);
            val.str_val = std::move(s);
            val.raw.assign(val.str_val.begin(), val.str_val.end());
            break;
        }
        case gguf_metadata_type::ARRAY:
            val = read_metadata_array(ptr, buf_end);
            break;
    }
    return val;
}

// Forward declaration — defined below after gguf_load_metadata
static std::unique_ptr<GGUFile> gguf_parse_buffer(
        const uint8_t* data, size_t buf_size, size_t file_size);

std::unique_ptr<GGUFile> gguf_load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open: " << path << std::endl;
        return nullptr;
    }
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(file_size);
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    if (!file) {
        std::cerr << "Failed to read: " << path << std::endl;
        return nullptr;
    }

    auto gguf = gguf_parse_buffer(data.data(), file_size, file_size);
    if (!gguf) return nullptr;

    // Copy tensor data
    uint64_t tdo = gguf->tensor_data_offset;
    if (tdo < file_size) {
        size_t tensor_data_size = file_size - tdo;
        gguf->tensor_data.resize(tensor_data_size);
        std::memcpy(gguf->tensor_data.data(), data.data() + tdo, tensor_data_size);
    }

    return gguf;
}

// Shared parse logic — parses from an in-memory buffer (does NOT copy tensor data).
// All reads are bounds-checked against buf_end = data + buf_size.
static std::unique_ptr<GGUFile> gguf_parse_buffer(
        const uint8_t* data, size_t buf_size, size_t file_size) {
    const uint8_t* buf_end = data + buf_size;
    const uint8_t* ptr     = data;

    if (buf_size < 24) return nullptr;  // header too small
    uint32_t magic = read_u32_le(ptr); ptr += 4;
    if (magic != GGUF_MAGIC) {
        std::cerr << "Invalid GGUF magic: " << std::hex << magic << std::dec << std::endl;
        return nullptr;
    }
    uint32_t version = read_u32_le(ptr); ptr += 4;
    if (version != GGUF_VERSION) {
        std::cerr << "Unsupported GGUF version: " << version << std::endl;
        return nullptr;
    }

    auto gguf = std::make_unique<GGUFile>();
    uint64_t n_tensors  = read_u64_le(ptr); ptr += 8;
    uint64_t n_metadata = read_u64_le(ptr); ptr += 8;

    for (uint64_t i = 0; i < n_metadata && ptr < buf_end; ++i) {
        std::string key;
        if (!read_string(ptr, buf_end, key)) break;
        MetadataValue val = read_metadata_value(ptr, buf_end);
        gguf->metadata[std::move(key)] = std::move(val);
    }

    gguf->tensors.reserve(static_cast<size_t>(std::min(n_tensors, uint64_t(100000))));
    for (uint64_t i = 0; i < n_tensors && ptr < buf_end; ++i) {
        TensorInfo info;
        if (!read_string(ptr, buf_end, info.name)) break;
        if (buf_end - ptr < 4) break;
        uint32_t n_dims = read_u32_le(ptr); ptr += 4;
        if (n_dims > 8) break;  // sanity cap
        info.shape.resize(n_dims);
        if (buf_end - ptr < static_cast<ptrdiff_t>(n_dims * 8 + 8)) break;
        for (uint32_t d = 0; d < n_dims; ++d) {
            info.shape[d] = read_u64_le(ptr); ptr += 8;
        }
        uint32_t type_val = read_u32_le(ptr); ptr += 4;
        info.type = (ggml_type)type_val;
        info.offset = read_u64_le(ptr); ptr += 8;
        gguf->tensors.push_back(std::move(info));
    }

    uint32_t alignment = 32;
    if (gguf->has_metadata("general.alignment"))
        alignment = gguf->get_metadata("general.alignment").as_u32();

    uint64_t tensor_data_offset = ptr - data;
    tensor_data_offset = (tensor_data_offset + alignment - 1) / alignment * alignment;
    gguf->tensor_data_offset = tensor_data_offset;

    return gguf;
}

std::unique_ptr<GGUFile> gguf_load_metadata(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open: " << path << std::endl;
        return nullptr;
    }
    file.seekg(0, std::ios::end);
    size_t file_size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    // For a 128K-vocab Llama 3.1 BPE tokenizer, metadata is ~15-20 MB.
    // 64 MB gives comfortable headroom for any realistic model variant.
    static constexpr size_t MAX_META_READ = 64u * 1024u * 1024u;
    size_t read_size = std::min(file_size, MAX_META_READ);

    std::vector<uint8_t> data(read_size, 0);
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(read_size));
    size_t actually_read = static_cast<size_t>(file.gcount());
    if (actually_read < std::min(read_size, file_size)) {
        std::cerr << "[Tokenizer] Short read on GGUF metadata: "
                  << actually_read << " / " << read_size << "\n";
        return nullptr;
    }

    // Trim to what was actually read (defensive for partial reads at EOF)
    return gguf_parse_buffer(data.data(), actually_read, file_size);
    // tensor_data stays empty — caller must not use this for weight loading
}

void gguf_print_info(const GGUFile& gguf) {
    std::cout << "=== GGUF Info ===" << std::endl;
    std::cout << "Metadata entries: " << gguf.metadata.size() << std::endl;
    std::cout << "Tensor count: " << gguf.tensors.size() << std::endl;
    std::cout << "Tensor data offset: " << gguf.tensor_data_offset << std::endl;
    std::cout << "Tensor data size: " << gguf.tensor_data.size() << std::endl;

    if (gguf.has_metadata("general.architecture")) {
        std::cout << "Architecture: " << gguf.get_metadata("general.architecture").as_string() << std::endl;
    }
    if (gguf.has_metadata("general.name")) {
        std::cout << "Name: " << gguf.get_metadata("general.name").as_string() << std::endl;
    }
    if (gguf.has_metadata("llama.block_count")) {
        std::cout << "Block count: " << gguf.get_metadata("llama.block_count").as_u32() << std::endl;
    }
    if (gguf.has_metadata("llama.context_length")) {
        std::cout << "Context length: " << gguf.get_metadata("llama.context_length").as_u32() << std::endl;
    }
    if (gguf.has_metadata("llama.embedding_length")) {
        std::cout << "Embedding length: " << gguf.get_metadata("llama.embedding_length").as_u32() << std::endl;
    }

    std::cout << "\nTensors:" << std::endl;
    for (const auto& t : gguf.tensors) {
        std::cout << "  " << t.name;
        std::cout << " [";
        for (size_t i = 0; i < t.shape.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << t.shape[i];
        }
        std::cout << "] type=" << (int)t.type;
        std::cout << " offset=" << t.offset;
        std::cout << std::endl;
    }
}

size_t ggml_type_size(ggml_type type) {
    switch (type) {
        case ggml_type::F32:  return 4;
        case ggml_type::F16:  return 2;
        case ggml_type::Q4_0: return 2;  // 32 weights per 18 bytes ~= 2
        case ggml_type::Q4_1: return 2;
        case ggml_type::Q5_0: return 2;
        case ggml_type::Q5_1: return 2;
        case ggml_type::Q8_0: return 2;
        case ggml_type::Q8_1: return 2;
        case ggml_type::Q2_K: return 1;
        case ggml_type::Q3_K: return 1;
        case ggml_type::Q4_K: return 1;
        case ggml_type::Q5_K: return 1;
        case ggml_type::Q6_K: return 1;
        case ggml_type::Q8_K: return 1;
        case ggml_type::BF16: return 2;
        case ggml_type::I2_S: return 1;  // 2 bits per weight, packed
        default: return 1;
    }
}

size_t ggml_block_size(ggml_type type) {
    switch (type) {
        case ggml_type::F32:  return 1;
        case ggml_type::F16:  return 1;
        case ggml_type::BF16: return 1;
        case ggml_type::I2_S: return 256;  // 256 weights per block (64 bytes)
        default: return 32;  // Most K-quants use 256 or 32
    }
}

} // namespace ternative
