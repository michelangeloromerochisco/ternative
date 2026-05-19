#include "ternative/gguf_write.h"
#include "ternative/tensor.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <string>
#include <iostream>

namespace ternative {

// ============================================================
// Low-level binary write helpers
// ============================================================

static void wbytes(FILE* f, const void* p, size_t n) {
    if (fwrite(p, 1, n, f) != n)
        throw std::runtime_error("gguf_write: I/O error");
}

static void wu8 (FILE* f, uint8_t  v) { wbytes(f, &v, 1); }
static void wu32(FILE* f, uint32_t v) { wbytes(f, &v, 4); }
static void wi32(FILE* f, int32_t  v) { wbytes(f, &v, 4); }
static void wu64(FILE* f, uint64_t v) { wbytes(f, &v, 8); }
static void wf32(FILE* f, float    v) { wbytes(f, &v, 4); }
static void wbool(FILE* f, bool    v) { uint8_t b = v ? 1 : 0; wbytes(f, &b, 1); }

static void wstring(FILE* f, const std::string& s) {
    uint64_t len = s.size();
    wu64(f, len);
    wbytes(f, s.data(), len);
}

// Pad file position to a multiple of `align`; fill with zero bytes.
static uint64_t wpad(FILE* f, uint32_t align) {
    uint64_t pos = static_cast<uint64_t>(ftell(f));
    uint64_t rem = pos % align;
    if (rem == 0) return pos;
    uint64_t pad = align - rem;
    static const uint8_t zeros[32] = {};
    wbytes(f, zeros, static_cast<size_t>(pad));
    return pos + pad;
}

// ============================================================
// Metadata key-value writers
// ============================================================

// GGUF metadata type tags
enum : uint32_t {
    MV_UINT8   = 0, MV_INT8   = 1, MV_UINT16 = 2, MV_INT16  = 3,
    MV_UINT32  = 4, MV_INT32  = 5, MV_FLOAT32= 6, MV_BOOL   = 7,
    MV_STRING  = 8, MV_ARRAY  = 9, MV_UINT64 = 10,MV_INT64  = 11,
    MV_FLOAT64 = 12
};

static void wkv_u32(FILE* f, const char* key, uint32_t v) {
    wstring(f, key); wu32(f, MV_UINT32); wu32(f, v);
}
static void wkv_i32(FILE* f, const char* key, int32_t v) {
    wstring(f, key); wu32(f, MV_INT32); wi32(f, v);
}
static void wkv_f32(FILE* f, const char* key, float v) {
    wstring(f, key); wu32(f, MV_FLOAT32); wf32(f, v);
}
static void wkv_str(FILE* f, const char* key, const std::string& v) {
    wstring(f, key); wu32(f, MV_STRING); wstring(f, v);
}
static void wkv_bool(FILE* f, const char* key, bool v) {
    wstring(f, key); wu32(f, MV_BOOL); wbool(f, v);
}

// Write a MetadataValue (for passthrough of tokenizer arrays from base GGUF)
static void write_mv_value(FILE* f, const MetadataValue& mv);

static void write_mv_value(FILE* f, const MetadataValue& mv) {
    switch (mv.type) {
        case gguf_metadata_type::UINT8:
        case gguf_metadata_type::INT8:
        case gguf_metadata_type::BOOL:
            wbytes(f, mv.raw.data(), 1); break;
        case gguf_metadata_type::UINT16:
        case gguf_metadata_type::INT16:
            wbytes(f, mv.raw.data(), 2); break;
        case gguf_metadata_type::UINT32:
        case gguf_metadata_type::INT32:
        case gguf_metadata_type::FLOAT32:
            wbytes(f, mv.raw.data(), 4); break;
        case gguf_metadata_type::UINT64:
        case gguf_metadata_type::INT64:
        case gguf_metadata_type::FLOAT64:
            wbytes(f, mv.raw.data(), 8); break;
        case gguf_metadata_type::STRING:
            wstring(f, mv.str_val); break;
        case gguf_metadata_type::ARRAY: {
            if (mv.array_val.empty()) { wu32(f, MV_UINT8); wu64(f, 0); break; }
            wu32(f, static_cast<uint32_t>(mv.array_val[0].type));
            wu64(f, static_cast<uint64_t>(mv.array_val.size()));
            for (const auto& elem : mv.array_val)
                write_mv_value(f, elem);
            break;
        }
    }
}

// Pass through a key-value pair from the base GGUFile without decoding it.
static void wkv_passthrough(FILE* f, const std::string& key, const MetadataValue& mv) {
    wstring(f, key);
    wu32(f, static_cast<uint32_t>(mv.type));
    write_mv_value(f, mv);
}

// ============================================================
// Tokenizer keys to passthrough from base GGUF (in priority order)
// ============================================================

static const char* TOKENIZER_KEYS[] = {
    "tokenizer.ggml.model",
    "tokenizer.ggml.pre",
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.scores",
    "tokenizer.ggml.merges",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.unknown_token_id",
    "tokenizer.chat_template",
    nullptr
};

static int count_tokenizer_keys(const GGUFile& g) {
    int n = 0;
    for (int i = 0; TOKENIZER_KEYS[i]; ++i)
        if (g.has_metadata(TOKENIZER_KEYS[i])) ++n;
    return n;
}

// ============================================================
// Tensor plan
// ============================================================

struct TensorEntry {
    std::string name;
    std::vector<uint64_t> dims;  // GGUF order = ternative order (innermost first)
    const uint8_t* data;
    uint64_t nbytes;
    uint64_t offset_in_data;    // 32-byte aligned, relative to tensor data section
};

static std::string blk(int i, const char* suffix) {
    return "blk." + std::to_string(i) + "." + suffix;
}

static TensorEntry make_entry(const std::string& name, const Tensor& t) {
    TensorEntry e;
    e.name  = name;
    e.data  = t.data.data();
    e.nbytes = t.data.size();
    for (auto d : t.shape) e.dims.push_back(static_cast<uint64_t>(d));
    // nbytes sanity check: all cached tensors should be F16
    if (t.type == ggml_type::F16) {
        uint64_t expected = t.n_elements() * 2;
        if (e.nbytes != expected)
            throw std::runtime_error("gguf_write: tensor " + name +
                " has unexpected byte count");
    }
    return e;
}

static std::vector<TensorEntry> build_plan(const Model& m) {
    std::vector<TensorEntry> plan;
    plan.reserve(11 * m.config.num_layers + 3);

    // Global tensors
    plan.push_back(make_entry("token_embd.weight", m.tok_embd));
    plan.push_back(make_entry("output_norm.weight", m.output_norm));
    // output.weight omitted when tie_embeddings (llama.cpp falls back to tok_embd)

    // Per-layer tensors (canonical llama.cpp order)
    for (int i = 0; i < m.config.num_layers; ++i) {
        const auto& lw = m.layers[i];
        plan.push_back(make_entry(blk(i, "attn_norm.weight"),   lw.attn_norm));
        plan.push_back(make_entry(blk(i, "attn_q.weight"),      lw.wq));
        plan.push_back(make_entry(blk(i, "attn_k.weight"),      lw.wk));
        plan.push_back(make_entry(blk(i, "attn_v.weight"),      lw.wv));
        plan.push_back(make_entry(blk(i, "attn_output.weight"), lw.wo));
        plan.push_back(make_entry(blk(i, "ffn_norm.weight"),    lw.ffn_norm));
        plan.push_back(make_entry(blk(i, "ffn_gate.weight"),    lw.w_gate));
        plan.push_back(make_entry(blk(i, "ffn_up.weight"),      lw.w_up));
        plan.push_back(make_entry(blk(i, "ffn_down.weight"),    lw.w_down));
        // attn_sub_norm and ffn_sub_norm are BitNet-only; dropped for llama.cpp
    }

    // Compute aligned offsets within the tensor data section
    constexpr uint32_t ALIGN = 32;
    uint64_t offset = 0;
    for (auto& e : plan) {
        e.offset_in_data = offset;
        // Next entry starts after this data, padded to ALIGN
        uint64_t end = offset + e.nbytes;
        uint64_t rem = end % ALIGN;
        offset = (rem == 0) ? end : end + (ALIGN - rem);
    }

    return plan;
}

// ============================================================
// Top-level export
// ============================================================

void gguf_write_f16(const Model&      model,
                    const GGUFile&    base_meta,
                    const std::string& path,
                    bool verbose)
{
    if (model.layers.empty())
        throw std::runtime_error("gguf_write: model has no layers (not loaded?)");
    if (model.layers[0].wq.type != ggml_type::F16)
        throw std::runtime_error("gguf_write: weights must be F16 (load from .tvcache first)");

    if (verbose)
        std::cout << "[Export] Building tensor plan (" << model.config.num_layers
                  << " layers)...\n" << std::flush;

    auto plan = build_plan(model);

    // Count metadata entries
    const ModelConfig& cfg = model.config;
    const int  n_fixed_kv  = 14;                            // architecture + config keys
    const int  n_tok_kv    = count_tokenizer_keys(base_meta);
    const uint64_t n_meta  = static_cast<uint64_t>(n_fixed_kv + n_tok_kv);
    const uint64_t n_tensors = static_cast<uint64_t>(plan.size());

    if (verbose)
        std::cout << "[Export] Writing " << n_tensors << " tensors, "
                  << n_meta << " metadata keys to: " << path << "\n" << std::flush;

    FILE* f = fopen(path.c_str(), "wb");
    if (!f)
        throw std::runtime_error("gguf_write: cannot open " + path);

    try {
        // ── 1. Header ──────────────────────────────────────────────────
        const uint32_t MAGIC   = 0x46554747u;  // "GGUF"
        const uint32_t VERSION = 3u;
        wu32(f, MAGIC);
        wu32(f, VERSION);
        wu64(f, n_tensors);
        wu64(f, n_meta);

        // ── 2. Metadata ────────────────────────────────────────────────
        // Architecture
        wkv_str(f,  "general.architecture",                    "llama");
        wkv_str(f,  "general.name",                            "orchid");
        wkv_u32(f,  "general.file_type",                       1u);  // all-F16
        wkv_u32(f,  "general.quantization_version",            2u);
        // Model dimensions
        wkv_u32(f,  "llama.block_count",                       (uint32_t)cfg.num_layers);
        wkv_u32(f,  "llama.context_length",                    (uint32_t)cfg.max_seq_len);
        wkv_u32(f,  "llama.embedding_length",                  (uint32_t)cfg.hidden_size);
        wkv_u32(f,  "llama.feed_forward_length",               (uint32_t)cfg.intermediate_size);
        wkv_u32(f,  "llama.attention.head_count",              (uint32_t)cfg.num_heads);
        wkv_u32(f,  "llama.attention.head_count_kv",           (uint32_t)cfg.num_kv_heads);
        wkv_f32(f,  "llama.attention.layer_norm_rms_epsilon",  cfg.rms_norm_eps);
        wkv_f32(f,  "llama.rope.freq_base",                    cfg.rope_theta);
        wkv_u32(f,  "llama.rope.dimension_count",              (uint32_t)cfg.head_dim);
        wkv_u32(f,  "llama.vocab_size",                        (uint32_t)cfg.vocab_size);
        // Tokenizer: verbatim passthrough from base GGUF
        for (int i = 0; TOKENIZER_KEYS[i]; ++i) {
            if (base_meta.has_metadata(TOKENIZER_KEYS[i]))
                wkv_passthrough(f, TOKENIZER_KEYS[i], base_meta.get_metadata(TOKENIZER_KEYS[i]));
        }

        // ── 3. Tensor info table ───────────────────────────────────────
        for (const auto& e : plan) {
            wstring(f, e.name);
            wu32(f, static_cast<uint32_t>(e.dims.size()));
            for (uint64_t d : e.dims) wu64(f, d);
            wu32(f, 1u);          // type = F16
            wu64(f, e.offset_in_data);
        }

        // ── 4. Align to 32 bytes → start of tensor data section ───────
        wpad(f, 32u);

        // ── 5. Tensor data ─────────────────────────────────────────────
        uint64_t written = 0;
        uint64_t total = n_tensors;
        for (const auto& e : plan) {
            wbytes(f, e.data, static_cast<size_t>(e.nbytes));
            wpad(f, 32u);
            if (verbose && (++written % 50 == 0 || written == total))
                std::cout << "[Export] " << written << "/" << total << " tensors\r" << std::flush;
        }
        if (verbose) std::cout << "\n[Export] Done. File: " << path << "\n";

    } catch (...) {
        fclose(f);
        throw;
    }
    fclose(f);
}

} // namespace ternative
