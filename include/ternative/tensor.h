#pragma once

#include "gguf.h"

#include <vector>
#include <cstdint>
#include <cstddef>
#include <string>
#include <memory>
#include <cassert>

namespace ternative {

// Half-precision float (F16) — simple struct, conversion to/from float
struct f16_t {
    uint16_t bits;
    f16_t() = default;
    explicit f16_t(uint16_t b) : bits(b) {}
    explicit f16_t(float f);
    float to_float() const;
};

// Brain float (BF16) — 1 sign + 8 exp + 7 mantissa, same range as F32
struct bf16_t {
    uint16_t bits;
    bf16_t() = default;
    explicit bf16_t(uint16_t b) : bits(b) {}
    explicit bf16_t(float f);
    float to_float() const;
};

// Tensor storage — owns its CPU memory; gpu_ptr is non-owning (managed by CUDABackend)
struct Tensor {
    std::vector<int64_t> shape;
    ggml_type type = ggml_type::F32;
    std::vector<uint8_t> data;
    float scale = 1.0f;  // Per-tensor scale for quantized types

    // Non-owning pointer to GPU-resident copy of this tensor's data (void* to avoid
    // including CUDA headers here).  Set by CUDABackend::offload_layer(); null = CPU only.
    void* gpu_ptr = nullptr;
    bool on_gpu() const { return gpu_ptr != nullptr; }
    bool gpu_is_int8 = false;   // true when gpu_ptr holds INT8 data (scale stored in Tensor::scale)

    Tensor() = default;
    Tensor(std::vector<int64_t> s, ggml_type t);

    size_t n_elements() const;
    size_t n_bytes() const;
    size_t element_size() const;

    // Typed access (unsafe — caller must ensure type matches)
    template<typename T>
    T* ptr() {
        return reinterpret_cast<T*>(data.data());
    }
    template<typename T>
    const T* ptr() const {
        return reinterpret_cast<const T*>(data.data());
    }

    // Direct float pointer access (for BLAS)
    float* f32_data() { return reinterpret_cast<float*>(data.data()); }
    const float* f32_data() const { return reinterpret_cast<const float*>(data.data()); }

    // Float access helpers
    float get_float(size_t idx) const;
    void set_float(size_t idx, float val);

    // Create a new tensor with same shape but different type
    Tensor clone_empty(ggml_type new_type) const;

    // Create a view (shares data, different shape)
    Tensor view(const std::vector<int64_t>& new_shape) const;
};

// ===== Basic tensor ops =====

// Element-wise add: out = a + b
void tensor_add(const Tensor& a, const Tensor& b, Tensor& out);

// Element-wise mul: out = a * b
void tensor_mul(const Tensor& a, const Tensor& b, Tensor& out);

// Matrix multiply: out = a @ b (standard GEMM, a=[M,K], b=[K,N], out=[M,N])
// For weights, b follows GGUF convention: [in_features, out_features]
void tensor_matmul(const Tensor& a, const Tensor& b, Tensor& out);

// RMSNorm: out = x / sqrt(mean(x^2) + eps) * weight
void tensor_rms_norm(const Tensor& x, const Tensor& weight, float eps, Tensor& out);

// Softmax over last dimension
void tensor_softmax(const Tensor& x, Tensor& out);

// ReLU squared: out = max(0, x)^2
void tensor_relu2(const Tensor& x, Tensor& out);

// RoPE (Rotary Position Embedding) — applied in-place to Q and K
// q shape: [n_heads, seq_len, head_dim]
// k shape: [n_kv_heads, seq_len, head_dim]
void tensor_rope(Tensor& q, Tensor& k, int pos, float theta);

// Batched RoPE for prefill — q/k in [seq_len, n_heads/n_kv_heads, head_dim] row-major layout
void tensor_rope_batched(Tensor& q, Tensor& k, int start_pos, float theta);

// Quantize F32/F16 tensor to I2_S (ternary {-1, 0, +1})
// Returns a new tensor with type I2_S
Tensor tensor_quantize_i2_s(const Tensor& src);

// Dequantize I2_S tensor to F32
// dst must be pre-allocated with shape matching src and type F32
void tensor_dequantize_i2_s(const Tensor& src, Tensor& dst);

// Dequantize I2_S to a temporary float buffer (for reference matmul)
std::vector<float> tensor_dequantize_i2_s_to_floats(const Tensor& src);

} // namespace ternative
