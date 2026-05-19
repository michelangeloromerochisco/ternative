#pragma once

#include "tensor.h"
#include <vector>
#include <cstddef>

namespace ternative {

struct ModelConfig; // fwd-decl for CUDABackend::init_gpu_activations

// Compute backend abstraction
class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;

    virtual void matmul(const Tensor& a, const Tensor& b, Tensor& out) = 0;
    virtual void rms_norm(const Tensor& x, const Tensor& weight, float eps, Tensor& out) = 0;
    virtual void rope(Tensor& q, Tensor& k, int pos, float theta) = 0;
    virtual void softmax(const Tensor& x, Tensor& out) = 0;
    virtual void relu2(const Tensor& x, Tensor& out) = 0;
    virtual void add(const Tensor& a, const Tensor& b, Tensor& out) = 0;
    virtual void mul(const Tensor& a, const Tensor& b, Tensor& out) = 0;

    // Batch GEMV: upload 'a' once, run N matmuls, sync once.
    // Default: sequential calls to matmul(). CUDA override batches kernel launches.
    virtual void matmul_batch(const Tensor& a,
                              const std::vector<const Tensor*>& bs,
                              std::vector<Tensor*>& outs) {
        for (size_t i = 0; i < bs.size(); ++i)
            matmul(a, *bs[i], *outs[i]);
    }
};

// CPU reference backend (always available)
class CPUBackend : public ComputeBackend {
public:
    void matmul(const Tensor& a, const Tensor& b, Tensor& out) override;
    void rms_norm(const Tensor& x, const Tensor& weight, float eps, Tensor& out) override;
    void rope(Tensor& q, Tensor& k, int pos, float theta) override;
    void softmax(const Tensor& x, Tensor& out) override;
    void relu2(const Tensor& x, Tensor& out) override;
    void add(const Tensor& a, const Tensor& b, Tensor& out) override;
    void mul(const Tensor& a, const Tensor& b, Tensor& out) override;
};

// CUDA backend (available if compiled with -DTERNATIVE_CUDA)
#ifdef TERNATIVE_USE_CUDA
class CUDABackend : public ComputeBackend {
public:
    CUDABackend();
    ~CUDABackend();

    // matmul: if b.on_gpu(), runs CUDA GEMV; otherwise falls through to CPU AVX2.
    void matmul(const Tensor& a, const Tensor& b, Tensor& out) override;

    // Batch GEMV: one upload, N kernels, one sync, N downloads.
    // Requires all bs[i].on_gpu() == true. Falls back to sequential matmul() otherwise.
    void matmul_batch(const Tensor& a,
                      const std::vector<const Tensor*>& bs,
                      std::vector<Tensor*>& outs) override;

    // All other ops run on CPU (they are cheap compared to GEMV).
    void rms_norm(const Tensor& x, const Tensor& weight, float eps, Tensor& out) override;
    void rope(Tensor& q, Tensor& k, int pos, float theta) override;
    void softmax(const Tensor& x, Tensor& out) override;
    void relu2(const Tensor& x, Tensor& out) override;
    void add(const Tensor& a, const Tensor& b, Tensor& out) override;
    void mul(const Tensor& a, const Tensor& b, Tensor& out) override;

    // GPU memory management ───────────────────────────────────────────────
    // Query free VRAM in bytes.
    static size_t free_vram_bytes();

    // Allocate a GPU buffer of `bytes` bytes and return the device pointer.
    // Ownership transfers to the caller; free with release_gpu_mem().
    void* alloc_gpu_mem(size_t bytes);

    // Copy `bytes` bytes from host `src` to a previously allocated device `dst`.
    void upload(void* dst, const void* src, size_t bytes);

    // Free a device buffer previously returned by alloc_gpu_mem().
    void release_gpu_mem(void* ptr);

    // ── GPU-resident activation pipeline (Phase 1) ────────────────────────
    // Allocate device + pinned staging buffers used by Model::forward_gpu().
    // Idempotent; safe to call once per model load.
    void init_gpu_activations(const ModelConfig& cfg);
    bool has_gpu_activations() const;

    // Device pointer accessors (returned device memory is owned by this backend).
    float* d_x();        // hidden state buffer [hidden_size]
    float* d_norm();     // norm/temp buffer [max(hidden, intermediate)]
    float* d_q();        // [n_heads * head_dim]
    float* d_k();        // [n_kv_heads * head_dim]
    float* d_v();        // [n_kv_heads * head_dim]
    float* d_gate();     // [intermediate_size] (also reused for attn_proj temp)
    float* d_up();       // [intermediate_size]

    // Pinned host staging buffers (allocated by init_gpu_activations).
    float* h_qkv();      // sized for (n_heads + 2*n_kv_heads) * head_dim floats
    float* h_attn();     // [hidden_size]
    float* h_logit();    // [vocab_size]

    // Element-wise GPU ops (no host transfers).
    void add_d(float* d_out, const float* d_a, const float* d_b, int N);
    void mul_d(float* d_out, const float* d_a, const float* d_b, int N);
    void relu2_d(float* d_out, const float* d_x, int N);

    // RMS norm on device tensors (eps/weight live on the device).
    void rms_norm_d(float* d_out, const float* d_x, const float* d_w,
                    float eps, int N);

    // Device-to-device GEMV: y[N] = W[N,K] × x[K]. W is F16 device-resident
    // (passed as void* to keep this header free of CUDA includes).
    void gemv_d2d(float* d_out, const float* d_in, const void* d_W_f16,
                  int K, int N);

    // INT8 variant: y[N] = scale * W_int8[N,K] × x[K]. W_int8 is INT8 device-resident.
    void gemv_d2d_int8(float* d_out, const float* d_in, const void* d_W_int8,
                       float scale, int K, int N);

    // Synchronous H<->D transfer helpers (use pinned host buffers for speed).
    void upload_pinned(float* d_dst, const float* h_src, size_t n_floats);
    void download_pinned(float* h_dst, const float* d_src, size_t n_floats);

    // cudaDeviceSynchronize() wrapper.
    void sync_d();

    // ── Phase 4: GPU-resident RoPE, KV cache, and attention ───────────────

    // Apply RoPE in-place to q [n_heads, head_dim] and k [n_kv_heads, head_dim].
    // pos: position in sequence. No sync needed after (same stream).
    void rope_inplace_d(float* d_q, float* d_k, int pos, float theta,
                        int n_heads, int n_kv_heads, int head_dim);

    // Write k and v for the current token into the GPU KV cache at position pos.
    // layer: transformer layer index. No sync needed after (same stream).
    void kv_write_d(const float* d_k, const float* d_v, int layer, int pos);

    // Single-query decode attention: reads from GPU KV cache, writes attn_out.
    // kv_len = pos + 1 (length of filled KV cache including current token).
    // d_out should have space for [n_heads * head_dim] floats.
    // No sync needed after (same stream).
    void attention_d(const float* d_q, int layer, int kv_len, float* d_out,
                     int n_heads, int n_kv_heads, int head_dim, float scale);

    // Reset the GPU KV cache to zero (call at start of each generation).
    void clear_kv_d();

    // True when GPU KV cache is allocated and has capacity >= min_seq.
    bool has_gpu_kv(int min_seq = 1) const;
    int  gpu_kv_max_seq() const;

    // Copy `n_positions × Hk` floats from host k/v buffers into the GPU KV
    // cache slot for `layer`. Used after CPU prefill to seed the GPU cache so
    // subsequent forward_gpu() calls see the prompt's k/v tensors.
    void upload_kv_layer_d(int layer, int n_positions,
                           const float* h_k, const float* h_v);

    // Download `n_floats` from the GPU KV cache slot at (layer, pos) into h_dst.
    // Debug helper.
    void download_kv_slot_d(int layer, int pos, int n_floats, float* h_dst);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    CPUBackend cpu_;   // fallback for non-GPU tensors and non-GEMV ops
};
#endif

} // namespace ternative
