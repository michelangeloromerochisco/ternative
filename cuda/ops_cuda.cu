// ternative.cpp — CUDA backend
// All GEMV ops for GPU-resident F16 weight matrices.
// Non-GEMV ops (RMS norm, RoPE, softmax, add, mul) fall through to CPUBackend.
//
// Design: partial layer offload.
//   - At startup, Model::load() queries free VRAM and offloads as many
//     transformer layers as fit (RTX 3050 4 GB → typically 22–26 layers).
//   - Each offloaded Tensor has tensor.gpu_ptr set to its CUDA device pointer.
//   - CUDABackend::matmul() checks b.on_gpu() and routes accordingly.
//   - Transfer overhead (10 KB activation per boundary) is negligible vs
//     the 29–34 ms saved by running GEMVs at 112 GB/s instead of ~40 GB/s.

#include "ternative/ops.h"
#include "ternative/tensor.h"
#include "ternative/model.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <algorithm>

// ============================================================
// Error checking
// ============================================================

#define CUDA_CHECK(call) do {                                           \
    cudaError_t _e = (call);                                            \
    if (_e != cudaSuccess) {                                            \
        throw std::runtime_error(std::string("[CUDA] ") +              \
            cudaGetErrorString(_e) + " at " __FILE__ ":" +             \
            std::to_string(__LINE__));                                  \
    }                                                                   \
} while(0)

// ============================================================
// GEMV kernel: y[N] = W[N,K] × x[K]
//   W: F16 device, x: F32 device, y: F32 device
//   One warp (32 threads) per output element for coalesced reads.
//   4 warps per block (128 threads) for occupancy.
// ============================================================

__global__ void gemv_f16_f32_kernel(
        const __half* __restrict__ W,
        const float*  __restrict__ x,
        float* __restrict__ y,
        int K, int N)
{
    // warp_id = output index; lane = thread within warp
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane    = threadIdx.x & 31;
    if (warp_id >= N) return;

    const __half* w = W + (int64_t)warp_id * K;
    float sum = 0.f;

    // Stride by warp width (32) so all lanes in a warp read consecutive addresses
    // → one 64-byte cache line satisfies 32 F16 loads (perfect coalescing).
    for (int k = lane; k < K; k += 32)
        sum = __fmaf_rn(__half2float(__ldg(w + k)), x[k], sum);

    // Warp shuffle reduction
#pragma unroll
    for (int s = 16; s > 0; s >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, s);

    if (lane == 0) y[warp_id] = sum;
}

// ============================================================
// INT8 GEMV: y[N] = scale * W_int8[N,K] × x[K]
//   W_int8: INT8 device [N,K] column-major (GGUF layout: W[k,n] = data[k + n*K])
//   x: F32 device [K]
//   y: F32 device [N]
//   One warp per output element (same layout as F16 kernel).
// ============================================================

__global__ void gemv_int8_f32_kernel(
        const int8_t* __restrict__ W,
        const float*  __restrict__ x,
        float* __restrict__ y,
        float scale,
        int K, int N)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane    = threadIdx.x & 31;
    if (warp_id >= N) return;

    const int8_t* w = W + (int64_t)warp_id * K;
    float sum = 0.f;

    for (int k = lane; k < K; k += 32)
        sum = __fmaf_rn((float)__ldg(w + k), x[k], sum);

#pragma unroll
    for (int s = 16; s > 0; s >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, s);

    if (lane == 0) y[warp_id] = sum * scale;
}

// ============================================================
// Phase 1 GPU-resident element-wise / norm kernels
// ============================================================

__global__ void add_d_kernel(const float* __restrict__ a,
                             const float* __restrict__ b,
                             float* __restrict__ out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = a[i] + b[i];
}

__global__ void mul_d_kernel(const float* __restrict__ a,
                             const float* __restrict__ b,
                             float* __restrict__ out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = a[i] * b[i];
}

__global__ void relu2_d_kernel(const float* __restrict__ x,
                               float* __restrict__ out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float v = fmaxf(x[i], 0.0f);
        out[i] = v * v;
    }
}

// RMS norm: out[i] = x[i] * rsqrt(mean(x^2) + eps) * w[i].
// One block, 256 threads, shared memory reduction over N (N <= ~7K).
__global__ void rms_norm_d_kernel(const float* __restrict__ x,
                                  const float* __restrict__ w,
                                  float eps,
                                  float* __restrict__ out, int N) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;

    float ss = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float v = x[i];
        ss += v * v;
    }
    sdata[tid] = ss;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float inv_rms = rsqrtf(sdata[0] / (float)N + eps);
    for (int i = tid; i < N; i += blockDim.x) {
        out[i] = x[i] * inv_rms * w[i];
    }
}

// ============================================================
// Phase 4 GPU-resident RoPE, KV cache, decode attention kernels
// ============================================================

// Apply rotary embedding in-place to q [n_heads, head_dim] and k [n_kv_heads, head_dim].
// Uses the "NeoX split-halves" convention to match CPU `tensor_rope`:
//   pair (ic, ic + head_dim/2), with freq = theta^(-(2*ic)/head_dim).
// Launch: <<<n_heads + n_kv_heads, head_dim/2>>>
__global__ void rope_inplace_kernel(
        float* __restrict__ q, float* __restrict__ k,
        int pos, float theta,
        int n_heads, int n_kv_heads, int head_dim) {
    int head = blockIdx.x;
    int ic   = threadIdx.x;          // 0 .. head_dim/2 - 1
    int total_heads = n_heads + n_kv_heads;
    int half = head_dim / 2;
    if (head >= total_heads || ic >= half) return;

    bool is_q = (head < n_heads);
    float* vec = is_q ? (q + head * head_dim)
                      : (k + (head - n_heads) * head_dim);

    // Match CPU: freq uses i = 2*ic (the even index) → exponent = (2*ic)/head_dim.
    float freq  = 1.0f / powf(theta, 2.0f * (float)ic / (float)head_dim);
    float angle = (float)pos * freq;
    float cv = __cosf(angle), sv = __sinf(angle);
    float v0 = vec[ic];
    float v1 = vec[ic + half];
    vec[ic]        = v0 * cv - v1 * sv;
    vec[ic + half] = v0 * sv + v1 * cv;
}

// Write current-token k and v into the GPU KV cache at position `pos`.
__global__ void kv_write_kernel(
        const float* __restrict__ k, const float* __restrict__ v,
        float* __restrict__ k_cache_layer, float* __restrict__ v_cache_layer,
        int pos, int Hk, int /*gpu_max_seq*/) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Hk) return;
    k_cache_layer[(int64_t)pos * Hk + i] = k[i];
    v_cache_layer[(int64_t)pos * Hk + i] = v[i];
}

// Single-query causal attention for decode (one new token).
//   Launch: <<<n_heads, 128, kv_len * sizeof(float)>>>
__global__ void decode_attention_kernel(
        const float* __restrict__ q,
        const float* __restrict__ k_cache_layer,
        const float* __restrict__ v_cache_layer,
        float* __restrict__ out,
        int n_heads, int /*n_kv_heads*/, int head_dim,
        int kv_len, int Hk, float scale, int group) {
    extern __shared__ float scores[];   // [kv_len]

    int h   = blockIdx.x;               // query head
    int tid = threadIdx.x;
    int kv_h = h / group;               // corresponding KV head (GQA)

    const float* q_h   = q   + h * head_dim;
    float*       out_h = out + h * head_dim;

    // Step 1: QKᵀ scores (parallel over t).
    for (int t = tid; t < kv_len; t += blockDim.x) {
        const float* k_t = k_cache_layer + (int64_t)t * Hk + kv_h * head_dim;
        float dot = 0.0f;
        // head_dim is a multiple of 4 for BitNet (128); unroll 4-wide.
        for (int d = 0; d < head_dim; d += 4) {
            dot += q_h[d]     * __ldg(k_t + d);
            dot += q_h[d + 1] * __ldg(k_t + d + 1);
            dot += q_h[d + 2] * __ldg(k_t + d + 2);
            dot += q_h[d + 3] * __ldg(k_t + d + 3);
        }
        scores[t] = dot * scale;
    }
    __syncthreads();

    // Step 2: softmax (sequential by thread 0 — kv_len ≤ gpu_max_seq, small).
    if (tid == 0) {
        float mx = scores[0];
        for (int t = 1; t < kv_len; ++t) if (scores[t] > mx) mx = scores[t];
        float sm = 0.0f;
        for (int t = 0; t < kv_len; ++t) {
            scores[t] = __expf(scores[t] - mx);
            sm += scores[t];
        }
        float inv = 1.0f / sm;
        for (int t = 0; t < kv_len; ++t) scores[t] *= inv;
    }
    __syncthreads();

    // Step 3: weighted sum of V (parallel over d).
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < kv_len; ++t) {
            acc += scores[t] * __ldg(v_cache_layer + (int64_t)t * Hk
                                     + kv_h * head_dim + d);
        }
        out_h[d] = acc;
    }
}

// ============================================================
// Batched GEMV: y[M,N] = W[N,K] × x[M,K]
//   W: F16 device (GGUF layout: [K,N] column-major, i.e. W[k,n] = data[k + n*K])
//   x: F32 device [M,K] row-major
//   y: F32 device [M,N] row-major
//   One warp per (m, n) output element. gridDim.x = ceil(M*N / 4).
// ============================================================

__global__ void gemm_f16_f32_kernel(
        const __half* __restrict__ W,
        const float*  __restrict__ x,
        float* __restrict__ y,
        int M, int K, int N)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane    = threadIdx.x & 31;
    if (warp_id >= M * N) return;

    const int m = warp_id / N;   // which input row
    const int n = warp_id % N;   // which output neuron

    const __half* w  = W + (int64_t)n * K;   // column n of W (GGUF: [K,N] col-major)
    const float*  xi = x + (int64_t)m * K;   // row m of activations

    float sum = 0.f;
    for (int k = lane; k < K; k += 32)
        sum = __fmaf_rn(__half2float(__ldg(w + k)), xi[k], sum);

#pragma unroll
    for (int s = 16; s > 0; s >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, s);

    if (lane == 0) y[(int64_t)m * N + n] = sum;
}

// ============================================================
// CUDABackend::Impl
// ============================================================

namespace ternative {

// Maximum batch size for matmul_batch (QKV=3, Gate+Up=2).
static constexpr int MAX_BATCH_OUTS = 3;
// Max output elements: intermediate_size=6912 (largest single GEMV output).
static constexpr size_t MAX_OUT_ELEMS = 6912;
// Max input elements: intermediate_size=6912 (ffn_in for Gate+Up batch).
static constexpr size_t MAX_ACT_ELEMS = 128256; // vocab logits = largest activation

class CUDABackend::Impl {
public:
    Impl() {
        // Device buffers: one shared input + one per batch output slot.
        CUDA_CHECK(cudaMalloc(&d_act_in_,  MAX_ACT_ELEMS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_act_out_, MAX_ACT_ELEMS * sizeof(float)));
        for (int i = 0; i < MAX_BATCH_OUTS; ++i)
            CUDA_CHECK(cudaMalloc(&d_batch_out_[i], MAX_OUT_ELEMS * sizeof(float)));

        // Pinned host staging buffers: eliminate the pageable→device double-copy.
        // For 10-27 KB activations this saves ~40-80µs per transfer on Windows.
        CUDA_CHECK(cudaMallocHost(&h_act_in_,  MAX_ACT_ELEMS * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&h_act_out_, MAX_ACT_ELEMS * sizeof(float)));
        for (int i = 0; i < MAX_BATCH_OUTS; ++i)
            CUDA_CHECK(cudaMallocHost(&h_batch_out_[i], MAX_OUT_ELEMS * sizeof(float)));
    }

    ~Impl() {
        cudaFree(d_act_in_);
        cudaFree(d_act_out_);
        for (int i = 0; i < MAX_BATCH_OUTS; ++i) cudaFree(d_batch_out_[i]);
        cudaFreeHost(h_act_in_);
        cudaFreeHost(h_act_out_);
        for (int i = 0; i < MAX_BATCH_OUTS; ++i) cudaFreeHost(h_batch_out_[i]);

        // Phase 1 buffers
        if (d_x_res_)      cudaFree(d_x_res_);
        if (d_norm_out_)   cudaFree(d_norm_out_);
        if (d_q_res_)      cudaFree(d_q_res_);
        if (d_k_res_)      cudaFree(d_k_res_);
        if (d_v_res_)      cudaFree(d_v_res_);
        if (d_gate_res_)   cudaFree(d_gate_res_);
        if (d_up_res_)     cudaFree(d_up_res_);
        if (h_qkv_stage_)  cudaFreeHost(h_qkv_stage_);
        if (h_attn_stage_) cudaFreeHost(h_attn_stage_);
        if (h_logit_stage_)cudaFreeHost(h_logit_stage_);

        // Phase 4 KV cache
        if (d_k_cache_) cudaFree(d_k_cache_);
        if (d_v_cache_) cudaFree(d_v_cache_);

        // Phase 3 fix: pre-allocated GEMM buffers
        if (d_gemm_in_)  cudaFree(d_gemm_in_);
        if (d_gemm_out_) cudaFree(d_gemm_out_);
    }

    void init_gpu_activations(int hidden, int n_heads, int n_kv_heads,
                              int head_dim, int intermediate, int vocab,
                              int n_layers) {
        if (gpu_act_ready_) return;
        int Hq = n_heads    * head_dim;
        int Hk = n_kv_heads * head_dim;
        CUDA_CHECK(cudaMalloc(&d_x_res_,    hidden * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_norm_out_, std::max(hidden, intermediate) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_q_res_,    Hq * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_k_res_,    Hk * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_v_res_,    Hk * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gate_res_, intermediate * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_up_res_,   intermediate * sizeof(float)));

        // h_qkv_stage holds Q (Hq) + K (Hk) + V (Hk) packed contiguously.
        CUDA_CHECK(cudaMallocHost(&h_qkv_stage_, (Hq + 2 * Hk) * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&h_attn_stage_, std::max(hidden, Hq) * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&h_logit_stage_, vocab * sizeof(float)));

        // ── Phase 4: GPU KV cache ────────────────────────────────────────
        // 30 layers × 1024 seq × 640 floats × 2 (k+v) = ~157 MB.
        const int GPU_MAX_SEQ = 1024;
        gpu_max_seq_  = GPU_MAX_SEQ;
        n_layers_gpu_ = n_layers;
        Hk_           = Hk;

        size_t kv_elems = (size_t)n_layers * GPU_MAX_SEQ * Hk;
        CUDA_CHECK(cudaMalloc(&d_k_cache_, kv_elems * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_v_cache_, kv_elems * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_k_cache_, 0, kv_elems * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_v_cache_, 0, kv_elems * sizeof(float)));

        // Pre-allocate GEMM buffers (Phase 3 fix — eliminates per-call cudaMalloc)
        CUDA_CHECK(cudaMalloc(&d_gemm_in_,
            (size_t)GEMM_MAX_M * GEMM_MAX_SIZE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gemm_out_,
            (size_t)GEMM_MAX_M * GEMM_MAX_SIZE * sizeof(float)));
        gemm_bufs_ready_ = true;

        gpu_act_ready_ = true;
    }

    // Single GEMV with pinned staging. Used for wo and w_down (unique inputs).
    void gemv(const Tensor& a, const Tensor& b, Tensor& out) {
        const int M = (int)a.shape[0];
        const int K = (int)a.shape[1];
        const int N = (int)b.shape[1];

        const float*  x_cpu = a.ptr<float>();
        const __half* W_gpu = static_cast<const __half*>(b.gpu_ptr);
        float*        y_cpu = out.ptr<float>();

        if (M == 1) {
            // Upload via pinned staging → DMA is direct, avoids internal double-copy.
            std::memcpy(h_act_in_, x_cpu, K * sizeof(float));
            CUDA_CHECK(cudaMemcpy(d_act_in_, h_act_in_, K * sizeof(float), cudaMemcpyHostToDevice));

            const int blocks = (N + 3) / 4;
            gemv_f16_f32_kernel<<<blocks, 128>>>(W_gpu, d_act_in_, d_act_out_, K, N);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // Download via pinned staging.
            CUDA_CHECK(cudaMemcpy(h_act_out_, d_act_out_, N * sizeof(float), cudaMemcpyDeviceToHost));
            std::memcpy(y_cpu, h_act_out_, N * sizeof(float));
        } else {
            // Batched prefill: row-by-row (infrequent; not performance-critical).
            for (int m = 0; m < M; ++m) {
                const float* xm = x_cpu + m * K;
                float*       ym = y_cpu + m * N;
                std::memcpy(h_act_in_, xm, K * sizeof(float));
                CUDA_CHECK(cudaMemcpy(d_act_in_, h_act_in_, K * sizeof(float), cudaMemcpyHostToDevice));
                const int blocks = (N + 3) / 4;
                gemv_f16_f32_kernel<<<blocks, 128>>>(W_gpu, d_act_in_, d_act_out_, K, N);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(h_act_out_, d_act_out_, N * sizeof(float), cudaMemcpyDeviceToHost));
                std::memcpy(ym, h_act_out_, N * sizeof(float));
            }
        }
    }

    // Batched GEMV for prefill (M > 1). Single upload → single kernel → single download.
    // Uses pre-allocated device buffers to avoid per-call cudaMalloc/cudaFree overhead
    // (~5ms each on Windows WDDM — 420 allocs per prefill = ~2s wasted).
    void gemm(const Tensor& a, const Tensor& b, Tensor& out) {
        const int M = (int)a.shape[0];
        const int K = (int)a.shape[1];
        const int N = (int)b.shape[1];

        const float*  x_cpu = a.ptr<float>();
        float*        y_cpu = out.ptr<float>();

        const size_t x_bytes = (size_t)M * K * sizeof(float);
        const size_t y_bytes = (size_t)M * N * sizeof(float);

        // Use pre-allocated buffers when available (always after init_gpu_activations).
        // Fall back to dynamic allocation only during the brief window before init.
        float* d_x;
        float* d_y;
        bool dynamic = !gemm_bufs_ready_ ||
                       M > GEMM_MAX_M || K > GEMM_MAX_SIZE || N > GEMM_MAX_SIZE;
        if (dynamic) {
            CUDA_CHECK(cudaMalloc(&d_x, x_bytes));
            CUDA_CHECK(cudaMalloc(&d_y, y_bytes));
        } else {
            d_x = d_gemm_in_;
            d_y = d_gemm_out_;
        }

        // Get correct weight pointer (F16 or INT8)
        const void* W_raw = b.gpu_ptr;

        CUDA_CHECK(cudaMemcpy(d_x, x_cpu, x_bytes, cudaMemcpyHostToDevice));

        if (!b.gpu_is_int8) {
            const __half* W_gpu = static_cast<const __half*>(W_raw);
            const int total_warps = M * N;
            const int blocks = (total_warps + 3) / 4;
            gemm_f16_f32_kernel<<<blocks, 128>>>(W_gpu, d_x, d_y, M, K, N);
        } else {
            // INT8 weights: use row-by-row INT8 GEMV (rare path for INT8 layers during prefill)
            const int8_t* W_i8 = static_cast<const int8_t*>(W_raw);
            const float   scale = b.scale;
            const int blocks = (N + 3) / 4;
            for (int m = 0; m < M; ++m) {
                gemv_int8_f32_kernel<<<blocks, 128>>>(W_i8, d_x + m * K,
                                                      d_y + m * N, scale, K, N);
            }
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(y_cpu, d_y, y_bytes, cudaMemcpyDeviceToHost));

        if (dynamic) { cudaFree(d_x); cudaFree(d_y); }
    }

    // Batch GEMV: same input 'a', up to MAX_BATCH_OUTS outputs.
    // One upload → N kernels (no sync between) → one sync → N downloads.
    // Reduces WDDM sync overhead by (N-1) syncs compared to N sequential gemv() calls.
    void gemv_batch(const Tensor& a,
                    const std::vector<const Tensor*>& bs,
                    std::vector<Tensor*>& outs) {
        const int count = (int)bs.size();
        if (count == 0) return;
        if (count > MAX_BATCH_OUTS) {
            // Fallback: sequential (should not happen given current call sites).
            for (int i = 0; i < count; ++i) gemv(*outs[i], *bs[i], *outs[i]);
            return;
        }
        // Verify all weights are GPU-resident (should always be true for offloaded layers).
        for (int i = 0; i < count; ++i) {
            if (!bs[i]->on_gpu()) {
                for (int j = 0; j < count; ++j) gemv(a, *bs[j], *outs[j]);
                return;
            }
        }

        const int K = (int)a.shape[1]; // Input dimension
        const float* x_cpu = a.ptr<float>();

        // Upload input activation once via pinned staging.
        std::memcpy(h_act_in_, x_cpu, K * sizeof(float));
        CUDA_CHECK(cudaMemcpy(d_act_in_, h_act_in_, K * sizeof(float), cudaMemcpyHostToDevice));

        // Launch N kernels on the same stream without syncing between them.
        // CUDA stream ensures sequential execution; each kernel writes to its own d_batch_out_[i].
        for (int i = 0; i < count; ++i) {
            const int N = (int)bs[i]->shape[1];
            const __half* W_gpu = static_cast<const __half*>(bs[i]->gpu_ptr);
            const int blocks = (N + 3) / 4;
            gemv_f16_f32_kernel<<<blocks, 128>>>(W_gpu, d_act_in_, d_batch_out_[i], K, N);
            CUDA_CHECK(cudaGetLastError());
        }

        // One sync for all N kernels.
        CUDA_CHECK(cudaDeviceSynchronize());

        // Download all outputs.
        for (int i = 0; i < count; ++i) {
            const int N = (int)bs[i]->shape[1];
            float* y_cpu = outs[i]->ptr<float>();
            CUDA_CHECK(cudaMemcpy(h_batch_out_[i], d_batch_out_[i], N * sizeof(float), cudaMemcpyDeviceToHost));
            std::memcpy(y_cpu, h_batch_out_[i], N * sizeof(float));
        }
    }

    // INT8 GEMV: y[N] = scale * W_int8[N,K] × x[K]
    void gemv_d2d_int8(float* d_out, const float* d_in, const void* d_W_int8,
                       float scale, int K, int N) {
        const int8_t* W = static_cast<const int8_t*>(d_W_int8);
        const int blocks = (N + 3) / 4;
        gemv_int8_f32_kernel<<<blocks, 128>>>(W, d_in, d_out, scale, K, N);
        CUDA_CHECK(cudaGetLastError());
    }

    // Device buffers
    float*  d_act_in_  = nullptr;
    float*  d_act_out_ = nullptr;
    float*  d_batch_out_[MAX_BATCH_OUTS] = {};

    // Pinned host staging buffers
    float*  h_act_in_  = nullptr;
    float*  h_act_out_ = nullptr;
    float*  h_batch_out_[MAX_BATCH_OUTS] = {};

    // ── Phase 1: GPU-resident decode pipeline ───────────────────────────
    bool   gpu_act_ready_ = false;
    float* d_x_res_       = nullptr;
    float* d_norm_out_    = nullptr;
    float* d_q_res_       = nullptr;
    float* d_k_res_       = nullptr;
    float* d_v_res_       = nullptr;
    float* d_gate_res_    = nullptr;
    float* d_up_res_      = nullptr;
    float* h_qkv_stage_   = nullptr;
    float* h_attn_stage_  = nullptr;
    float* h_logit_stage_ = nullptr;

    // ── Phase 4: GPU-resident KV cache ──────────────────────────────────
    float* d_k_cache_     = nullptr;
    float* d_v_cache_     = nullptr;
    int    gpu_max_seq_   = 0;
    int    n_layers_gpu_  = 0;
    int    Hk_            = 0;

    // ── Phase 3 fix: pre-allocated prefill GEMM buffers ─────────────────
    // Eliminates per-call cudaMalloc/cudaFree which cost ~5 ms each on
    // Windows WDDM — 420 allocations per 30-token prefill = ~2 s overhead.
    // Sized for GEMM_MAX_M rows × GEMM_MAX_SIZE cols.
    static constexpr int GEMM_MAX_M    = 2048;  // max prefill sequence length
    static constexpr int GEMM_MAX_SIZE = 6912;  // max(intermediate_size, hidden_size)
    float* d_gemm_in_  = nullptr;   // [GEMM_MAX_M × GEMM_MAX_SIZE] device
    float* d_gemm_out_ = nullptr;   // [GEMM_MAX_M × GEMM_MAX_SIZE] device
    bool   gemm_bufs_ready_ = false;
};

// ============================================================
// CUDABackend public interface
// ============================================================

CUDABackend::CUDABackend() : impl_(std::make_unique<Impl>()) {}
CUDABackend::~CUDABackend() = default;

void CUDABackend::matmul(const Tensor& a, const Tensor& b, Tensor& out) {
    if (b.on_gpu() && !b.gpu_is_int8 && a.type == ggml_type::F32
                   && b.type == ggml_type::F16 && out.type == ggml_type::F32) {
        const int M = (int)a.shape[0];
        if (M == 1) {
            impl_->gemv(a, b, out);   // decode path: single token, fast GEMV
        } else {
            impl_->gemm(a, b, out);   // prefill path: batched GEMV, single upload/download
        }
    } else if (b.on_gpu() && b.gpu_is_int8 && a.type == ggml_type::F32
                          && out.type == ggml_type::F32) {
        // INT8 GEMV/GEMM via pinned-staging upload + per-row kernel + download.
        const int M = (int)a.shape[0];
        const int K = (int)a.shape[1];
        const int N = (int)b.shape[1];
        const float* x_cpu = a.ptr<float>();
        float*       y_cpu = out.ptr<float>();
        for (int m = 0; m < M; ++m) {
            std::memcpy(impl_->h_act_in_, x_cpu + (size_t)m * K, K * sizeof(float));
            CUDA_CHECK(cudaMemcpy(impl_->d_act_in_, impl_->h_act_in_,
                                  K * sizeof(float), cudaMemcpyHostToDevice));
            impl_->gemv_d2d_int8(impl_->d_act_out_, impl_->d_act_in_,
                                 b.gpu_ptr, b.scale, K, N);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(impl_->h_act_out_, impl_->d_act_out_,
                                  N * sizeof(float), cudaMemcpyDeviceToHost));
            std::memcpy(y_cpu + (size_t)m * N, impl_->h_act_out_, N * sizeof(float));
        }
    } else {
        cpu_.matmul(a, b, out);
    }
}

void CUDABackend::matmul_batch(const Tensor& a,
                               const std::vector<const Tensor*>& bs,
                               std::vector<Tensor*>& outs) {
    // Delegate to the batch-optimized path if all weights are GPU F16 (not INT8).
    bool all_f16_gpu = !bs.empty() && a.type == ggml_type::F32;
    for (size_t i = 0; all_f16_gpu && i < bs.size(); ++i) {
        if (!bs[i]->on_gpu() || bs[i]->gpu_is_int8 || bs[i]->type != ggml_type::F16)
            all_f16_gpu = false;
    }
    if (all_f16_gpu) {
        impl_->gemv_batch(a, bs, outs);
    } else {
        // Fallback: per-tensor matmul (handles INT8 GPU path, F16 GPU path, and CPU).
        for (size_t i = 0; i < bs.size(); ++i)
            matmul(a, *bs[i], *outs[i]);
    }
}

// All non-GEMV ops delegate to CPUBackend — they are tiny vs GEMV.
void CUDABackend::rms_norm(const Tensor& x, const Tensor& w, float eps, Tensor& out) {
    cpu_.rms_norm(x, w, eps, out);
}
void CUDABackend::rope(Tensor& q, Tensor& k, int pos, float theta) {
    cpu_.rope(q, k, pos, theta);
}
void CUDABackend::softmax(const Tensor& x, Tensor& out) {
    cpu_.softmax(x, out);
}
void CUDABackend::relu2(const Tensor& x, Tensor& out) {
    cpu_.relu2(x, out);
}
void CUDABackend::add(const Tensor& a, const Tensor& b, Tensor& out) {
    cpu_.add(a, b, out);
}
void CUDABackend::mul(const Tensor& a, const Tensor& b, Tensor& out) {
    cpu_.mul(a, b, out);
}

// ── Memory management helpers used by Model::load() ──────────────────────

size_t CUDABackend::free_vram_bytes() {
    size_t free_bytes = 0, total_bytes = 0;
    cudaError_t e = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (e != cudaSuccess) return 0;
    return free_bytes;
}

void* CUDABackend::alloc_gpu_mem(size_t bytes) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    return ptr;
}

void CUDABackend::upload(void* dst, const void* src, size_t bytes) {
    CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}

void CUDABackend::release_gpu_mem(void* ptr) {
    if (ptr) cudaFree(ptr);
}

// ── Phase 1: GPU-resident pipeline ──────────────────────────────────────

void CUDABackend::init_gpu_activations(const ModelConfig& cfg) {
    impl_->init_gpu_activations(cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads,
                                cfg.head_dim, cfg.intermediate_size, cfg.vocab_size,
                                cfg.num_layers);
}
bool CUDABackend::has_gpu_activations() const { return impl_->gpu_act_ready_; }

float* CUDABackend::d_x()     { return impl_->d_x_res_; }
float* CUDABackend::d_norm()  { return impl_->d_norm_out_; }
float* CUDABackend::d_q()     { return impl_->d_q_res_; }
float* CUDABackend::d_k()     { return impl_->d_k_res_; }
float* CUDABackend::d_v()     { return impl_->d_v_res_; }
float* CUDABackend::d_gate()  { return impl_->d_gate_res_; }
float* CUDABackend::d_up()    { return impl_->d_up_res_; }
float* CUDABackend::h_qkv()   { return impl_->h_qkv_stage_; }
float* CUDABackend::h_attn()  { return impl_->h_attn_stage_; }
float* CUDABackend::h_logit() { return impl_->h_logit_stage_; }

void CUDABackend::add_d(float* d_out, const float* d_a, const float* d_b, int N) {
    int blocks = (N + 255) / 256;
    add_d_kernel<<<blocks, 256>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaGetLastError());
}
void CUDABackend::mul_d(float* d_out, const float* d_a, const float* d_b, int N) {
    int blocks = (N + 255) / 256;
    mul_d_kernel<<<blocks, 256>>>(d_a, d_b, d_out, N);
    CUDA_CHECK(cudaGetLastError());
}
void CUDABackend::relu2_d(float* d_out, const float* d_x, int N) {
    int blocks = (N + 255) / 256;
    relu2_d_kernel<<<blocks, 256>>>(d_x, d_out, N);
    CUDA_CHECK(cudaGetLastError());
}
void CUDABackend::rms_norm_d(float* d_out, const float* d_x, const float* d_w,
                              float eps, int N) {
    rms_norm_d_kernel<<<1, 256>>>(d_x, d_w, eps, d_out, N);
    CUDA_CHECK(cudaGetLastError());
}
void CUDABackend::gemv_d2d(float* d_out, const float* d_in, const void* d_W_f16,
                            int K, int N) {
    const __half* W = static_cast<const __half*>(d_W_f16);
    int blocks = (N + 3) / 4;
    gemv_f16_f32_kernel<<<blocks, 128>>>(W, d_in, d_out, K, N);
    CUDA_CHECK(cudaGetLastError());
}
void CUDABackend::gemv_d2d_int8(float* d_out, const float* d_in,
                                 const void* d_W_int8, float scale, int K, int N) {
    impl_->gemv_d2d_int8(d_out, d_in, d_W_int8, scale, K, N);
}
void CUDABackend::upload_pinned(float* d_dst, const float* h_src, size_t n) {
    CUDA_CHECK(cudaMemcpy(d_dst, h_src, n * sizeof(float), cudaMemcpyHostToDevice));
}
void CUDABackend::download_pinned(float* h_dst, const float* d_src, size_t n) {
    CUDA_CHECK(cudaMemcpy(h_dst, d_src, n * sizeof(float), cudaMemcpyDeviceToHost));
}
void CUDABackend::sync_d() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ── Phase 4: GPU-resident RoPE, KV cache, decode attention ──────────────

void CUDABackend::rope_inplace_d(float* d_q, float* d_k, int pos, float theta,
                                  int n_heads, int n_kv_heads, int head_dim) {
    int total = n_heads + n_kv_heads;
    rope_inplace_kernel<<<total, head_dim / 2>>>(
        d_q, d_k, pos, theta, n_heads, n_kv_heads, head_dim);
    CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::kv_write_d(const float* d_k, const float* d_v,
                              int layer, int pos) {
    int Hk  = impl_->Hk_;
    int mxs = impl_->gpu_max_seq_;
    float* k_layer = impl_->d_k_cache_ + (int64_t)layer * mxs * Hk;
    float* v_layer = impl_->d_v_cache_ + (int64_t)layer * mxs * Hk;
    int blocks = (Hk + 127) / 128;
    kv_write_kernel<<<blocks, 128>>>(d_k, d_v, k_layer, v_layer, pos, Hk, mxs);
    CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::attention_d(const float* d_q, int layer, int kv_len,
                               float* d_out, int n_heads, int n_kv_heads,
                               int head_dim, float scale) {
    int Hk  = impl_->Hk_;
    int mxs = impl_->gpu_max_seq_;
    const float* k_layer = impl_->d_k_cache_ + (int64_t)layer * mxs * Hk;
    const float* v_layer = impl_->d_v_cache_ + (int64_t)layer * mxs * Hk;
    int group = n_heads / n_kv_heads;
    int shared_bytes = kv_len * sizeof(float);
    decode_attention_kernel<<<n_heads, 128, shared_bytes>>>(
        d_q, k_layer, v_layer, d_out,
        n_heads, n_kv_heads, head_dim, kv_len, Hk, scale, group);
    CUDA_CHECK(cudaGetLastError());
}

void CUDABackend::clear_kv_d() {
    if (!impl_->d_k_cache_) return;
    size_t elems = (size_t)impl_->n_layers_gpu_ * impl_->gpu_max_seq_ * impl_->Hk_;
    CUDA_CHECK(cudaMemsetAsync(impl_->d_k_cache_, 0, elems * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(impl_->d_v_cache_, 0, elems * sizeof(float)));
}

bool CUDABackend::has_gpu_kv(int min_seq) const {
    return impl_->d_k_cache_ != nullptr && impl_->gpu_max_seq_ >= min_seq;
}

int CUDABackend::gpu_kv_max_seq() const { return impl_->gpu_max_seq_; }

void CUDABackend::download_kv_slot_d(int layer, int pos, int n_floats, float* h_dst) {
    if (!impl_->d_k_cache_) return;
    int Hk  = impl_->Hk_;
    int mxs = impl_->gpu_max_seq_;
    const float* k_layer = impl_->d_k_cache_ + (int64_t)layer * mxs * Hk + (int64_t)pos * Hk;
    CUDA_CHECK(cudaMemcpy(h_dst, k_layer, n_floats * sizeof(float), cudaMemcpyDeviceToHost));
}

void CUDABackend::upload_kv_layer_d(int layer, int n_positions,
                                     const float* h_k, const float* h_v) {
    if (!impl_->d_k_cache_ || n_positions <= 0) return;
    int Hk  = impl_->Hk_;
    int mxs = impl_->gpu_max_seq_;
    if (n_positions > mxs) n_positions = mxs;
    size_t bytes = (size_t)n_positions * Hk * sizeof(float);
    float* k_layer = impl_->d_k_cache_ + (int64_t)layer * mxs * Hk;
    float* v_layer = impl_->d_v_cache_ + (int64_t)layer * mxs * Hk;
    CUDA_CHECK(cudaMemcpy(k_layer, h_k, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(v_layer, h_v, bytes, cudaMemcpyHostToDevice));
}

} // namespace ternative
