#include "ternative/model.h"
#include "ternative/gguf.h"
#include "ternative/ops.h"
#include "ternative/sampler.h"
#include "ternative/cache.h"

#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstdlib>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ternative {

#ifdef __AVX2__
static inline float hsum256_ps(__m256 v) {
    __m128 vsum = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    vsum = _mm_add_ps(vsum, _mm_movehl_ps(vsum, vsum));
    vsum = _mm_add_ss(vsum, _mm_movehdup_ps(vsum));
    return _mm_cvtss_f32(vsum);
}
#endif

} // namespace ternative

namespace ternative {

// === ActivationArena ===

void ActivationArena::init(const ModelConfig& cfg) {
    int H  = cfg.hidden_size;
    int I  = cfg.intermediate_size;
    int Hq = cfg.num_heads    * cfg.head_dim;
    int Hk = cfg.num_kv_heads * cfg.head_dim;

    t_attn_in     = Tensor({1, H},  ggml_type::F32);
    t_q           = Tensor({1, Hq}, ggml_type::F32);
    t_k           = Tensor({1, Hk}, ggml_type::F32);
    t_v           = Tensor({1, Hk}, ggml_type::F32);
    t_attn_proj   = Tensor({1, H},  ggml_type::F32);
    t_residual    = Tensor({1, H},  ggml_type::F32);
    t_ffn_in      = Tensor({1, H},  ggml_type::F32);
    t_gate        = Tensor({1, I},  ggml_type::F32);
    t_up          = Tensor({1, I},  ggml_type::F32);
    t_gate_relu2  = Tensor({1, I},  ggml_type::F32);
    t_ffn_mid     = Tensor({1, I},  ggml_type::F32);
    t_ffn_out     = Tensor({1, H},  ggml_type::F32);
    t_normed      = Tensor({1, H},  ggml_type::F32);
    initialized = true;
}

// === Model destructor (frees GPU-resident norm weights) ===

Model::~Model() {
#ifdef TERNATIVE_USE_CUDA
    auto* cuda = dynamic_cast<CUDABackend*>(backend.get());
    if (cuda) {
        for (auto& lw : layers) {
            if (lw.d_attn_norm)     cuda->release_gpu_mem(lw.d_attn_norm);
            if (lw.d_ffn_norm)      cuda->release_gpu_mem(lw.d_ffn_norm);
            if (lw.d_attn_sub_norm) cuda->release_gpu_mem(lw.d_attn_sub_norm);
            if (lw.d_ffn_sub_norm)  cuda->release_gpu_mem(lw.d_ffn_sub_norm);
            // Weight matrices (wq/wk/wv/wo/w_gate/w_up/w_down) are tracked via
            // the per-Tensor gpu_ptr; the Tensor destructor does not free them
            // (gpu_ptr is non-owning by contract). Free them here.
            for (Tensor* t : {&lw.wq, &lw.wk, &lw.wv, &lw.wo,
                              &lw.w_gate, &lw.w_up, &lw.w_down}) {
                if (t->gpu_ptr) { cuda->release_gpu_mem(t->gpu_ptr); t->gpu_ptr = nullptr; }
            }
        }
        if (d_output_norm) cuda->release_gpu_mem(d_output_norm);
    }
#endif
}

// === KVCache ===

void KVCache::init(const ModelConfig& config, int cap) {
    n_layers = config.num_layers;
    n_kv_heads = config.num_kv_heads;
    head_dim = config.head_dim;
    capacity = cap;
    seq_len = 0;

    k_cache.resize(n_layers);
    v_cache.resize(n_layers);
    for (int i = 0; i < n_layers; ++i) {
        k_cache[i] = Tensor({capacity, n_kv_heads, head_dim}, ggml_type::F32);
        v_cache[i] = Tensor({capacity, n_kv_heads, head_dim}, ggml_type::F32);
    }
}

void KVCache::clear() {
    seq_len = 0;
}

void KVCache::resize(int new_seq_len) {
    seq_len = new_seq_len;
}

void KVCache::update(int layer, int pos, const Tensor& k, const Tensor& v) {
    // k, v shape: [seq_len, n_kv_heads, head_dim]
    int64_t seq = k.shape[0];
    int64_t kv_heads = k.shape[1];
    int64_t hdim = k.shape[2];

    const float* kp = k.ptr<float>();
    const float* vp = v.ptr<float>();
    float* k_dst = k_cache[layer].ptr<float>();
    float* v_dst = v_cache[layer].ptr<float>();

    for (int64_t s = 0; s < seq; ++s) {
        int p = pos + (int)s;
        if (p >= capacity) break;
        for (int64_t h = 0; h < kv_heads; ++h) {
            size_t src_off = (s * kv_heads + h) * hdim;
            size_t dst_off = (p * n_kv_heads + h) * head_dim;
            std::memcpy(k_dst + dst_off, kp + src_off, hdim * sizeof(float));
            std::memcpy(v_dst + dst_off, vp + src_off, hdim * sizeof(float));
        }
    }
}

// === Model loading helpers ===

static Tensor load_tensor_from_gguf(const GGUFile& gguf, const std::string& name) {
    const TensorInfo* info = gguf.find_tensor(name);
    if (!info) {
        std::cerr << "Tensor not found: " << name << std::endl;
        return Tensor();
    }

    std::vector<int64_t> shape;
    shape.reserve(info->shape.size());
    for (auto d : info->shape) shape.push_back(static_cast<int64_t>(d));
    Tensor t(std::move(shape), info->type);
    size_t tensor_size = t.n_bytes();
    size_t src_offset = info->offset;

    if (src_offset + tensor_size > gguf.tensor_data.size()) {
        std::cerr << "Tensor data out of bounds: " << name << std::endl;
        return Tensor();
    }

    std::memcpy(t.data.data(), gguf.tensor_data.data() + src_offset, tensor_size);

    // Recover per-tensor scale for I2_S so consumers don't need to read from data.
    if (t.type == ggml_type::I2_S) {
        size_t n_blocks = (t.data.size() >= sizeof(float)) ? (t.data.size() - sizeof(float)) / 32 : 0;
        const float* scale_ptr = reinterpret_cast<const float*>(t.data.data() + n_blocks * 32);
        t.scale = *scale_ptr;
    }

    // Keep F16 weights as F16 — the AVX2+F16C GEMM kernel converts on-the-fly.
    // This halves memory bandwidth vs pre-converting to F32 at load time.

    return t;
}

static bool has_tensor(const GGUFile& gguf, const std::string& name) {
    return gguf.find_tensor(name) != nullptr;
}

static std::string blk_name(int layer, const std::string& suffix) {
    return "blk." + std::to_string(layer) + "." + suffix;
}

// === LoRA merge ===

static Tensor matmul_lora(const Tensor& a, const Tensor& b) {
    // Both a and b are loaded from GGUF and follow GGUF column-major storage:
    //   element(i0, i1) = i1 * ne0 + i0
    // a: [M, K], b: [K, N]
    // Result stored in same GGUF column-major format.
    assert(a.shape.size() == 2);
    assert(b.shape.size() == 2);
    int64_t M = a.shape[0];
    int64_t K = a.shape[1];
    int64_t N = b.shape[1];
    assert(b.shape[0] == K);

    Tensor out({M, N}, ggml_type::F32);
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += a.get_float(k * M + m) * b.get_float(n * K + k);
            }
            out.set_float(n * M + m, sum);
        }
    }
    return out;
}

static void apply_lora(Tensor& base, const Tensor& lora_a, const Tensor& lora_b, float alpha, int rank) {
    // GGUF LoRA stores:
    //   lora_a: [in_features, rank]
    //   lora_b: [rank, out_features]
    // delta = lora_a @ lora_b = [in_features, out_features]
    // base in GGUF is [in_features, out_features]

    assert(base.shape.size() == 2);
    assert(lora_a.shape.size() == 2);
    assert(lora_b.shape.size() == 2);

    std::cerr << "[apply_lora] base=" << base.shape[0] << "x" << base.shape[1]
              << " lora_a=" << lora_a.shape[0] << "x" << lora_a.shape[1]
              << " lora_b=" << lora_b.shape[0] << "x" << lora_b.shape[1]
              << " alpha=" << alpha << " rank=" << rank << std::endl;

    Tensor delta = matmul_lora(lora_a, lora_b);
    std::cerr << "[apply_lora] delta=" << delta.shape[0] << "x" << delta.shape[1] << std::endl;

    assert(base.shape == delta.shape);

    float scale = alpha / rank;

    if (base.type == ggml_type::I2_S) {
        // I2_S ternary quantization is too coarse for LoRA-merged weights:
        // a tiny delta can snap a zero to ±scale, destroying the weight.
        // Merge in F32 and keep the result as F16 (F16 GEMM path exists).
        auto original_shape = base.shape;
        Tensor f32_base(base.shape, ggml_type::F32);
        tensor_dequantize_i2_s(base, f32_base);
        size_t n = f32_base.n_elements();
        float scale_factor = scale;
        for (size_t i = 0; i < n; ++i) {
            float bv = f32_base.get_float(i);
            float dv = delta.get_float(i) * scale_factor;
            f32_base.set_float(i, bv + dv);
        }
        // Convert merged weights to F16 for memory efficiency (4GB vs 8GB)
        Tensor f16_merged(original_shape, ggml_type::F16);
        for (size_t i = 0; i < n; ++i) {
            f16_merged.set_float(i, f32_base.get_float(i));
        }
        base = std::move(f16_merged);
        return;
    } else {
        assert(base.type == ggml_type::F32 || base.type == ggml_type::F16 || base.type == ggml_type::BF16);
        size_t n = base.n_elements();
        for (size_t i = 0; i < n; ++i) {
            float bv = base.get_float(i);
            float dv = delta.get_float(i) * scale;
            base.set_float(i, bv + dv);
        }
    }
}

// === GPU backend init ===

#ifdef TERNATIVE_USE_CUDA
// Attempt GPU offload; silently falls back to CPU if VRAM is insufficient or init fails.
static void init_gpu_backend(Model& model) {
    try {
        auto cuda_backend = std::make_unique<CUDABackend>();
        const size_t VRAM_MARGIN = 200ull << 20; // 200 MB safety margin
        // Reserve VRAM for the GPU KV cache + activation pipeline (allocated AFTER
        // the offload loop by init_gpu_activations). For BitNet 2B-4T this is
        // 30 layers × 1024 seq × 640 floats × 2 (k+v) ≈ 157 MB plus ~15 MB
        // activation/staging buffers — round up to 180 MB to be safe.
        const size_t VRAM_KV_RESERVE = 180ull << 20;
        size_t free_vram = CUDABackend::free_vram_bytes();
        const size_t total_reserve = VRAM_MARGIN + VRAM_KV_RESERVE;

        if (free_vram > total_reserve) {
            size_t budget = free_vram - total_reserve;
            int n_offloaded = 0;

            // Only offload layers whose weight matrices are F16 — the GPU GEMV
            // kernel only handles F16. I2_S/BF16/F32 base models would have to
            // fall through to the CPU path (which is why the LoRA-merged path
            // converts I2_S → F16 in apply_lora).
            bool weights_f16 = !model.layers.empty()
                && model.layers[0].wq.type == ggml_type::F16
                && model.layers[0].wk.type == ggml_type::F16;

            // ── Pre-plan F16 vs INT8 split so all 30 layers fit if at all possible ──
            // Per-layer sizes (computed from layer 0 for budget estimation).
            int max_f16_layers = model.config.num_layers;
            if (weights_f16 && !model.layers.empty()) {
                auto& lw0 = model.layers[0];
                size_t layer_bytes_f16 = 0;
                size_t layer_bytes_int8 = 0;
                for (auto* t : {&lw0.wq, &lw0.wk, &lw0.wv, &lw0.wo,
                                &lw0.w_gate, &lw0.w_up, &lw0.w_down}) {
                    layer_bytes_f16  += t->data.size();
                    layer_bytes_int8 += t->n_elements() * sizeof(int8_t);
                }
                size_t norm_bytes = lw0.attn_norm.data.size() + lw0.ffn_norm.data.size();
                if (!lw0.attn_sub_norm.shape.empty()) norm_bytes += lw0.attn_sub_norm.data.size();
                if (!lw0.ffn_sub_norm.shape.empty())  norm_bytes += lw0.ffn_sub_norm.data.size();
                layer_bytes_f16  += norm_bytes;
                layer_bytes_int8 += norm_bytes;

                const int N = model.config.num_layers;
                if (layer_bytes_int8 * (size_t)N <= budget) {
                    // All-INT8 fits: maximize F16 count subject to remaining budget.
                    size_t remaining = budget - layer_bytes_int8 * (size_t)N;
                    size_t extra_per_f16 = layer_bytes_f16 - layer_bytes_int8;
                    int k = extra_per_f16 ? (int)(remaining / extra_per_f16) : N;
                    if (k > N) k = N;
                    max_f16_layers = k;
                } else {
                    // Not all 30 fit even in INT8: F16-first greedy covers as many as possible.
                    max_f16_layers = N;
                }
            }

            for (int i = 0; weights_f16 && i < max_f16_layers && i < model.config.num_layers; ++i) {
                auto& lw = model.layers[i];
                size_t layer_bytes = 0;
                for (auto* t : {&lw.wq, &lw.wk, &lw.wv, &lw.wo,
                                &lw.w_gate, &lw.w_up, &lw.w_down})
                    layer_bytes += t->data.size();
                // Norm weights are tiny but include them so the accounting is exact.
                layer_bytes += lw.attn_norm.data.size() + lw.ffn_norm.data.size();
                if (!lw.attn_sub_norm.shape.empty()) layer_bytes += lw.attn_sub_norm.data.size();
                if (!lw.ffn_sub_norm.shape.empty())  layer_bytes += lw.ffn_sub_norm.data.size();
                if (layer_bytes > budget) break;

                for (auto* t : {&lw.wq, &lw.wk, &lw.wv, &lw.wo,
                                &lw.w_gate, &lw.w_up, &lw.w_down}) {
                    t->gpu_ptr = cuda_backend->alloc_gpu_mem(t->data.size());
                    cuda_backend->upload(t->gpu_ptr, t->data.data(), t->data.size());
                }
                // Upload F32 norm weights so rms_norm_d can read them on-device.
                lw.d_attn_norm = cuda_backend->alloc_gpu_mem(lw.attn_norm.data.size());
                cuda_backend->upload(lw.d_attn_norm, lw.attn_norm.data.data(), lw.attn_norm.data.size());
                lw.d_ffn_norm = cuda_backend->alloc_gpu_mem(lw.ffn_norm.data.size());
                cuda_backend->upload(lw.d_ffn_norm, lw.ffn_norm.data.data(), lw.ffn_norm.data.size());
                if (!lw.attn_sub_norm.shape.empty()) {
                    lw.d_attn_sub_norm = cuda_backend->alloc_gpu_mem(lw.attn_sub_norm.data.size());
                    cuda_backend->upload(lw.d_attn_sub_norm, lw.attn_sub_norm.data.data(), lw.attn_sub_norm.data.size());
                }
                if (!lw.ffn_sub_norm.shape.empty()) {
                    lw.d_ffn_sub_norm = cuda_backend->alloc_gpu_mem(lw.ffn_sub_norm.data.size());
                    cuda_backend->upload(lw.d_ffn_sub_norm, lw.ffn_sub_norm.data.data(), lw.ffn_sub_norm.data.size());
                }
                budget -= layer_bytes;
                ++n_offloaded;
            }

            // ── INT8 pass: try to fit remaining layers at half the VRAM per layer ──────
            // Layers that didn't fit in F16 (141 MB each) may fit in INT8 (~70 MB each).
            // Quality impact is negligible for near-ternary BitNet weights.
            for (int i = n_offloaded; weights_f16 && i < model.config.num_layers; ++i) {
                auto& lw = model.layers[i];

                // Estimate INT8 budget: weight matrices only (norm weights remain F32)
                size_t layer_bytes_int8 = 0;
                for (auto* t : {&lw.wq, &lw.wk, &lw.wv, &lw.wo,
                                &lw.w_gate, &lw.w_up, &lw.w_down}) {
                    if (t->type == ggml_type::F16)
                        layer_bytes_int8 += t->n_elements() * sizeof(int8_t);
                }
                // Add norm weights (stay F32, tiny)
                layer_bytes_int8 += lw.attn_norm.data.size() + lw.ffn_norm.data.size();
                if (!lw.attn_sub_norm.shape.empty()) layer_bytes_int8 += lw.attn_sub_norm.data.size();
                if (!lw.ffn_sub_norm.shape.empty())  layer_bytes_int8 += lw.ffn_sub_norm.data.size();

                if (layer_bytes_int8 > budget) break;

                // Quantize each weight tensor to INT8 symmetric (per-tensor scale)
                for (auto* t : {&lw.wq, &lw.wk, &lw.wv, &lw.wo,
                                &lw.w_gate, &lw.w_up, &lw.w_down}) {
                    if (t->type != ggml_type::F16) continue;  // skip non-F16 (shouldn't happen)

                    const size_t n_elems = t->n_elements();

                    // Find max absolute value for symmetric quantization
                    float max_abs = 0.0f;
                    const f16_t* fp = t->ptr<f16_t>();
                    for (size_t j = 0; j < n_elems; ++j) {
                        float v = fp[j].to_float();
                        if (v < 0) v = -v;
                        if (v > max_abs) max_abs = v;
                    }
                    if (max_abs == 0.0f) max_abs = 1.0f;

                    const float q_scale = max_abs / 127.0f;
                    t->scale = q_scale;         // store dequantization scale in existing field

                    // Convert F16 → INT8
                    std::vector<int8_t> int8_data(n_elems);
                    for (size_t j = 0; j < n_elems; ++j) {
                        float v = fp[j].to_float() / q_scale;
                        if (v >  127.f) v =  127.f;
                        if (v < -127.f) v = -127.f;
                        int8_data[j] = static_cast<int8_t>(std::round(v));
                    }

                    // Upload INT8 to GPU
                    const size_t int8_bytes = (size_t)n_elems * sizeof(int8_t);
                    t->gpu_ptr = cuda_backend->alloc_gpu_mem(int8_bytes);
                    cuda_backend->upload(t->gpu_ptr, int8_data.data(), int8_bytes);
                    t->gpu_is_int8 = true;

                    budget -= int8_bytes;
                }

                // Upload norm weights (F32, same as F16 layers)
                lw.d_attn_norm = cuda_backend->alloc_gpu_mem(lw.attn_norm.data.size());
                cuda_backend->upload(lw.d_attn_norm, lw.attn_norm.data.data(), lw.attn_norm.data.size());
                lw.d_ffn_norm = cuda_backend->alloc_gpu_mem(lw.ffn_norm.data.size());
                cuda_backend->upload(lw.d_ffn_norm, lw.ffn_norm.data.data(), lw.ffn_norm.data.size());
                if (!lw.attn_sub_norm.shape.empty()) {
                    lw.d_attn_sub_norm = cuda_backend->alloc_gpu_mem(lw.attn_sub_norm.data.size());
                    cuda_backend->upload(lw.d_attn_sub_norm, lw.attn_sub_norm.data.data(), lw.attn_sub_norm.data.size());
                }
                if (!lw.ffn_sub_norm.shape.empty()) {
                    lw.d_ffn_sub_norm = cuda_backend->alloc_gpu_mem(lw.ffn_sub_norm.data.size());
                    cuda_backend->upload(lw.d_ffn_sub_norm, lw.ffn_sub_norm.data.data(), lw.ffn_sub_norm.data.size());
                }
                budget -= lw.attn_norm.data.size() + lw.ffn_norm.data.size();
                if (!lw.attn_sub_norm.shape.empty()) budget -= lw.attn_sub_norm.data.size();
                if (!lw.ffn_sub_norm.shape.empty())  budget -= lw.ffn_sub_norm.data.size();

                ++n_offloaded;
                std::cerr << "[GPU] Layer " << i << " offloaded as INT8\n";
            }

            model.n_gpu_layers = n_offloaded;

            // Upload output_norm first (only meaningful when ALL layers are on GPU),
            // then init Phase 1+4 activation/KV buffers when ANY layers are
            // offloaded. The CPU tail (layers ≥ n_gpu_layers) keeps using the
            // CPU KV cache; the GPU KV cache only covers layers 0..n_gpu_layers-1.
            if (n_offloaded == model.config.num_layers) {
                model.d_output_norm = cuda_backend->alloc_gpu_mem(model.output_norm.data.size());
                cuda_backend->upload(model.d_output_norm,
                                     model.output_norm.data.data(),
                                     model.output_norm.data.size());
            }
            if (n_offloaded > 0) {
                cuda_backend->init_gpu_activations(model.config);
            }

            std::cout << "[GPU] Offloaded " << n_offloaded << "/"
                      << model.config.num_layers << " layers ("
                      << (free_vram - budget) / (1024*1024) << " MB VRAM used)\n";
            if (!weights_f16) {
                std::cout << "[GPU] Base model weights are not F16 (likely I2_S) — "
                             "CPU-only path. Use a LoRA-merged GGUF (F16) for GPU.\n";
            } else if (n_offloaded == model.config.num_layers) {
                std::cout << "[GPU] All layers offloaded — GPU-resident decode path enabled\n";
            } else if (n_offloaded > 0) {
                std::cout << "[GPU] Partial offload (" << n_offloaded
                          << " GPU layers) — GPU KV cache + "
                          << (model.config.num_layers - n_offloaded)
                          << " CPU tail layers\n";
            }
        } else {
            std::cout << "[GPU] Insufficient VRAM ("
                      << free_vram / (1024*1024) << " MB free) — CPU only\n";
        }
        model.backend = std::move(cuda_backend);
    } catch (const std::exception& e) {
        std::cerr << "[GPU] CUDA init failed: " << e.what() << " — CPU fallback\n";
        model.backend = std::make_unique<CPUBackend>();
    }
}
#endif

// === Model::load ===

std::unique_ptr<Model> Model::load(
    const std::string& base_gguf_path,
    const std::vector<std::string>& lora_gguf_paths,
    bool use_cuda
) {
    auto base_gguf = gguf_load(base_gguf_path);
    if (!base_gguf) {
        std::cerr << "Failed to load base GGUF: " << base_gguf_path << std::endl;
        return nullptr;
    }

    auto model = std::make_unique<Model>();

    // Read config from metadata
    if (base_gguf->has_metadata("llama.block_count")) {
        model->config.num_layers = base_gguf->get_metadata("llama.block_count").as_u32();
    }
    if (base_gguf->has_metadata("llama.context_length")) {
        model->config.max_seq_len = base_gguf->get_metadata("llama.context_length").as_u32();
    }
    if (base_gguf->has_metadata("llama.embedding_length")) {
        model->config.hidden_size = base_gguf->get_metadata("llama.embedding_length").as_u32();
    }
    if (base_gguf->has_metadata("llama.feed_forward_length")) {
        model->config.intermediate_size = base_gguf->get_metadata("llama.feed_forward_length").as_u32();
    }
    if (base_gguf->has_metadata("llama.head_count")) {
        model->config.num_heads = base_gguf->get_metadata("llama.head_count").as_u32();
    }
    if (base_gguf->has_metadata("llama.head_count_kv")) {
        model->config.num_kv_heads = base_gguf->get_metadata("llama.head_count_kv").as_u32();
    }
    if (base_gguf->has_metadata("llama.rope.freq_base")) {
        model->config.rope_theta = base_gguf->get_metadata("llama.rope.freq_base").as_f32();
    }
    if (base_gguf->has_metadata("llama.attention.layer_norm_rms_eps")) {
        model->config.rms_norm_eps = base_gguf->get_metadata("llama.attention.layer_norm_rms_eps").as_f32();
    }

    model->config.head_dim = model->config.hidden_size / model->config.num_heads;

    std::cout << "Loading model: "
              << model->config.num_layers << " layers, "
              << model->config.hidden_size << " hidden, "
              << model->config.num_heads << " heads, "
              << model->config.num_kv_heads << " kv heads, "
              << model->config.head_dim << " head_dim"
              << std::endl;

    // ── Cache check ────────────────────────────────────────────────────────
    // Check for a previously saved .tvcache (skips the ~8-min LoRA merge).
    // Disable with env TERNATIVE_NO_CACHE=1.
    bool use_cache = (getenv("TERNATIVE_NO_CACHE") == nullptr);
    std::string cache_path_str;
    CacheKey cache_key;

    if (use_cache && !lora_gguf_paths.empty()) {
        std::string lora0 = lora_gguf_paths[0];
        std::cerr << "[Cache] Computing key (hashing files, may take a few seconds)...\n";
        cache_key = ModelCache::compute_key(base_gguf_path, lora0);
        cache_key.lora_alpha = 32.0f; // default; updated after LoRA load if needed
        cache_path_str = ModelCache::cache_path(cache_key);
        std::cerr << "[Cache] Path: " << cache_path_str << "\n";

        // Preallocate layers so try_load can populate them
        model->layers.resize(model->config.num_layers);

        // Try loading from cache. Free GGUF first only if cache HIT to avoid
        // having both 1.2GB GGUF + 4.6GB cache in RAM simultaneously.
        // On MISS, keep base_gguf alive for the normal load path below.
        base_gguf.reset();  // free GGUF to make room for 4.6GB cache

        if (ModelCache::try_load(*model, cache_path_str)) {
            std::cerr << "[Cache] HIT — skipping LoRA merge\n";
#ifdef TERNATIVE_USE_CUDA
            if (use_cuda) init_gpu_backend(*model);
            else          model->backend = std::make_unique<CPUBackend>();
#else
            model->backend = std::make_unique<CPUBackend>();
#endif
            model->kv_cache.init(model->config, model->config.max_seq_len);
            model->arena.init(model->config);
            return model;
        }

        std::cerr << "[Cache] MISS — performing full merge (will cache on completion)\n";
        // Reload base GGUF since we freed it above
        base_gguf = gguf_load(base_gguf_path);
        if (!base_gguf) {
            std::cerr << "Failed to reload base GGUF after cache miss: " << base_gguf_path << "\n";
            return nullptr;
        }
    }
    // ── End cache check ────────────────────────────────────────────────────

    // ── No-LoRA cache (base model only) ────────────────────────────────────
    // When no adapter is used, cache the raw base-model tensors so subsequent
    // loads skip the full GGUF read (saves ~2 min on the 1.8 GB I2_S file).
    if (use_cache && lora_gguf_paths.empty()) {
        std::cerr << "[Cache] Computing base-only key...\n";
        // Reuse the same CacheKey but with zero lora fields so the path differs
        // from any LoRA cache.
        CacheKey base_key = ModelCache::compute_key(base_gguf_path, "");
        base_key.lora_alpha = 0.0f;
        base_key.lora_rank  = 0;
        std::string base_cache_path = ModelCache::cache_path(base_key);
        std::cerr << "[Cache] Base path: " << base_cache_path << "\n";

        model->layers.resize(model->config.num_layers);

        // Free the just-loaded GGUF to make room for the cache (same pattern as
        // the LoRA cache block above).
        base_gguf.reset();

        if (ModelCache::try_load(*model, base_cache_path)) {
            std::cerr << "[Cache] HIT (base) — skipping GGUF tensor load\n";
#ifdef TERNATIVE_USE_CUDA
            if (use_cuda) init_gpu_backend(*model);
            else          model->backend = std::make_unique<CPUBackend>();
#else
            model->backend = std::make_unique<CPUBackend>();
#endif
            model->kv_cache.init(model->config, model->config.max_seq_len);
            model->arena.init(model->config);
            return model;
        }

        std::cerr << "[Cache] MISS (base) — loading from GGUF\n";
        // Reload GGUF since we freed it.
        base_gguf = gguf_load(base_gguf_path);
        if (!base_gguf) {
            std::cerr << "Failed to reload base GGUF after cache miss: " << base_gguf_path << "\n";
            return nullptr;
        }

        // After the normal tensor load below, save to cache. Set a flag.
        // We detect "we're in the no-LoRA cache path" by checking the saved key.
        // Use a lambda-style approach: save at the end of Model::load.
        // Store the path in a local so we can save after the normal load.
        // We do this below at the "// ── Save base cache ──" marker.
        cache_path_str = base_cache_path;  // reuse the existing local variable
    }
    // ── End no-LoRA cache check ──────────────────────────────────────────────

    // Load embeddings
    model->tok_embd = load_tensor_from_gguf(*base_gguf, "token_embd.weight");
    if (model->tok_embd.shape.size() >= 2) {
        model->config.vocab_size = static_cast<int>(model->tok_embd.shape[1]);
    }
    model->output_norm = load_tensor_from_gguf(*base_gguf, "output_norm.weight");

    // Load layers
    model->layers.resize(model->config.num_layers);
    for (int i = 0; i < model->config.num_layers; ++i) {
        auto& layer = model->layers[i];
        layer.attn_norm = load_tensor_from_gguf(*base_gguf, blk_name(i, "attn_norm.weight"));
        layer.wq = load_tensor_from_gguf(*base_gguf, blk_name(i, "attn_q.weight"));
        layer.wk = load_tensor_from_gguf(*base_gguf, blk_name(i, "attn_k.weight"));
        layer.wv = load_tensor_from_gguf(*base_gguf, blk_name(i, "attn_v.weight"));
        layer.wo = load_tensor_from_gguf(*base_gguf, blk_name(i, "attn_output.weight"));
        layer.ffn_norm = load_tensor_from_gguf(*base_gguf, blk_name(i, "ffn_norm.weight"));
        layer.w_gate = load_tensor_from_gguf(*base_gguf, blk_name(i, "ffn_gate.weight"));
        layer.w_up = load_tensor_from_gguf(*base_gguf, blk_name(i, "ffn_up.weight"));
        layer.w_down = load_tensor_from_gguf(*base_gguf, blk_name(i, "ffn_down.weight"));

        // Optional BitNet sub_norm layers
        if (has_tensor(*base_gguf, blk_name(i, "attn_sub_norm.weight"))) {
            layer.attn_sub_norm = load_tensor_from_gguf(*base_gguf, blk_name(i, "attn_sub_norm.weight"));
        }
        if (has_tensor(*base_gguf, blk_name(i, "ffn_sub_norm.weight"))) {
            layer.ffn_sub_norm = load_tensor_from_gguf(*base_gguf, blk_name(i, "ffn_sub_norm.weight"));
        }
    }

    // Load and apply LoRA adapters in order
    for (const auto& lora_gguf_path : lora_gguf_paths) {
        if (lora_gguf_path.empty()) continue;

        auto lora_gguf = gguf_load(lora_gguf_path);
        if (!lora_gguf) {
            std::cerr << "Failed to load LoRA GGUF: " << lora_gguf_path << std::endl;
            return nullptr;
        }

        model->has_lora = true;

        // Read LoRA metadata
        float alpha = 32.0f; // default
        if (lora_gguf->has_metadata("adapter.lora.alpha")) {
            alpha = lora_gguf->get_metadata("adapter.lora.alpha").as_f32();
        }
        if (model->lora_alpha == 0.0f) {
            model->lora_alpha = alpha;
        }

        // Infer rank from the first available lora_a tensor
        int rank = 16;
        for (int i = 0; i < model->config.num_layers; ++i) {
            std::string q_lora_a = blk_name(i, "attn_q.weight.lora_a");
            if (has_tensor(*lora_gguf, q_lora_a)) {
                const TensorInfo* info = lora_gguf->find_tensor(q_lora_a);
                if (info && info->shape.size() >= 2) {
                    rank = static_cast<int>(info->shape[1]);
                }
                break;
            }
        }
        model->lora_rank = rank;

        std::cout << "Applying LoRA adapter: " << lora_gguf_path
                  << " (alpha=" << alpha << ", rank=" << rank << ")..." << std::endl;

        for (int i = 0; i < model->config.num_layers; ++i) {
            auto& layer = model->layers[i];

            std::string q_lora_a = blk_name(i, "attn_q.weight.lora_a");
            std::string q_lora_b = blk_name(i, "attn_q.weight.lora_b");
            if (has_tensor(*lora_gguf, q_lora_a) && has_tensor(*lora_gguf, q_lora_b)) {
                apply_lora(layer.wq,
                    load_tensor_from_gguf(*lora_gguf, q_lora_a),
                    load_tensor_from_gguf(*lora_gguf, q_lora_b),
                    alpha, rank);
            }

            std::string k_lora_a = blk_name(i, "attn_k.weight.lora_a");
            std::string k_lora_b = blk_name(i, "attn_k.weight.lora_b");
            if (has_tensor(*lora_gguf, k_lora_a) && has_tensor(*lora_gguf, k_lora_b)) {
                apply_lora(layer.wk,
                    load_tensor_from_gguf(*lora_gguf, k_lora_a),
                    load_tensor_from_gguf(*lora_gguf, k_lora_b),
                    alpha, rank);
            }

            std::string v_lora_a = blk_name(i, "attn_v.weight.lora_a");
            std::string v_lora_b = blk_name(i, "attn_v.weight.lora_b");
            if (has_tensor(*lora_gguf, v_lora_a) && has_tensor(*lora_gguf, v_lora_b)) {
                apply_lora(layer.wv,
                    load_tensor_from_gguf(*lora_gguf, v_lora_a),
                    load_tensor_from_gguf(*lora_gguf, v_lora_b),
                    alpha, rank);
            }

            std::string o_lora_a = blk_name(i, "attn_output.weight.lora_a");
            std::string o_lora_b = blk_name(i, "attn_output.weight.lora_b");
            if (has_tensor(*lora_gguf, o_lora_a) && has_tensor(*lora_gguf, o_lora_b)) {
                apply_lora(layer.wo,
                    load_tensor_from_gguf(*lora_gguf, o_lora_a),
                    load_tensor_from_gguf(*lora_gguf, o_lora_b),
                    alpha, rank);
            }

            std::string gate_lora_a = blk_name(i, "ffn_gate.weight.lora_a");
            std::string gate_lora_b = blk_name(i, "ffn_gate.weight.lora_b");
            if (has_tensor(*lora_gguf, gate_lora_a) && has_tensor(*lora_gguf, gate_lora_b)) {
                apply_lora(layer.w_gate,
                    load_tensor_from_gguf(*lora_gguf, gate_lora_a),
                    load_tensor_from_gguf(*lora_gguf, gate_lora_b),
                    alpha, rank);
            }

            std::string up_lora_a = blk_name(i, "ffn_up.weight.lora_a");
            std::string up_lora_b = blk_name(i, "ffn_up.weight.lora_b");
            if (has_tensor(*lora_gguf, up_lora_a) && has_tensor(*lora_gguf, up_lora_b)) {
                apply_lora(layer.w_up,
                    load_tensor_from_gguf(*lora_gguf, up_lora_a),
                    load_tensor_from_gguf(*lora_gguf, up_lora_b),
                    alpha, rank);
            }

            std::string down_lora_a = blk_name(i, "ffn_down.weight.lora_a");
            std::string down_lora_b = blk_name(i, "ffn_down.weight.lora_b");
            if (has_tensor(*lora_gguf, down_lora_a) && has_tensor(*lora_gguf, down_lora_b)) {
                apply_lora(layer.w_down,
                    load_tensor_from_gguf(*lora_gguf, down_lora_a),
                    load_tensor_from_gguf(*lora_gguf, down_lora_b),
                    alpha, rank);
            }
        }

        std::cout << "LoRA merge complete: " << lora_gguf_path << std::endl;
    }

    // ── Save cache after successful merge ───────────────────────────────────
    if (use_cache && !cache_path_str.empty()) {
        // For LoRA path: cache_key was set during the LoRA block above.
        // For base-only path: cache_key is empty/zero (lora_alpha=0, lora_rank=0)
        // — that's fine, the path already differs via the key hash.
        if (!lora_gguf_paths.empty()) {
            cache_key.lora_alpha = model->lora_alpha;
            cache_key.lora_rank  = (uint32_t)model->lora_rank;
        }
        std::cerr << "[Cache] Saving to: " << cache_path_str << "\n";
        ModelCache::save(*model, cache_path_str, cache_key);
    }
    // ── End cache save ──────────────────────────────────────────────────────

    // Initialize backend
#ifdef TERNATIVE_USE_CUDA
    if (use_cuda) init_gpu_backend(*model);
    else          model->backend = std::make_unique<CPUBackend>();
#else
    model->backend = std::make_unique<CPUBackend>();
#endif

    // Initialize KV cache
    model->kv_cache.init(model->config, model->config.max_seq_len);

    // Pre-allocate forward() activation buffers (Phase 0).
    model->arena.init(model->config);

    return model;
}

// === Forward pass ===

static Tensor extract_embedding(const Tensor& tok_embd, int token_id) {
    // GGUF token_embd shape: [hidden_size, vocab_size]
    int64_t hidden_size = tok_embd.shape[0];
    Tensor out({1, hidden_size}, ggml_type::F32);
    float* dst = out.ptr<float>();
    if (tok_embd.type == ggml_type::F16) {
        const f16_t* src = tok_embd.ptr<f16_t>() + token_id * hidden_size;
        for (int64_t i = 0; i < hidden_size; ++i) {
            dst[i] = src[i].to_float();
        }
    } else if (tok_embd.type == ggml_type::BF16) {
        const bf16_t* src = tok_embd.ptr<bf16_t>() + token_id * hidden_size;
        for (int64_t i = 0; i < hidden_size; ++i) {
            dst[i] = src[i].to_float();
        }
    } else {
        const float* src = tok_embd.ptr<float>() + token_id * hidden_size;
        std::memcpy(dst, src, hidden_size * sizeof(float));
    }
    return out;
}

static Tensor extract_embeddings_batched(const Tensor& tok_embd, const std::vector<int>& token_ids) {
    int64_t hidden_size = tok_embd.shape[0];
    int64_t S = static_cast<int64_t>(token_ids.size());
    Tensor out({S, hidden_size}, ggml_type::F32);
    float* dst = out.ptr<float>();
    if (tok_embd.type == ggml_type::F16) {
        const f16_t* src = tok_embd.ptr<f16_t>();
        for (int64_t s = 0; s < S; ++s) {
            int tid = token_ids[static_cast<size_t>(s)];
            const f16_t* src_row = src + tid * hidden_size;
            float* dst_row = dst + s * hidden_size;
            for (int64_t i = 0; i < hidden_size; ++i) {
                dst_row[i] = src_row[i].to_float();
            }
        }
    } else if (tok_embd.type == ggml_type::BF16) {
        const bf16_t* src = tok_embd.ptr<bf16_t>();
        for (int64_t s = 0; s < S; ++s) {
            int tid = token_ids[static_cast<size_t>(s)];
            const bf16_t* src_row = src + tid * hidden_size;
            float* dst_row = dst + s * hidden_size;
            for (int64_t i = 0; i < hidden_size; ++i) {
                dst_row[i] = src_row[i].to_float();
            }
        }
    } else {
        const float* src = tok_embd.ptr<float>();
        for (int64_t s = 0; s < S; ++s) {
            int tid = token_ids[static_cast<size_t>(s)];
            const float* src_row = src + tid * hidden_size;
            float* dst_row = dst + s * hidden_size;
            std::memcpy(dst_row, src_row, hidden_size * sizeof(float));
        }
    }
    return out;
}

Tensor Model::forward(int token_id, int pos, bool prefill) {
    if (pos >= kv_cache.capacity) {
        std::cerr << "[Model::forward] Position " << pos << " exceeds KV cache capacity " << kv_cache.capacity << std::endl;
        return Tensor({1, config.vocab_size}, ggml_type::F32);
    }

#ifdef TERNATIVE_USE_CUDA
    // Phase 1+4: GPU-resident decode. Engages whenever ANY layer is offloaded.
    // Phase 4 keeps Q/K/V/attention entirely on the GPU for offloaded layers
    // (no per-layer D2H), and falls through to a CPU tail for any remaining
    // layers. When all layers are offloaded, the final norm + logits download
    // happens once instead of per-layer.
    if (!prefill && n_gpu_layers > 0) {
        auto* cuda = dynamic_cast<CUDABackend*>(backend.get());
        if (cuda && cuda->has_gpu_activations() && cuda->has_gpu_kv(pos + 1)) {
            return forward_gpu(token_id, pos);
        }
    }
#endif

    const auto& cfg = config;
    int n_heads = cfg.num_heads;
    int n_kv_heads = cfg.num_kv_heads;
    int head_dim = cfg.head_dim;

    // Embeddings
    Tensor x = extract_embedding(tok_embd, token_id);

    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        const auto& w = layers[layer];
        // === Attention ===
        Tensor attn_in = x.clone_empty(ggml_type::F32);
        backend->rms_norm(x, w.attn_norm, cfg.rms_norm_eps, attn_in);

        // QKV projections — batched to amortize the CUDA sync across all three.
        Tensor q = Tensor({1, n_heads * head_dim}, ggml_type::F32);
        Tensor k = Tensor({1, n_kv_heads * head_dim}, ggml_type::F32);
        Tensor v = Tensor({1, n_kv_heads * head_dim}, ggml_type::F32);

        {
            std::vector<const Tensor*> qkv_ws = {&w.wq, &w.wk, &w.wv};
            std::vector<Tensor*>       qkv_os = {&q, &k, &v};
            backend->matmul_batch(attn_in, qkv_ws, qkv_os);
        }

        // Reshape for RoPE: [n_heads, seq_len=1, head_dim]
        q = q.view({n_heads, 1, head_dim});
        k = k.view({n_kv_heads, 1, head_dim});
        v = v.view({n_kv_heads, 1, head_dim});

        // Apply RoPE
        backend->rope(q, k, pos, cfg.rope_theta);

        // Re-shape KV for cache: [seq_len=1, n_kv_heads, head_dim]
        k = k.view({1, n_kv_heads, head_dim});
        v = v.view({1, n_kv_heads, head_dim});

        // During prefill, token attends to itself (include current KV in attention).
        // During generation, token only attends to past tokens.
        if (prefill) {
            kv_cache.update(layer, pos, k, v);
        }

        // Attention: softmax(QK^T / sqrt(d)) @ V
        int kv_len = prefill ? (pos + 1) : pos;

        Tensor scores({1, n_heads, kv_len}, ggml_type::F32);
        float scale = 1.0f / std::sqrt((float)head_dim);

        const float* qp = q.ptr<float>();
        const float* k_cache_p = kv_cache.k_cache[layer].ptr<float>();
        float* sp = scores.ptr<float>();

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int h = 0; h < n_heads; ++h) {
            int kv_h = h / (n_heads / n_kv_heads);  // GQA mapping
            const float* q_head = qp + h * head_dim;
            for (int t = 0; t < kv_len; ++t) {
                const float* k_head = k_cache_p + t * n_kv_heads * head_dim + kv_h * head_dim;
                float dot = 0.0f;
#ifdef __AVX2__
                __m256 sum_vec = _mm256_setzero_ps();
                int d = 0;
                for (; d + 7 < head_dim; d += 8) {
                    __m256 qv = _mm256_loadu_ps(q_head + d);
                    __m256 kv = _mm256_loadu_ps(k_head + d);
                    sum_vec = _mm256_fmadd_ps(qv, kv, sum_vec);
                }
                dot = hsum256_ps(sum_vec);
                for (; d < head_dim; ++d) {
                    dot += q_head[d] * k_head[d];
                }
#else
                for (int d = 0; d < head_dim; ++d) {
                    dot += q_head[d] * k_head[d];
                }
#endif
                sp[h * kv_len + t] = dot * scale;
            }
        }

        // Softmax over kv_len dimension
        Tensor attn_weights = scores.clone_empty(ggml_type::F32);
        float* awp = attn_weights.ptr<float>();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int h = 0; h < n_heads; ++h) {
            float max_val = -1e30f;
            for (int t = 0; t < kv_len; ++t) {
                max_val = std::max(max_val, sp[h * kv_len + t]);
            }
            float sum = 0.0f;
            for (int t = 0; t < kv_len; ++t) {
                float v = std::exp(sp[h * kv_len + t] - max_val);
                awp[h * kv_len + t] = v;
                sum += v;
            }
            float inv_sum = 1.0f / sum;
            for (int t = 0; t < kv_len; ++t) {
                awp[h * kv_len + t] *= inv_sum;
            }
        }

        // Weighted sum of V
        Tensor attn_out({1, n_heads, head_dim}, ggml_type::F32);
        float* aop = attn_out.ptr<float>();
        const float* v_cache_p = kv_cache.v_cache[layer].ptr<float>();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int h = 0; h < n_heads; ++h) {
            int kv_h = h / (n_heads / n_kv_heads);
            float* out_head = aop + h * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                out_head[d] = 0.0f;
            }
            for (int t = 0; t < kv_len; ++t) {
                float w = awp[h * kv_len + t];
                const float* v_head = v_cache_p + t * n_kv_heads * head_dim + kv_h * head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    out_head[d] += w * v_head[d];
                }
            }
        }

        attn_out = attn_out.view({1, n_heads * head_dim});

        // Store KV for future tokens during generation
        if (!prefill) {
            kv_cache.update(layer, pos, k, v);
        }

        // attn_sub_norm (BitNet extra)
        if (!w.attn_sub_norm.shape.empty()) {
            Tensor normed = attn_out.clone_empty(ggml_type::F32);
            backend->rms_norm(attn_out, w.attn_sub_norm, cfg.rms_norm_eps, normed);
            attn_out = std::move(normed);
        }

        // Output projection
        Tensor attn_proj = Tensor({1, cfg.hidden_size}, ggml_type::F32);
        backend->matmul(attn_out, w.wo, attn_proj);

        // Residual
        Tensor residual = x.clone_empty(ggml_type::F32);
        backend->add(x, attn_proj, residual);

        // === FFN ===
        Tensor ffn_in = residual.clone_empty(ggml_type::F32);
        backend->rms_norm(residual, w.ffn_norm, cfg.rms_norm_eps, ffn_in);

        // Gate and Up projections — batched (same input ffn_in, one sync).
        Tensor gate = Tensor({1, cfg.intermediate_size}, ggml_type::F32);
        Tensor up = Tensor({1, cfg.intermediate_size}, ggml_type::F32);
        {
            std::vector<const Tensor*> gu_ws = {&w.w_gate, &w.w_up};
            std::vector<Tensor*>       gu_os = {&gate, &up};
            backend->matmul_batch(ffn_in, gu_ws, gu_os);
        }

        // ReLU²(gate) * up
        Tensor gate_relu2 = gate.clone_empty(ggml_type::F32);
        backend->relu2(gate, gate_relu2);
        Tensor ffn_mid = gate_relu2.clone_empty(ggml_type::F32);
        backend->mul(gate_relu2, up, ffn_mid);

        // ffn_sub_norm (BitNet extra)
        if (!w.ffn_sub_norm.shape.empty()) {
            Tensor normed = ffn_mid.clone_empty(ggml_type::F32);
            backend->rms_norm(ffn_mid, w.ffn_sub_norm, cfg.rms_norm_eps, normed);
            ffn_mid = std::move(normed);
        }

        // Down projection
        Tensor ffn_out = Tensor({1, cfg.hidden_size}, ggml_type::F32);
        backend->matmul(ffn_mid, w.w_down, ffn_out);

        // Residual
        backend->add(residual, ffn_out, x);
    }

    // Final norm
    Tensor normed = x.clone_empty(ggml_type::F32);
    backend->rms_norm(x, output_norm, cfg.rms_norm_eps, normed);

    // Output projection (tied embeddings)
    // logits = normed @ tok_embd
    // normed: [1, hidden_size], tok_embd: [hidden_size, vocab_size]
    // logits: [1, vocab_size]
    Tensor logits({1, cfg.vocab_size}, ggml_type::F32);
    backend->matmul(normed, tok_embd, logits);

    return logits;
}

// === Batched prefill ===

static void compute_batched_attention(
    int S, int n_heads, int n_kv_heads, int head_dim,
    const float* qp, const float* k_cache_p, const float* v_cache_p,
    float* aop
) {
    float scale = 1.0f / std::sqrt((float)head_dim);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (int sh = 0; sh < S * n_heads; ++sh) {
        int s = sh / n_heads;
        int h = sh % n_heads;
        int kv_h = h / (n_heads / n_kv_heads);
        int kv_len = s + 1;
            const float* q_head = qp + (int64_t)s * n_heads * head_dim + h * head_dim;
            float* out_head = aop + (int64_t)s * n_heads * head_dim + h * head_dim;

            std::vector<float> scores(kv_len);
            for (int t = 0; t < kv_len; ++t) {
                const float* k_head = k_cache_p + (int64_t)t * n_kv_heads * head_dim + kv_h * head_dim;
                float dot = 0.0f;
#ifdef __AVX2__
                __m256 sum_vec = _mm256_setzero_ps();
                int d = 0;
                for (; d + 7 < head_dim; d += 8) {
                    __m256 qv = _mm256_loadu_ps(q_head + d);
                    __m256 kv = _mm256_loadu_ps(k_head + d);
                    sum_vec = _mm256_fmadd_ps(qv, kv, sum_vec);
                }
                dot = hsum256_ps(sum_vec);
                for (; d < head_dim; ++d) {
                    dot += q_head[d] * k_head[d];
                }
#else
                for (int d = 0; d < head_dim; ++d) {
                    dot += q_head[d] * k_head[d];
                }
#endif
                scores[t] = dot * scale;
            }

            float max_val = -1e30f;
            for (int t = 0; t < kv_len; ++t) {
                if (scores[t] > max_val) max_val = scores[t];
            }
            float sum = 0.0f;
            for (int t = 0; t < kv_len; ++t) {
                scores[t] = std::exp(scores[t] - max_val);
                sum += scores[t];
            }
            float inv_sum = 1.0f / sum;
            for (int t = 0; t < kv_len; ++t) {
                scores[t] *= inv_sum;
            }

            for (int d = 0; d < head_dim; ++d) {
                out_head[d] = 0.0f;
            }
            for (int t = 0; t < kv_len; ++t) {
                float w = scores[t];
                const float* v_head = v_cache_p + (int64_t)t * n_kv_heads * head_dim + kv_h * head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    out_head[d] += w * v_head[d];
                }
            }
    }
}

Tensor Model::forward_prefill(const std::vector<int>& prompt_tokens) {
    int S = (int)prompt_tokens.size();
    if (S == 0) {
        return Tensor({0, config.vocab_size}, ggml_type::F32);
    }
    if (S > kv_cache.capacity) {
        std::cerr << "[Model::forward_prefill] Prompt length " << S
                  << " exceeds KV cache capacity " << kv_cache.capacity << std::endl;
        return Tensor({0, config.vocab_size}, ggml_type::F32);
    }

    const auto& cfg = config;
    int n_heads = cfg.num_heads;
    int n_kv_heads = cfg.num_kv_heads;
    int head_dim = cfg.head_dim;

    Tensor x = extract_embeddings_batched(tok_embd, prompt_tokens);

    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        const auto& w = layers[layer];

        Tensor attn_in = x.clone_empty(ggml_type::F32);
        backend->rms_norm(x, w.attn_norm, cfg.rms_norm_eps, attn_in);

        Tensor q = Tensor({S, n_heads * head_dim}, ggml_type::F32);
        Tensor k = Tensor({S, n_kv_heads * head_dim}, ggml_type::F32);
        Tensor v = Tensor({S, n_kv_heads * head_dim}, ggml_type::F32);

        backend->matmul(attn_in, w.wq, q);
        backend->matmul(attn_in, w.wk, k);
        backend->matmul(attn_in, w.wv, v);

        q = q.view({S, n_heads, head_dim});
        k = k.view({S, n_kv_heads, head_dim});
        v = v.view({S, n_kv_heads, head_dim});

        tensor_rope_batched(q, k, 0, cfg.rope_theta);

        kv_cache.update(layer, 0, k, v);

        Tensor attn_out({S, n_heads, head_dim}, ggml_type::F32);
        compute_batched_attention(
            S, n_heads, n_kv_heads, head_dim,
            q.ptr<float>(),
            kv_cache.k_cache[layer].ptr<float>(),
            kv_cache.v_cache[layer].ptr<float>(),
            attn_out.ptr<float>()
        );

        attn_out = attn_out.view({S, n_heads * head_dim});

        if (!w.attn_sub_norm.shape.empty()) {
            Tensor normed = attn_out.clone_empty(ggml_type::F32);
            backend->rms_norm(attn_out, w.attn_sub_norm, cfg.rms_norm_eps, normed);
            attn_out = std::move(normed);
        }

        Tensor attn_proj = Tensor({S, cfg.hidden_size}, ggml_type::F32);
        backend->matmul(attn_out, w.wo, attn_proj);

        Tensor residual = x.clone_empty(ggml_type::F32);
        backend->add(x, attn_proj, residual);

        Tensor ffn_in = residual.clone_empty(ggml_type::F32);
        backend->rms_norm(residual, w.ffn_norm, cfg.rms_norm_eps, ffn_in);

        Tensor gate = Tensor({S, cfg.intermediate_size}, ggml_type::F32);
        Tensor up = Tensor({S, cfg.intermediate_size}, ggml_type::F32);
        backend->matmul(ffn_in, w.w_gate, gate);
        backend->matmul(ffn_in, w.w_up, up);

        Tensor gate_relu2 = gate.clone_empty(ggml_type::F32);
        backend->relu2(gate, gate_relu2);
        Tensor ffn_mid = gate_relu2.clone_empty(ggml_type::F32);
        backend->mul(gate_relu2, up, ffn_mid);

        if (!w.ffn_sub_norm.shape.empty()) {
            Tensor normed = ffn_mid.clone_empty(ggml_type::F32);
            backend->rms_norm(ffn_mid, w.ffn_sub_norm, cfg.rms_norm_eps, normed);
            ffn_mid = std::move(normed);
        }

        Tensor ffn_out = Tensor({S, cfg.hidden_size}, ggml_type::F32);
        backend->matmul(ffn_mid, w.w_down, ffn_out);

        backend->add(residual, ffn_out, x);
    }

    Tensor normed = x.clone_empty(ggml_type::F32);
    backend->rms_norm(x, output_norm, cfg.rms_norm_eps, normed);

    Tensor logits({S, cfg.vocab_size}, ggml_type::F32);
    backend->matmul(normed, tok_embd, logits);

    return logits;
}

// Like forward_prefill but returns final-norm hidden states [S, H] without
// the logit projection. score_logprobs uses this to compute logits selectively.
Tensor Model::forward_prefill_hidden(const std::vector<int>& prompt_tokens) {
    int S = (int)prompt_tokens.size();
    if (S == 0) return Tensor({0, config.hidden_size}, ggml_type::F32);
    if (S > kv_cache.capacity) {
        std::cerr << "[forward_prefill_hidden] Prompt length " << S
                  << " exceeds KV cache capacity\n";
        return Tensor({0, config.hidden_size}, ggml_type::F32);
    }

    const auto& cfg = config;
    int n_heads = cfg.num_heads;
    int n_kv_heads = cfg.num_kv_heads;
    int head_dim = cfg.head_dim;

    Tensor x = extract_embeddings_batched(tok_embd, prompt_tokens);

    for (int layer = 0; layer < cfg.num_layers; ++layer) {
        const auto& w = layers[layer];
        Tensor attn_in = x.clone_empty(ggml_type::F32);
        backend->rms_norm(x, w.attn_norm, cfg.rms_norm_eps, attn_in);

        Tensor q({S, n_heads * head_dim}, ggml_type::F32);
        Tensor k({S, n_kv_heads * head_dim}, ggml_type::F32);
        Tensor v({S, n_kv_heads * head_dim}, ggml_type::F32);
        backend->matmul(attn_in, w.wq, q);
        backend->matmul(attn_in, w.wk, k);
        backend->matmul(attn_in, w.wv, v);

        q = q.view({S, n_heads, head_dim});
        k = k.view({S, n_kv_heads, head_dim});
        v = v.view({S, n_kv_heads, head_dim});
        tensor_rope_batched(q, k, 0, cfg.rope_theta);
        kv_cache.update(layer, 0, k, v);

        Tensor attn_out({S, n_heads, head_dim}, ggml_type::F32);
        compute_batched_attention(S, n_heads, n_kv_heads, head_dim,
            q.ptr<float>(), kv_cache.k_cache[layer].ptr<float>(),
            kv_cache.v_cache[layer].ptr<float>(), attn_out.ptr<float>());
        attn_out = attn_out.view({S, n_heads * head_dim});

        if (!w.attn_sub_norm.shape.empty()) {
            Tensor normed = attn_out.clone_empty(ggml_type::F32);
            backend->rms_norm(attn_out, w.attn_sub_norm, cfg.rms_norm_eps, normed);
            attn_out = std::move(normed);
        }

        Tensor attn_proj({S, cfg.hidden_size}, ggml_type::F32);
        backend->matmul(attn_out, w.wo, attn_proj);

        Tensor residual = x.clone_empty(ggml_type::F32);
        backend->add(x, attn_proj, residual);

        Tensor ffn_in = residual.clone_empty(ggml_type::F32);
        backend->rms_norm(residual, w.ffn_norm, cfg.rms_norm_eps, ffn_in);

        Tensor gate({S, cfg.intermediate_size}, ggml_type::F32);
        Tensor up  ({S, cfg.intermediate_size}, ggml_type::F32);
        backend->matmul(ffn_in, w.w_gate, gate);
        backend->matmul(ffn_in, w.w_up,   up);

        Tensor gate_r2 = gate.clone_empty(ggml_type::F32);
        backend->relu2(gate, gate_r2);
        Tensor ffn_mid = gate_r2.clone_empty(ggml_type::F32);
        backend->mul(gate_r2, up, ffn_mid);

        if (!w.ffn_sub_norm.shape.empty()) {
            Tensor normed = ffn_mid.clone_empty(ggml_type::F32);
            backend->rms_norm(ffn_mid, w.ffn_sub_norm, cfg.rms_norm_eps, normed);
            ffn_mid = std::move(normed);
        }

        Tensor ffn_out({S, cfg.hidden_size}, ggml_type::F32);
        backend->matmul(ffn_mid, w.w_down, ffn_out);
        backend->add(residual, ffn_out, x);
    }

    // Apply final norm and return hidden states — NO logit projection.
    Tensor normed = x.clone_empty(ggml_type::F32);
    backend->rms_norm(x, output_norm, cfg.rms_norm_eps, normed);
    return normed;  // [S, H]
}

#ifdef TERNATIVE_USE_CUDA
// === Phase 4: seed GPU KV cache from CPU cache after prefill ===
void Model::sync_kv_to_gpu(int n_positions) {
    if (n_positions <= 0 || n_gpu_layers <= 0) return;
    auto* cuda = dynamic_cast<CUDABackend*>(backend.get());
    if (!cuda || !cuda->has_gpu_kv(n_positions)) return;
    for (int layer = 0; layer < n_gpu_layers; ++layer) {
        const float* k_src = kv_cache.k_cache[layer].ptr<float>();
        const float* v_src = kv_cache.v_cache[layer].ptr<float>();
        cuda->upload_kv_layer_d(layer, n_positions, k_src, v_src);
    }
}

// === Phase 1: GPU-resident decode ===
//
// Keeps the hidden state d_x on the device for the whole transformer.  Only
// RoPE + attention + final logits cross PCIe (the latter only because we keep
// tok_embd on the CPU — 655 MB would not fit on a 4 GB GPU).  Reduces per-token
// sync count from ~210 to ~91.
//
// Layout invariants (BitNet 2B-4T):
//   hidden_size H = 2560, intermediate I = 6912, vocab V = 128256
//   Hq = n_heads*head_dim = 20*128 = 2560
//   Hk = n_kv_heads*head_dim = 5*128 = 640
//
// Device-buffer aliasing (allocated once in CUDABackend::init_gpu_activations):
//   d_x   [H]           — residual stream, persists across layers
//   d_norm[max(H,I)]    — scratch for rms_norm output AND attn-projection upload
//   d_q   [Hq]          — Q activations
//   d_k   [Hk]          — K activations
//   d_v   [Hk]          — V activations
//   d_gate[I]           — gate activation AND temp for w_o / w_down output
//                         (safe because H<=I, and we add into d_x immediately)
//   d_up  [I]           — up activation
Tensor Model::forward_gpu(int token_id, int pos) {
    auto* cuda = static_cast<CUDABackend*>(backend.get());
    const auto& cfg = config;
    const int H  = cfg.hidden_size;
    const int I  = cfg.intermediate_size;
    const int V  = cfg.vocab_size;
    const int Hq = cfg.num_heads    * cfg.head_dim;
    const int Hk = cfg.num_kv_heads * cfg.head_dim;
    const int head_dim   = cfg.head_dim;
    const int n_heads    = cfg.num_heads;
    const int n_kv_heads = cfg.num_kv_heads;

    float* dx   = cuda->d_x();
    float* dnrm = cuda->d_norm();
    float* dq   = cuda->d_q();
    float* dk   = cuda->d_k();
    float* dv   = cuda->d_v();
    float* dgt  = cuda->d_gate();
    float* dup  = cuda->d_up();

    // ── 1. Embedding lookup (CPU dequant → pinned → device) ────────────────
    {
        float* h_stage = cuda->h_attn();
        if (tok_embd.type == ggml_type::F16) {
            const f16_t* src = tok_embd.ptr<f16_t>() + (int64_t)token_id * H;
            for (int i = 0; i < H; ++i) h_stage[i] = src[i].to_float();
        } else if (tok_embd.type == ggml_type::BF16) {
            const bf16_t* src = tok_embd.ptr<bf16_t>() + (int64_t)token_id * H;
            for (int i = 0; i < H; ++i) h_stage[i] = src[i].to_float();
        } else { // F32
            const float* src = tok_embd.ptr<float>() + (int64_t)token_id * H;
            std::memcpy(h_stage, src, H * sizeof(float));
        }
        cuda->upload_pinned(dx, h_stage, H);
    }

    // Dispatch GEMV based on weight quantization (F16 vs INT8 GPU layers).
    auto dispatch = [&](float* out, const float* in, const Tensor& wt, int K, int N) {
        if (wt.gpu_is_int8)
            cuda->gemv_d2d_int8(out, in, wt.gpu_ptr, wt.scale, K, N);
        else
            cuda->gemv_d2d(out, in, wt.gpu_ptr, K, N);
    };

    // ── 2. Transformer layers (Phase 4: zero per-layer D2H) ───────────────
    for (int layer = 0; layer < n_gpu_layers; ++layer) {
        const auto& w = layers[layer];

        // (a) attn_norm + Q/K/V projections — all GPU, no sync.
        cuda->rms_norm_d(dnrm, dx, (const float*)w.d_attn_norm, cfg.rms_norm_eps, H);
        dispatch(dq, dnrm, w.wq, H, Hq);
        dispatch(dk, dnrm, w.wk, H, Hk);
        dispatch(dv, dnrm, w.wv, H, Hk);

        // (b) RoPE in-place on the GPU.
        cuda->rope_inplace_d(dq, dk, pos, cfg.rope_theta,
                             n_heads, n_kv_heads, head_dim);

        // (c) Write current-token k/v into GPU KV cache at slot `pos`.
        cuda->kv_write_d(dk, dv, layer, pos);

        // (d) Decode attention reads KV cache → writes attn_out into dnrm.
        float attn_scale = 1.0f / std::sqrt((float)head_dim);
        cuda->attention_d(dq, layer, pos + 1, dnrm,
                          n_heads, n_kv_heads, head_dim, attn_scale);

        // (e) Optional attn_sub_norm (GPU).
        const float* attn_proj_in = dnrm;
        if (w.d_attn_sub_norm != nullptr && !w.attn_sub_norm.shape.empty()) {
            cuda->rms_norm_d(dq, dnrm, (const float*)w.d_attn_sub_norm,
                             cfg.rms_norm_eps, Hq);
            attn_proj_in = dq;
        }

        // (f) wo projection → residual add (all GPU).
        dispatch(dgt, attn_proj_in, w.wo, Hq, H);
        cuda->add_d(dx, dx, dgt, H);

        // (g) ffn_norm + Gate/Up projections.
        cuda->rms_norm_d(dnrm, dx, (const float*)w.d_ffn_norm, cfg.rms_norm_eps, H);
        dispatch(dgt, dnrm, w.w_gate, H, I);
        dispatch(dup, dnrm, w.w_up,   H, I);

        // (h) ReLU²(gate) * up.
        cuda->relu2_d(dgt, dgt, I);
        cuda->mul_d(dgt, dgt, dup, I);

        // (i) GPU ffn_sub_norm + w_down + residual.
        if (!w.ffn_sub_norm.shape.empty() && w.d_ffn_sub_norm != nullptr) {
            cuda->rms_norm_d(dnrm, dgt, (const float*)w.d_ffn_sub_norm,
                             cfg.rms_norm_eps, I);
            dispatch(dgt, dnrm, w.w_down, I, H);
            cuda->add_d(dx, dx, dgt, H);
        } else {
            dispatch(dnrm, dgt, w.w_down, I, H);
            cuda->add_d(dx, dx, dnrm, H);
        }
    }

    // ── 2b. CPU tail: layers [n_gpu_layers, num_layers) not GPU-resident. ──
    // One sync + one D2H download hands the hidden state off to the CPU path.
    if (n_gpu_layers < cfg.num_layers) {
        // Sync and download the GPU-resident hidden state.
        cuda->sync_d();
        float* h_x = cuda->h_attn();
        cuda->download_pinned(h_x, dx, H);

        // Wrap in a Tensor for the existing CPU operators.
        Tensor x_cpu({1, (int64_t)H}, ggml_type::F32);
        std::memcpy(x_cpu.ptr<float>(), h_x, H * sizeof(float));

        for (int layer = n_gpu_layers; layer < cfg.num_layers; ++layer) {
            const auto& w = layers[layer];

            // Attention
            Tensor attn_in = x_cpu.clone_empty(ggml_type::F32);
            backend->rms_norm(x_cpu, w.attn_norm, cfg.rms_norm_eps, attn_in);

            Tensor q_t({1, (int64_t)(n_heads * head_dim)},    ggml_type::F32);
            Tensor k_t({1, (int64_t)(n_kv_heads * head_dim)}, ggml_type::F32);
            Tensor v_t({1, (int64_t)(n_kv_heads * head_dim)}, ggml_type::F32);
            backend->matmul(attn_in, w.wq, q_t);
            backend->matmul(attn_in, w.wk, k_t);
            backend->matmul(attn_in, w.wv, v_t);

            q_t = q_t.view({(int64_t)n_heads,    1, (int64_t)head_dim});
            k_t = k_t.view({(int64_t)n_kv_heads, 1, (int64_t)head_dim});
            v_t = v_t.view({(int64_t)n_kv_heads, 1, (int64_t)head_dim});
            backend->rope(q_t, k_t, pos, cfg.rope_theta);

            k_t = k_t.view({1, (int64_t)n_kv_heads, (int64_t)head_dim});
            v_t = v_t.view({1, (int64_t)n_kv_heads, (int64_t)head_dim});
            kv_cache.update(layer, pos, k_t, v_t);

            int kv_len = pos + 1;
            Tensor scores_t({1, (int64_t)n_heads, (int64_t)kv_len}, ggml_type::F32);
            float attn_scale = 1.0f / std::sqrt((float)head_dim);
            const float* qp = q_t.ptr<float>();
            const float* kcp = kv_cache.k_cache[layer].ptr<float>();
            float* sp = scores_t.ptr<float>();
            for (int h = 0; h < n_heads; ++h) {
                int kv_h = h / (n_heads / n_kv_heads);
                const float* q_head = qp + h * head_dim;
                for (int t = 0; t < kv_len; ++t) {
                    const float* k_head = kcp + t * n_kv_heads * head_dim + kv_h * head_dim;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) dot += q_head[d] * k_head[d];
                    sp[h * kv_len + t] = dot * attn_scale;
                }
            }
            Tensor aw_t = scores_t.clone_empty(ggml_type::F32);
            float* awp = aw_t.ptr<float>();
            for (int h = 0; h < n_heads; ++h) {
                float mx = -1e30f;
                for (int t = 0; t < kv_len; ++t) if (sp[h*kv_len+t] > mx) mx = sp[h*kv_len+t];
                float sm = 0.0f;
                for (int t = 0; t < kv_len; ++t) { awp[h*kv_len+t]=std::exp(sp[h*kv_len+t]-mx); sm+=awp[h*kv_len+t]; }
                float inv_sm = 1.0f / sm;
                for (int t = 0; t < kv_len; ++t) awp[h*kv_len+t] *= inv_sm;
            }
            Tensor attn_out_t({1, (int64_t)n_heads, (int64_t)head_dim}, ggml_type::F32);
            float* aop = attn_out_t.ptr<float>();
            const float* vcp = kv_cache.v_cache[layer].ptr<float>();
            for (int h = 0; h < n_heads; ++h) {
                int kv_h = h / (n_heads / n_kv_heads);
                float* out_head = aop + h * head_dim;
                for (int d = 0; d < head_dim; ++d) out_head[d] = 0.0f;
                for (int t = 0; t < kv_len; ++t) {
                    float wt = awp[h*kv_len+t];
                    const float* v_head = vcp + t * n_kv_heads * head_dim + kv_h * head_dim;
                    for (int d = 0; d < head_dim; ++d) out_head[d] += wt * v_head[d];
                }
            }
            attn_out_t = attn_out_t.view({1, (int64_t)(n_heads * head_dim)});

            if (!w.attn_sub_norm.shape.empty()) {
                Tensor normed_a = attn_out_t.clone_empty(ggml_type::F32);
                backend->rms_norm(attn_out_t, w.attn_sub_norm, cfg.rms_norm_eps, normed_a);
                attn_out_t = std::move(normed_a);
            }

            Tensor attn_proj_t({1, (int64_t)H}, ggml_type::F32);
            backend->matmul(attn_out_t, w.wo, attn_proj_t);
            Tensor res_t = x_cpu.clone_empty(ggml_type::F32);
            backend->add(x_cpu, attn_proj_t, res_t);

            // FFN
            Tensor ffn_in_t = res_t.clone_empty(ggml_type::F32);
            backend->rms_norm(res_t, w.ffn_norm, cfg.rms_norm_eps, ffn_in_t);
            Tensor gate_t({1, (int64_t)I}, ggml_type::F32);
            Tensor up_t  ({1, (int64_t)I}, ggml_type::F32);
            backend->matmul(ffn_in_t, w.w_gate, gate_t);
            backend->matmul(ffn_in_t, w.w_up,   up_t);
            Tensor gate_r2_t = gate_t.clone_empty(ggml_type::F32);
            backend->relu2(gate_t, gate_r2_t);
            Tensor ffn_mid_t = gate_r2_t.clone_empty(ggml_type::F32);
            backend->mul(gate_r2_t, up_t, ffn_mid_t);
            if (!w.ffn_sub_norm.shape.empty()) {
                Tensor normed_f = ffn_mid_t.clone_empty(ggml_type::F32);
                backend->rms_norm(ffn_mid_t, w.ffn_sub_norm, cfg.rms_norm_eps, normed_f);
                ffn_mid_t = std::move(normed_f);
            }
            Tensor ffn_out_t({1, (int64_t)H}, ggml_type::F32);
            backend->matmul(ffn_mid_t, w.w_down, ffn_out_t);
            backend->add(res_t, ffn_out_t, x_cpu);
        }

        // Final norm + logits on CPU.
        Tensor normed_cpu = x_cpu.clone_empty(ggml_type::F32);
        backend->rms_norm(x_cpu, output_norm, cfg.rms_norm_eps, normed_cpu);
        Tensor logits({1, (int64_t)V}, ggml_type::F32);
        backend->matmul(normed_cpu, tok_embd, logits);
        return logits;
    }
    // ── End CPU tail ─────────────────────────────────────────────────────────

    // ── 3. Final norm (GPU) + download. The download is the implicit sync. ─
    cuda->rms_norm_d(dnrm, dx, (const float*)d_output_norm, cfg.rms_norm_eps, H);

    // ── 4. Logits: GEMV on CPU because tok_embd stays in system RAM. ───────
    float* h_nrm = cuda->h_attn();
    cuda->download_pinned(h_nrm, dnrm, H);

    Tensor normed_cpu({1, (int64_t)H}, ggml_type::F32);
    std::memcpy(normed_cpu.ptr<float>(), h_nrm, H * sizeof(float));
    Tensor logits({1, V}, ggml_type::F32);
    backend->matmul(normed_cpu, tok_embd, logits);
    return logits;
}
#endif

// === Generation ===

std::vector<int> Model::generate(
    const std::vector<int>& prompt_tokens,
    int max_new_tokens,
    float temperature,
    float top_p,
    int top_k,
    int eos_token
) {
    std::vector<int> tokens = prompt_tokens;
    kv_cache.clear();
#ifdef TERNATIVE_USE_CUDA
    {
        auto* cuda = dynamic_cast<CUDABackend*>(backend.get());
        if (cuda && cuda->has_gpu_kv()) cuda->clear_kv_d();
    }
#endif

    // Forward prompt tokens (batched prefill)
    Tensor logits;
    if (!prompt_tokens.empty()) {
        logits = forward_prefill(prompt_tokens);
        kv_cache.seq_len = (int)prompt_tokens.size();
#ifdef TERNATIVE_USE_CUDA
        sync_kv_to_gpu(kv_cache.seq_len);
#endif
    } else {
        std::cerr << "[Model::generate] Warning: empty prompt tokens, returning empty output.\n";
        return {};
    }

    // Generate new tokens
    for (int i = 0; i < max_new_tokens; ++i) {
        SamplerConfig cfg;
        cfg.temperature = temperature;
        cfg.top_p = top_p;
        cfg.top_k = top_k;

        const float* logits_ptr = logits.ptr<float>();
        if (logits.shape.size() >= 2 && logits.shape[0] > 1) {
            // Batched prefill returns [S, vocab_size]; use last position
            logits_ptr += (logits.shape[0] - 1) * config.vocab_size;
        }

        int token_id = sample_token(logits_ptr, config.vocab_size, cfg, tokens);
        tokens.push_back(token_id);

        if (token_id == eos_token) break;
        if (kv_cache.seq_len >= config.max_seq_len) break;

        // Forward the just-sampled token to get logits for the next iteration
        logits = forward(token_id, kv_cache.seq_len, false);
        kv_cache.seq_len += 1;
    }

    return tokens;
}

Model::GenerationResult Model::generate_with_logprobs(
    const std::vector<int>& prompt_tokens,
    int max_new_tokens,
    float temperature,
    float top_p,
    int top_k,
    int eos_token,
    int num_logprobs
) {
    GenerationResult result;
    result.tokens = prompt_tokens;
    kv_cache.clear();
#ifdef TERNATIVE_USE_CUDA
    {
        auto* cuda = dynamic_cast<CUDABackend*>(backend.get());
        if (cuda && cuda->has_gpu_kv()) cuda->clear_kv_d();
    }
#endif

    // Forward prompt tokens (batched prefill)
    Tensor logits;
    if (!prompt_tokens.empty()) {
        logits = forward_prefill(prompt_tokens);
        kv_cache.seq_len = (int)prompt_tokens.size();
#ifdef TERNATIVE_USE_CUDA
        sync_kv_to_gpu(kv_cache.seq_len);
#endif
    } else {
        std::cerr << "[Model::generate_with_logprobs] Warning: empty prompt tokens.\n";
        return result;
    }

    for (int i = 0; i < max_new_tokens; ++i) {
        const float* logits_ptr = logits.ptr<float>();
        if (logits.shape.size() >= 2 && logits.shape[0] > 1) {
            logits_ptr += (logits.shape[0] - 1) * config.vocab_size;
        }

        // Compute log-softmax for logprobs
        int V = config.vocab_size;
        float max_logit = -1e30f;
        for (int v = 0; v < V; ++v) {
            if (logits_ptr[v] > max_logit) max_logit = logits_ptr[v];
        }
        float sum_exp = 0.0f;
        std::vector<float> log_probs(V);
        for (int v = 0; v < V; ++v) {
            log_probs[v] = logits_ptr[v] - max_logit;
            sum_exp += std::exp(log_probs[v]);
        }
        float log_sum_exp = std::log(sum_exp);
        for (int v = 0; v < V; ++v) {
            log_probs[v] -= log_sum_exp;  // Now log_probs[v] = log P(token v)
        }

        // Sample token
        SamplerConfig cfg;
        cfg.temperature = temperature;
        cfg.top_p = top_p;
        cfg.top_k = top_k;
        int token_id = sample_token(logits_ptr, V, cfg, result.tokens);

        // Find top-K logprobs for this token
        struct TopEntry { int id; float lp; };
        std::vector<TopEntry> top_entries(V);
        for (int v = 0; v < V; ++v) {
            top_entries[v] = {v, log_probs[v]};
        }
        std::partial_sort(top_entries.begin(), top_entries.begin() + std::min(num_logprobs, V), top_entries.end(),
            [](const TopEntry& a, const TopEntry& b) { return a.lp > b.lp; });

        TokenLogProb tlp;
        tlp.token_id = token_id;
        tlp.logprob = log_probs[token_id];
        tlp.text = "";

        // Store top-K logprobs
        int k = std::min(num_logprobs, V);
        tlp.top_logprobs.resize(k);
        for (int j = 0; j < k; ++j) {
            tlp.top_logprobs[j].token_id = top_entries[j].id;
            tlp.top_logprobs[j].logprob = top_entries[j].lp;
            tlp.top_logprobs[j].text = "";
        }

        result.tokens.push_back(token_id);
        result.logprobs.push_back(tlp);

        if (token_id == eos_token) break;
        if (kv_cache.seq_len >= config.max_seq_len) break;

        logits = forward(token_id, kv_cache.seq_len, false);
        kv_cache.seq_len += 1;
    }

    return result;
}

// Score a full sequence: compute log-probability for each token given the preceding context.
// Does NOT generate — evaluates logprobs for tokens in the input.
// start_pos: only compute logprobs for positions >= start_pos (0 = all).
// This avoids computing the expensive tok_embd logit GEMV for context-only positions.
Model::GenerationResult Model::score_logprobs(
    const std::vector<int>& tokens,
    int num_logprobs,
    int start_pos
) {
    GenerationResult result;
    result.tokens = tokens;
    kv_cache.clear();
#ifdef TERNATIVE_USE_CUDA
    {
        auto* cuda = dynamic_cast<CUDABackend*>(backend.get());
        if (cuda && cuda->has_gpu_kv()) cuda->clear_kv_d();
    }
#endif

    if (tokens.empty()) return result;

    int V = config.vocab_size;
    int T = (int)tokens.size();
    int first_pos = std::max(1, start_pos);  // earliest position needing logprobs

    // Prefill: run the full transformer for all tokens (needed for correct KV cache).
    // Then compute logits ONLY at positions [first_pos-1 .. T-1] (the positions that
    // predict the requested continuation tokens). This cuts tok_embd GEMV from T to
    // (T - first_pos + 1) calls — a significant saving when start_pos > 0.
    Tensor all_hidden = forward_prefill_hidden(tokens);  // [T, H], no logit GEMV
    kv_cache.seq_len = T;

    for (int pos = first_pos; pos < T; ++pos) {
        // Compute logits for position pos-1 → predicts token at pos
        const int H = config.hidden_size;
        Tensor h_row({1, H}, ggml_type::F32);
        std::memcpy(h_row.ptr<float>(),
                    all_hidden.ptr<float>() + (int64_t)(pos - 1) * H,
                    H * sizeof(float));

        // Final norm + logit projection (one token at a time)
        Tensor normed_row = h_row.clone_empty(ggml_type::F32);
        backend->rms_norm(h_row, output_norm, config.rms_norm_eps, normed_row);
        Tensor logits_row({1, V}, ggml_type::F32);
        backend->matmul(normed_row, tok_embd, logits_row);

        const float* logits_ptr = logits_row.ptr<float>();

        // Compute log-softmax
        float max_logit = -1e30f;
        for (int v = 0; v < V; ++v) {
            if (logits_ptr[v] > max_logit) max_logit = logits_ptr[v];
        }
        float sum_exp = 0.0f;
        std::vector<float> log_probs(V);
        for (int v = 0; v < V; ++v) {
            log_probs[v] = logits_ptr[v] - max_logit;
            sum_exp += std::exp(log_probs[v]);
        }
        float log_sum_exp = std::log(sum_exp);
        for (int v = 0; v < V; ++v) {
            log_probs[v] -= log_sum_exp;
        }

        // Find top-K
        struct TopEntry { int id; float lp; };
        std::vector<TopEntry> top_entries(V);
        for (int v = 0; v < V; ++v) {
            top_entries[v] = {v, log_probs[v]};
        }
        int k = std::min(num_logprobs, V);
        std::partial_sort(top_entries.begin(), top_entries.begin() + k, top_entries.end(),
            [](const TopEntry& a, const TopEntry& b) { return a.lp > b.lp; });

        int token_id = tokens[pos];
        TokenLogProb tlp;
        tlp.token_id = token_id;
        tlp.logprob = log_probs[token_id];
        tlp.text = "";
        tlp.top_logprobs.resize(k);
        for (int j = 0; j < k; ++j) {
            tlp.top_logprobs[j].token_id = top_entries[j].id;
            tlp.top_logprobs[j].logprob = top_entries[j].lp;
            tlp.top_logprobs[j].text = "";
        }
        result.logprobs.push_back(tlp);
    }

    return result;
}

} // namespace ternative
