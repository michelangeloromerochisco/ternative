#include "ternative/tensor.h"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <string>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ternative {

// === AVX2 helpers ===

#ifdef __AVX2__
static inline float hsum256_ps(__m256 v) {
    __m128 vsum = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    vsum = _mm_add_ps(vsum, _mm_movehl_ps(vsum, vsum));
    vsum = _mm_add_ss(vsum, _mm_movehdup_ps(vsum));
    return _mm_cvtss_f32(vsum);
}
#endif

// === f16_t ===

static inline uint16_t float_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 31) & 0x1;
    uint32_t exp = (x >> 23) & 0xFF;
    uint32_t frac = x & 0x7FFFFF;

    if (exp == 0xFF) {
        // Inf/NaN
        return (sign << 15) | 0x7C00 | (frac >> 13);
    }

    int32_t new_exp = (int32_t)exp - 127 + 15;
    if (new_exp >= 31) {
        // Overflow to inf
        return (sign << 15) | 0x7C00;
    } else if (new_exp <= 0) {
        // Underflow to subnormal or zero
        if (new_exp < -10) return sign << 15;
        frac = (frac | 0x800000) >> (1 - new_exp);
        return (sign << 15) | (frac >> 13);
    }

    return (sign << 15) | ((uint16_t)new_exp << 10) | (frac >> 13);
}

static inline float f16_to_float(uint16_t h) {
    uint16_t sign = (h >> 15) & 0x1;
    uint16_t exp = (h >> 10) & 0x1F;
    uint16_t frac = h & 0x3FF;

    uint32_t result;
    if (exp == 0x1F) {
        result = (sign << 31) | 0x7F800000 | ((uint32_t)frac << 13);
    } else if (exp == 0) {
        if (frac == 0) {
            result = sign << 31;
        } else {
            // Subnormal
            uint32_t m = frac;
            int e = -14;
            while ((m & 0x400) == 0) {
                m <<= 1;
                e--;
            }
            result = (sign << 31) | ((uint32_t)(e + 127) << 23) | ((m & 0x3FF) << 13);
        }
    } else {
        result = (sign << 31) | ((uint32_t)(exp + 112) << 23) | ((uint32_t)frac << 13);
    }

    float f;
    std::memcpy(&f, &result, sizeof(f));
    return f;
}

f16_t::f16_t(float f) : bits(float_to_f16(f)) {}
float f16_t::to_float() const { return f16_to_float(bits); }

// === bf16_t ===

bf16_t::bf16_t(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    // BF16: keep top 16 bits of F32 (sign, 8-bit exponent, 7-bit mantissa)
    bits = static_cast<uint16_t>(x >> 16);
}

float bf16_t::to_float() const {
    uint32_t x = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &x, sizeof(f));
    return f;
}

// === Tensor ===

Tensor::Tensor(std::vector<int64_t> s, ggml_type t)
    : shape(std::move(s)), type(t) {
    size_t total = n_elements();
    if (type == ggml_type::I2_S) {
        // I2_S: packed in blocks of 128 weights (32 bytes per block), plus scale
        constexpr int QK_I2 = 128;
        size_t n_blocks = (total + QK_I2 - 1) / QK_I2;
        data.resize(n_blocks * (QK_I2 / 4) + sizeof(float));
    } else {
        data.resize(total * element_size());
    }
}

size_t Tensor::n_elements() const {
    size_t n = 1;
    for (auto d : shape) n *= (size_t)d;
    return n;
}

size_t Tensor::n_bytes() const {
    return data.size();
}

size_t Tensor::element_size() const {
    return ggml_type_size(type);
}

float Tensor::get_float(size_t idx) const {
    switch (type) {
        case ggml_type::F32: {
            const float* p = ptr<float>();
            return p[idx];
        }
        case ggml_type::F16: {
            const f16_t* p = ptr<f16_t>();
            return p[idx].to_float();
        }
        case ggml_type::BF16: {
            const bf16_t* p = ptr<bf16_t>();
            return p[idx].to_float();
        }
        default:
            // Quantized types not supported for direct float access
            return 0.0f;
    }
}

void Tensor::set_float(size_t idx, float val) {
    switch (type) {
        case ggml_type::F32: {
            float* p = ptr<float>();
            p[idx] = val;
            break;
        }
        case ggml_type::F16: {
            f16_t* p = ptr<f16_t>();
            p[idx] = f16_t(val);
            break;
        }
        case ggml_type::BF16: {
            bf16_t* p = ptr<bf16_t>();
            p[idx] = bf16_t(val);
            break;
        }
        default:
            break;
    }
}

Tensor Tensor::clone_empty(ggml_type new_type) const {
    return Tensor(shape, new_type);
}

Tensor Tensor::view(const std::vector<int64_t>& new_shape) const {
    Tensor v;
    v.shape = new_shape;
    v.type = type;
    v.data = data;  // Share data
    v.scale = scale;
    return v;
}

// === Basic tensor ops ===

void tensor_add(const Tensor& a, const Tensor& b, Tensor& out) {
    size_t n = a.n_elements();
    assert(n == b.n_elements());
    assert(n == out.n_elements());

    if (a.type == ggml_type::F32 && b.type == ggml_type::F32 && out.type == ggml_type::F32) {
        const float* ap = a.ptr<float>();
        const float* bp = b.ptr<float>();
        float* op = out.ptr<float>();
#ifdef __AVX2__
        size_t i = 0;
        for (; i + 7 < n; i += 8) {
            __m256 av = _mm256_loadu_ps(ap + i);
            __m256 bv = _mm256_loadu_ps(bp + i);
            _mm256_storeu_ps(op + i, _mm256_add_ps(av, bv));
        }
        for (; i < n; ++i) {
            op[i] = ap[i] + bp[i];
        }
#else
        for (size_t i = 0; i < n; ++i) {
            op[i] = ap[i] + bp[i];
        }
#endif
        return;
    }

    for (size_t i = 0; i < n; ++i) {
        float av = a.get_float(i);
        float bv = b.get_float(i);
        out.set_float(i, av + bv);
    }
}

void tensor_mul(const Tensor& a, const Tensor& b, Tensor& out) {
    size_t n = a.n_elements();
    assert(n == b.n_elements());
    assert(n == out.n_elements());

    if (a.type == ggml_type::F32 && b.type == ggml_type::F32 && out.type == ggml_type::F32) {
        const float* ap = a.ptr<float>();
        const float* bp = b.ptr<float>();
        float* op = out.ptr<float>();
#ifdef __AVX2__
        size_t i = 0;
        for (; i + 7 < n; i += 8) {
            __m256 av = _mm256_loadu_ps(ap + i);
            __m256 bv = _mm256_loadu_ps(bp + i);
            _mm256_storeu_ps(op + i, _mm256_mul_ps(av, bv));
        }
        for (; i < n; ++i) {
            op[i] = ap[i] * bp[i];
        }
#else
        for (size_t i = 0; i < n; ++i) {
            op[i] = ap[i] * bp[i];
        }
#endif
        return;
    }

    for (size_t i = 0; i < n; ++i) {
        float av = a.get_float(i);
        float bv = b.get_float(i);
        out.set_float(i, av * bv);
    }
}

void tensor_matmul(const Tensor& a, const Tensor& b, Tensor& out) {
    // a: [M, K]
    // b: [K, N] (GGUF convention: [in_features, out_features])
    // out: [M, N]
    assert(a.shape.size() == 2);
    assert(b.shape.size() == 2);
    assert(out.shape.size() == 2);

    int64_t M = a.shape[0];
    int64_t K = a.shape[1];
    int64_t N = b.shape[1];
    assert(b.shape[0] == K);
    assert(out.shape[0] == M);
    assert(out.shape[1] == N);

    // Fast path: F32 activations x I2_S weights -> F32 output (lowest bandwidth)
    if (a.type == ggml_type::F32 && b.type == ggml_type::I2_S && out.type == ggml_type::F32) {
        const float* ap = a.ptr<float>();
        float* op = out.ptr<float>();
        const uint8_t* b_data = b.data.data();

        constexpr int QK_I2 = 128;
        int64_t blocks_per_col = K / QK_I2;
        size_t col_stride_bytes = blocks_per_col * (QK_I2 / 4); // 32 bytes per block

        float scale = b.scale;
        if (scale == 0.0f) {
            size_t n_blocks_total = (b.n_elements() + QK_I2 - 1) / QK_I2;
            const float* scale_ptr = reinterpret_cast<const float*>(b.data.data() + n_blocks_total * 32);
            scale = *scale_ptr;
        }

        static const float map2bit[4] = {-1.0f, 0.0f, +1.0f, 0.0f};

        // Unified I2_S GEMM: dequantize each column once, then compute dot products
        // for all M rows with AVX2. For M==1 this is comparable to the old block-by-block
        // approach but with less temp-buffer churn; for M>1 it eliminates redundant
        // dequantization, which was the prefill bottleneck.
#ifdef _OPENMP
#pragma omp parallel
{
        // thread_local avoids a heap alloc per call; reused across tokens
        thread_local std::vector<float> temp;
        if ((int64_t)temp.size() < K) temp.resize(static_cast<size_t>(K));
#pragma omp for schedule(static)
#else
        thread_local std::vector<float> temp;
        if ((int64_t)temp.size() < K) temp.resize(static_cast<size_t>(K));
#endif
        for (int64_t n = 0; n < N; ++n) {
            const uint8_t* col_packed = b_data + n * col_stride_bytes;

            // Dequantize full column into temp[K]
            for (int64_t blk = 0; blk < blocks_per_col; ++blk) {
                const uint8_t* x = col_packed + blk * 32;
                float* t = temp.data() + blk * QK_I2;
                for (int gp = 0; gp < 32; ++gp) {
                    uint8_t byte = x[gp];
                    t[gp + 0 * 32] = scale * map2bit[(byte >> 6) & 0x3];
                    t[gp + 1 * 32] = scale * map2bit[(byte >> 4) & 0x3];
                    t[gp + 2 * 32] = scale * map2bit[(byte >> 2) & 0x3];
                    t[gp + 3 * 32] = scale * map2bit[(byte >> 0) & 0x3];
                }
            }

            // Compute dot products for all rows against dequantized column
            for (int64_t m = 0; m < M; ++m) {
                const float* a_row = ap + m * K;
                float sum = 0.0f;
#ifdef __AVX2__
                __m256 sum_vec = _mm256_setzero_ps();
                int64_t k = 0;
                for (; k + 7 < K; k += 8) {
                    __m256 tv = _mm256_loadu_ps(temp.data() + k);
                    __m256 av = _mm256_loadu_ps(a_row + k);
                    sum_vec = _mm256_fmadd_ps(tv, av, sum_vec);
                }
                sum = hsum256_ps(sum_vec);
                for (; k < K; ++k) {
                    sum += temp[static_cast<size_t>(k)] * a_row[k];
                }
#else
                for (int64_t k = 0; k < K; ++k) {
                    sum += temp[static_cast<size_t>(k)] * a_row[k];
                }
#endif
                op[m * N + n] = sum;
            }
        }
#ifdef _OPENMP
}
#endif
        return;
    }

    // Fast path: F32 activations x F16 weights -> F32 output (common inference case)
    // v1.2: N-tiled GEMV with N-parallel OpenMP for M=1 (single-token generation).
    //
    // Previous code parallelised over M (the row count).  For generation M=1, that
    // left 7 of 8 cores idle and created 2560 independent memory streams — far more
    // than the hardware prefetcher (8-12 streams) can handle, causing cache-miss
    // stalls on every column switch.
    //
    // Fix: process N_TILE=8 output neurons simultaneously per k-strip.
    //  • The activation register (av) is loaded ONCE and used for all 8 columns.
    //  • 8 streams is within the prefetcher's capability → hardware prefetch works.
    //  • OpenMP tiles are distributed across all cores → full CPU utilisation.
    if (a.type == ggml_type::F32 && b.type == ggml_type::F16 && out.type == ggml_type::F32) {
        const float* ap = a.ptr<float>();
        const f16_t* bp = b.ptr<f16_t>();
        float* op = out.ptr<float>();

#ifdef __AVX2__
        if (M == 1) {
            // ── N-tiled GEMV (single-token generation) ──────────────────────
            const float* a_row = ap;
            constexpr int64_t NT = 8;  // columns processed per tile
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int64_t nb = 0; nb < N; nb += NT) {
                const int64_t nc = (nb + NT <= N) ? NT : (N - nb);

                if (nc == NT) {
                    // ── Full tile: 2× unroll + software prefetch ─────────────
                    // Prefetch distance: 512 bytes = 256 f16 = 32 loop iterations ahead.
                    // Covers ~200 ns of DRAM latency at 3200 MHz DDR4.
                    const f16_t* c0 = bp + (nb+0)*K;
                    const f16_t* c1 = bp + (nb+1)*K;
                    const f16_t* c2 = bp + (nb+2)*K;
                    const f16_t* c3 = bp + (nb+3)*K;
                    const f16_t* c4 = bp + (nb+4)*K;
                    const f16_t* c5 = bp + (nb+5)*K;
                    const f16_t* c6 = bp + (nb+6)*K;
                    const f16_t* c7 = bp + (nb+7)*K;

                    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
                    __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
                    __m256 s4 = _mm256_setzero_ps(), s5 = _mm256_setzero_ps();
                    __m256 s6 = _mm256_setzero_ps(), s7 = _mm256_setzero_ps();

                    int64_t k = 0;
                    // 2× unrolled: process 16 activation elements per iteration,
                    // hiding cvtph_ps latency (5 cyc) behind the FMA pipeline.
                    for (; k + 15 < K; k += 16) {
                        // Prefetch next cache lines for all 8 columns
                        _mm_prefetch((const char*)(c0 + k + 256), _MM_HINT_T0);
                        _mm_prefetch((const char*)(c1 + k + 256), _MM_HINT_T0);
                        _mm_prefetch((const char*)(c2 + k + 256), _MM_HINT_T0);
                        _mm_prefetch((const char*)(c3 + k + 256), _MM_HINT_T0);
                        _mm_prefetch((const char*)(c4 + k + 256), _MM_HINT_T0);
                        _mm_prefetch((const char*)(c5 + k + 256), _MM_HINT_T0);
                        _mm_prefetch((const char*)(c6 + k + 256), _MM_HINT_T0);
                        _mm_prefetch((const char*)(c7 + k + 256), _MM_HINT_T0);

                        __m256 av0 = _mm256_loadu_ps(a_row + k);
                        __m256 av1 = _mm256_loadu_ps(a_row + k + 8);

                        s0 = _mm256_fmadd_ps(av0, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(c0+k))),   s0);
                        s0 = _mm256_fmadd_ps(av1, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(c0+k+8))), s0);
                        s1 = _mm256_fmadd_ps(av0, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(c1+k))),   s1);
                        s1 = _mm256_fmadd_ps(av1, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(c1+k+8))), s1);
                        s2 = _mm256_fmadd_ps(av0, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(c2+k))),   s2);
                        s2 = _mm256_fmadd_ps(av1, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(c2+k+8))), s2);
                        s3 = _mm256_fmadd_ps(av0, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(c3+k))),   s3);
                        s3 = _mm256_fmadd_ps(av1, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(c3+k+8))), s3);
                        s4 = _mm256_fmadd_ps(av0, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(c4+k))),   s4);
                        s4 = _mm256_fmadd_ps(av1, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(c4+k+8))), s4);
                        s5 = _mm256_fmadd_ps(av0, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(c5+k))),   s5);
                        s5 = _mm256_fmadd_ps(av1, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(c5+k+8))), s5);
                        s6 = _mm256_fmadd_ps(av0, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(c6+k))),   s6);
                        s6 = _mm256_fmadd_ps(av1, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(c6+k+8))), s6);
                        s7 = _mm256_fmadd_ps(av0, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(c7+k))),   s7);
                        s7 = _mm256_fmadd_ps(av1, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(c7+k+8))), s7);
                    }
                    // Single-step tail (handles K % 16 remainder, usually 0 for dim=2560)
                    for (; k + 7 < K; k += 8) {
                        __m256 av = _mm256_loadu_ps(a_row + k);
                        s0 = _mm256_fmadd_ps(av, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(c0+k))), s0);
                        s1 = _mm256_fmadd_ps(av, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(c1+k))), s1);
                        s2 = _mm256_fmadd_ps(av, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(c2+k))), s2);
                        s3 = _mm256_fmadd_ps(av, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(c3+k))), s3);
                        s4 = _mm256_fmadd_ps(av, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(c4+k))), s4);
                        s5 = _mm256_fmadd_ps(av, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(c5+k))), s5);
                        s6 = _mm256_fmadd_ps(av, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(c6+k))), s6);
                        s7 = _mm256_fmadd_ps(av, _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(c7+k))), s7);
                    }
                    float r0=hsum256_ps(s0), r1=hsum256_ps(s1);
                    float r2=hsum256_ps(s2), r3=hsum256_ps(s3);
                    float r4=hsum256_ps(s4), r5=hsum256_ps(s5);
                    float r6=hsum256_ps(s6), r7=hsum256_ps(s7);
                    for (; k < K; ++k) {
                        float av = a_row[k];
                        r0 += av * c0[k].to_float(); r1 += av * c1[k].to_float();
                        r2 += av * c2[k].to_float(); r3 += av * c3[k].to_float();
                        r4 += av * c4[k].to_float(); r5 += av * c5[k].to_float();
                        r6 += av * c6[k].to_float(); r7 += av * c7[k].to_float();
                    }
                    op[nb+0]=r0; op[nb+1]=r1; op[nb+2]=r2; op[nb+3]=r3;
                    op[nb+4]=r4; op[nb+5]=r5; op[nb+6]=r6; op[nb+7]=r7;
                } else {
                    // ── Partial tail tile (at most one tile, when N % 8 != 0) ──
                    __m256 sv[NT];
                    for (int64_t i = 0; i < nc; ++i) sv[i] = _mm256_setzero_ps();

                    int64_t k = 0;
                    for (; k + 7 < K; k += 8) {
                        __m256 av = _mm256_loadu_ps(a_row + k);
                        for (int64_t i = 0; i < nc; ++i) {
                            const f16_t* bc = bp + (nb+i)*K;
                            sv[i] = _mm256_fmadd_ps(av, _mm256_cvtph_ps(
                                _mm_loadu_si128((__m128i*)(bc+k))), sv[i]);
                        }
                    }
                    for (int64_t i = 0; i < nc; ++i) {
                        float sum = hsum256_ps(sv[i]);
                        const f16_t* bc = bp + (nb+i)*K;
                        for (int64_t ki = k; ki < K; ++ki)
                            sum += a_row[ki] * bc[ki].to_float();
                        op[nb+i] = sum;
                    }
                }
            }
            return;
        }

        // ── M > 1 (prefill): parallelise over M-rows ─────────────────────────
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t m = 0; m < M; ++m) {
            const float* a_row = ap + m * K;
            for (int64_t n = 0; n < N; ++n) {
                const f16_t* b_col = bp + n * K;
                __m256 sv = _mm256_setzero_ps();
                int64_t k = 0;
                for (; k + 7 < K; k += 8) {
                    __m256 av = _mm256_loadu_ps(a_row + k);
                    __m256 bv = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(b_col+k)));
                    sv = _mm256_fmadd_ps(av, bv, sv);
                }
                float sum = hsum256_ps(sv);
                for (; k < K; ++k) sum += a_row[k] * b_col[k].to_float();
                op[m * N + n] = sum;
            }
        }
#else
        // ── Scalar fallback (no AVX2) ──────────────────────────────────────
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t m = 0; m < M; ++m) {
            const float* a_row = ap + m * K;
            for (int64_t n = 0; n < N; ++n) {
                const f16_t* b_col = bp + n * K;
                float sum = 0.0f;
                for (int64_t k = 0; k < K; ++k)
                    sum += a_row[k] * b_col[k].to_float();
                op[m * N + n] = sum;
            }
        }
#endif
        return;
    }

    // Fast path: F32 activations x BF16 weights -> F32 output (BitNet uses BF16)
    // BF16 -> F32 is trivial: just shift the 16-bit value left by 16 bits.
    if (a.type == ggml_type::F32 && b.type == ggml_type::BF16 && out.type == ggml_type::F32) {
        const float* ap = a.ptr<float>();
        const uint16_t* bp = b.ptr<uint16_t>();  // BF16 stored as uint16_t
        float* op = out.ptr<float>();

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t m = 0; m < M; ++m) {
            const float* a_row = ap + m * K;
            for (int64_t n = 0; n < N; ++n) {
                const uint16_t* b_col = bp + n * K;
#ifdef __AVX2__
                __m256 sum_vec = _mm256_setzero_ps();
                int64_t k = 0;
                for (; k + 7 < K; k += 8) {
                    // Load 8 BF16 values and convert to F32 via integer shift
                    __m128i b16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b_col + k));
                    __m256i b32i = _mm256_cvtepu16_epi32(b16);
                    __m256i b_shifted = _mm256_slli_epi32(b32i, 16);
                    __m256 bv = _mm256_castsi256_ps(b_shifted);
                    __m256 av = _mm256_loadu_ps(a_row + k);
                    sum_vec = _mm256_fmadd_ps(av, bv, sum_vec);
                }
                float sum = hsum256_ps(sum_vec);
                for (; k < K; ++k) {
                    uint32_t bits = (uint32_t)b_col[k] << 16;
                    float bf;
                    std::memcpy(&bf, &bits, sizeof(bf));
                    sum += a_row[k] * bf;
                }
#else
                float sum = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    uint32_t bits = (uint32_t)b_col[k] << 16;
                    float bf;
                    std::memcpy(&bf, &bits, sizeof(bf));
                    sum += a_row[k] * bf;
                }
#endif
                op[m * N + n] = sum;
            }
        }
        return;
    }

    // Fast path: all F32 tensors
    if (a.type == ggml_type::F32 && b.type == ggml_type::F32 && out.type == ggml_type::F32) {
        const float* ap = a.ptr<float>();
        const float* bp = b.ptr<float>();
        float* op = out.ptr<float>();

#ifdef __AVX2__
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t m = 0; m < M; ++m) {
            const float* a_row = ap + m * K;
            for (int64_t n = 0; n < N; ++n) {
                const float* b_col = bp + n * K;
                __m256 sum_vec = _mm256_setzero_ps();
                int64_t k = 0;
                for (; k + 7 < K; k += 8) {
                    __m256 av = _mm256_loadu_ps(a_row + k);
                    __m256 bv = _mm256_loadu_ps(b_col + k);
                    sum_vec = _mm256_fmadd_ps(av, bv, sum_vec);
                }
                float sum = hsum256_ps(sum_vec);
                for (; k < K; ++k) {
                    sum += a_row[k] * b_col[k];
                }
                op[m * N + n] = sum;
            }
        }
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int64_t m = 0; m < M; ++m) {
            const float* a_row = ap + m * K;
            for (int64_t n = 0; n < N; ++n) {
                const float* b_col = bp + n * K;
                float sum = 0.0f;
                for (int64_t k = 0; k < K; ++k) {
                    sum += a_row[k] * b_col[k];
                }
                op[m * N + n] = sum;
            }
        }
#endif
        return;
    }

    // Slow path: mixed types or quantized tensors
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                float av = a.get_float(k * M + m);
                float bv = b.get_float(n * K + k);
                sum += av * bv;
            }
            out.set_float(n * M + m, sum);
        }
    }
}

void tensor_rms_norm(const Tensor& x, const Tensor& weight, float eps, Tensor& out) {
    assert(x.shape.size() >= 1);
    assert(weight.shape.size() == 1);
    int64_t n = x.shape.back();
    assert(weight.shape[0] == n);
    assert(out.shape == x.shape);

    size_t total = x.n_elements();
    size_t n_rows = total / n;

    for (size_t row = 0; row < n_rows; ++row) {
        float ss = 0.0f;

        if (x.type == ggml_type::F32 && weight.type == ggml_type::F32 && out.type == ggml_type::F32) {
            const float* xp = x.ptr<float>() + row * n;
            const float* wp = weight.ptr<float>();
            float* op = out.ptr<float>() + row * n;

#ifdef __AVX2__
            {
                __m256 sum_vec = _mm256_setzero_ps();
                int64_t i = 0;
                for (; i + 7 < n; i += 8) {
                    __m256 xv = _mm256_loadu_ps(xp + i);
                    sum_vec = _mm256_fmadd_ps(xv, xv, sum_vec);
                }
                ss = hsum256_ps(sum_vec);
                for (; i < n; ++i) {
                    ss += xp[i] * xp[i];
                }
            }
#else
            for (int64_t i = 0; i < n; ++i) {
                ss += xp[i] * xp[i];
            }
#endif
            float rms = std::sqrt(ss / n + eps);
            float inv_rms = 1.0f / rms;

#ifdef __AVX2__
            {
                __m256 inv_rms_vec = _mm256_set1_ps(inv_rms);
                int64_t i = 0;
                for (; i + 7 < n; i += 8) {
                    __m256 xv = _mm256_loadu_ps(xp + i);
                    __m256 wv = _mm256_loadu_ps(wp + i);
                    __m256 ov = _mm256_mul_ps(xv, _mm256_mul_ps(inv_rms_vec, wv));
                    _mm256_storeu_ps(op + i, ov);
                }
                for (; i < n; ++i) {
                    op[i] = xp[i] * inv_rms * wp[i];
                }
            }
#else
            for (int64_t i = 0; i < n; ++i) {
                op[i] = xp[i] * inv_rms * wp[i];
            }
#endif
        } else {
            for (int64_t i = 0; i < n; ++i) {
                float v = x.get_float(row * n + i);
                ss += v * v;
            }
            float rms = std::sqrt(ss / n + eps);
            float inv_rms = 1.0f / rms;
            for (int64_t i = 0; i < n; ++i) {
                float v = x.get_float(row * n + i);
                float w = weight.get_float(i);
                out.set_float(row * n + i, v * inv_rms * w);
            }
        }
    }
}

void tensor_softmax(const Tensor& x, Tensor& out) {
    assert(x.shape.size() >= 1);
    assert(out.shape == x.shape);

    int64_t n = x.shape.back();
    size_t total = x.n_elements();
    size_t n_rows = total / n;

    for (size_t row = 0; row < n_rows; ++row) {
        float max_val = -1e30f;

        if (x.type == ggml_type::F32 && out.type == ggml_type::F32) {
            const float* xp = x.ptr<float>() + row * n;
            float* op = out.ptr<float>() + row * n;

#ifdef __AVX2__
            {
                __m256 max_vec = _mm256_set1_ps(-1e30f);
                int64_t i = 0;
                for (; i + 7 < n; i += 8) {
                    __m256 xv = _mm256_loadu_ps(xp + i);
                    max_vec = _mm256_max_ps(max_vec, xv);
                }
                alignas(32) float max_vals[8];
                _mm256_storeu_ps(max_vals, max_vec);
                for (int j = 0; j < 8; ++j) {
                    max_val = std::max(max_val, max_vals[j]);
                }
                for (; i < n; ++i) {
                    max_val = std::max(max_val, xp[i]);
                }
            }
#else
            for (int64_t i = 0; i < n; ++i) {
                max_val = std::max(max_val, xp[i]);
            }
#endif

            float sum = 0.0f;
            for (int64_t i = 0; i < n; ++i) {
                float v = std::exp(xp[i] - max_val);
                op[i] = v;
                sum += v;
            }

            float inv_sum = 1.0f / sum;
#ifdef __AVX2__
            {
                __m256 inv_sum_vec = _mm256_set1_ps(inv_sum);
                int64_t i = 0;
                for (; i + 7 < n; i += 8) {
                    __m256 ov = _mm256_loadu_ps(op + i);
                    ov = _mm256_mul_ps(ov, inv_sum_vec);
                    _mm256_storeu_ps(op + i, ov);
                }
                for (; i < n; ++i) {
                    op[i] *= inv_sum;
                }
            }
#else
            for (int64_t i = 0; i < n; ++i) {
                op[i] *= inv_sum;
            }
#endif
        } else {
            for (int64_t i = 0; i < n; ++i) {
                max_val = std::max(max_val, x.get_float(row * n + i));
            }
            float sum = 0.0f;
            for (int64_t i = 0; i < n; ++i) {
                float v = std::exp(x.get_float(row * n + i) - max_val);
                out.set_float(row * n + i, v);
                sum += v;
            }
            float inv_sum = 1.0f / sum;
            for (int64_t i = 0; i < n; ++i) {
                out.set_float(row * n + i, out.get_float(row * n + i) * inv_sum);
            }
        }
    }
}

void tensor_relu2(const Tensor& x, Tensor& out) {
    assert(out.shape == x.shape);
    size_t n = x.n_elements();

    if (x.type == ggml_type::F32 && out.type == ggml_type::F32) {
        const float* xp = x.ptr<float>();
        float* op = out.ptr<float>();
#ifdef __AVX2__
        __m256 zero = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 7 < n; i += 8) {
            __m256 xv = _mm256_loadu_ps(xp + i);
            __m256 mv = _mm256_max_ps(xv, zero);
            __m256 ov = _mm256_mul_ps(mv, mv);
            _mm256_storeu_ps(op + i, ov);
        }
        for (; i < n; ++i) {
            float v = xp[i];
            if (v < 0.0f) v = 0.0f;
            op[i] = v * v;
        }
#else
        for (size_t i = 0; i < n; ++i) {
            float v = xp[i];
            if (v < 0.0f) v = 0.0f;
            op[i] = v * v;
        }
#endif
        return;
    }

    for (size_t i = 0; i < n; ++i) {
        float v = x.get_float(i);
        if (v < 0.0f) v = 0.0f;
        out.set_float(i, v * v);
    }
}

static inline float rope_compute_cos(int pos, int i, int head_dim, float theta) {
    float freq = 1.0f / std::pow(theta, (float)i / (float)head_dim);
    return std::cos(pos * freq);
}

static inline float rope_compute_sin(int pos, int i, int head_dim, float theta) {
    float freq = 1.0f / std::pow(theta, (float)i / (float)head_dim);
    return std::sin(pos * freq);
}

void tensor_rope(Tensor& q, Tensor& k, int pos, float theta) {
    // q shape: [n_heads, seq_len, head_dim]
    // k shape: [n_kv_heads, seq_len, head_dim]
    assert(q.shape.size() == 3);
    assert(k.shape.size() == 3);

    int64_t n_heads = q.shape[0];
    int64_t q_seq = q.shape[1];
    int64_t head_dim = q.shape[2];

    int64_t n_kv_heads = k.shape[0];
    int64_t k_seq = k.shape[1];
    assert(k.shape[2] == head_dim);

    int64_t half_dim = head_dim / 2;
    for (int64_t h = 0; h < n_heads; ++h) {
        for (int64_t s = 0; s < q_seq; ++s) {
            for (int64_t i = 0; i < head_dim; i += 2) {
                int64_t ic = i / 2;
                // NeoX RoPE: pair (ic, ic + half_dim)
                int64_t idx0 = h * q_seq * head_dim + s * head_dim + ic;
                int64_t idx1 = idx0 + half_dim;
                float x0 = q.get_float(idx0);
                float x1 = q.get_float(idx1);
                float cos_val = rope_compute_cos(pos + s, i, head_dim, theta);
                float sin_val = rope_compute_sin(pos + s, i, head_dim, theta);
                q.set_float(idx0, x0 * cos_val - x1 * sin_val);
                q.set_float(idx1, x0 * sin_val + x1 * cos_val);
            }
        }
    }

    for (int64_t h = 0; h < n_kv_heads; ++h) {
        for (int64_t s = 0; s < k_seq; ++s) {
            for (int64_t i = 0; i < head_dim; i += 2) {
                int64_t ic = i / 2;
                // NeoX RoPE: pair (ic, ic + half_dim)
                int64_t idx0 = h * k_seq * head_dim + s * head_dim + ic;
                int64_t idx1 = idx0 + half_dim;
                float x0 = k.get_float(idx0);
                float x1 = k.get_float(idx1);
                float cos_val = rope_compute_cos(pos + s, i, head_dim, theta);
                float sin_val = rope_compute_sin(pos + s, i, head_dim, theta);
                k.set_float(idx0, x0 * cos_val - x1 * sin_val);
                k.set_float(idx1, x0 * sin_val + x1 * cos_val);
            }
        }
    }
}

void tensor_rope_batched(Tensor& q, Tensor& k, int start_pos, float theta) {
    // q shape: [seq_len, n_heads, head_dim]
    // k shape: [seq_len, n_kv_heads, head_dim]
    assert(q.shape.size() == 3);
    assert(k.shape.size() == 3);

    int64_t S = q.shape[0];
    int64_t n_heads = q.shape[1];
    int64_t head_dim = q.shape[2];

    int64_t n_kv_heads = k.shape[1];
    assert(k.shape[0] == S);
    assert(k.shape[2] == head_dim);

    int64_t half_dim = head_dim / 2;
    float* qp = q.ptr<float>();
    for (int64_t s = 0; s < S; ++s) {
        for (int64_t h = 0; h < n_heads; ++h) {
            for (int64_t i = 0; i < head_dim; i += 2) {
                int64_t ic = i / 2;
                // NeoX RoPE: pair (ic, ic + half_dim)
                int64_t idx0 = s * n_heads * head_dim + h * head_dim + ic;
                int64_t idx1 = idx0 + half_dim;
                float x0 = qp[idx0];
                float x1 = qp[idx1];
                float cos_val = rope_compute_cos(start_pos + (int)s, (int)i, (int)head_dim, theta);
                float sin_val = rope_compute_sin(start_pos + (int)s, (int)i, (int)head_dim, theta);
                qp[idx0] = x0 * cos_val - x1 * sin_val;
                qp[idx1] = x0 * sin_val + x1 * cos_val;
            }
        }
    }

    float* kp = k.ptr<float>();
    for (int64_t s = 0; s < S; ++s) {
        for (int64_t h = 0; h < n_kv_heads; ++h) {
            for (int64_t i = 0; i < head_dim; i += 2) {
                int64_t ic = i / 2;
                // NeoX RoPE: pair (ic, ic + half_dim)
                int64_t idx0 = s * n_kv_heads * head_dim + h * head_dim + ic;
                int64_t idx1 = idx0 + half_dim;
                float x0 = kp[idx0];
                float x1 = kp[idx1];
                float cos_val = rope_compute_cos(start_pos + (int)s, (int)i, (int)head_dim, theta);
                float sin_val = rope_compute_sin(start_pos + (int)s, (int)i, (int)head_dim, theta);
                kp[idx0] = x0 * cos_val - x1 * sin_val;
                kp[idx1] = x0 * sin_val + x1 * cos_val;
            }
        }
    }
}

// === I2_S quantization / dequantization ===

Tensor tensor_quantize_i2_s(const Tensor& src) {
    assert(src.type == ggml_type::F32 || src.type == ggml_type::F16);
    size_t n = src.n_elements();
    Tensor dst(std::vector<int64_t>{(int64_t)n}, ggml_type::I2_S);

    // Microsoft BitNet format: scale = max(abs(src))
    float scale = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        scale = std::max(scale, std::abs(src.get_float(i)));
    }
    if (scale < 1e-6f) scale = 1e-6f;

    // Quantize to ternary codes: -1 -> 0, 0 -> 1, +1 -> 2
    std::vector<uint8_t> q8(n);
    for (size_t i = 0; i < n; ++i) {
        float v = src.get_float(i);
        if (std::abs(v) < 1e-6f) {
            q8[i] = 1;  // 0
        } else if (v > 0.0f) {
            q8[i] = 2;  // +1
        } else {
            q8[i] = 0;  // -1
        }
    }

    // Pack in blocks of 128. Byte gp holds:
    //   weight[gp + 0*32] in bits [7:6]
    //   weight[gp + 1*32] in bits [5:4]
    //   weight[gp + 2*32] in bits [3:2]
    //   weight[gp + 3*32] in bits [1:0]
    constexpr int QK_I2 = 128;
    uint8_t* i2_weight = dst.data.data();
    size_t n_blocks = (n + QK_I2 - 1) / QK_I2;

    for (size_t block = 0; block < n_blocks; ++block) {
        size_t block_start = block * QK_I2;
        size_t block_end = std::min(block_start + QK_I2, n);
        size_t block_n = block_end - block_start;

        size_t cols0 = block_n >= 32 ? 32 : block_n;
        size_t cols1 = block_n >= 64 ? 32 : (block_n > 32 ? block_n - 32 : 0);
        size_t cols2 = block_n >= 96 ? 32 : (block_n > 64 ? block_n - 64 : 0);
        size_t cols3 = block_n >= 128 ? 32 : (block_n > 96 ? block_n - 96 : 0);

        for (size_t gp = 0; gp < 32; ++gp) {
            uint8_t packed = 0;
            if (gp < cols0) packed |= (q8[block_start + gp + 0 * 32] << 6);
            if (gp < cols1) packed |= (q8[block_start + gp + 1 * 32] << 4);
            if (gp < cols2) packed |= (q8[block_start + gp + 2 * 32] << 2);
            if (gp < cols3) packed |= (q8[block_start + gp + 3 * 32] << 0);
            i2_weight[block * 32 + gp] = packed;
        }
    }

    // Store scale after the packed weight blocks (block-aligned, not n/4)
    float* scale_ptr = reinterpret_cast<float*>(dst.data.data() + n_blocks * 32);
    *scale_ptr = scale;
    dst.scale = scale;

    return dst;
}

std::vector<float> tensor_dequantize_i2_s_to_floats(const Tensor& src) {
    assert(src.type == ggml_type::I2_S);
    size_t n = src.n_elements();
    std::vector<float> result(n);

    // Microsoft BitNet format: scale stored after packed weight blocks
    constexpr int QK_I2 = 128;
    size_t n_blocks = (n + QK_I2 - 1) / QK_I2;
    float scale = src.scale;
    if (scale == 0.0f) {
        const float* scale_ptr = reinterpret_cast<const float*>(src.data.data() + n_blocks * 32);
        scale = *scale_ptr;
    }

    static const float map2bit[4] = {-1.0f, 0.0f, +1.0f, 0.0f};

    const uint8_t* x = src.data.data();
    size_t done = 0;

    while (done < n) {
        size_t blk_e = (n - done >= QK_I2) ? QK_I2 : (n - done);
        size_t cols0 = blk_e >= 32 ? 32 : blk_e;
        size_t cols1 = blk_e >= 64 ? 32 : (blk_e > 32 ? blk_e - 32 : 0);
        size_t cols2 = blk_e >= 96 ? 32 : (blk_e > 64 ? blk_e - 64 : 0);
        size_t cols3 = blk_e >= 128 ? 32 : (blk_e > 96 ? blk_e - 96 : 0);

        for (size_t gp = 0; gp < 32; ++gp) {
            uint8_t b = x[gp];
            uint8_t c0 = (b >> 6) & 0x3;
            uint8_t c1 = (b >> 4) & 0x3;
            uint8_t c2 = (b >> 2) & 0x3;
            uint8_t c3 = (b >> 0) & 0x3;

            if (gp < cols0) result[done + 0 * 32 + gp] = scale * map2bit[c0];
            if (gp < cols1) result[done + 1 * 32 + gp] = scale * map2bit[c1];
            if (gp < cols2) result[done + 2 * 32 + gp] = scale * map2bit[c2];
            if (gp < cols3) result[done + 3 * 32 + gp] = scale * map2bit[c3];
        }

        x += 32;
        done += blk_e;
    }

    return result;
}

void tensor_dequantize_i2_s(const Tensor& src, Tensor& dst) {
    assert(dst.type == ggml_type::F32 || dst.type == ggml_type::F16);
    auto floats = tensor_dequantize_i2_s_to_floats(src);
    size_t n = std::min(floats.size(), dst.n_elements());
    for (size_t i = 0; i < n; ++i) {
        dst.set_float(i, floats[i]);
    }
}

} // namespace ternative
