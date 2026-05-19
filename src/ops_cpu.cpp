#include "ternative/ops.h"
#include "ternative/tensor.h"

#include <cmath>
#include <algorithm>

#ifdef TERNATIVE_USE_OPENBLAS
#include <cblas.h>
#endif

namespace ternative {

void CPUBackend::matmul(const Tensor& a, const Tensor& b, Tensor& out) {
#ifdef TERNATIVE_USE_OPENBLAS
    if (a.type == ggml_type::F32 && b.type == ggml_type::F32 && out.type == ggml_type::F32) {
        assert(a.shape.size() == 2 && b.shape.size() == 2 && out.shape.size() == 2);
        int64_t M = a.shape[0];
        int64_t K = a.shape[1];
        int64_t N = b.shape[1];
        assert(b.shape[0] == K);
        assert(out.shape[0] == M);
        assert(out.shape[1] == N);
        // a is row-major MxK; b is column-major KxN (GGUF) which equals row-major NxK transposed.
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)M, (int)N, (int)K,
                    1.0f, a.f32_data(), (int)K,
                    b.f32_data(), (int)K,
                    0.0f, out.f32_data(), (int)N);
        return;
    }
#endif
    tensor_matmul(a, b, out);
}

void CPUBackend::rms_norm(const Tensor& x, const Tensor& weight, float eps, Tensor& out) {
    tensor_rms_norm(x, weight, eps, out);
}

void CPUBackend::rope(Tensor& q, Tensor& k, int pos, float theta) {
    tensor_rope(q, k, pos, theta);
}

void CPUBackend::softmax(const Tensor& x, Tensor& out) {
    tensor_softmax(x, out);
}

void CPUBackend::relu2(const Tensor& x, Tensor& out) {
    tensor_relu2(x, out);
}

void CPUBackend::add(const Tensor& a, const Tensor& b, Tensor& out) {
    tensor_add(a, b, out);
}

void CPUBackend::mul(const Tensor& a, const Tensor& b, Tensor& out) {
    tensor_mul(a, b, out);
}

} // namespace ternative
