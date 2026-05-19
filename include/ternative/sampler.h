#pragma once

#include <vector>
#include <cstdint>

namespace ternative {

struct SamplerConfig {
    float temperature = 0.8f;
    float top_p = 0.9f;
    int top_k = 40;
    int max_tokens = 512;
    float repetition_penalty = 1.1f;
};

// Sample a token from logits
// logits: [vocab_size] float array
// prompt_tokens: tokens seen so far (for repetition penalty)
int sample_token(
    const float* logits,
    int vocab_size,
    const SamplerConfig& cfg,
    const std::vector<int>& prompt_tokens
);

} // namespace ternative
