#include "ternative/sampler.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

namespace ternative {

int sample_token(
    const float* logits,
    int vocab_size,
    const SamplerConfig& cfg,
    const std::vector<int>& prompt_tokens)
{
    // Make a mutable copy of logits for processing
    std::vector<float> probs(logits, logits + vocab_size);

    // 1. Apply temperature scaling
    if (cfg.temperature > 0.0f && cfg.temperature != 1.0f) {
        for (int i = 0; i < vocab_size; ++i) {
            probs[i] /= cfg.temperature;
        }
    }

    // 2. Apply repetition penalty
    if (cfg.repetition_penalty > 1.0f && !prompt_tokens.empty()) {
        std::vector<bool> penalized(vocab_size, false);
        for (int token_id : prompt_tokens) {
            if (token_id >= 0 && token_id < vocab_size && !penalized[token_id]) {
                // Divide logit by penalty (equivalent to subtracting log(penalty) in log-space)
                probs[token_id] /= cfg.repetition_penalty;
                penalized[token_id] = true;
            }
        }
    }

    // 3. Convert to probabilities using numerically stable softmax
    float max_logit = *std::max_element(probs.begin(), probs.end());
    double sum_exp = 0.0;
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp(probs[i] - max_logit);
        sum_exp += static_cast<double>(probs[i]);
    }
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] = static_cast<float>(static_cast<double>(probs[i]) / sum_exp);
    }

    // 4. Top-k filtering: keep only top k logits, set others to probability 0
    if (cfg.top_k > 0 && cfg.top_k < vocab_size) {
        std::vector<std::pair<float, int>> indexed;
        indexed.reserve(vocab_size);
        for (int i = 0; i < vocab_size; ++i) {
            indexed.emplace_back(probs[i], i);
        }
        std::nth_element(
            indexed.begin(),
            indexed.begin() + cfg.top_k,
            indexed.end(),
            std::greater<std::pair<float, int>>()
        );
        std::vector<bool> in_top_k(vocab_size, false);
        for (int i = 0; i < cfg.top_k; ++i) {
            in_top_k[indexed[i].second] = true;
        }
        for (int i = 0; i < vocab_size; ++i) {
            if (!in_top_k[i]) {
                probs[i] = 0.0f;
            }
        }
    }

    // 5. Top-p (nucleus) filtering: keep smallest set whose cumulative probability >= top_p
    if (cfg.top_p > 0.0f && cfg.top_p < 1.0f) {
        std::vector<std::pair<float, int>> indexed;
        indexed.reserve(vocab_size);
        for (int i = 0; i < vocab_size; ++i) {
            if (probs[i] > 0.0f) {
                indexed.emplace_back(probs[i], i);
            }
        }
        std::sort(
            indexed.begin(),
            indexed.end(),
            std::greater<std::pair<float, int>>()
        );
        float cumsum = 0.0f;
        std::vector<bool> in_nucleus(vocab_size, false);
        for (const auto& p : indexed) {
            cumsum += p.first;
            in_nucleus[p.second] = true;
            if (cumsum >= cfg.top_p) {
                break;
            }
        }
        for (int i = 0; i < vocab_size; ++i) {
            if (!in_nucleus[i]) {
                probs[i] = 0.0f;
            }
        }
    }

    // 6. Re-normalize probabilities after filtering
    sum_exp = 0.0;
    for (int i = 0; i < vocab_size; ++i) {
        sum_exp += static_cast<double>(probs[i]);
    }
    if (sum_exp <= 0.0) {
        // Fallback: if everything was filtered out, return the highest logit
        int best = 0;
        float best_logit = logits[0];
        for (int i = 1; i < vocab_size; ++i) {
            if (logits[i] > best_logit) {
                best_logit = logits[i];
                best = i;
            }
        }
        return best;
    }
    for (int i = 0; i < vocab_size; ++i) {
        probs[i] = static_cast<float>(static_cast<double>(probs[i]) / sum_exp);
    }

    // 7. Sample from the remaining distribution
    static thread_local std::mt19937 gen([]() {
        std::random_device rd;
        return std::mt19937(rd());
    }());

    // Build cumulative distribution for discrete sampling
    std::vector<float> cdf(vocab_size);
    float acc = 0.0f;
    for (int i = 0; i < vocab_size; ++i) {
        acc += probs[i];
        cdf[i] = acc;
    }

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float sample = dist(gen);

    for (int i = 0; i < vocab_size; ++i) {
        if (sample <= cdf[i]) {
            return i;
        }
    }
    return vocab_size - 1;
}

} // namespace ternative
