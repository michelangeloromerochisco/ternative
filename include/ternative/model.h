#pragma once

#include "tensor.h"
#include "ops.h"

#include <memory>
#include <vector>
#include <string>

namespace ternative {

// BitNet b1.58 model configuration
struct ModelConfig {
    int vocab_size = 128256;
    int hidden_size = 2560;
    int num_layers = 30;
    int num_heads = 20;
    int num_kv_heads = 5;
    int head_dim = 128;
    int intermediate_size = 6912;
    int max_seq_len = 4096;
    float rope_theta = 500000.0f;
    float rms_norm_eps = 1e-5f;
    bool tie_embeddings = true;
};

// Single transformer layer weights
struct LayerWeights {
    // Attention
    Tensor attn_norm;      // [hidden_size]
    Tensor wq;             // [hidden_size, hidden_size]  (or transposed)
    Tensor wk;             // [hidden_size, num_kv_heads * head_dim]
    Tensor wv;             // [hidden_size, num_kv_heads * head_dim]
    Tensor wo;             // [num_heads * head_dim, hidden_size]
    Tensor attn_sub_norm;  // [hidden_size] — BitNet extra

    // FFN
    Tensor ffn_norm;       // [hidden_size]
    Tensor w_gate;         // [hidden_size, intermediate_size]
    Tensor w_up;           // [hidden_size, intermediate_size]
    Tensor w_down;         // [intermediate_size, hidden_size]
    Tensor ffn_sub_norm;   // [intermediate_size] — BitNet extra

    // GPU-resident F32 norm-weight copies, populated when this layer is offloaded
    // to the CUDA backend. Owned by the CUDABackend (freed via release_gpu_mem in
    // Model destructor). Non-null only when the corresponding norm exists.
    void* d_attn_norm = nullptr;
    void* d_ffn_norm = nullptr;
    void* d_attn_sub_norm = nullptr;
    void* d_ffn_sub_norm = nullptr;
};

// LoRA adapter weights for a single layer
struct LayerLoRA {
    Tensor wq_a, wq_b;
    Tensor wk_a, wk_b;
    Tensor wv_a, wv_b;
    Tensor wo_a, wo_b;
    Tensor w_gate_a, w_gate_b;
    Tensor w_up_a, w_up_b;
    Tensor w_down_a, w_down_b;
};

// KV cache
struct KVCache {
    int n_layers = 0;
    int n_kv_heads = 0;
    int head_dim = 0;
    int capacity = 0;
    int seq_len = 0;  // Current sequence length

    std::vector<Tensor> k_cache;  // [n_layers][capacity, n_kv_heads, head_dim]
    std::vector<Tensor> v_cache;  // [n_layers][capacity, n_kv_heads, head_dim]

    void init(const ModelConfig& config, int capacity);
    void clear();
    void resize(int new_seq_len);
    void update(int layer, int pos, const Tensor& k, const Tensor& v);
};

// Pre-allocated activation buffers reused on every Model::forward() call to
// avoid 15+ vector heap allocations per token. Lifetime tied to the Model.
struct ActivationArena {
    bool initialized = false;
    Tensor t_attn_in;
    Tensor t_q;
    Tensor t_k;
    Tensor t_v;
    Tensor t_attn_proj;
    Tensor t_residual;
    Tensor t_ffn_in;
    Tensor t_gate;
    Tensor t_up;
    Tensor t_gate_relu2;
    Tensor t_ffn_mid;
    Tensor t_ffn_out;
    Tensor t_normed;

    void init(const ModelConfig& cfg);
};

// Full model
struct Model {
    ModelConfig config;
    Tensor tok_embd;       // [hidden_size, vocab_size] (GGUF column-major)
    Tensor output_norm;    // [hidden_size]

    std::vector<LayerWeights> layers;

    // Optional: loaded from LoRA adapter
    float lora_alpha = 0.0f;
    int lora_rank = 0;
    std::vector<LayerLoRA> lora_layers;
    bool has_lora = false;

    // Runtime state
    std::unique_ptr<ComputeBackend> backend;
    KVCache kv_cache;

    // Phase 0: pre-allocated CPU activation buffers (avoids per-token heap churn).
    ActivationArena arena;

    // Phase 1: GPU-resident output_norm (F32). Non-null when CUDABackend offloaded
    // at least one layer. Freed in ~Model via the backend's release_gpu_mem.
    void* d_output_norm = nullptr;

    // Number of leading layers that have GPU-resident weights AND norm copies.
    // Phase 1 forward_gpu() only runs when this equals config.num_layers.
    int n_gpu_layers = 0;

    // Free GPU-resident norm-weight copies (called from destructor).
    ~Model();
    Model() = default;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    // Load from GGUF files
    static std::unique_ptr<Model> load(
        const std::string& base_gguf_path,
        const std::vector<std::string>& lora_gguf_paths = {},
        bool use_cuda = false
    );

    // Forward pass for a single token
    // token_id: input token
    // pos: position in sequence
    // prefill: true during prompt processing (token attends to itself),
    //          false during autoregressive generation (token only attends to past)
    // Returns logits [vocab_size]
    Tensor forward(int token_id, int pos, bool prefill = false);

#ifdef TERNATIVE_USE_CUDA
    // GPU-resident single-token decode path. Activations stay on the device
    // throughout the transformer; only RoPE + attention + final logits cross
    // the PCIe boundary. Falls back to forward() if any layer is not fully
    // GPU-resident. Logits are returned on CPU.
    Tensor forward_gpu(int token_id, int pos);

    // After CPU forward_prefill() populates kv_cache, copy the first
    // `n_positions` slots of layers [0, n_gpu_layers) into the GPU KV cache so
    // subsequent forward_gpu() calls see the prompt's keys/values.
    void sync_kv_to_gpu(int n_positions);
#endif

    // Batched forward pass for the entire prompt.
    // Returns logits [prompt_len, vocab_size] — caller uses last row for sampling.
    Tensor forward_prefill(const std::vector<int>& prompt_tokens);

    // Like forward_prefill but returns final-norm hidden states [prompt_len, hidden_size]
    // without computing the logit projection (tok_embd GEMV). Used by score_logprobs
    // to compute logits selectively per-position instead of for the full batch.
    Tensor forward_prefill_hidden(const std::vector<int>& prompt_tokens);

    // Generate text from prompt tokens
    std::vector<int> generate(
        const std::vector<int>& prompt_tokens,
        int max_new_tokens,
        float temperature = 0.8f,
        float top_p = 0.9f,
        int top_k = 40,
        int eos_token = 128001
    );

    // Log-probability info for a single generated token
    struct TokenLogProb {
        int token_id;
        float logprob;
        std::string text;
        // Top-K logprobs for this position (sorted descending by logprob)
        struct TopEntry {
            int token_id;
            float logprob;
            std::string text;
        };
        std::vector<TopEntry> top_logprobs;
    };

    // Generate text with per-token log probabilities
    struct GenerationResult {
        std::vector<int> tokens;           // All tokens (prompt + generated)
        std::vector<TokenLogProb> logprobs; // Log probs for generated tokens only
    };

    GenerationResult generate_with_logprobs(
        const std::vector<int>& prompt_tokens,
        int max_new_tokens,
        float temperature = 0.8f,
        float top_p = 0.9f,
        int top_k = 40,
        int eos_token = 128001,
        int num_logprobs = 5  // Top-K logprobs to return per token
    );

    // Score a full sequence: compute log-probs for each token given context.
    // Does NOT generate — evaluates logprobs for tokens in the input.
    // start_pos: only compute logprobs for positions >= start_pos (default 0 = all).
    // Using start_pos = split avoids computing logits for context-only tokens,
    // cutting the expensive tok_embd GEMV from S × 18ms to (S-split) × 18ms.
    GenerationResult score_logprobs(
        const std::vector<int>& tokens,
        int num_logprobs = 5,
        int start_pos = 0
    );
};

} // namespace ternative
