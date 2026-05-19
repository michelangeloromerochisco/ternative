#pragma once

#include <cstdint>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace ternative {

// Native BPE tokenizer — loads directly from the base GGUF file.
// Implements the GPT-2 / tiktoken byte-level BPE used by Llama 3.1 / BitNet b1.58.
// Zero Python subprocess dependency; encode/decode run in <1 ms.
class Tokenizer {
public:
    // Load tokenizer vocab and BPE merges from the base model GGUF.
    // Uses gguf_load_metadata (reads ~10 MB, not the full 1.3 GB).
    // Falls back to ORCHID_MODEL_PATH env var or "orchid/models/base/bitnet-gguf/ggml-model-i2_s.gguf"
    // if path is empty.
    bool load(const std::string& gguf_path = "");

    // Encode text → token IDs.  add_bos prepends <|begin_of_text|> (128000).
    std::vector<int> encode(const std::string& text, bool add_bos = true) const;

    // Decode token IDs → UTF-8 text.  skip_special strips control tokens (id ≥ 128000).
    std::string decode(const std::vector<int>& tokens, bool skip_special = true) const;

    // Special token IDs (Llama 3.1 / BitNet b1.58 values)
    int bos_token_id() const { return bos_id_; }
    int eos_token_id() const { return eos_id_; }
    int eot_token_id() const { return eot_id_; }  // <|eot_id|> = 128009

    // Build a Llama 3.1 instruct prompt string from (role, content) pairs.
    // BOS is NOT included here — encode(text, add_bos=true) adds it.
    std::string apply_chat_template(
        const std::vector<std::pair<std::string, std::string>>& messages) const;

    bool is_loaded() const { return loaded_; }
    size_t vocab_size() const { return id_to_token_.size(); }

private:
    bool loaded_ = false;

    // Vocabulary
    std::vector<std::string>          id_to_token_;   // token_id → piece (unicode repr)
    std::unordered_map<std::string, int> token_to_id_; // piece → token_id

    // BPE merge rank table: "piece_a piece_b" (space-separated) → rank (lower = higher priority)
    std::unordered_map<std::string, int> merge_rank_;

    // GPT-2 byte ↔ unicode encoding tables
    // byte_to_char_[b] = UTF-8 string of the unicode codepoint that represents byte b
    std::string byte_to_char_[256];
    std::unordered_map<std::string, uint8_t> char_to_byte_;

    // IDs of all control / special tokens (id ≥ 128000 for Llama 3)
    std::set<int> special_ids_;

    // Special-token string → ID (for fast lookup during encode)
    std::unordered_map<std::string, int> special_token_map_;

    int bos_id_ = 128000;
    int eos_id_ = 128001;
    int eot_id_ = 128009;

    void init_byte_encoding();

    // Split text into pre-tokenizer pieces (simplified Llama 3.1 regex, UTF-8 aware)
    std::vector<std::string> pre_tokenize(const std::string& text) const;

    // BPE encode a single pre-tokenized piece (no special tokens inside)
    std::vector<int> bpe_encode_piece(const std::string& piece) const;
};

} // namespace ternative
