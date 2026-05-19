#include "ternative/tokenizer.h"
#include "ternative/gguf.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <unordered_set>

namespace ternative {

// ============================================================
// UTF-8 helpers
// ============================================================

// Encode a Unicode codepoint as a UTF-8 std::string (1-4 bytes)
static std::string cp_to_utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += (char)cp;
    } else if (cp < 0x800) {
        s += (char)(0xC0 | (cp >> 6));
        s += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += (char)(0xE0 | (cp >> 12));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    } else {
        s += (char)(0xF0 | (cp >> 18));
        s += (char)(0x80 | ((cp >> 12) & 0x3F));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    }
    return s;
}

// Decode one UTF-8 character at p, set len to byte length consumed, return codepoint.
// end is the first byte past the valid buffer — len is clamped so p+len <= end.
static uint32_t utf8_decode(const char* p, const char* end, int& len) {
    auto u = [](const char c) { return static_cast<uint8_t>(c); };
    uint8_t c0 = u(p[0]);
    if (c0 < 0x80) {
        len = 1; return c0;
    }
    if (c0 < 0xE0) {
        len = (p + 2 <= end) ? 2 : 1;
        if (len < 2) return c0;
        return ((c0 & 0x1F) << 6) | (u(p[1]) & 0x3F);
    }
    if (c0 < 0xF0) {
        len = (p + 3 <= end) ? 3 : 1;
        if (len < 3) return c0;
        return ((c0 & 0x0F) << 12) | ((u(p[1]) & 0x3F) << 6) | (u(p[2]) & 0x3F);
    }
    len = (p + 4 <= end) ? 4 : 1;
    if (len < 4) return c0;
    return ((c0 & 0x07) << 18) | ((u(p[1]) & 0x3F) << 12) | ((u(p[2]) & 0x3F) << 6) | (u(p[3]) & 0x3F);
}

// ============================================================
// GPT-2 byte ↔ unicode encoding table
// ============================================================
// Python reference:
//   bs = range(33,127) + range(161,173) + range(174,256)   # "printable" bytes
//   remaining bytes → chr(256), chr(257), ...
// Space (32) → chr(288) = U+0120 = Ġ
// 0-31, 127, 128-160, 173 → U+0100 through U+0123 (68 entries)

void Tokenizer::init_byte_encoding() {
    bool mapped[256] = {};

    auto add = [&](int b, uint32_t cp) {
        byte_to_char_[b] = cp_to_utf8(cp);
        char_to_byte_[byte_to_char_[b]] = static_cast<uint8_t>(b);
        mapped[b] = true;
    };

    // Printable ASCII: 33-126
    for (int i = 33; i <= 126; ++i) add(i, i);
    // Latin-1 printable: 161-172 (¡ to ¬)
    for (int i = 161; i <= 172; ++i) add(i, i);
    // Latin-1 printable: 174-255 (® to ÿ)
    for (int i = 174; i <= 255; ++i) add(i, i);

    // Remaining 68 bytes (0-32, 127, 128-160, 173) → U+0100+
    int n = 0;
    for (int i = 0; i < 256; ++i) {
        if (!mapped[i]) {
            add(i, 0x100 + n);
            ++n;
        }
    }
    // After this: space (32) → U+0120 (Ġ), tab (9) → U+0109, etc.
}

// ============================================================
// Load from GGUF
// ============================================================

bool Tokenizer::load(const std::string& gguf_path) {
    std::string path = gguf_path;
    if (path.empty()) {
        const char* env = std::getenv("ORCHID_MODEL_PATH");
        if (env) {
            // env may point to the HF model dir — we need the GGUF file
            path = std::string(env) + "/ggml-model-i2_s.gguf";
        } else {
            path = "orchid/models/base/bitnet-gguf/ggml-model-i2_s.gguf";
        }
    }

    // Metadata-only load: reads ~10 MB instead of the full 1.3 GB
    auto gguf = gguf_load_metadata(path);
    if (!gguf) {
        std::cerr << "[Tokenizer] Failed to load GGUF metadata from: " << path << "\n";
        return false;
    }

    // ── Vocabulary ────────────────────────────────────────────
    if (!gguf->has_metadata("tokenizer.ggml.tokens")) {
        std::cerr << "[Tokenizer] No tokenizer.ggml.tokens in GGUF\n";
        return false;
    }
    const auto& tok_meta = gguf->get_metadata("tokenizer.ggml.tokens");
    if (tok_meta.type != gguf_metadata_type::ARRAY) {
        std::cerr << "[Tokenizer] tokenizer.ggml.tokens is not an array\n";
        return false;
    }

    const size_t vsz = tok_meta.array_val.size();
    id_to_token_.resize(vsz);
    token_to_id_.reserve(vsz);
    for (size_t i = 0; i < vsz; ++i) {
        id_to_token_[i] = tok_meta.array_val[i].as_string();
        token_to_id_[id_to_token_[i]] = static_cast<int>(i);
    }

    // ── BPE merges ────────────────────────────────────────────
    if (!gguf->has_metadata("tokenizer.ggml.merges")) {
        std::cerr << "[Tokenizer] No tokenizer.ggml.merges in GGUF\n";
        return false;
    }
    const auto& merge_meta = gguf->get_metadata("tokenizer.ggml.merges");
    if (merge_meta.type == gguf_metadata_type::ARRAY) {
        merge_rank_.reserve(merge_meta.array_val.size());
        for (size_t i = 0; i < merge_meta.array_val.size(); ++i) {
            // Stored as "piece_a piece_b" — used directly as the lookup key
            merge_rank_[merge_meta.array_val[i].as_string()] = static_cast<int>(i);
        }
    }
    if (merge_rank_.empty()) {
        std::cerr << "[Tokenizer] BPE merges array is empty\n";
        return false;
    }

    // ── Special token IDs from metadata ──────────────────────
    if (gguf->has_metadata("tokenizer.ggml.bos_token_id"))
        bos_id_ = static_cast<int>(gguf->get_metadata("tokenizer.ggml.bos_token_id").as_u32());
    if (gguf->has_metadata("tokenizer.ggml.eos_token_id"))
        eos_id_ = static_cast<int>(gguf->get_metadata("tokenizer.ggml.eos_token_id").as_u32());

    // ── Mark special tokens (Llama 3: id ≥ 128000) ────────────
    for (int id = 128000; id < static_cast<int>(vsz) && id < 128256; ++id) {
        special_ids_.insert(id);
        special_token_map_[id_to_token_[id]] = id;
    }
    // Also mark token types 3 (control) and 4 (byte) as special if present
    if (gguf->has_metadata("tokenizer.ggml.token_type")) {
        const auto& tt = gguf->get_metadata("tokenizer.ggml.token_type");
        if (tt.type == gguf_metadata_type::ARRAY) {
            for (size_t i = 0; i < std::min(vsz, tt.array_val.size()); ++i) {
                int t = tt.array_val[i].as_i32();
                if (t == 3) special_ids_.insert(static_cast<int>(i));
            }
        }
    }

    // ── Byte encoding tables ──────────────────────────────────
    init_byte_encoding();

    loaded_ = true;
    std::cout << "[Tokenizer] Native BPE loaded: vocab=" << vsz
              << " merges=" << merge_rank_.size()
              << " special=" << special_ids_.size() << "\n";
    return true;
}

// ============================================================
// Pre-tokenizer
// ============================================================
// Approximates the Llama 3.1 tiktoken regex:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)
//   | [^\r\n\p{L}\p{N}]? \p{L}+
//   | \p{N}{1,3}
//   | ' '? [^\s\p{L}\p{N}]+ [\r\n]*
//   | \s* [\r\n]+
//   | \s+
// UTF-8 high bytes (≥0x80) are classified as letters for correctness with Spanish.

static char byte_cls(uint8_t b) {
    if (b == '\n' || b == '\r') return 'R';            // newline
    if (b == ' ' || b == '\t') return 'S';             // space
    if (b >= '0' && b <= '9') return 'N';              // digit
    if ((b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z') || b >= 0x80) return 'L'; // letter
    return 'P';                                         // punctuation / symbol
}

// Advance p over one complete UTF-8 character (1–4 bytes)
static const uint8_t* utf8_next(const uint8_t* p, const uint8_t* end) {
    if (p >= end) return p;
    uint8_t c = *p;
    if (c < 0x80)  return p + 1;
    if (c < 0xE0)  return p + ((p + 2 <= end) ? 2 : 1);
    if (c < 0xF0)  return p + ((p + 3 <= end) ? 3 : 1);
    return p + ((p + 4 <= end) ? 4 : 1);
}

std::vector<std::string> Tokenizer::pre_tokenize(const std::string& text) const {
    std::vector<std::string> pieces;
    const uint8_t* p   = reinterpret_cast<const uint8_t*>(text.data());
    const uint8_t* end = p + text.size();

    while (p < end) {
        const uint8_t* start = p;

        // ── Apostrophe contractions ────────────────────────────
        // Match 's  't  're  've  'm  'll  'd  (case-insensitive)
        if (*p == '\'' && p + 1 < end) {
            char n1 = static_cast<char>(std::tolower(p[1]));
            if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') {
                pieces.emplace_back(reinterpret_cast<const char*>(start), 2);
                p += 2; continue;
            }
            if ((n1 == 'r' || n1 == 'v' || n1 == 'l') && p + 2 < end) {
                char n2 = static_cast<char>(std::tolower(p[2]));
                if ((n1 == 'r' && n2 == 'e') ||
                    (n1 == 'v' && n2 == 'e') ||
                    (n1 == 'l' && n2 == 'l')) {
                    pieces.emplace_back(reinterpret_cast<const char*>(start), 3);
                    p += 3; continue;
                }
            }
        }

        // ── Optional-space + letters ───────────────────────────
        // Matches: ' '? <letter-sequence>
        // Space is included in the piece (becomes Ġ in BPE space)
        {
            bool has_sp = (*p == ' ');
            const uint8_t* q = has_sp ? p + 1 : p;
            if (q < end && byte_cls(*q) == 'L') {
                while (q < end && byte_cls(*q) == 'L')
                    q = utf8_next(q, end);
                pieces.emplace_back(reinterpret_cast<const char*>(p),
                                    q - p);
                p = q; continue;
            }
        }

        // ── 1–3 digits (no leading space) ─────────────────────
        if (byte_cls(*p) == 'N') {
            int n = 0;
            while (p < end && byte_cls(*p) == 'N' && n < 3) { ++p; ++n; }
            pieces.emplace_back(reinterpret_cast<const char*>(start), p - start);
            continue;
        }

        // ── Newlines (with optional preceding spaces) ──────────
        if (byte_cls(*p) == 'R') {
            while (p < end && (byte_cls(*p) == 'R' || byte_cls(*p) == 'S')) ++p;
            pieces.emplace_back(reinterpret_cast<const char*>(start), p - start);
            continue;
        }

        // ── Optional-space + punctuation ──────────────────────
        {
            bool has_sp = (*p == ' ');
            const uint8_t* q = has_sp ? p + 1 : p;
            if (q < end && byte_cls(*q) == 'P') {
                while (q < end && byte_cls(*q) == 'P') ++q;
                // Consume optional trailing newline(s)
                while (q < end && byte_cls(*q) == 'R') ++q;
                pieces.emplace_back(reinterpret_cast<const char*>(p), q - p);
                p = q; continue;
            }
        }

        // ── Remaining spaces ───────────────────────────────────
        if (byte_cls(*p) == 'S') {
            while (p < end && byte_cls(*p) == 'S') ++p;
            pieces.emplace_back(reinterpret_cast<const char*>(start), p - start);
            continue;
        }

        // ── Fallback: single byte ──────────────────────────────
        pieces.emplace_back(reinterpret_cast<const char*>(p), 1);
        ++p;
    }

    return pieces;
}

// ============================================================
// BPE encode one pre-tokenized piece
// ============================================================

std::vector<int> Tokenizer::bpe_encode_piece(const std::string& piece) const {
    if (piece.empty()) return {};

    // Convert each raw byte to its unicode representation string
    std::vector<std::string> syms;
    syms.reserve(piece.size());
    for (uint8_t b : piece)
        syms.push_back(byte_to_char_[b]);

    // Greedy BPE: repeatedly find the pair with the lowest merge rank and apply it
    while (syms.size() > 1) {
        int best_rank = INT_MAX;
        int best_i    = -1;

        for (int i = 0; i + 1 < static_cast<int>(syms.size()); ++i) {
            // Merge key: "sym_i sym_{i+1}" (literal ASCII space separator)
            std::string key = syms[i] + ' ' + syms[i + 1];
            auto it = merge_rank_.find(key);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_i    = i;
            }
        }

        if (best_i < 0) break;  // No applicable merge remains

        syms[best_i] += syms[best_i + 1];
        syms.erase(syms.begin() + best_i + 1);
    }

    // Map each merged symbol to its token ID
    std::vector<int> ids;
    ids.reserve(syms.size());
    for (const auto& sym : syms) {
        auto it = token_to_id_.find(sym);
        if (it != token_to_id_.end()) {
            ids.push_back(it->second);
        } else {
            // Byte fallback: sym is already a unicode-encoded merged symbol.
            // Each character in sym (as a UTF-8 codepoint, NOT a raw byte) is
            // the unicode representation of one original byte — look each up individually.
            const char* q    = sym.data();
            const char* qend = q + sym.size();
            while (q < qend) {
                int clen = 0;
                utf8_decode(q, qend, clen);
                if (clen <= 0) { ++q; continue; }
                std::string ch(q, clen);
                auto jt = token_to_id_.find(ch);
                if (jt != token_to_id_.end())
                    ids.push_back(jt->second);
                q += clen;
            }
        }
    }
    return ids;
}

// ============================================================
// Encode
// ============================================================

std::vector<int> Tokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int> result;
    if (add_bos) result.push_back(bos_id_);
    if (text.empty()) return result;

    // Build sorted special-token list (longest first — greedy match).
    // Cache as a sorted vector on the stack for this call.
    struct Sp { const std::string* str; int id; };
    std::vector<Sp> specials;
    specials.reserve(special_token_map_.size());
    for (const auto& kv : special_token_map_)
        specials.push_back({&kv.first, kv.second});
    std::sort(specials.begin(), specials.end(),
              [](const Sp& a, const Sp& b){ return a.str->size() > b.str->size(); });

    // Build a fast first-char set: chars that can begin any special token.
    // All Llama 3.1 specials start with '<', so this set is almost always {'<'}.
    // The inner scan loop checks this before doing the full string compare — reduces
    // per-byte overhead from O(256 comparisons) to O(1 char check) for typical text.
    std::unordered_set<char> special_lead_chars;
    special_lead_chars.reserve(8);
    for (const auto& sp : specials)
        if (!sp.str->empty()) special_lead_chars.insert((*sp.str)[0]);

    // Scan the text, emitting special tokens directly and BPE-encoding the rest
    size_t i = 0;
    while (i < text.size()) {
        // Fast path: current byte can't begin any special token
        if (!special_lead_chars.count(text[i])) {
            // Collect segment until we hit a potential special-token lead char or end
            size_t seg_end = i + 1;
            while (seg_end < text.size() && !special_lead_chars.count(text[seg_end]))
                ++seg_end;
            std::string seg = text.substr(i, seg_end - i);
            for (const auto& piece : pre_tokenize(seg)) {
                auto toks = bpe_encode_piece(piece);
                result.insert(result.end(), toks.begin(), toks.end());
            }
            i = seg_end;
            continue;
        }

        // Current byte matches a special-token lead char — try longest match
        bool found = false;
        for (const auto& sp : specials) {
            const std::string& s = *sp.str;
            if (s.size() <= text.size() - i &&
                text.compare(i, s.size(), s) == 0) {
                result.push_back(sp.id);
                i += s.size();
                found = true;
                break;
            }
        }
        if (found) continue;

        // Lead char matched but no special token — emit this char as regular text
        // Collect until next special-lead-char candidate
        size_t seg_end = i + 1;
        while (seg_end < text.size() && !special_lead_chars.count(text[seg_end]))
            ++seg_end;
        std::string seg = text.substr(i, seg_end - i);
        for (const auto& piece : pre_tokenize(seg)) {
            auto toks = bpe_encode_piece(piece);
            result.insert(result.end(), toks.begin(), toks.end());
        }
        i = seg_end;
    }

    return result;
}

// ============================================================
// Decode
// ============================================================

std::string Tokenizer::decode(const std::vector<int>& tokens, bool skip_special) const {
    std::string result;
    for (int id : tokens) {
        if (id < 0 || id >= static_cast<int>(id_to_token_.size())) continue;
        if (skip_special && special_ids_.count(id)) continue;

        const std::string& piece = id_to_token_[id];

        // Walk the piece character by character (UTF-8 codepoints),
        // converting each unicode char back to the raw byte it represents.
        const char* p   = piece.data();
        const char* pend = p + piece.size();
        while (p < pend) {
            int clen = 0;
            utf8_decode(p, pend, clen);
            if (clen <= 0) { ++p; continue; }
            std::string key(p, clen);
            auto it = char_to_byte_.find(key);
            if (it != char_to_byte_.end()) {
                result += static_cast<char>(it->second);
            } else {
                // Pass through unchanged — chars outside the byte-unicode mapping
                // (shouldn't occur for well-formed Llama 3.1 vocab entries)
                result.append(p, clen);
            }
            p += clen;
        }
    }
    return result;
}

// ============================================================
// Chat template
// ============================================================

std::string Tokenizer::apply_chat_template(
    const std::vector<std::pair<std::string, std::string>>& messages) const
{
    // Llama 3.1 instruct format.
    // NOTE: \n\n (double newline) between the header and content is REQUIRED.
    // Single \n corrupts the context framing — the model was trained with \n\n.
    // BOS (<|begin_of_text|>) is added by encode(text, add_bos=true).
    std::ostringstream oss;
    for (const auto& msg : messages) {
        oss << "<|start_header_id|>" << msg.first << "<|end_header_id|>\n\n"
            << msg.second << "<|eot_id|>";
    }
    oss << "<|start_header_id|>assistant<|end_header_id|>\n\n";
    return oss.str();
}

} // namespace ternative
