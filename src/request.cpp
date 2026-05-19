#include "ternative/request.h"
#include "ternative/sampler.h"
#include "ternative/ops.h"

#include <iostream>
#include <sstream>

namespace ternative {

// ---------------------------------------------------------------------------
// RequestQueue
// ---------------------------------------------------------------------------
void RequestQueue::submit(std::shared_ptr<GenRequest> r) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        q_.push_back(std::move(r));
    }
    cv_.notify_one();
}

std::shared_ptr<GenRequest> RequestQueue::wait_pop() {
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [this]{ return !q_.empty() || stopped_; });
    if (q_.empty()) return nullptr;
    auto r = std::move(q_.front());
    q_.pop_front();
    return r;
}

void RequestQueue::shutdown() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stopped_ = true;
    }
    cv_.notify_all();
}

// ---------------------------------------------------------------------------
// StopMatcher
// ---------------------------------------------------------------------------
StopMatcher::StopMatcher(const std::vector<std::string>& patterns)
    : pats_(patterns) {}

int StopMatcher::feed(const std::string& piece) {
    if (pats_.empty()) return (int)piece.size();

    buf_ += piece;

    // Check for complete match
    for (const auto& p : pats_) {
        size_t pos = buf_.find(p);
        if (pos != std::string::npos) {
            // Complete match found. Return negative = stop triggered.
            // Trim buf_ to everything before the match.
            buf_ = buf_.substr(0, pos);
            return -1;
        }
    }

    // Check for partial match at end of buffer
    // Return safe prefix = buffer minus the longest partial match suffix
    size_t safe_len = buf_.size();
    for (const auto& p : pats_) {
        // Find the longest suffix of buf_ that is a prefix of p
        for (size_t cand = std::min(buf_.size(), p.size()); cand > 0; --cand) {
            if (buf_.size() >= cand &&
                buf_.substr(buf_.size() - cand) == p.substr(0, cand)) {
                safe_len = std::min(safe_len, buf_.size() - cand);
                break;
            }
        }
    }

    // Return how many bytes of 'piece' are safe to emit
    // (buf_ now contains piece appended; safe_len is offset in buf_)
    // We need the safe prefix relative to 'piece' not 'buf_'
    size_t prev_buf_size = buf_.size() - piece.size();
    int safe_in_piece = (int)safe_len - (int)prev_buf_size;
    return std::max(0, safe_in_piece);
}

std::string StopMatcher::flush_buffer() {
    std::string out = std::move(buf_);
    buf_.clear();
    return out;
}

// ---------------------------------------------------------------------------
// Worker
// ---------------------------------------------------------------------------
Worker::Worker(Model& model, Tokenizer& tok, RequestQueue& queue)
    : model_(model), tok_(tok), queue_(queue) {}

Worker::~Worker() { stop(); }

void Worker::start() {
    running_ = true;
    thread_ = std::thread([this]{ run(); });
}

void Worker::stop() {
    running_ = false;
    if (thread_.joinable()) thread_.join();
}

void Worker::run() {
    while (running_) {
        auto req = queue_.wait_pop();
        if (!req) break;
        try {
            process(*req);
        } catch (const std::exception& e) {
            report_error(*req, e.what());
        }
    }
}

void Worker::process(GenRequest& req) {
    if (req.prompt_tokens.empty()) {
        finish(req, "stop", "");
        return;
    }

    const int stop_token = tok_.eot_token_id();
    const int eos_token  = tok_.eos_token_id();

    // Prefill: process all prompt tokens in one batched forward pass
    model_.kv_cache.clear();
#ifdef TERNATIVE_USE_CUDA
    {
        auto* cuda = dynamic_cast<CUDABackend*>(model_.backend.get());
        if (cuda && cuda->has_gpu_kv()) cuda->clear_kv_d();
    }
#endif
    Tensor logits = model_.forward_prefill(req.prompt_tokens);
    model_.kv_cache.seq_len = (int)req.prompt_tokens.size();
#ifdef TERNATIVE_USE_CUDA
    model_.sync_kv_to_gpu(model_.kv_cache.seq_len);
#endif

    SamplerConfig scfg;
    scfg.temperature = req.temperature;
    scfg.top_p       = req.top_p;
    scfg.top_k       = req.top_k;

    StopMatcher sm(req.stop_sequences);
    std::vector<int> all_tokens = req.prompt_tokens;  // for repetition penalty
    std::vector<int> gen_tokens;
    std::string finish_reason = "length";

    for (int i = 0; i < req.max_tokens; ++i) {
        if (req.cancel) break;

        const float* logits_ptr = logits.ptr<float>();
        if (logits.shape.size() >= 2 && logits.shape[0] > 1)
            logits_ptr += (logits.shape[0] - 1) * model_.config.vocab_size;

        int token_id = sample_token(logits_ptr, model_.config.vocab_size, scfg, all_tokens);
        all_tokens.push_back(token_id);

        if (token_id == stop_token || token_id == eos_token) {
            finish_reason = "stop";
            break;
        }
        if (model_.kv_cache.seq_len >= model_.config.max_seq_len) break;

        gen_tokens.push_back(token_id);

        if (req.stream) {
            // Decode and emit this single token immediately
            std::string piece = tok_.decode({token_id}, false);
            int safe = sm.feed(piece);
            if (safe < 0) { finish_reason = "stop"; break; }
            if (safe > 0) emit_token(req, piece.substr(0, (size_t)safe), token_id);
        }

        logits = model_.forward(token_id, model_.kv_cache.seq_len, false);
        model_.kv_cache.seq_len += 1;
    }

    // Flush any held-back partial stop-sequence match
    if (req.stream) {
        std::string tail = sm.flush_buffer();
        if (!tail.empty()) emit_token(req, tail, -1);
    }

    // Non-streaming: decode all tokens at once and apply stop sequences
    std::string full_text;
    if (!req.stream) {
        std::string raw = tok_.decode(gen_tokens, true);
        for (char c : raw) {
            std::string piece(1, c);
            int r = sm.feed(piece);
            if (r < 0) { full_text += sm.flush_buffer(); break; }
            if (r > 0) full_text += piece;
        }
        if (full_text.empty() && req.stop_sequences.empty())
            full_text = raw;
    }

    finish(req, finish_reason, full_text);
}

void Worker::emit_token(GenRequest& req, const std::string& piece, int token_id) {
    if (!req.stream || !req.channel) return;
    TokenEvent ev;
    ev.kind     = TokenEvent::Kind::TOKEN;
    ev.piece    = piece;
    ev.token_id = token_id;
    req.channel->push(std::move(ev));
}

void Worker::finish(GenRequest& req, const std::string& reason,
                    const std::string& full_text) {
    if (req.stream) {
        TokenEvent ev;
        ev.kind          = TokenEvent::Kind::FINISH;
        ev.finish_reason = reason;
        req.channel->push(std::move(ev));
        req.channel->close();
    } else {
        try { req.result.set_value(full_text); } catch (...) {}
    }
}

void Worker::report_error(GenRequest& req, const std::string& msg) {
    if (req.stream && req.channel) {
        TokenEvent ev;
        ev.kind          = TokenEvent::Kind::ERR;
        ev.error_message = msg;
        req.channel->push(std::move(ev));
        req.channel->close();
    } else {
        try {
            req.result.set_exception(
                std::make_exception_ptr(std::runtime_error(msg)));
        } catch (...) {}
    }
}

} // namespace ternative
