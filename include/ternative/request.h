#pragma once

#include "stream.h"
#include "model.h"
#include "sampler.h"
#include "tokenizer.h"

#include <string>
#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <future>
#include <optional>
#include <thread>

namespace ternative {

// ---------------------------------------------------------------------------
// GenRequest — one inference request
// ---------------------------------------------------------------------------
struct GenRequest {
    std::string id;                          // "chatcmpl-..."
    std::vector<int> prompt_tokens;

    // Sampling params
    float temperature = 0.8f;
    float top_p       = 0.9f;
    int   top_k       = 40;
    int   max_tokens  = 512;
    std::vector<std::string> stop_sequences;

    // Stream vs block
    bool stream = false;
    std::shared_ptr<TokenChannel> channel;   // populated when stream=true
    std::promise<std::string> result;        // populated when stream=false

    std::atomic<bool> cancel{false};

    // Non-copyable because of atomic and promise
    GenRequest() = default;
    GenRequest(const GenRequest&) = delete;
    GenRequest& operator=(const GenRequest&) = delete;
};

// ---------------------------------------------------------------------------
// RequestQueue — thread-safe work queue
// ---------------------------------------------------------------------------
class RequestQueue {
public:
    void submit(std::shared_ptr<GenRequest> r);
    std::shared_ptr<GenRequest> wait_pop();
    void shutdown();

private:
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::shared_ptr<GenRequest>> q_;
    bool stopped_ = false;
};

// ---------------------------------------------------------------------------
// StopMatcher — streaming-safe multi-pattern stop sequence detector
// ---------------------------------------------------------------------------
class StopMatcher {
public:
    explicit StopMatcher(const std::vector<std::string>& patterns);

    // Feed a new text piece.  Returns the byte length of the safe prefix to emit
    // (may be less than piece.size() if a partial stop match is pending),
    // or -1 if a complete stop sequence was matched (caller should stop generation).
    int feed(const std::string& piece);

    // Any buffered text that turned out not to be a stop sequence.
    std::string flush_buffer();

private:
    std::vector<std::string> pats_;
    std::string buf_;  // pending partial match buffer
};

// ---------------------------------------------------------------------------
// Worker — single thread that drives the model
// ---------------------------------------------------------------------------
class Worker {
public:
    Worker(Model& model, Tokenizer& tok, RequestQueue& queue);
    ~Worker();

    void start();
    void stop();

private:
    void run();
    void process(GenRequest& req);
    void emit_token(GenRequest& req, const std::string& piece, int token_id);
    void finish(GenRequest& req, const std::string& reason, const std::string& full_text);
    void report_error(GenRequest& req, const std::string& msg);

    Model&        model_;
    Tokenizer&    tok_;
    RequestQueue& queue_;
    std::thread   thread_;
    std::atomic<bool> running_{false};
};

} // namespace ternative
