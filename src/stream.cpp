#include "ternative/stream.h"

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <winsock2.h>
#else
#  include <sys/socket.h>
#endif

#include <nlohmann/json.hpp>
#include <string>
#include <ctime>

using json = nlohmann::json;

namespace ternative {

// ---------------------------------------------------------------------------
// TokenChannel
// ---------------------------------------------------------------------------
void TokenChannel::push(TokenEvent ev) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_push_.wait(lk, [this]{ return (int)q_.size() < cap_ || closed_.load(); });
    if (closed_) return;
    q_.push_back(std::move(ev));
    cv_pop_.notify_one();
}

bool TokenChannel::pop(TokenEvent& out) {
    std::unique_lock<std::mutex> lk(mu_);
    cv_pop_.wait(lk, [this]{ return !q_.empty() || closed_.load(); });
    if (q_.empty()) return false;  // closed and empty
    out = std::move(q_.front());
    q_.pop_front();
    cv_push_.notify_one();
    return true;
}

void TokenChannel::close() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        closed_ = true;
    }
    cv_pop_.notify_all();
    cv_push_.notify_all();
}

// ---------------------------------------------------------------------------
// SseWriter
// ---------------------------------------------------------------------------
bool SseWriter::send_raw(const std::string& s) {
    const char* p = s.data();
    size_t rem = s.size();
    while (rem > 0) {
#ifdef _WIN32
        int n = ::send(sock_, p, static_cast<int>(rem), 0);
#else
        ssize_t n = ::send(sock_, p, rem, 0);
#endif
        if (n <= 0) return false;
        p += n; rem -= (size_t)n;
    }
    return true;
}

static std::string make_sse_data(const json& payload) {
    return "data: " + payload.dump() + "\n\n";
}

bool SseWriter::send_role(const std::string& id, const std::string& model_name) {
    json j = {
        {"id",      id},
        {"object",  "chat.completion.chunk"},
        {"created", (int64_t)std::time(nullptr)},
        {"model",   model_name},
        {"choices", json::array({
            {
                {"delta",         {{"role", "assistant"}, {"content", ""}}},
                {"index",         0},
                {"finish_reason", nullptr}
            }
        })}
    };
    return send_raw(make_sse_data(j));
}

bool SseWriter::send_chunk(const std::string& id, const std::string& model_name,
                           const std::string& delta) {
    json j = {
        {"id",      id},
        {"object",  "chat.completion.chunk"},
        {"created", (int64_t)std::time(nullptr)},
        {"model",   model_name},
        {"choices", json::array({
            {
                {"delta",         {{"content", delta}}},
                {"index",         0},
                {"finish_reason", nullptr}
            }
        })}
    };
    return send_raw(make_sse_data(j));
}

bool SseWriter::send_finish(const std::string& id, const std::string& model_name,
                            const std::string& finish_reason) {
    json j = {
        {"id",      id},
        {"object",  "chat.completion.chunk"},
        {"created", (int64_t)std::time(nullptr)},
        {"model",   model_name},
        {"choices", json::array({
            {
                {"delta",         json::object()},
                {"index",         0},
                {"finish_reason", finish_reason}
            }
        })}
    };
    return send_raw(make_sse_data(j));
}

bool SseWriter::send_done() {
    return send_raw("data: [DONE]\n\n");
}

} // namespace ternative
