#pragma once

#include <string>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>

#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
   using socket_t = SOCKET;
#else
   using socket_t = int;
#endif

namespace ternative {

// ---------------------------------------------------------------------------
// TokenEvent — one entry in the streaming pipeline
// ---------------------------------------------------------------------------
struct TokenEvent {
    enum class Kind { TOKEN, FINISH, ERR } kind = Kind::TOKEN;
    std::string  piece;           // detokenized text fragment (kind == TOKEN)
    int          token_id = -1;
    std::string  finish_reason;   // "stop" | "length" | "error" (kind == FINISH/ERROR)
    std::string  error_message;
};

// ---------------------------------------------------------------------------
// TokenChannel — thread-safe bounded queue between generator and SSE writer
// ---------------------------------------------------------------------------
class TokenChannel {
public:
    explicit TokenChannel(int capacity = 128) : cap_(capacity) {}

    // Push event. Blocks if channel is full (back-pressure).
    void push(TokenEvent ev);

    // Pop event. Blocks until available or channel is closed.
    // Returns false if channel is closed and empty.
    bool pop(TokenEvent& out);

    // Close the channel (unblocks any waiting pop()).
    void close();

    bool is_closed() const { return closed_.load(); }

private:
    int cap_;
    std::mutex mu_;
    std::condition_variable cv_push_;
    std::condition_variable cv_pop_;
    std::deque<TokenEvent> q_;
    std::atomic<bool> closed_{false};
};

// ---------------------------------------------------------------------------
// SseWriter — writes SSE events directly to a raw socket
// ---------------------------------------------------------------------------
class SseWriter {
public:
    explicit SseWriter(socket_t sock) : sock_(sock) {}

    // Send the role delta (first chunk, announces role=assistant)
    bool send_role(const std::string& id, const std::string& model_name);

    // Send a token chunk
    bool send_chunk(const std::string& id, const std::string& model_name,
                    const std::string& delta);

    // Send finish_reason + close the stream
    bool send_finish(const std::string& id, const std::string& model_name,
                     const std::string& finish_reason);

    // Send "data: [DONE]\n\n"
    bool send_done();

private:
    socket_t sock_;
    bool send_raw(const std::string& s);
};

} // namespace ternative
