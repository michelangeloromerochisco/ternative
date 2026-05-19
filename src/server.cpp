#include "ternative/server.h"
#include "ternative/model.h"
#include "ternative/tokenizer.h"
#include "ternative/sampler.h"
#include "ternative/request.h"
#include "ternative/stream.h"

#include <nlohmann/json.hpp>
#include <ctime>
#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <memory>
#include <future>
#include <thread>
#include <mutex>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>
#endif

using json = nlohmann::json;

namespace ternative {

#ifdef _WIN32
using socket_t = SOCKET;
const socket_t INVALID_SOCKET_VAL = INVALID_SOCKET;
#else
using socket_t = int;
const socket_t INVALID_SOCKET_VAL = -1;
#endif

namespace {

// ---------------------------------------------------------------------------
// Platform socket helpers
// ---------------------------------------------------------------------------
bool socket_platform_init() {
#ifdef _WIN32
    WSADATA wsa_data;
    return WSAStartup(MAKEWORD(2, 2), &wsa_data) == 0;
#else
    return true;
#endif
}

void socket_platform_cleanup() {
#ifdef _WIN32
    WSACleanup();
#endif
}

void close_socket(socket_t fd) {
#ifdef _WIN32
    if (fd != INVALID_SOCKET_VAL) {
        closesocket(fd);
    }
#else
    if (fd >= 0) {
        close(fd);
    }
#endif
}

bool send_all(socket_t fd, const char* data, size_t len) {
    size_t sent = 0;
    while (sent < len) {
#ifdef _WIN32
        int result = ::send(fd, data + sent, static_cast<int>(len - sent), 0);
#else
        ssize_t result = ::send(fd, data + sent, len - sent, 0);
#endif
        if (result <= 0) {
            return false;
        }
        sent += static_cast<size_t>(result);
    }
    return true;
}

int recv_some(socket_t fd, char* buf, size_t max_len) {
#ifdef _WIN32
    return ::recv(fd, buf, static_cast<int>(max_len), 0);
#else
    return static_cast<int>(::recv(fd, buf, max_len, 0));
#endif
}

// ---------------------------------------------------------------------------
// HTTP request parsing
// ---------------------------------------------------------------------------
struct HttpRequest {
    std::string method;
    std::string path;
    std::string body;
};

static constexpr size_t MAX_HEADER_SIZE = 65536;
static constexpr size_t MAX_BODY_SIZE = 8 * 1024 * 1024;

bool parse_http_request(socket_t client, HttpRequest& req) {
    std::string buffer;
    char temp[4096];

    // Read until we have the full HTTP header
    while (true) {
        int received = recv_some(client, temp, sizeof(temp) - 1);
        if (received <= 0) {
            return false;
        }
        temp[received] = '\0';
        buffer.append(temp, static_cast<size_t>(received));
        if (buffer.size() > MAX_HEADER_SIZE) {
            return false;
        }
        if (buffer.find("\r\n\r\n") != std::string::npos) {
            break;
        }
    }

    size_t header_end = buffer.find("\r\n\r\n");
    std::string header = buffer.substr(0, header_end);

    // Parse request line: METHOD PATH HTTP/1.1
    std::istringstream stream(header);
    stream >> req.method >> req.path;
    std::string version;
    stream >> version;

    // Parse headers to find Content-Length
    size_t content_length = 0;
    std::string line;
    std::getline(stream, line); // consume rest of first line
    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        size_t colon = line.find(':');
        if (colon != std::string::npos) {
            std::string key = line.substr(0, colon);
            std::string val = line.substr(colon + 1);
            while (!val.empty() && val.front() == ' ') {
                val.erase(val.begin());
            }
            if (key == "Content-Length" || key == "content-length") {
                try {
                    content_length = static_cast<size_t>(std::stoll(val));
                } catch (...) {
                    content_length = 0;
                }
            }
        }
    }

    if (content_length > MAX_BODY_SIZE) {
        return false;
    }

    // Extract any body already in the buffer
    size_t body_start = header_end + 4;
    size_t already_have = 0;
    if (buffer.size() > body_start) {
        already_have = buffer.size() - body_start;
        req.body = buffer.substr(body_start, already_have);
    }

    // Read remaining body bytes
    int remaining = static_cast<int>(content_length) - static_cast<int>(already_have);
    while (remaining > 0) {
        int to_read = std::min(remaining, static_cast<int>(sizeof(temp)));
        int received = recv_some(client, temp, static_cast<size_t>(to_read));
        if (received <= 0) {
            break;
        }
        req.body.append(temp, static_cast<size_t>(received));
        remaining -= received;
    }

    return true;
}

// ---------------------------------------------------------------------------
// HTTP response helpers
// ---------------------------------------------------------------------------
void send_response(socket_t client, int status, const std::string& status_text,
                   const std::string& content_type, const std::string& body) {
    std::ostringstream response;
    response << "HTTP/1.1 " << status << " " << status_text << "\r\n";
    response << "Content-Type: " << content_type << "\r\n";
    response << "Content-Length: " << body.size() << "\r\n";
    response << "Connection: close\r\n";
    response << "\r\n";
    response << body;
    std::string resp_str = response.str();
    send_all(client, resp_str.c_str(), resp_str.size());
}

void send_json_response(socket_t client, int status, const json& j) {
    send_response(client, status, (status == 200 ? "OK" : "Error"),
                  "application/json", j.dump());
}

void send_sse_headers(socket_t client) {
    std::string headers =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: keep-alive\r\n"
        "\r\n";
    send_all(client, headers.c_str(), headers.size());
}

void send_sse_event(socket_t client, const json& data) {
    std::string event = "data: " + data.dump() + "\n\n";
    send_all(client, event.c_str(), event.size());
}

void send_sse_done(socket_t client) {
    std::string done = "data: [DONE]\n\n";
    send_all(client, done.c_str(), done.size());
}

// ---------------------------------------------------------------------------
// Endpoint handlers
// ---------------------------------------------------------------------------
std::time_t now_unix() {
    return std::time(nullptr);
}

void handle_models(socket_t client) {
    json j = {
        {"object", "list"},
        {"data", json::array({
            {{"id", "orchid"}, {"object", "model"}}
        })}
    };
    send_json_response(client, 200, j);
}

void handle_completions(socket_t client, const HttpRequest& req,
                        Model* model, Tokenizer* tokenizer,
                        const ServerConfig& config) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception& e) {
        send_json_response(client, 400, {{"error", "Invalid JSON body"}});
        return;
    }

    std::string prompt = body.value("prompt", "");
    int max_tokens = body.value("max_tokens", config.default_max_tokens);
    float temperature = body.value("temperature", config.default_temperature);
    float top_p = body.value("top_p", config.default_top_p);
    int top_k = body.value("top_k", config.default_top_k);
    bool stream = body.value("stream", false);
    int logprobs_param = body.value("logprobs", 0);
    bool echo = body.value("echo", false);

    if (prompt.empty()) {
        send_json_response(client, 400, {{"error", "Missing prompt"}});
        return;
    }

    auto prompt_tokens = tokenizer->encode(prompt, true);
    if (static_cast<int>(prompt_tokens.size()) > config.max_ctx_size) {
        prompt_tokens.erase(
            prompt_tokens.begin(),
            prompt_tokens.begin() + (prompt_tokens.size() - config.max_ctx_size)
        );
    }

    std::time_t created = now_unix();

    if (logprobs_param > 0 && echo && max_tokens == 0) {
        // Echo scoring mode. Optional start_pos skips logit computation for context tokens
        // (bench_standard.py uses this to avoid computing logits at context positions).
        int start_pos = body.value("start_pos", 0);
        auto result = model->score_logprobs(prompt_tokens, logprobs_param, start_pos);

        // Build logprobs arrays covering all prompt tokens.
        // Position 0 has no predecessor → null logprob (matches OpenAI format).
        json tokens_arr       = json::array();
        json token_logprobs_arr = json::array();
        json top_logprobs_arr = json::array();

        // First token: no preceding context, logprob = null.
        if (!prompt_tokens.empty()) {
            tokens_arr.push_back(tokenizer->decode({prompt_tokens[0]}, false));
            token_logprobs_arr.push_back(nullptr);
            top_logprobs_arr.push_back(json::object());
        }
        // Remaining tokens: use scored logprobs.
        for (size_t i = 0; i < result.logprobs.size(); ++i) {
            const auto& tlp = result.logprobs[i];
            tokens_arr.push_back(tokenizer->decode({tlp.token_id}, false));
            token_logprobs_arr.push_back(tlp.logprob);
            json top_map = json::object();
            for (const auto& entry : tlp.top_logprobs) {
                top_map[tokenizer->decode({entry.token_id}, false)] = entry.logprob;
            }
            top_logprobs_arr.push_back(top_map);
        }

        json logprobs_json = {
            {"tokens",        tokens_arr},
            {"token_logprobs", token_logprobs_arr},
            {"top_logprobs",  top_logprobs_arr}
        };

        // Echo mode: return the full prompt as the generated text.
        std::string echo_text = tokenizer->decode(prompt_tokens, true);
        json response = {
            {"id", "cmpl-ternative"},
            {"object", "text_completion"},
            {"created", created},
            {"model", "orchid"},
            {"choices", json::array({
                {
                    {"text", echo_text},
                    {"index", 0},
                    {"finish_reason", "length"},
                    {"logprobs", logprobs_json}
                }
            })}
        };
        send_json_response(client, 200, response);
    } else if (logprobs_param > 0) {
        // Standard generate with per-token log-probabilities.
        auto result = model->generate_with_logprobs(
            prompt_tokens, max_tokens, temperature, top_p, top_k,
            tokenizer->eos_token_id(), logprobs_param
        );

        std::vector<int> generated_tokens;
        if (result.tokens.size() > prompt_tokens.size()) {
            generated_tokens.assign(
                result.tokens.begin() + prompt_tokens.size(),
                result.tokens.end()
            );
        }

        std::string text = tokenizer->decode(generated_tokens, true);

        json tokens_arr       = json::array();
        json token_logprobs_arr = json::array();
        json top_logprobs_arr = json::array();
        for (const auto& tlp : result.logprobs) {
            tokens_arr.push_back(tokenizer->decode({tlp.token_id}, false));
            token_logprobs_arr.push_back(tlp.logprob);
            json top_map = json::object();
            for (const auto& entry : tlp.top_logprobs) {
                top_map[tokenizer->decode({entry.token_id}, false)] = entry.logprob;
            }
            top_logprobs_arr.push_back(top_map);
        }
        json logprobs_json = {
            {"tokens",        tokens_arr},
            {"token_logprobs", token_logprobs_arr},
            {"top_logprobs",  top_logprobs_arr}
        };

        json response = {
            {"id", "cmpl-ternative"},
            {"object", "text_completion"},
            {"created", created},
            {"model", "orchid"},
            {"choices", json::array({
                {
                    {"text", text},
                    {"index", 0},
                    {"finish_reason", generated_tokens.empty() ? "length" :
                        (generated_tokens.back() == tokenizer->eos_token_id() ? "stop" : "length")},
                    {"logprobs", logprobs_json}
                }
            })}
        };
        send_json_response(client, 200, response);
    } else {
        auto output_tokens = model->generate(
            prompt_tokens, max_tokens, temperature, top_p, top_k,
            tokenizer->eos_token_id()
        );

        std::vector<int> generated_tokens;
        if (output_tokens.size() > prompt_tokens.size()) {
            generated_tokens.assign(
                output_tokens.begin() + prompt_tokens.size(),
                output_tokens.end()
            );
        }
        std::string text = tokenizer->decode(generated_tokens, true);

        if (stream) {
            send_sse_headers(client);
            json chunk = {
                {"id", "cmpl-ternative"},
                {"object", "text_completion"},
                {"created", created},
                {"model", "orchid"},
                {"choices", json::array({
                    {{"text", text}, {"index", 0}, {"finish_reason", "stop"}}
                })}
            };
            send_sse_event(client, chunk);
            send_sse_done(client);
        } else {
            json response = {
                {"id", "cmpl-ternative"},
                {"object", "text_completion"},
                {"created", created},
                {"model", "orchid"},
                {"choices", json::array({
                    {{"text", text}, {"index", 0}, {"finish_reason", "stop"}}
                })}
            };
            send_json_response(client, 200, response);
        }
    }
}

void handle_chat_completions(socket_t client, const HttpRequest& req,
                             Model* model, Tokenizer* tokenizer,
                             const ServerConfig& config,
                             RequestQueue* worker_queue = nullptr) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception& e) {
        send_json_response(client, 400, {{"error", "Invalid JSON body"}});
        return;
    }

    std::vector<std::pair<std::string, std::string>> messages;
    if (body.contains("messages") && body["messages"].is_array()) {
        for (const auto& msg : body["messages"]) {
            std::string role = msg.value("role", "user");
            std::string content = msg.value("content", "");
            messages.emplace_back(role, content);
        }
    }

    int max_tokens = body.value("max_tokens", config.default_max_tokens);
    float temperature = body.value("temperature", config.default_temperature);
    float top_p = body.value("top_p", config.default_top_p);
    int top_k = body.value("top_k", config.default_top_k);
    bool stream = body.value("stream", false);
    int logprobs_param = 0;
    if (body.contains("logprobs") && body["logprobs"].is_boolean()) {
        logprobs_param = body["logprobs"].get<bool>() ? 5 : 0;
    } else if (body.contains("logprobs") && body["logprobs"].is_number()) {
        logprobs_param = body["logprobs"].get<int>();
    }
    // OpenAI API: top_logprobs is separate
    if (body.contains("top_logprobs") && body["top_logprobs"].is_number()) {
        int tl = body["top_logprobs"].get<int>();
        if (tl > 0) logprobs_param = std::max(logprobs_param, tl);
    }

    std::string prompt_text = tokenizer->apply_chat_template(messages);
    auto prompt_tokens = tokenizer->encode(prompt_text, true);
    if (static_cast<int>(prompt_tokens.size()) > config.max_ctx_size) {
        prompt_tokens.erase(
            prompt_tokens.begin(),
            prompt_tokens.begin() + (prompt_tokens.size() - config.max_ctx_size)
        );
    }

    std::time_t created = now_unix();

    if (logprobs_param > 0) {
        auto result = model->generate_with_logprobs(
            prompt_tokens, max_tokens, temperature, top_p, top_k,
            tokenizer->eos_token_id(), logprobs_param
        );

        std::vector<int> generated_tokens;
        if (result.tokens.size() > prompt_tokens.size()) {
            generated_tokens.assign(
                result.tokens.begin() + prompt_tokens.size(),
                result.tokens.end()
            );
        }
        std::string text = tokenizer->decode(generated_tokens, true);

        // Build logprobs content for chat completions format
        json logprobs_content = json::object();
        {
            json tokens_arr = json::array();
            json token_logprobs_arr = json::array();
            json top_logprobs_arr = json::array();

            for (const auto& tlp : result.logprobs) {
                std::string tok_text = tokenizer->decode({tlp.token_id}, false);
                tokens_arr.push_back(tok_text);
                token_logprobs_arr.push_back(tlp.logprob);

                json top_arr = json::array();
                for (const auto& entry : tlp.top_logprobs) {
                    std::string entry_text = tokenizer->decode({entry.token_id}, false);
                    top_arr.push_back({{"token", entry_text}, {"logprob", entry.logprob}});
                }
                top_logprobs_arr.push_back(top_arr);
            }

            logprobs_content["tokens"] = tokens_arr;
            logprobs_content["token_logprobs"] = token_logprobs_arr;
            logprobs_content["top_logprobs"] = top_logprobs_arr;
        }

        json response = {
            {"id", "chatcmpl-ternative"},
            {"object", "chat.completion"},
            {"created", created},
            {"model", "orchid"},
            {"choices", json::array({
                {
                    {"message", {{"role", "assistant"}, {"content", text}}},
                    {"index", 0},
                    {"finish_reason", generated_tokens.empty() ? "length" :
                        (generated_tokens.back() == tokenizer->eos_token_id() ? "stop" : "length")},
                    {"logprobs", logprobs_content}
                }
            })}
        };
        send_json_response(client, 200, response);
    } else if (worker_queue && (stream || true)) {
        // ── Worker-based path (streaming or non-streaming via queue) ──────────
        auto gr = std::make_shared<GenRequest>();
        gr->id             = "chatcmpl-ternative";
        gr->prompt_tokens  = prompt_tokens;
        gr->temperature    = temperature;
        gr->top_p          = top_p;
        gr->top_k          = top_k;
        gr->max_tokens     = max_tokens;
        gr->stream         = stream;

        // Parse stop sequences
        if (body.contains("stop")) {
            if (body["stop"].is_string())
                gr->stop_sequences.push_back(body["stop"].get<std::string>());
            else if (body["stop"].is_array())
                for (const auto& s : body["stop"])
                    if (s.is_string()) gr->stop_sequences.push_back(s.get<std::string>());
        }

        if (stream) {
            gr->channel = std::make_shared<TokenChannel>();
            worker_queue->submit(gr);

            // Send SSE headers immediately
            send_sse_headers(client);
            SseWriter sse(client);
            sse.send_role(gr->id, config.model_name);

            // Drain token channel and write SSE events
            TokenEvent ev;
            while (gr->channel->pop(ev)) {
                switch (ev.kind) {
                    case TokenEvent::Kind::TOKEN:
                        if (!sse.send_chunk(gr->id, config.model_name, ev.piece)) {
                            gr->cancel = true;
                        }
                        break;
                    case TokenEvent::Kind::FINISH:
                        sse.send_finish(gr->id, config.model_name, ev.finish_reason);
                        sse.send_done();
                        return;
                    case TokenEvent::Kind::ERR:
                        sse.send_finish(gr->id, config.model_name, "error");
                        sse.send_done();
                        return;
                }
            }
            sse.send_done();
        } else {
            // Non-streaming via worker queue — serializes through the worker thread
            auto fut = gr->result.get_future();
            worker_queue->submit(gr);
            std::string text;
            try { text = fut.get(); } catch (...) { text = ""; }
            json response = {
                {"id",      gr->id},
                {"object",  "chat.completion"},
                {"created", created},
                {"model",   config.model_name},
                {"choices", json::array({
                    {
                        {"message",       {{"role", "assistant"}, {"content", text}}},
                        {"index",         0},
                        {"finish_reason", "stop"}
                    }
                })},
                {"usage", {
                    {"prompt_tokens",     (int)prompt_tokens.size()},
                    {"completion_tokens", 0},
                    {"total_tokens",      (int)prompt_tokens.size()}
                }}
            };
            send_json_response(client, 200, response);
        }
    } else {
        // ── Fallback: inline synchronous generation (no worker) ──────────────
        auto output_tokens = model->generate(
            prompt_tokens, max_tokens, temperature, top_p, top_k,
            tokenizer->eos_token_id()
        );
        std::vector<int> generated_tokens;
        if (output_tokens.size() > prompt_tokens.size())
            generated_tokens.assign(output_tokens.begin() + prompt_tokens.size(),
                                    output_tokens.end());
        std::string text = tokenizer->decode(generated_tokens, true);

        json response = {
            {"id",      "chatcmpl-ternative"},
            {"object",  "chat.completion"},
            {"created", created},
            {"model",   config.model_name},
            {"choices", json::array({
                {
                    {"message",       {{"role", "assistant"}, {"content", text}}},
                    {"index",         0},
                    {"finish_reason", "stop"}
                }
            })}
        };
        send_json_response(client, 200, response);
    }
}

void handle_logprobs(socket_t client, const HttpRequest& req,
                     Model* model, Tokenizer* tokenizer,
                     const ServerConfig& config) {
    json body;
    try {
        body = json::parse(req.body);
    } catch (const std::exception& e) {
        send_json_response(client, 400, {{"error", "Invalid JSON body"}});
        return;
    }

    std::string prompt = body.value("prompt", "");
    std::string continuation = body.value("continuation", "");
    int num_logprobs = body.value("num_logprobs", 5);

    if (prompt.empty()) {
        send_json_response(client, 400, {{"error", "Missing prompt"}});
        return;
    }

    // Encode prompt + continuation as a single sequence
    std::string full_text = prompt + continuation;
    auto full_tokens = tokenizer->encode(full_text, true);
    auto prompt_tokens = tokenizer->encode(prompt, true);

    if (static_cast<int>(full_tokens.size()) > config.max_ctx_size) {
        send_json_response(client, 400, {{"error", "Sequence too long"}});
        return;
    }

    int prompt_len = (int)prompt_tokens.size();

    // Score the full sequence to get log-probs for each position
    auto result = model->score_logprobs(full_tokens, num_logprobs);

    // Compute log-likelihood of the continuation tokens
    // result.logprobs[i] gives logprob for full_tokens[i+1] given full_tokens[0..i]
    // Continuation tokens start at position prompt_len in full_tokens
    // Their logprobs are in result.logprobs[prompt_len-1 .. end]
    double continuation_logprob = 0.0;
    int continuation_token_count = 0;
    json token_logprobs_arr = json::array();

    // result.logprobs[0] = logprob for full_tokens[1] given full_tokens[0]
    // result.logprobs[j] = logprob for full_tokens[j+1] given full_tokens[0..j]
    // So for continuation tokens at positions prompt_len .. end:
    // Their logprobs are at result.logprobs[prompt_len-1 .. end-1]
    for (int i = prompt_len - 1; i < (int)result.logprobs.size() && i < (int)full_tokens.size() - 1; ++i) {
        const auto& tlp = result.logprobs[i];
        continuation_logprob += tlp.logprob;
        continuation_token_count++;

        json tok_entry = {
            {"token_id", tlp.token_id},
            {"logprob", tlp.logprob},
            {"text", tokenizer->decode({tlp.token_id}, false)}
        };
        json top_arr = json::array();
        for (const auto& entry : tlp.top_logprobs) {
            top_arr.push_back({
                {"token_id", entry.token_id},
                {"logprob", entry.logprob},
                {"text", tokenizer->decode({entry.token_id}, false)}
            });
        }
        tok_entry["top_logprobs"] = top_arr;
        token_logprobs_arr.push_back(tok_entry);
    }

    json response = {
        {"id", "logprob-ternative"},
        {"object", "logprobs"},
        {"created", now_unix()},
        {"model", "orchid"},
        {"prompt_tokens", prompt_len},
        {"continuation_tokens", continuation_token_count},
        {"continuation_logprob", continuation_logprob},
        {"normalized_logprob", continuation_token_count > 0 ?
            continuation_logprob / continuation_token_count : 0.0},
        {"token_logprobs", token_logprobs_arr}
    };
    send_json_response(client, 200, response);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
void run_server(Model* model, Tokenizer* tokenizer, const ServerConfig& config) {
    if (!model || !tokenizer) {
        std::cerr << "[Server] Model and tokenizer must not be null\n";
        return;
    }

    // Start the generation worker thread
    RequestQueue queue;
    Worker worker(*model, *tokenizer, queue);
    worker.start();
    std::cerr << "[Server] Worker thread started\n";

    if (!socket_platform_init()) {
        std::cerr << "[Server] Failed to initialize socket platform\n";
        queue.shutdown();
        return;
    }

    socket_t server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == INVALID_SOCKET_VAL) {
        std::cerr << "[Server] Failed to create socket\n";
        socket_platform_cleanup();
        return;
    }

    int opt = 1;
#ifdef _WIN32
    ::setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR,
                 reinterpret_cast<const char*>(&opt), sizeof(opt));
#else
    ::setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<u_short>(config.port));

    if (config.host == "0.0.0.0") {
        addr.sin_addr.s_addr = INADDR_ANY;
    } else {
        addr.sin_addr.s_addr = ::inet_addr(config.host.c_str());
        if (addr.sin_addr.s_addr == INADDR_NONE) {
            std::cerr << "[Server] Invalid host address: " << config.host << "\n";
            close_socket(server_fd);
            socket_platform_cleanup();
            return;
        }
    }

    if (::bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        std::cerr << "[Server] Failed to bind to " << config.host << ":"
                  << config.port << "\n";
        close_socket(server_fd);
        socket_platform_cleanup();
        return;
    }

    if (::listen(server_fd, SOMAXCONN) != 0) {
        std::cerr << "[Server] Failed to listen\n";
        close_socket(server_fd);
        socket_platform_cleanup();
        return;
    }

    std::cout << "[Server] Listening on http://" << config.host << ":"
              << config.port << "/\n";

    // Serialise all model inference calls. The Model's KV cache is shared mutable
    // state — concurrent calls to generate/score_logprobs corrupt it.
    // Non-inference routes (GET /v1/models) never touch the model, so they are
    // passed through without the lock.
    std::mutex model_mutex;

    while (true) {
        sockaddr_in client_addr{};
#ifdef _WIN32
        int client_addr_len = sizeof(client_addr);
#else
        socklen_t client_addr_len = sizeof(client_addr);
#endif
        socket_t client_fd = ::accept(
            server_fd,
            reinterpret_cast<sockaddr*>(&client_addr),
            &client_addr_len
        );

        if (client_fd == INVALID_SOCKET_VAL) {
            continue;
        }

        std::thread([client_fd, model, tokenizer, &config, &queue, &model_mutex]() {
            HttpRequest req;
            if (parse_http_request(client_fd, req)) {
                if (req.method == "GET" && req.path == "/v1/models") {
                    handle_models(client_fd);                       // no model access
                } else if (req.method == "POST" && req.path == "/v1/tokenize") {
                    // Fast tokenize: no model inference, no mutex needed.
                    // bench_standard.py uses this to find the context/continuation
                    // split point without making a second echo scoring request.
                    try {
                        auto body = nlohmann::json::parse(req.body);
                        std::string text = body.value("text", "");
                        auto ids = tokenizer->encode(text, true);
                        nlohmann::json tokens_arr = nlohmann::json::array();
                        for (int id : ids)
                            tokens_arr.push_back(tokenizer->decode({id}, false));
                        send_json_response(client_fd, 200,
                            {{"tokens", tokens_arr}, {"count", (int)ids.size()}});
                    } catch (...) {
                        send_json_response(client_fd, 400,
                            {{"error", "Invalid JSON or tokenize failed"}});
                    }
                } else if (req.method == "POST" && req.path == "/v1/completions") {
                    std::lock_guard<std::mutex> lk(model_mutex);
                    handle_completions(client_fd, req, model, tokenizer, config);
                } else if (req.method == "POST" && req.path == "/v1/chat/completions") {
                    std::lock_guard<std::mutex> lk(model_mutex);
                    handle_chat_completions(client_fd, req, model, tokenizer, config, &queue);
                } else if (req.method == "POST" && req.path == "/v1/logprobs") {
                    std::lock_guard<std::mutex> lk(model_mutex);
                    handle_logprobs(client_fd, req, model, tokenizer, config);
                } else {
                    send_json_response(client_fd, 404,
                        {{"error", "Not found"}, {"path", req.path}});
                }
            }
            close_socket(client_fd);
        }).detach();
    }

    queue.shutdown();
    close_socket(server_fd);
    socket_platform_cleanup();
}

} // namespace ternative
