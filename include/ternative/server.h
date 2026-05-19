#pragma once
#include <string>
#include <functional>
#include <memory>

namespace ternative {

struct Model;
class Tokenizer;

struct ServerConfig {
    int port = 8080;
    std::string host = "0.0.0.0";
    int max_ctx_size = 4096;
    int default_max_tokens = 512;
    float default_temperature = 0.8f;
    float default_top_p = 0.9f;
    int default_top_k = 40;
    // Model name returned in API responses
    std::string model_name = "orchid";
};

void run_server(Model* model, Tokenizer* tokenizer, const ServerConfig& config);

} // namespace ternative
