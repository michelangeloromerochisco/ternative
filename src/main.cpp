#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <memory>
#include <chrono>

namespace {
    struct CompileFlags {
        CompileFlags() {
#ifdef __AVX2__
            std::cout << "[Compile] AVX2 enabled\n";
#else
            std::cout << "[Compile] AVX2 NOT enabled\n";
#endif
        }
    } compile_flags;
}

#ifdef _WIN32
#include <windows.h>
#include <vector>

static std::vector<std::string> get_utf8_args() {
    int argc;
    wchar_t** argvw = CommandLineToArgvW(GetCommandLineW(), &argc);
    std::vector<std::string> args;
    for (int i = 0; i < argc; ++i) {
        int len = WideCharToMultiByte(CP_UTF8, 0, argvw[i], -1, nullptr, 0, nullptr, nullptr);
        std::string arg(len, '\0');
        WideCharToMultiByte(CP_UTF8, 0, argvw[i], -1, arg.data(), len, nullptr, nullptr);
        if (!arg.empty() && arg.back() == '\0') arg.pop_back();
        args.push_back(std::move(arg));
    }
    LocalFree(argvw);
    return args;
}
#endif


#include "ternative/model.h"
#include "ternative/tokenizer.h"
#include "ternative/server.h"
#include "ternative/gguf.h"
#include "ternative/gguf_write.h"

static void print_banner() {
    std::cout << "==========================================\n";
    std::cout << "         TERNATIVE.CPP INFERENCE\n";
    std::cout << "==========================================\n";
}

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]\n"
              << "\n"
              << "Options:\n"
              << "  --model, -m <path>    Base GGUF model path (required)\n"
              << "  --lora <path>         LoRA adapter GGUF path (optional)\n"
              << "  --prompt, -p <text>  Prompt text (required for generate mode)\n"
              << "  --max-tokens <n>      Max new tokens (default: 512)\n"
              << "  --temperature <f>     Sampling temperature (default: 0.8)\n"
              << "  --top-p <f>           Nucleus sampling top-p (default: 0.9)\n"
              << "  --top-k <n>           Top-k sampling (default: 40)\n"
              << "  --server              Run HTTP server mode\n"
              << "  --port <n>            Server port (default: 8080)\n"
              << "  --info <path>         Print GGUF metadata and exit\n"
              << "  --system <text>       System prompt for chat mode\n"
              << "  --export-gguf <path>  Export merged F16 model as GGUF for llama.cpp\n"
              << "  --no-gpu              Disable GPU offload (CPU-only mode)\n"
              << "  --help, -h            Show this help\n";
}

int main(int argc_, char* argv_[]) {
#ifdef _WIN32
    auto utf8_args = get_utf8_args();
    int argc = (int)utf8_args.size();
    std::vector<char*> argv_ptrs;
    for (auto& arg : utf8_args) {
        argv_ptrs.push_back(arg.data());
    }
    char** argv = argv_ptrs.data();
#else
    int argc = argc_;
    char** argv = argv_;
#endif
    std::string model_path;
    std::vector<std::string> lora_paths;
    std::string prompt;
    std::string system_prompt;
    std::string info_path;
    std::string export_gguf_path;

    int max_tokens = 512;
    float temperature = 0.8f;
    float top_p = 0.9f;
    int top_k = 40;
    bool server_mode = false;
    int port = 8080;
    bool no_gpu = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if ((arg == "--model" || arg == "-m") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--lora" && i + 1 < argc) {
            lora_paths.push_back(argv[++i]);
        } else if ((arg == "--prompt" || arg == "-p") && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            try {
                max_tokens = std::stoi(argv[++i]);
            } catch (...) {
                std::cerr << "Invalid --max-tokens value\n";
                return 1;
            }
        } else if (arg == "--temperature" && i + 1 < argc) {
            try {
                temperature = std::stof(argv[++i]);
            } catch (...) {
                std::cerr << "Invalid --temperature value\n";
                return 1;
            }
        } else if (arg == "--top-p" && i + 1 < argc) {
            try {
                top_p = std::stof(argv[++i]);
            } catch (...) {
                std::cerr << "Invalid --top-p value\n";
                return 1;
            }
        } else if (arg == "--top-k" && i + 1 < argc) {
            try {
                top_k = std::stoi(argv[++i]);
            } catch (...) {
                std::cerr << "Invalid --top-k value\n";
                return 1;
            }
        } else if (arg == "--server") {
            server_mode = true;
        } else if (arg == "--port" && i + 1 < argc) {
            try {
                port = std::stoi(argv[++i]);
            } catch (...) {
                std::cerr << "Invalid --port value\n";
                return 1;
            }
        } else if (arg == "--info" && i + 1 < argc) {
            info_path = argv[++i];
        } else if (arg == "--system" && i + 1 < argc) {
            system_prompt = argv[++i];
        } else if (arg == "--export-gguf" && i + 1 < argc) {
            export_gguf_path = argv[++i];
        } else if (arg == "--no-gpu") {
            no_gpu = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // --info mode: print GGUF metadata and exit
    if (!info_path.empty()) {
        auto gguf = ternative::gguf_load(info_path);
        if (!gguf) {
            std::cerr << "Failed to load GGUF file: " << info_path << "\n";
            return 1;
        }
        ternative::gguf_print_info(*gguf);
        return 0;
    }

    // Export mode does not require --prompt or --server
    if (!export_gguf_path.empty() && model_path.empty()) {
        std::cerr << "Error: --export-gguf requires --model\n";
        return 1;
    }

    // Validate required arguments
    if (model_path.empty()) {
        std::cerr << "Error: --model is required\n";
        print_usage(argv[0]);
        return 1;
    }

    if (!server_mode && prompt.empty() && export_gguf_path.empty()) {
        std::cerr << "Error: --prompt is required in generate mode\n";
        print_usage(argv[0]);
        return 1;
    }

    print_banner();

    // Load model
    std::cout << "Loading model: " << model_path << "\n";
    std::unique_ptr<ternative::Model> model = ternative::Model::load(model_path, lora_paths, !no_gpu);
    if (!model) {
        std::cerr << "Failed to load model from " << model_path << "\n";
        return 1;
    }
    std::cout << "Model loaded. Layers: " << model->config.num_layers
              << ", Hidden: " << model->config.hidden_size << "\n";

    // Load tokenizer from the base GGUF (metadata-only, ~10 MB read)
    ternative::Tokenizer tokenizer;
    if (!tokenizer.load(model_path)) {
        std::cerr << "Failed to load tokenizer from: " << model_path << "\n";
        return 1;
    }
    std::cout << "Tokenizer loaded: vocab=" << tokenizer.vocab_size()
              << " bos=" << tokenizer.bos_token_id()
              << " eos=" << tokenizer.eos_token_id()
              << " eot=" << tokenizer.eot_token_id() << "\n";

    // Export mode: write merged F16 weights as GGUF for llama.cpp / llama-quantize
    if (!export_gguf_path.empty()) {
        std::cout << "Loading base GGUF metadata for tokenizer passthrough...\n";
        auto base_meta = ternative::gguf_load_metadata(model_path);
        if (!base_meta) {
            std::cerr << "Failed to load base GGUF metadata from: " << model_path << "\n";
            return 1;
        }
        try {
            ternative::gguf_write_f16(*model, *base_meta, export_gguf_path, true);
        } catch (const std::exception& e) {
            std::cerr << "Export failed: " << e.what() << "\n";
            return 1;
        }
        return 0;
    }

    // Server mode
    if (server_mode) {
        ternative::ServerConfig config;
        config.port = port;
        std::cout << "Starting server...\n";
        ternative::run_server(model.get(), &tokenizer, config);
        return 0;
    }

    // Generate mode
    std::string prompt_text;
    if (!system_prompt.empty()) {
        std::vector<std::pair<std::string, std::string>> messages;
        messages.emplace_back("system", system_prompt);
        messages.emplace_back("user", prompt);
        prompt_text = tokenizer.apply_chat_template(messages);
        std::cout << "\n--- Chat prompt ---\n" << prompt_text << "\n---\n";
    } else {
        prompt_text = prompt;
    }

    std::cout << "Encoding prompt...\n";
    std::vector<int> prompt_tokens = tokenizer.encode(prompt_text, true);
    std::cout << "Prompt tokens: " << prompt_tokens.size() << "\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<int> output_tokens = model->generate(
        prompt_tokens,
        max_tokens,
        temperature,
        top_p,
        top_k,
        tokenizer.eos_token_id()
    );
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();

    // Separate generated tokens from prompt tokens for clean output
    std::vector<int> generated_tokens;
    if (output_tokens.size() > prompt_tokens.size()) {
        generated_tokens.assign(
            output_tokens.begin() + prompt_tokens.size(),
            output_tokens.end()
        );
    }
    std::cout << "Generated tokens: " << generated_tokens.size() << "\n";
    if (elapsed_sec > 0 && generated_tokens.size() > 0) {
        double tps = generated_tokens.size() / elapsed_sec;
        double mspt = (elapsed_sec * 1000.0) / generated_tokens.size();
        std::cout << "Time: " << elapsed_sec << "s, " << tps << " t/s, " << mspt << " ms/tok\n";
    }

    std::string response = tokenizer.decode(generated_tokens, true);
    std::cout << "\n--- Response ---\n" << response << "\n---\n";
    std::cout.flush();

    return 0;
}
