#pragma once
#include "model.h"
#include "gguf.h"
#include <string>

namespace ternative {

// Write a ternative Model (post-cache-load, all weights F16) to a GGUF v3 file
// compatible with llama.cpp.  Tokenizer metadata is copied verbatim from the
// base model's GGUFile (loaded via gguf_load_metadata).
//
// Resulting file:
//  - Architecture: llama (standard Llama-style)
//  - All weight tensors: F16
//  - BitNet-specific sub_norm tensors are omitted (not in the llama arch)
//  - Estimated size: ~4.4 GB (can be further quantised with llama-quantize)
//
// Throws std::runtime_error on I/O failure or shape mismatch.
void gguf_write_f16(const Model&     model,
                    const GGUFile&   base_metadata,
                    const std::string& output_path,
                    bool verbose = true);

} // namespace ternative
