# ternative

**Inference engine for ternary-weight LLMs with runtime LoRA** — the llama.cpp of BitNet models.

[![CI](https://github.com/michelangeloromerochisco/ternative/actions/workflows/ci.yml/badge.svg)](https://github.com/michelangeloromerochisco/ternative/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)](README.md)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green)](README.md)
[![C++](https://img.shields.io/badge/C++-17-blue)](CMakeLists.txt)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20452163.svg)](https://doi.org/10.5281/zenodo.20452163)

Loads a BitNet I2_S base GGUF + a separate LoRA adapter GGUF, merges them at full F32 precision, and serves the result via an OpenAI-compatible HTTP server — on GPU or CPU-only hardware.

---

## Why ternative?

Standard inference stacks cannot serve LoRA-fine-tuned ternary models correctly:

| Engine | BitNet I2_S | Runtime LoRA | I2_S + LoRA | Server |
|--------|:-----------:|:------------:|:-----------:|:------:|
| llama.cpp | ⚠️ type-36 error | ✓ (Q4/Q8 only) | ✗ | Via llama-server |
| bitnet.cpp | ✓ native kernels | ✗ no adapter path | ✗ | ✗ |
| **ternative** | ✓ | ✓ full precision | ✓ | ✓ built-in |

**The root problem**: merging a LoRA adapter into an I2_S base and re-quantizing rounds every delta to zero — the fine-tuning is silently discarded (delta magnitude ≈ 10⁻⁵ vs base weight magnitude ≈ 1.2). ternative avoids this by keeping the LoRA separate and applying it at full F32 precision at load time.

---

## Performance

Benchmarked with [Orchid 1.0](https://github.com/michelangeloromerochisco/orchid-1.0) (2B ternary, all 30 layers) on an RTX 3050 laptop (4 GB VRAM):

| Mode | Decode | Notes |
|------|-------:|-------|
| GPU — all 30 layers | **~6–7 tok/s** | 14 layers F16 + 16 layers INT8, 3296 MB VRAM |
| CPU-only (`--no-gpu`) | **~6 tok/s** | AVX2 F16C GEMM + OpenMP |

**Prefill**: batched GEMV kernel — scoring 100 prompt tokens takes milliseconds instead of seconds (~50× faster than the naive row-by-row path).

---

## Quick Start

### Requirements

- **CPU-only**: CMake 3.18+, C++17 compiler (MSVC 2022 / GCC 11+), 8 GB RAM
- **GPU**: additionally CUDA 12.x and the CUDA toolkit

> **macOS not supported.** The engine uses x86/x64 AVX2 intrinsics throughout. macOS ARM (Apple Silicon) requires a separate ARM-native backend — tracked in the roadmap below. macOS Intel may build but is not tested.

### Build

```bash
# Linux
git clone --depth 1 https://github.com/michelangeloromerochisco/ternative
cd ternative
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --parallel

# Windows (PowerShell)
git clone --depth 1 https://github.com/michelangeloromerochisco/ternative
cd ternative
cmake -B build -DCMAKE_BUILD_TYPE=Release; cmake --build build --parallel
```

**GPU build** — add `-DTERNATIVE_CUDA=ON`:
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTERNATIVE_CUDA=ON
cmake --build build --parallel
```

### Generate text

```bash
# Download Orchid 1.0 weights (Linux)
huggingface-cli download MicheRomChis/orchid-1.0 \
  ggml-model-i2_s.gguf dpo_aligned-lora.gguf \
  --local-dir ./models

# Run (Linux)
./build/ternative \
  --model ./models/ggml-model-i2_s.gguf \
  --lora  ./models/dpo_aligned-lora.gguf \
  --prompt "What is the capital of Colombia?"
```

### OpenAI-compatible server

```bash
./build/ternative \
  --model ./models/ggml-model-i2_s.gguf \
  --lora  ./models/dpo_aligned-lora.gguf \
  --server --port 8080
```

Any OpenAI client works:
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")
response = client.chat.completions.create(
    model="orchid",
    messages=[{"role": "user", "content": "Explain quantum entanglement simply."}]
)
print(response.choices[0].message.content)
```

---

## How It Works

```
I2_S base GGUF (~1.1 GB on disk)
        │
        ▼
Dequantize I2_S → F32
        │
        ▼
Load LoRA adapter GGUF
Apply delta: W = W_base_f32 + (lora_B @ lora_A) × α/r
        │
        ▼
Cast → F16  [cached to disk as .tvcache for fast reload]
        │
        ▼
GPU offload (mixed precision):
  Layers 0..k  → F16  (141 MB/layer)
  Layers k..N  → INT8 (~70 MB/layer, auto when VRAM limited)
  GPU KV-cache → 1024 tokens × 30 layers
        │
        ▼
Forward pass (fully GPU-resident per token):
  embed → [rms_norm → QKV → RoPE → attention → FFN] × 30 → logits
```

---

## Flags

```
--model, -m <path>    Base GGUF model path (required)
--lora <path>         LoRA adapter GGUF path (optional, repeatable)
--prompt, -p <text>   Prompt text (generate mode)
--max-tokens <n>      Max new tokens to generate (default: 512)
--temperature <f>     Sampling temperature (default: 0.8)
--top-p <f>           Nucleus sampling cutoff (default: 0.9)
--top-k <n>           Top-k sampling (default: 40)
--server              Run OpenAI-compatible HTTP server
--port <n>            Server port (default: 8080)
--no-gpu              Disable GPU offload, force CPU-only
--export-gguf <path>  Export merged F16 model as GGUF
--info <path>         Print GGUF metadata and exit
```

---

## Supported Models

| Model | Format | LoRA | Status |
|-------|--------|------|--------|
| [Orchid 1.0](https://huggingface.co/MicheRomChis/orchid-1.0) | I2_S + GGUF LoRA | ✓ | ✅ Production |
| BitNet b1.58-2B-4T (base) | I2_S | — | ✅ |
| Terse (upcoming) | I2_S + extensions | ✓ | 🔜 |

---

## Project Structure

```
ternative/
├── cuda/               CUDA kernels (GEMV, attention, RoPE, KV-cache, INT8)
├── include/ternative/  Public C++ headers
├── scripts/            Build helpers (build.sh, build.ps1, build.bat)
├── src/                Engine implementation
│   ├── model.cpp       Forward pass, GPU offload, LoRA merge, cache
│   ├── server.cpp      OpenAI-compatible HTTP server
│   ├── tokenizer.cpp   BPE tokenizer with GGUF vocab
│   └── ...
├── CMakeLists.txt      Build configuration
├── NOTICE              Third-party copyright notices
└── run_server.bat      Quick server launcher (Windows)
```

---

## Roadmap

- [x] GGUF v3 loader + I2_S tensor ops
- [x] LoRA merge (F32 precision, zero rounding loss)
- [x] CPU inference (AVX2 F16C GEMM, OpenMP, ~6 tok/s)
- [x] OpenAI-compatible server (`/v1/chat/completions`, `/v1/completions` with `logprobs`/`echo`)
- [x] GPU-resident forward pass (RoPE, KV-cache, attention all on GPU)
- [x] INT8 auto-quantization (symmetric per-tensor, overflow handling)
- [x] Batched GEMV prefill kernel (~50× faster prompt processing)
- [x] GPU KV-cache (1024-token capacity)
- [ ] cuBLAS GEMM for large-batch prefill
- [ ] Metal backend (Apple Silicon)
- [ ] Terse model support (MoE, KDA, MLA, recurrent depth)
- [ ] Python bindings (`ternative-py`)

---

## Third-Party Licenses

| Component | License | Copyright |
|-----------|---------|-----------|
| [llama.cpp](https://github.com/ggerganov/llama.cpp) — GGUF format conventions | MIT | Copyright (c) 2023 Georgi Gerganov |
| [nlohmann/json](https://github.com/nlohmann/json) — JSON parsing | MIT | Copyright (c) 2013-2022 Niels Lohmann |
| [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS) — CPU BLAS | BSD-3-Clause | Copyright (c) The OpenBLAS Project |
| [BitNet b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) — model architecture reference | MIT | Copyright (c) Microsoft Corporation |

See [NOTICE](NOTICE) for full copyright texts.

---

## Citation

If you use ternative in your research, please cite:

​```bibtex
@software{romerochisco2026ternative,
  title     = {ternative: Inference Engine for Ternary-Weight LLMs with Runtime LoRA},
  author    = {Romero Chisco, Michelangelo},
  year      = {2026},
  url       = {https://github.com/michelangeloromerochisco/ternative},
  license   = {Apache-2.0}
}
​```

---

## License

Apache 2.0 — free for research and commercial use.

## Acknowledgments

- **Microsoft Research** — BitNet b1.58 architecture and the I2_S ternary weight format
- **Georgi Gerganov and the llama.cpp project** — GGUF format specification and loader conventions
- **[Orchid 1.0](https://github.com/michelangeloromerochisco/orchid-1.0)** — the model that motivated this engine
