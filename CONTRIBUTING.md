# Contributing to ternative

Thank you for your interest. ternative is a focused inference engine — contributions that keep it lean and correct are welcome.

## What fits

- **Bug fixes** — especially platform-specific build failures, correctness issues, memory safety
- **Performance improvements** — kernel optimizations, better memory layout, faster prefill
- **New backend support** — Metal (Apple Silicon), ARM NEON, ROCm
- **Model support** — new ternary model architectures that use the GGUF format
- **Documentation** — build instructions, flag documentation, architecture explanations

## What doesn't fit (for now)

- Feature flags that bloat the binary for most users
- New quantization formats unrelated to ternary models
- Python wrappers (tracked separately as `ternative-py`)

## How to contribute

1. **Open an issue first** for any non-trivial change. Describe what you're building and why.
2. **Fork and branch** — branch names like `fix/windows-avx2`, `feat/metal-backend`
3. **Keep diffs small** — one concern per PR
4. **Test on at least one platform** — paste your build output and a sample inference run
5. **Follow the existing code style** — C++17, snake_case, RAII, no unnecessary heap allocation

## Build and test

```bash
# CPU build (baseline)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# GPU build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DTERNATIVE_CUDA=ON
cmake --build build --parallel

# Run tests
ctest --test-dir build --output-on-failure
```

## License

By contributing you agree your changes are licensed under Apache 2.0.
