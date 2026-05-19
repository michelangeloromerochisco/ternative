---
name: Bug report
about: Something doesn't work
title: '[BUG] '
labels: bug
assignees: ''
---

**Platform**
- OS: [e.g. Windows 11, Ubuntu 22.04] — macOS not currently supported
- GPU: [e.g. RTX 3050 4GB, or CPU-only]
- CUDA version: [if GPU build]
- Compiler: [MSVC 2022 / GCC 12 / Clang 17]

**Model files**
- Base GGUF: [e.g. ggml-model-i2_s.gguf from MicheRomChis/orchid-1.0]
- LoRA GGUF: [e.g. dpo_aligned-lora.gguf]

**Command run**
```
./build/ternative --model ... --lora ... --prompt "..."
```

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened. Paste the full terminal output.

**Build log** (if build failed)
```
paste cmake / make output here
```
