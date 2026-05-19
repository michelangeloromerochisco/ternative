@echo off
cd /d "C:\Users\miche\Desktop\synaptic projects\LLM\ternative.cpp"
build\Release\ternative.exe --server --port 8093 --model "..\orchid\models\base\bitnet-bf16.gguf" --lora "..\orchid\models\orchid\dpo_aligned_lora.gguf"