#!/bin/sh
# Start the project-local Qwen3 server. Stop with Ctrl-C.
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$project_dir/.local/ollama/llama-server" \
    --model "$project_dir/.cache/chat-models/qwen3-0.6b-q4_k_m.gguf" \
    --host 127.0.0.1 --port 11434 --ctx-size 32768 --threads 4 --parallel 1 \
    --chat-template-kwargs '{"enable_thinking":false}'
