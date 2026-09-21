#!/usr/bin/env bash
# Serve Qwen3.5-4B locally with llama.cpp, on an OpenAI-compatible endpoint.
#
#   brew install llama.cpp        # macOS; builds with Metal
#   ./serve.sh
#
# Then http://127.0.0.1:8080 is both a browser UI and an API that the rest of
# this folder talks to. Nothing leaves the machine.
set -euo pipefail

REPO="unsloth/Qwen3.5-4B-GGUF"
FILE="Qwen3.5-4B-Q4_K_M.gguf"        # 2.7 GB: 4-bit weights, the usual default
DIR="$(cd "$(dirname "$0")/../models" && pwd)"
PORT="${PORT:-8080}"

if [ ! -f "$DIR/$FILE" ]; then
  echo "Downloading $FILE (2.7 GB) into $DIR ..."
  python3 - "$REPO" "$FILE" "$DIR" <<'PY'
import sys
from huggingface_hub import hf_hub_download
print(hf_hub_download(sys.argv[1], sys.argv[2], local_dir=sys.argv[3]))
PY
fi

# --alias      the name the OpenAI client asks for
# --ctx-size   how much context to reserve; the KV cache from Lecture 8 is paid here
# --jinja      use the model's own chat template, which is what makes
#              chat_template_kwargs={"enable_thinking": false} work
# --n-gpu-layers 99   put every layer on the Metal GPU
exec llama-server \
  --model "$DIR/$FILE" \
  --alias qwen3.5-4b \
  --host 127.0.0.1 --port "$PORT" \
  --ctx-size 16384 \
  --n-gpu-layers 99 \
  --jinja \
  --temp 0.7 --top-p 0.8 --top-k 20
