#!/usr/bin/env bash
# HidraChat Worker — start script
set -e

echo "⚡ HidraChat Worker"

# Optional: set env vars here
# export HIDRACHAT_WORKER_NAME="my-gpu-worker"
# export HIDRACHAT_WORKER_EMAIL="you@email.com"
# export HIDRACHAT_N_GPU_LAYERS=35     # set > 0 to use GPU
# export HIDRACHAT_THREADS=8
# export HIDRACHAT_RAM_GB=8
# export HIDRACHAT_SEARXNG_URL="http://127.0.0.1:8092"  # SearXNG local instance

python3 worker.py "$@"
