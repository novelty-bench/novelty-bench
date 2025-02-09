#!/bin/bash

set -Eeuo pipefail

MODEL=$1
PORT=$2

# Start VLLM server
exec vllm serve $MODEL \
    --port $PORT \
    --tensor-parallel-size $(nvidia-smi -L | wc -l) \
    --max-model-len 8192 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.95
