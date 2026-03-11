#!/bin/bash
# Training-aligned inference for AIgiSE + AReaL.
#
# Runs SeCodePLT through the SAME code path as training rollouts
# (ArealOpenAI -> ArealLlm -> ADK -> AIgiSE), useful for diagnosing
# tool call parsing and model capability issues.
#
# Usage:
#   bash run_infer_aligned.sh                      # defaults
#   bash run_infer_aligned.sh arvo:65380           # custom task
#   SGLANG_PORT=31000 GPU=1 bash run_infer_aligned.sh  # custom port/gpu

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-UCSB-SURFI/VulnLLM-R-7B}"
GPU="${GPU:-4}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
TASK_ID="${1:-arvo:53052}"

export CUDA_VISIBLE_DEVICES="$GPU"

# --- Start SGLang server if not already running ---
if curl -sf "http://localhost:${SGLANG_PORT}/health" > /dev/null 2>&1; then
    echo "SGLang server already running on port ${SGLANG_PORT}"
else
    echo "Starting SGLang server on GPU ${GPU}, port ${SGLANG_PORT}..."
    uv run -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --tp-size 1 \
        --port "$SGLANG_PORT" \
        --mem-fraction-static 0.9 \
        --context-length 32768 \
        --log-level warning &
    SGLANG_PID=$!
    trap "echo 'Stopping SGLang server...'; kill $SGLANG_PID 2>/dev/null; wait $SGLANG_PID 2>/dev/null" EXIT

    echo "Waiting for SGLang server to be ready..."
    for i in $(seq 1 60); do
        if curl -sf "http://localhost:${SGLANG_PORT}/health" > /dev/null 2>&1; then
            echo "SGLang server ready (took ${i}s)"
            break
        fi
        sleep 1
    done
    if ! curl -sf "http://localhost:${SGLANG_PORT}/health" > /dev/null 2>&1; then
        echo "ERROR: SGLang server failed to start after 60s"
        exit 1
    fi
fi

# --- Run training-aligned inference ---
uv run examples/aigise/infer_aligned.py \
    --model_path "$MODEL_PATH" \
    --sglang_addr "http://localhost:${SGLANG_PORT}" \
    --task_id "$TASK_ID" \
    --max_new_tokens 2048 \
    --max_turns 20 \
    --tool_call_parser qwen25 \
    --reasoning_parser qwen3
