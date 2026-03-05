#!/bin/bash
# AIgiSE GRPO Multi-Turn RL Training Script
#
# Usage:
#   ./examples/aigise/run_aigise_grpo.sh                    # 4-GPU default
#   ./examples/aigise/run_aigise_grpo.sh --trial debug_v15  # custom trial name
#   GPUS=2,3 NGPU=2 ./examples/aigise/run_aigise_grpo.sh   # 2-GPU override
#
# Prerequisites:
#   - Kill stale processes: pkill -9 -f sglang; pkill -9 -f rpc_server
#   - Check GPU availability: nvidia-smi

set -euo pipefail

# --- Configurable via environment variables ---
GPUS="${GPUS:-2,3,5,6}"          # CUDA_VISIBLE_DEVICES
NGPU="${NGPU:-4}"                # n_gpus_per_node
TRIAL="${TRIAL:-}"               # trial_name (auto-generated if empty)
BATCH_SIZE="${BATCH_SIZE:-2}"    # train batch size
MAX_CONCURRENT="${MAX_CONCURRENT:-4}"  # max concurrent rollouts
ALLOCATION="${ALLOCATION:-sglang:d1p1t2+fsdp:d2p1t1}"  # SGLang TP=2 inference + FSDP DP=2 TP=1 training

# --- Parse CLI args ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --trial)   TRIAL="$2"; shift 2 ;;
        --gpus)    GPUS="$2"; shift 2 ;;
        --ngpu)    NGPU="$2"; shift 2 ;;
        --batch)   BATCH_SIZE="$2"; shift 2 ;;
        *)         echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Auto-generate trial name if not provided
if [ -z "$TRIAL" ]; then
    TRIAL="run_$(date +%m%d_%H%M%S)"
fi

echo "=== AIgiSE GRPO Training ==="
echo "  GPUs:        $GPUS ($NGPU total)"
echo "  Trial:       $TRIAL"
echo "  Batch size:  $BATCH_SIZE"
echo "  Allocation:  $ALLOCATION"
echo "  Concurrent:  $MAX_CONCURRENT"
echo ""

# --- Kill stale processes ---
echo "Cleaning up stale processes..."
pkill -9 -f sglang 2>/dev/null || true
pkill -9 -f rpc_server 2>/dev/null || true
sleep 2

# --- Run ---
CUDA_VISIBLE_DEVICES=$GPUS uv run examples/aigise/aigise_rl_mt.py \
    --config examples/aigise/aigise_grpo_mt.yaml \
    scheduler.type=local \
    trial_name="$TRIAL" \
    allocation_mode="$ALLOCATION" \
    cluster.n_nodes=1 \
    cluster.n_gpus_per_node="$NGPU" \
    train_dataset.batch_size="$BATCH_SIZE" \
    rollout.max_concurrent_rollouts="$MAX_CONCURRENT" \
    rollout.max_head_offpolicyness=0
