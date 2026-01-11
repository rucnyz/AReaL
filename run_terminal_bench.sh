export DEV_NODE_URL=http://10.32.80.102:8081

python -m areal.launcher.local \
      examples/terminal_bench/terminal_bench_rl.py \
      --config examples/terminal_bench/terminal_bench_config.yaml \
      experiment_name=terminal-bench-test \
      trial_name=qwen3-32b-test \
      actor.path=/wekafs/kazhu/Qwen2-5-Coder-32B-sft-3000-agent-diverse-real-5ep-5e-6 \
      agent_config.use_remote=true +sglang.attention_backend=triton