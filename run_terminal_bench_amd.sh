export DEV_NODE_URL=http://10.32.80.102:8081

python -m areal.launcher.local \
      examples/terminal_bench/terminal_bench_rl.py \
      --config examples/terminal_bench/terminal_bench_config.yaml \
      experiment_name=terminal-bench-test \
      trial_name=qwen3-32b-test \
      actor.path=yuzhounie/sft_qwen32b \
      agent_config.use_remote=true