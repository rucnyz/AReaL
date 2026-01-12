python -m areal.launcher.local \
      examples/terminal_bench/terminal_bench_rl.py \
      --config examples/terminal_bench/terminal_bench_config.yaml \
      experiment_name=terminal-bench-test-local \
      trial_name=qwen3-32b-test-local \
      actor.path=yuzhounie/sft_qwen32b \
      agent_config.use_remote=false