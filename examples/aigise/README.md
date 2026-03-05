# AIgiSE RL Training with AReaL

Train AIgiSE security analysis agents using AReaL's GRPO pipeline.

## Quick Start

```bash
# From AReaL root
./examples/aigise/run_aigise_grpo.sh --trial my_experiment
```

See [AReaL-Training wiki](https://github.com/opensage-agent/AIgiSE/blob/main/docs/wiki/AReaL-Training.md) for full setup, configuration, and known issues.

## Files

| File | Role |
|------|------|
| `aigise_grpo_mt.yaml` | Training config (model, generation_kwargs, etc.) |
| `run_aigise_grpo.sh` | Launch script (GPU selection, process cleanup) |
| `aigise_rl_mt.py` | Entry point |
| `workflow.py` | RL workflow: rollout orchestration, trajectory logging |
| `configs.py` | Dataclass config with `generation_kwargs` support |
