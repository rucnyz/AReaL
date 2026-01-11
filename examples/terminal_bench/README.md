# Terminal Bench Integration for AReaL

## Quick Start

### Remote Mode (When Docker is NOT available locally, e.g., inside a container)

**Step 1: Start Dev Node Server** (on a machine with Docker access)
```bash
# On the dev node machine (must have Docker running)
cd /path/to/areal
python terminal_agent_utils/dev_node_server.py --host 0.0.0.0 --port 8081
```

**Step 2: Run Training** (on the training node)
```bash
# Option A: Set environment variable
export DEV_NODE_URL=http://<dev-node-ip>:8080

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m areal.launcher.local \
    examples/terminal_bench/terminal_bench_rl.py \
    --config examples/terminal_bench/terminal_bench_config.yaml \
    experiment_name=terminal-bench-test \
    trial_name=qwen3-8b-test \
    actor.path=Qwen/Qwen3-8B \
    +sglang.attention_backend=triton \
    stats_logger.wandb.mode=disabled

# Option B: Use command line args
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m areal.launcher.local \
    examples/terminal_bench/terminal_bench_rl.py \
    --config examples/terminal_bench/terminal_bench_config.yaml \
    experiment_name=terminal-bench-test \
    trial_name=qwen3-8b-test \
    actor.path=Qwen/Qwen3-8B \
    agent_config.use_remote=true \
    agent_config.dev_node_url=http://<dev-node-ip>:8080 \
    +sglang.attention_backend=triton \
    stats_logger.wandb.mode=disabled
```

### Local Docker Mode (When Docker IS available locally)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m areal.launcher.local \
    examples/terminal_bench/terminal_bench_rl.py \
    --config examples/terminal_bench/terminal_bench_config.yaml \
    experiment_name=terminal-bench-test \
    trial_name=qwen3-32b-test \
    actor.path=yuzhounie/sft_qwen32b \
    agent_config.use_remote=false
```

### AMD GPU (ROCm) Notes
- Use `+sglang.attention_backend=triton` (FA3 not supported on AMD)
- Use `stats_logger.wandb.mode=disabled` if wandb not configured

### Monitor Docker Containers
```bash
watch -n 1 'echo "Containers: $(docker ps --filter name=tb__ -q | wc -l) / 64"'
./monitor_docker.sh
python plot_docker.py docker_count.csv
```

---

This example demonstrates how to train an RL agent to solve terminal-based tasks using the AReaL framework. The agent learns to interact with a bash terminal to complete software engineering tasks, with rewards derived from automated test execution.

## Architecture

The system supports two modes:

### Local Docker Mode
```
┌─────────────────────────────────────┐
│           Training Node             │
│  ┌─────────┐      ┌──────────────┐  │
│  │  AReaL  │ ───> │ Local Docker │  │
│  └─────────┘      └──────────────┘  │
└─────────────────────────────────────┘
```

### Remote Dev Node Mode
```
┌─────────────────┐         HTTP API          ┌──────────────┐
│  Training Node  │ ────────────────────────> │  Dev Node    │
│   (AReaL)       │ <──────────────────────── │  (Docker)    │
└─────────────────┘      JSON Responses       └──────────────┘
```

**Mode Selection:**
- `use_remote: null` (default): Auto-detect from `DEV_NODE_URL` environment variable
- `use_remote: true`: Force remote mode
- `use_remote: false`: Force local Docker mode

## Prerequisites

### For Local Docker Mode

Ensure Docker is installed and running on the training machine:

```bash
# Check Docker is running
docker info

# Ensure you have permissions (or use sudo)
docker ps
```

### For Remote Dev Node Mode

The dev node server must be running to handle Docker container management:

```bash
# On the dev node machine
cd /path/to/kaijie_verl
python terminal_agent_utils/dev_node_server.py --host 0.0.0.0 --port 8080
```

### 2. Task Dataset

Prepare your task dataset in Parquet format. The dataset should have the following fields:
- `task_name`: Name of the task (corresponds to task directory name)
- `prompt`: Task instruction (string or list of messages)
- `extra_info`: Optional metadata dict

Example structure:
```
tasks_base_path/
├── task_name_1/
│   ├── docker-compose.yaml
│   ├── run-tests.sh
│   └── tests/
│       └── test_outputs.py
├── task_name_2/
│   └── ...
```

### 3. Dataset Conversion

If you have tasks from terminal-bench format, convert them to Parquet:

```python
import pandas as pd

data = []
for task_dir in tasks_base_path.iterdir():
    if task_dir.is_dir():
        task_yaml = task_dir / "task.yaml"
        if task_yaml.exists():
            with open(task_yaml) as f:
                task_info = yaml.safe_load(f)
            data.append({
                "task_name": task_dir.name,
                "prompt": task_info.get("instruction", ""),
                "extra_info": {"difficulty": task_info.get("difficulty", "medium")}
            })

df = pd.DataFrame(data)
df.to_parquet("train.parquet")
```

## Configuration

Edit `terminal_bench_config.yaml` to set your paths and parameters:

```yaml
# Dataset paths
dataset_path: /path/to/terminal_bench/train.parquet
valid_dataset_path: /path/to/terminal_bench/test.parquet

# Agent configuration
agent_config:
  dev_node_url: http://dev-node-ip:8080   # Dev node server URL (for remote mode)
  tasks_base_path: /path/to/tasks          # Base path to task directories
  max_turns: 10                            # Max interaction turns
  command_timeout_sec: 30.0                # Command execution timeout
  test_timeout_sec: 300.0                  # Test execution timeout
  max_output_bytes: 10000                  # Max output size per command
  use_remote: null                         # null=auto, true=remote, false=local

# Model
actor:
  path: Qwen/Qwen2.5-7B-Instruct           # Model to train

# Generation
gconfig:
  n_samples: 2                             # Samples per task
  max_new_tokens: 2048                     # Max tokens per generation
  temperature: 0.7                         # Sampling temperature
```

## Launch Training

### Local Docker Mode

```bash
# No DEV_NODE_URL set = automatic local mode
python -m areal.launcher.local examples/terminal_bench/terminal_bench_rl.py \
    --config examples/terminal_bench/terminal_bench_config.yaml \
    experiment_name=terminal-bench-local \
    trial_name=trial1 \
    agent_config.tasks_base_path=/data/terminal-bench/tasks \
    dataset_path=/data/terminal-bench/train.parquet
```

Or explicitly set local mode:
```bash
python -m areal.launcher.local examples/terminal_bench/terminal_bench_rl.py \
    --config examples/terminal_bench/terminal_bench_config.yaml \
    agent_config.use_remote=false \
    agent_config.tasks_base_path=/data/terminal-bench/tasks \
    dataset_path=/data/terminal-bench/train.parquet
```

### Remote Dev Node Mode

```bash
# Set DEV_NODE_URL for automatic remote mode
export DEV_NODE_URL=http://10.0.0.1:8080

python -m areal.launcher.local examples/terminal_bench/terminal_bench_rl.py \
    --config examples/terminal_bench/terminal_bench_config.yaml \
    experiment_name=terminal-bench-remote \
    trial_name=trial1 \
    agent_config.tasks_base_path=/data/terminal-bench/tasks \
    dataset_path=/data/terminal-bench/train.parquet
```

Or explicitly set remote mode:
```bash
python -m areal.launcher.local examples/terminal_bench/terminal_bench_rl.py \
    --config examples/terminal_bench/terminal_bench_config.yaml \
    agent_config.use_remote=true \
    agent_config.dev_node_url=http://10.0.0.1:8080 \
    agent_config.tasks_base_path=/data/terminal-bench/tasks \
    dataset_path=/data/terminal-bench/train.parquet
```

### Multi-Node Training (Remote Mode Recommended)

```bash
export DEV_NODE_URL=http://dev-node:8080

python -m areal.launcher.local examples/terminal_bench/terminal_bench_rl.py \
    --config examples/terminal_bench/terminal_bench_config.yaml \
    cluster.n_nodes=2 \
    cluster.n_gpus_per_node=8 \
    experiment_name=terminal-bench-multi \
    agent_config.tasks_base_path=/shared/tasks \
    dataset_path=/shared/train.parquet
```

### With Wandb Logging

```bash
python -m areal.launcher.local examples/terminal_bench/terminal_bench_rl.py \
    --config examples/terminal_bench/terminal_bench_config.yaml \
    stats_logger.wandb.mode=online \
    stats_logger.wandb.project=terminal-bench \
    stats_logger.wandb.name=exp1
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `agent_config.use_remote` | Terminal mode: `null`=auto, `true`=remote, `false`=local | `null` |
| `agent_config.dev_node_url` | URL of the dev node server (for remote mode) | `http://localhost:8080` |
| `agent_config.tasks_base_path` | Base path to task directories | - |
| `agent_config.max_turns` | Maximum interaction turns | 10 |
| `agent_config.command_timeout_sec` | Timeout for command execution | 30.0 |
| `agent_config.test_timeout_sec` | Timeout for test execution | 300.0 |
| `turn_discount` | Reward discount factor per turn | 0.9 |
| `gconfig.n_samples` | Number of rollout samples per task | 2 |
| `gconfig.temperature` | Sampling temperature | 0.7 |

## Reward Computation

The reward is computed based on test execution results:

```
reward = passed_tests / total_tests
```

Multi-turn reward propagation uses backward discounting:
```
reward[t] = reward[t] + reward[t+1] * turn_discount
```

## Trajectory Logging

Agent trajectories are saved to `running_logs/` for debugging:

```json
{
  "task_name": "example_task",
  "request_id": "abc12345",
  "reward_score": 0.8,
  "parser_results": {"test_1": "passed", "test_2": "passed", "test_3": "failed"},
  "num_user_turns": 5,
  "num_assistant_turns": 4,
  "trajectory": [
    {"type": "user_input", "content": "..."},
    {"type": "assistant", "content": "...", "tool_calls": [...]},
    {"type": "observation", "command": "ls -la", "content": "..."},
    ...
  ]
}
```

## Troubleshooting

### Connection Refused
- Ensure dev node server is running
- Check firewall settings
- Verify `dev_node_url` is correct

### Docker Errors
- Check Docker is installed and running on dev node
- Verify task docker-compose.yaml files are valid
- Check container name conflicts (cleanup old containers)

### Test Timeout
- Increase `test_timeout_sec` for long-running tests
- Check test script for infinite loops
- Verify test dependencies are installed

### Out of Memory
- Reduce `gconfig.n_samples`
- Reduce `rollout.max_concurrent_rollouts`
- Use a smaller model

## File Structure

```
examples/terminal_bench/
├── __init__.py
├── README.md
├── terminal_bench_rl.py          # Main workflow and agent
├── terminal_bench_config.yaml    # Training configuration
├── reward.py                     # Pytest result parsing
├── dataset.py                    # Parquet dataset loading
└── remote_terminal/              # Terminal utilities (local & remote)
    ├── __init__.py
    ├── remote_docker_client.py   # HTTP client for remote mode
    ├── remote_terminal.py        # RemoteTerminal, RemoteTmuxSession
    ├── docker_compose_manager.py # Docker Compose manager for local mode
    └── local_terminal.py         # Terminal, TmuxSession for local mode
```

## References

- Original terminal-bench implementation: verl repository
- AReaL multi-turn example: `examples/multi_turn_math/gsm8k_rl_mt.py`
