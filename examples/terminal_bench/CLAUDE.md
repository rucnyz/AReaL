# Terminal Bench 本地测试进展

## 测试目标
测试 terminal bench 在 areal 中使用本地 Docker 模式运行，使用 有空闲的GPU 和 Qwen3-4B 模型。
CUDA_VISIBLE_DEVICES 需要是空闲的GPU，例如 0,1,2,3。
你可以修改任何terminal_bench_config.yaml中的配置。以及terminal_bench文件夹下的代码，只要你觉得有必要

## 数据位置 (已配置在 terminal_bench_config.yaml 中)
- Parquet: `data/verl_rl_tasks_t100/train_filtered2.parquet`
- Tasks: `data/rl_tasks_t100/`

## 运行命令
```bash
cd ~/projects/areal

# 需要先测试环境
which python
# 结果需要是 ~/miniconda3/envs/areal/bin/python

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m areal.launcher.local \
    examples/terminal_bench/terminal_bench_rl.py \
    --config examples/terminal_bench/terminal_bench_config.yaml \
    experiment_name=terminal-bench-test \
    trial_name=qwen3-8b-test \
    actor.path=Qwen/Qwen3-8B \
    agent_config.use_remote=false
```

## 配置文件说明
`terminal_bench_config.yaml` 已预配置：
- `dataset_path` / `valid_dataset_path`: 数据 parquet 路径
- `agent_config.tasks_base_path`: 任务目录
- `cluster.n_gpus_per_node=4`: GPU 数量
- `allocation_mode=sglang:d2p1t1+d2p1t1`: 2 GPU inference + 2 GPU training
- `cluster.fileroot`: 日志输出到 `running_logs/` (相对于 areal 根目录)

如需修改，直接编辑 `terminal_bench_config.yaml` 即可。


### 2025-01: "Unexpected prompt format for task" 错误
**问题**: Parquet 文件中的 `prompt` 字段是 numpy ndarray 类型，而 `dataset.py` 只处理 str 和 list 类型。

**解决方案**: 在 `dataset.py` 中添加 numpy ndarray 到 list 的转换:
```python
# Convert numpy array to list if needed
if isinstance(prompt, np.ndarray):
    prompt = prompt.tolist()
```

## 配置说明
- `allocation_mode="sglang:d2p1t1+d2p1t1"`: 2 GPU for sglang inference + 2 GPU for training
- `cluster.n_gpus_per_node=4`: 指定节点 GPU 数量
- `agent_config.use_remote=false`: 强制使用本地 Docker 模式

## 注意事项
1. 必须先判断是否在 areal 环境中，否则 PATH 中不包含正确的 Python/torchrun
2. 日志输出位置: `areal/running_logs/` (相对于 areal 根目录)
