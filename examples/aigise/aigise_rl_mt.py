"""
AIgiSE Multi-Turn RL Training with AReaL.

This module integrates AIgiSE's agent framework into AReaL's PPO training pipeline,
following the same pattern as gsm8k_rl_mt.py and camel/train.py.

Architecture:
    AReaL PPOTrainer
         │
         ▼
    AIgiSERLWorkflow.arun_episode(engine, data)
         │
         ├── Create ArealOpenAI clients (n_trajs)
         │
         ├── AIgiSEAgent.run_agent(data, client)
         │       ├── Create ArealLlm(openai_client=client)
         │       ├── aigise_client.init_session() → RLSession
         │       ├── session.areal_generate(data, model) → result
         │       └── client.set_last_reward(reward)
         │
         └── Export completions for PPO training

Usage:
    python examples/aigise/aigise_rl_mt.py --config examples/aigise/aigise_grpo_mt.yaml

YAML config example:
    agent_run_args:
        agent_name: vul_agent_static_tools
        benchmark_name: secodeplt
        max_turns: 5
    export_style: concat
"""

import os
import sys

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import load_expr_config
from areal.trainer import PPOTrainer
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.stats_logger import StatsLogger
from examples.aigise.configs import AIgiSEGRPOConfig


def get_aigise_dataset(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerFast,
    split: str = "train",
):
    """Load AIgiSE dataset from HuggingFace Hub or local file.

    Supports:
    - HuggingFace Hub: "username/dataset_name"
    - Local jsonl: "path/to/data.jsonl"
    - Local json: "path/to/data.json"
    - Local parquet: "path/to/data.parquet"

    Args:
        dataset_path: HuggingFace dataset name or path to local file
        tokenizer: HuggingFace tokenizer (for optional filtering)
        split: Dataset split to load (default: "train")

    Returns:
        HuggingFace Dataset
    """
    # Check if it's a local file
    if os.path.exists(dataset_path):
        # Determine file type from extension
        if dataset_path.endswith(".jsonl") or dataset_path.endswith(".json"):
            dataset = load_dataset(
                path="json",
                split="train",
                data_files=dataset_path,
            )
        elif dataset_path.endswith(".parquet"):
            dataset = load_dataset(
                path="parquet",
                split="train",
                data_files=dataset_path,
            )
        else:
            # Try loading as generic dataset directory
            dataset = load_dataset(path=dataset_path, split=split)
    else:
        # Assume it's a HuggingFace Hub dataset
        dataset = load_dataset(path=dataset_path, split=split)

    return dataset


def main(args):
    config, _ = load_expr_config(args, AIgiSEGRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # Load dataset directly from jsonl (like tongyi_deepresearch/train.py)
    train_dataset = get_aigise_dataset(config.train_dataset.path, tokenizer=tokenizer)

    # Extract agent args from config
    agent_name = config.agent_run_args.get("agent_name", "vul_agent_static_tools")
    benchmark_name = config.agent_run_args.get("benchmark_name", "secodeplt")
    max_turns = config.agent_run_args.get("max_turns", None)
    log_path = StatsLogger.get_log_path(config.stats_logger)

    # All kwargs must be serializable (strings, dicts, primitives)
    # so remote RPC workers can re-instantiate the workflow.
    workflow_kwargs = dict(
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        agent_name=agent_name,
        benchmark_name=benchmark_name,
        max_turns=max_turns,
        dump_dir=os.path.join(log_path, "generated"),
        export_style=config.export_style,
        tool_call_parser=config.tool_call_parser,
        reasoning_parser=config.reasoning_parser,
        log_raw_conversation=config.log_raw_conversation,
        model_name=config.tokenizer_path,
    )

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=None,  # No validation dataset for agent RL
    ) as trainer:
        # Run training (no eval_workflow, like tongyi_deepresearch)
        trainer.train(
            workflow="examples.aigise.workflow.AIgiSERLWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow=None,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
