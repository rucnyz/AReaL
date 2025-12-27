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

import asyncio
import json
import os
import sys
import uuid
from dataclasses import dataclass, field

import aigise
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters, GRPOConfig, load_expr_config
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.adk import ArealLlm
from areal.experimental.openai import ArealOpenAI
from areal.experimental.trainer import PPOTrainer
from areal.utils import stats_tracker
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.stats_logger import StatsLogger


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
                path = "json",
                split = "train",
                data_files = dataset_path,
            )
        elif dataset_path.endswith(".parquet"):
            dataset = load_dataset(
                path = "parquet",
                split = "train",
                data_files = dataset_path,
            )
        else:
            # Try loading as generic dataset directory
            dataset = load_dataset(path = dataset_path, split = split)
    else:
        # Assume it's a HuggingFace Hub dataset
        dataset = load_dataset(path = dataset_path, split = split)

    return dataset


class AIgiSERLWorkflow(RolloutWorkflow):
    """Multi-turn RL workflow for AIgiSE tasks.

    Follows the same pattern as MultiturnRLVRWorkflow in gsm8k_rl_mt.py.
    """

    def __init__(
            self,
            gconfig: GenerationHyperparameters,
            tokenizer: PreTrainedTokenizerFast,
            agent_name: str = "vul_agent_static_tools",
            benchmark_name: str = "secodeplt",
            dump_dir: str | None = None,
            rollout_stat_scope: str = "rollout",
            export_style: str = "concat",
            tool_call_parser: str = "qwen25",
            reasoning_parser: str = "qwen3-thinking",
            log_raw_conversation: bool = False,
    ):
        self.n_trajs = gconfig.n_samples
        self.tokenizer = tokenizer
        self.dump_dir = dump_dir
        self.rollout_stat_scope = rollout_stat_scope
        self.export_style = export_style
        self.max_new_tokens = gconfig.max_new_tokens
        self.tool_call_parser = tool_call_parser
        self.reasoning_parser = reasoning_parser
        self.log_raw_conversation = log_raw_conversation

        if export_style not in ["individual", "concat"]:
            raise ValueError(f"Invalid export style: {export_style}")
        self.chat_template_type = "concat" if export_style == "concat" else "hf"

        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok = True)

        # Create AIgiSE client
        self._aigise_client = aigise.create(agent_name, benchmark_name)

    def _create_log_callback(self, traj_dir: str):
        """Create a callback function for logging raw conversations to JSON files.

        Args:
            traj_dir: Directory to save JSON files for this trajectory.

        Returns:
            A callback function that saves each turn as a JSON file.
        """
        turn_count = [0]  # Use list to allow modification in closure

        def log_turn(input_text: str, output_text: str):
            turn_data = {
                "turn": turn_count[0],
                "input": input_text,
                "output": output_text,
            }
            json_path = os.path.join(traj_dir, f"turn_{turn_count[0]:03d}.json")
            with open(json_path, "w", encoding = "utf-8") as f:
                json.dump(turn_data, f, ensure_ascii = False, indent = 2)
            turn_count[0] += 1

        return log_turn

    async def _run_trajectory(self, data: dict, client: ArealOpenAI) -> float:
        """Run a single trajectory using AIgiSE agent."""
        on_generate = None

        if self.log_raw_conversation and self.dump_dir is not None:
            # Create a unique directory for this trajectory
            traj_id = uuid.uuid4().hex[:8]
            data_id = data.get("id", "unknown")
            traj_dir = os.path.join(self.dump_dir, "raw_conversations", f"{data_id}_{traj_id}")
            os.makedirs(traj_dir, exist_ok = True)
            on_generate = self._create_log_callback(traj_dir)

        model = ArealLlm(
            openai_client = client,
            default_max_tokens = self.max_new_tokens,
            on_generate = on_generate,
        )

        with self._aigise_client.init_session() as session:
            result = await session.areal_generate(data = data, model = model)

        reward = result.get("reward", 0.0)
        client.set_last_reward(reward)
        return reward

    async def arun_episode(self, engine, data) -> dict:
        clients = [
            ArealOpenAI(
                engine = engine,
                tokenizer = self.tokenizer,
                tool_call_parser = self.tool_call_parser,
                reasoning_parser = self.reasoning_parser,
                chat_template_type = self.chat_template_type,
            )
            for _ in range(self.n_trajs)
        ]

        # Collect trajectories
        rewards = await asyncio.gather(
            *[self._run_trajectory(data, client) for client in clients]
        )

        for reward in rewards:
            stats_tracker.get(self.rollout_stat_scope).scalar(reward = reward)

        completions_with_reward = {}
        for client in clients:
            client.apply_reward_discount(turn_discount = 0.9)
            completions = client.export_interactions(style = self.export_style)
            completions_with_reward.update(completions)

        return completions_with_reward


@dataclass
class AIgiSEGRPOConfig(GRPOConfig):
    """Configuration for AIgiSE multi-turn GRPO training.

    Uses agent_run_args dict for flexible agent configuration,
    following the same pattern as gsm8k_rl_mt.py.
    """

    agent_run_args: dict = field(
        default_factory = dict,
        metadata = {"help": "Arguments for AIgiSE agent (agent_name, benchmark_name)."},
    )
    export_style: str = field(
        default = "concat",
        metadata = {"help": "Export style for completions: 'concat' or 'individual'."},
    )
    tool_call_parser: str = field(
        default = "qwen25",
        metadata = {"help": "Tool call parser for sglang. Options: qwen25, llama3, mistral, deepseekv3."},
    )
    reasoning_parser: str = field(
        default = "qwen3-thinking",
        metadata = {"help": "Reasoning parser for sglang. Options: qwen3-thinking."},
    )
    log_raw_conversation: bool = field(
        default = False,
        metadata = {"help": "Whether to log raw input/output text for each turn."},
    )


def main(args):
    config, _ = load_expr_config(args, AIgiSEGRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    # Load dataset directly from jsonl (like tongyi_deepresearch/train.py)
    train_dataset = get_aigise_dataset(config.train_dataset.path, tokenizer = tokenizer)

    with PPOTrainer(
            config,
            train_dataset = train_dataset,
            valid_dataset = None,  # No validation dataset for agent RL
    ) as trainer:
        # Extract agent args from config
        agent_name = config.agent_run_args.get("agent_name", "vul_agent_static_tools")
        benchmark_name = config.agent_run_args.get("benchmark_name", "secodeplt")
        log_path = StatsLogger.get_log_path(config.stats_logger)

        workflow = AIgiSERLWorkflow(
            gconfig = config.gconfig,
            tokenizer = trainer.tokenizer,
            agent_name = agent_name,
            benchmark_name = benchmark_name,
            dump_dir = os.path.join(log_path, "generated"),
            export_style = config.export_style,
            tool_call_parser = config.tool_call_parser,
            reasoning_parser = config.reasoning_parser,
            log_raw_conversation = config.log_raw_conversation,
        )

        # Run training (no eval_workflow, like tongyi_deepresearch)
        trainer.train(workflow, eval_workflow = None)


if __name__ == "__main__":
    main(sys.argv[1:])
