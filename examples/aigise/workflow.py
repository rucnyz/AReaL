"""AIgiSE Multi-Turn RL Workflow for AReaL."""

import json
import os
import uuid

import aigise
from transformers import PreTrainedTokenizerFast

from areal import workflow_context
from areal.api.cli_args import GenerationHyperparameters
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.adk import ArealLlm
from areal.experimental.openai import ArealOpenAI
from areal.utils import stats_tracker


class AIgiSERLWorkflow(RolloutWorkflow):
    """Multi-turn RL workflow for AIgiSE tasks.

    Follows the same pattern as MultiturnRLVRWorkflow in gsm8k_rl_mt.py.
    All __init__ args must be serializable (strings, dicts, primitives)
    so the workflow can be re-instantiated on remote RPC workers.
    """

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        agent_name: str = "vul_agent_static_tools",
        benchmark_name: str = "secodeplt",
        max_turns: int | None = None,
        dump_dir: str | None = None,
        export_style: str = "concat",
        tool_call_parser: str = "qwen25",
        reasoning_parser: str = "qwen3-thinking",
        log_raw_conversation: bool = False,
        model_name: str | None = None,
    ):
        if isinstance(tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(tokenizer)

        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.dump_dir = dump_dir
        self.export_style = export_style
        self.max_new_tokens = gconfig.max_new_tokens
        self.tool_call_parser = tool_call_parser
        self.reasoning_parser = reasoning_parser
        self.log_raw_conversation = log_raw_conversation

        if export_style not in ["individual", "concat"]:
            raise ValueError(f"Invalid export style: {export_style}")
        self.chat_template_type = "concat" if export_style == "concat" else "hf"

        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

        # Create AIgiSE client
        self._aigise_client = aigise.create(
            agent_name, benchmark_name, model_name=model_name
        )

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
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(turn_data, f, ensure_ascii=False, indent=2)
            turn_count[0] += 1

        return log_turn

    async def _run_trajectory(self, data: dict, client: ArealOpenAI) -> float:
        """Run a single trajectory using AIgiSE agent."""
        on_generate = None

        if self.log_raw_conversation and self.dump_dir is not None:
            # Create a unique directory for this trajectory
            traj_id = uuid.uuid4().hex[:8]
            data_id = data.get("id", "unknown")
            traj_dir = os.path.join(
                self.dump_dir, "raw_conversations", f"{data_id}_{traj_id}"
            )
            os.makedirs(traj_dir, exist_ok=True)
            on_generate = self._create_log_callback(traj_dir)

        model = ArealLlm(
            openai_client=client,
            default_max_tokens=self.max_new_tokens,
            on_generate=on_generate,
        )

        generate_kwargs = {}
        if self.max_turns is not None:
            generate_kwargs["max_turns"] = self.max_turns

        with self._aigise_client.init_session() as session:
            result = await session.areal_generate(
                data=data, model=model, **generate_kwargs
            )

        reward = result.get("reward", 0.0)
        client.set_last_reward(reward)
        return reward

    async def arun_episode(self, engine, data) -> dict:
        client = ArealOpenAI(
            engine=engine,
            tokenizer=self.tokenizer,
            tool_call_parser=self.tool_call_parser,
            reasoning_parser=self.reasoning_parser,
            chat_template_type=self.chat_template_type,
        )

        reward = await self._run_trajectory(data, client)
        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=reward)

        client.apply_reward_discount(turn_discount=0.9)
        completions_with_reward = client.export_interactions(style=self.export_style)
        return completions_with_reward
