"""AIgiSE Multi-Turn RL Workflow for AReaL."""

import json
import logging
import os
import uuid
from datetime import datetime

import aigise
from transformers import PreTrainedTokenizerFast

from areal import workflow_context
from areal.api.cli_args import GenerationHyperparameters
from areal.api.workflow_api import RolloutWorkflow
from areal.experimental.adk import ArealLlm
from areal.experimental.openai import ArealOpenAI
from areal.utils import stats_tracker

logger = logging.getLogger(__name__)


def _debug_log(msg: str) -> None:
    """Write debug message to a file for post-run analysis."""
    log_path = os.environ.get("AREAL_DEBUG_LOG", "/tmp/areal_llm_debug.log")
    try:
        with open(log_path, "a") as f:
            f.write(f"[{datetime.now().isoformat()}] [Workflow] {msg}\n")
    except Exception:
        pass


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
        create_kwargs: dict | None = None,
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
        self.create_kwargs = create_kwargs or {}

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
        data_id = data.get("id", "unknown")
        on_generate = None
        traj_dir = None

        if self.log_raw_conversation and self.dump_dir is not None:
            # Create a unique directory for this trajectory
            traj_id = uuid.uuid4().hex[:8]
            traj_dir = os.path.join(
                self.dump_dir, "raw_conversations", f"{data_id}_{traj_id}"
            )
            os.makedirs(traj_dir, exist_ok=True)
            on_generate = self._create_log_callback(traj_dir)

        model = ArealLlm(
            openai_client=client,
            default_max_tokens=self.max_new_tokens,
            on_generate=on_generate,
            create_kwargs=self.create_kwargs,
        )

        generate_kwargs = {}
        if self.max_turns is not None:
            generate_kwargs["max_turns"] = self.max_turns

        _debug_log(f"=== Trajectory START data_id={data_id} ===")

        with self._aigise_client.init_session() as session:
            result = await session.areal_generate(
                data=data, model=model, **generate_kwargs
            )

        reward = result.get("reward", 0.0)
        client.set_last_reward(reward)

        # --- Debug: log trajectory summary ---
        n_interactions = len(client._cache)
        self._log_trajectory_summary(
            data_id, client, reward, result, traj_dir
        )

        return reward

    def _log_trajectory_summary(
        self,
        data_id: str,
        client: ArealOpenAI,
        reward: float,
        result: dict,
        traj_dir: str | None,
    ) -> None:
        """Log a comprehensive trajectory summary for debugging."""
        cache = client._cache
        n_interactions = len(cache)

        # Collect per-turn info
        turn_summaries = []
        for i, (iid, interaction) in enumerate(cache.items()):
            completion = getattr(interaction, "completion", None)
            finish_reason = "?"
            if completion and completion.choices:
                finish_reason = completion.choices[0].finish_reason or "?"

            output_msgs = getattr(interaction, "output_message_list", None) or []
            tool_calls_in_turn = []
            text_in_turn = ""
            for msg in output_msgs:
                content = msg.get("content", "") or ""
                if content:
                    text_in_turn += content
                for tc in (msg.get("tool_calls", None) or []):
                    func = tc.get("function", {})
                    tool_calls_in_turn.append(func.get("name", "?"))

            turn_summaries.append(
                f"  Turn {i}: finish={finish_reason}, "
                f"tools={tool_calls_in_turn}, "
                f"text_len={len(text_in_turn)}"
            )

        summary = (
            f"=== Trajectory Summary data_id={data_id} ===\n"
            f"  reward: {reward}\n"
            f"  n_turns: {n_interactions}\n"
            f"  has_error: {'error' in result}\n"
            + "\n".join(turn_summaries)
            + f"\n=== End Summary ==="
        )

        _debug_log(summary)
        logger.info(summary)

        # Also dump full trajectory to a JSON file for detailed analysis
        if traj_dir:
            self._dump_trajectory_json(data_id, client, reward, result, traj_dir)

    def _dump_trajectory_json(
        self,
        data_id: str,
        client: ArealOpenAI,
        reward: float,
        result: dict,
        traj_dir: str,
    ) -> None:
        """Dump full trajectory to JSON files for human analysis.

        Saves two files:
        - trajectory_summary.json: lightweight overview (turn count, tools, reward)
        - trajectory_full.json: complete conversation with all messages, tool calls,
          tool responses, and per-turn rewards — everything needed to replay/analyze
          the agent's behavior
        """
        try:
            turns_summary = []
            turns_full = []

            for i, (iid, interaction) in enumerate(client._cache.items()):
                input_msgs = getattr(interaction, "messages", []) or []
                output_msgs = getattr(interaction, "output_message_list", []) or []
                completion = getattr(interaction, "completion", None)
                finish_reason = "?"
                if completion and completion.choices:
                    finish_reason = completion.choices[0].finish_reason or "?"

                # Extract tool call names for summary
                tool_names = []
                for msg in output_msgs:
                    for tc in (msg.get("tool_calls", None) or []):
                        func = tc.get("function", {})
                        tool_names.append(func.get("name", "?"))

                turns_summary.append({
                    "interaction_id": iid,
                    "n_input_messages": len(input_msgs),
                    "finish_reason": finish_reason,
                    "tool_calls": tool_names,
                    "reward": getattr(interaction, "reward", None),
                })

                # For full trajectory, include all messages.
                # input_msgs of turn i contains the full conversation history up to
                # that turn, including previous assistant responses and tool results.
                # To avoid duplication, we only save "new" messages since last turn:
                # the tool responses (from environment) + any new user messages.
                if i == 0:
                    # First turn: save all input messages (system + user prompt)
                    new_input_msgs = input_msgs
                else:
                    # Subsequent turns: save only messages added since last turn
                    prev_input_count = turns_full[-1]["_input_msg_count"]
                    new_input_msgs = input_msgs[prev_input_count:]

                turns_full.append({
                    "turn": i,
                    "interaction_id": iid,
                    "finish_reason": finish_reason,
                    "new_messages": new_input_msgs,  # tool responses, new user msgs
                    "assistant_response": output_msgs,  # assistant output with tool_calls
                    "reward": getattr(interaction, "reward", None),
                    "_input_msg_count": len(input_msgs),  # bookkeeping, stripped later
                })

            # Strip bookkeeping field
            for t in turns_full:
                t.pop("_input_msg_count", None)

            # Summary file (lightweight)
            summary_data = {
                "data_id": data_id,
                "reward": reward,
                "n_turns": len(turns_summary),
                "timestamp": datetime.now().isoformat(),
                "turns": turns_summary,
            }
            summary_path = os.path.join(traj_dir, "trajectory_summary.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2, default=str)

            # Full trajectory file (for human analysis)
            full_data = {
                "data_id": data_id,
                "reward": reward,
                "n_turns": len(turns_full),
                "timestamp": datetime.now().isoformat(),
                "task_data": {
                    k: v for k, v in (result.get("task_data", None) or {}).items()
                } if result.get("task_data") else None,
                "turns": turns_full,
            }
            full_path = os.path.join(traj_dir, "trajectory_full.json")
            with open(full_path, "w", encoding="utf-8") as f:
                json.dump(full_data, f, ensure_ascii=False, indent=2, default=str)

        except Exception as e:
            _debug_log(f"Failed to dump trajectory JSON: {e}")

    async def arun_episode(self, engine, data) -> dict:
        data_id = data.get("id", "unknown") if isinstance(data, dict) else "?"
        _debug_log(f"=== arun_episode START data_id={data_id} ===")

        client = ArealOpenAI(
            engine=engine,
            tokenizer=self.tokenizer,
            tool_call_parser=self.tool_call_parser,
            reasoning_parser=self.reasoning_parser,
            chat_template_type=self.chat_template_type,
        )

        reward = await self._run_trajectory(data, client)
        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=reward)

        _debug_log(
            f"arun_episode: reward={reward}, "
            f"n_interactions={len(client._cache)}, "
            f"applying reward discount..."
        )

        client.apply_reward_discount(turn_discount=0.9)
        completions_with_reward = client.export_interactions(style=self.export_style)

        # Log export stats
        n_exported = len(completions_with_reward)
        total_tokens = 0
        for iid, interaction in completions_with_reward.items():
            mr = getattr(interaction, "model_response", None)
            if mr:
                total_tokens += len(getattr(mr, "input_tokens", []))
                total_tokens += len(getattr(mr, "output_tokens_without_stop", []))

        _debug_log(
            f"=== arun_episode DONE data_id={data_id} "
            f"reward={reward} exported={n_exported} "
            f"total_tokens={total_tokens} ==="
        )

        return completions_with_reward
