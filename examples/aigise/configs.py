from dataclasses import dataclass, field

from areal.api.cli_args import GRPOConfig


@dataclass
class AIgiSEGRPOConfig(GRPOConfig):
    """Configuration for AIgiSE multi-turn GRPO training.

    Uses agent_run_args dict for flexible agent configuration,
    following the same pattern as gsm8k_rl_mt.py.
    """

    agent_run_args: dict = field(
        default_factory=dict,
        metadata={"help": "Arguments for AIgiSE agent (agent_name, benchmark_name)."},
    )
    export_style: str = field(
        default="concat",
        metadata={"help": "Export style for completions: 'concat' or 'individual'."},
    )
    tool_call_parser: str = field(
        default="qwen25",
        metadata={
            "help": "Tool call parser for sglang. Options: qwen25, llama3, mistral, deepseekv3."
        },
    )
    reasoning_parser: str = field(
        default="qwen3-thinking",
        metadata={"help": "Reasoning parser for sglang. Options: qwen3-thinking."},
    )
    log_raw_conversation: bool = field(
        default=False,
        metadata={"help": "Whether to log raw input/output text for each turn."},
    )
