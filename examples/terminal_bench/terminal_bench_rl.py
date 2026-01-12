"""Terminal Bench RL Training for AReaL.

This module implements multi-turn terminal interaction training using
the AReaL framework. It supports both local Docker and remote dev node
modes for container management and test execution.

Usage:
    python -m areal.launcher.local examples/terminal_bench/terminal_bench_rl.py \
        --config examples/terminal_bench/terminal_bench_config.yaml
"""

import asyncio
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

from openai.types.chat import ChatCompletion
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters, GRPOConfig, load_expr_config
from areal.api.workflow_api import RolloutWorkflow
from areal.core import workflow_context
from areal.experimental.openai import ArealOpenAI
from areal.experimental.trainer import PPOTrainer
from areal.utils import stats_tracker
from areal.utils.hf_utils import load_hf_tokenizer
from areal.utils.logging import LOGGER_COLORS_EXACT, getLogger

# Register logger colors for terminal_bench (avoid faint white on dark terminals)
LOGGER_COLORS_EXACT["TerminalBenchWorkflow"] = "light_purple"  # Like other workflows
LOGGER_COLORS_EXACT["LauncherUtils"] = "blue"  # Like other launcher components
LOGGER_COLORS_EXACT["SGLangWrapper"] = "light_cyan"  # Like other inference engines
LOGGER_COLORS_EXACT["VLLMWrapper"] = "light_cyan"  # Like other inference engines

try:
    from .dataset import load_terminal_bench_dataset
    from .remote_terminal import RemoteTerminal, RemoteTmuxSession, Terminal, TmuxSession
    from .remote_terminal.remote_docker_client import RemoteDockerClient
    from .reward import UnitTestStatus, compute_reward, parse_pytest_results
except ImportError:
    # When running directly with torchrun, use absolute imports
    from examples.terminal_bench.dataset import load_terminal_bench_dataset
    from examples.terminal_bench.remote_terminal import RemoteTerminal, RemoteTmuxSession, Terminal, TmuxSession
    from examples.terminal_bench.remote_terminal.remote_docker_client import RemoteDockerClient
    from examples.terminal_bench.reward import UnitTestStatus, compute_reward, parse_pytest_results

# Type alias for terminal classes
TerminalType = Union[Terminal, RemoteTerminal]
TmuxSessionType = Union[TmuxSession, RemoteTmuxSession]

logger = getLogger("TerminalBenchWorkflow", type_="colored")

# Bash tool definition in OpenAI function calling format
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command in the terminal. Use this to run shell commands, install packages, edit files, or perform any terminal operation needed to complete the task.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                }
            },
            "required": ["command"],
        },
    },
}


@dataclass
class TerminalAgentConfig:
    """Configuration for Terminal Bench agent.

    Supports both local Docker mode and remote dev node mode.
    Mode selection:
    - If use_remote=True: Uses remote dev node via HTTP
    - If use_remote=False: Uses local Docker directly
    - If use_remote=None (default): Auto-detect based on DEV_NODE_URL env var
    """

    dev_node_url: str = "http://localhost:8080"
    tasks_base_path: str = "/path/to/terminal-bench/tasks"
    max_turns: int = 10
    command_timeout_sec: float = 30.0
    test_timeout_sec: float = 300.0
    max_output_bytes: int = 10000
    no_rebuild: bool = False
    cleanup: bool = True
    save_trajectory: bool = True
    use_remote: bool | None = None  # None = auto-detect from DEV_NODE_URL env var
    max_containers: int | None = None  # Max concurrent Docker containers (None = unlimited)
    # Trajectory save path (set by trainer based on experiment config)
    trajectory_dir: str | None = None  # e.g. "running_logs/exp_name/trial_name/trajectories"


class TerminalBenchAgent:
    """Multi-turn agent for terminal bench tasks.

    Uses OpenAI function calling format for bash tool execution,
    integrating with AReaL's ArealOpenAI client.
    """

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        config: TerminalAgentConfig,
    ):
        """Initialize Terminal Bench agent.

        Args:
            gconfig: Generation hyperparameters.
            config: Agent configuration.
        """
        self.gconfig = gconfig
        self.config = config
        self._use_remote: bool | None = None
        self._remote_health_client: RemoteDockerClient | None = None

    _container_semaphore: asyncio.Semaphore | None = None
    _container_semaphore_limit: int | None = None

    @classmethod
    def _get_container_semaphore(cls, limit: int) -> asyncio.Semaphore:
        if cls._container_semaphore is None or cls._container_semaphore_limit != limit:
            cls._container_semaphore = asyncio.Semaphore(limit)
            cls._container_semaphore_limit = limit
        return cls._container_semaphore

    @asynccontextmanager
    async def _container_slot(self, request_id: str):
        if self.config.max_containers is None:
            yield
            return
        if self._should_use_remote():
            await self._wait_for_remote_slot(request_id)
            yield
            return
        semaphore = self._get_container_semaphore(self.config.max_containers)
        wait_start = time.monotonic()
        await semaphore.acquire()
        waited = time.monotonic() - wait_start
        if waited > 1.0:
            logger.info(
                "[Docker] %s waited %.2fs for local slot (max=%s)",
                request_id,
                waited,
                self.config.max_containers,
            )
        try:
            yield
        finally:
            semaphore.release()

    def _get_remote_health_client(self) -> RemoteDockerClient:
        if self._remote_health_client is None:
            dev_node_url = os.environ.get("DEV_NODE_URL") or self.config.dev_node_url
            self._remote_health_client = RemoteDockerClient(
                dev_node_url=dev_node_url,
                timeout=5.0,
                max_retries=1,
                retry_delay=0.5,
            )
        return self._remote_health_client

    async def _wait_for_remote_slot(self, request_id: str) -> None:
        poll_interval = 1.0
        wait_count = 0
        while True:
            try:
                health = await self._run_blocking(
                    self._get_remote_health_client().health_check
                )
            except Exception as e:
                logger.warning(
                    "[Docker] %s remote health check failed: %s", request_id, e
                )
                return

            active = health.get("active_containers")
            if active is None:
                logger.warning(
                    "[Docker] %s remote health missing active_containers", request_id
                )
                return

            if active < self.config.max_containers:
                if wait_count > 0:
                    logger.info(
                        "[Docker] %s remote slot acquired (active=%s, max=%s)",
                        request_id,
                        active,
                        self.config.max_containers,
                    )
                return

            wait_count += 1
            if wait_count == 1 or wait_count % 10 == 0:
                logger.info(
                    "[Docker] %s waiting for remote slot (active=%s, max=%s, waited=%.1fs)",
                    request_id,
                    active,
                    self.config.max_containers,
                    wait_count * poll_interval,
                )

            await asyncio.sleep(poll_interval)

    async def _run_blocking(self, func, *args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    async def run_agent(
        self,
        data: dict[str, Any],
        client: ArealOpenAI,
    ) -> float:
        """Run agent loop for a single terminal bench task.

        Args:
            data: Dict with keys: messages, task_name, task_path
            client: ArealOpenAI client for generation

        Returns:
            Reward score from test execution
        """
        messages = deepcopy(data["messages"])
        task_path = Path(data["task_path"])
        task_name = data.get("task_name", task_path.name)
        request_id = uuid.uuid4().hex[:8]

        trajectory: list[dict[str, Any]] = []
        trajectory.append({"type": "user_input", "content": messages[-1]["content"]})
        num_user_turns = 1  # Initial prompt counts as 1
        num_assistant_turns = 0
        last_response_id = None

        terminal = None
        terminal_started = False
        error_context = None

        try:
            async with self._container_slot(request_id):
                terminal = self._create_terminal(task_path, request_id)
                start_start = time.monotonic()
                try:
                    await self._run_blocking(terminal.start)
                    terminal_started = True
                except Exception as e:
                    error_context = f"Container start failed: {e}"
                    logger.warning(f"[Docker] {request_id} terminal.start() failed: {e}")
                    raise
                logger.info(
                    f"[Docker] {request_id} terminal.start() took {time.monotonic() - start_start:.2f}s"
                )

                try:
                    session_start = time.monotonic()
                    session = await self._run_blocking(
                        terminal.create_session,
                        "agent",
                        as_configured_user=True,
                    )
                    logger.info(
                        f"[Docker] {request_id} create_session took {time.monotonic() - session_start:.2f}s"
                    )

                    for turn in range(self.config.max_turns):
                        inference_start = time.monotonic()
                        logger.debug(
                            f"[Agent] {request_id} turn {turn}: calling inference..."
                        )
                        response: ChatCompletion = await client.chat.completions.create(
                            messages=messages,
                            tools=[BASH_TOOL],
                            tool_choice="auto",
                            **self.gconfig.to_openai_args_dict(),
                        )
                        logger.info(
                            f"[Agent] {request_id} turn {turn}: inference took "
                            f"{time.monotonic() - inference_start:.2f}s"
                        )

                        message = response.choices[0].message
                        last_response_id = response.id
                        messages.append(message)
                        num_assistant_turns += 1

                        trajectory.append(
                            {
                                "type": "assistant",
                                "content": message.content,
                                "tool_calls": (
                                    [
                                        {
                                            "id": tc.id,
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments,
                                        }
                                        for tc in message.tool_calls
                                    ]
                                    if message.tool_calls
                                    else None
                                ),
                            }
                        )

                        tool_calls = message.tool_calls or []
                        if not tool_calls:
                            break

                        for tool_call in tool_calls:
                            if tool_call.function.name != "bash":
                                continue
                            try:
                                args = json.loads(tool_call.function.arguments)
                            except json.JSONDecodeError:
                                args = {"command": "echo 'Invalid JSON arguments'"}

                            command = args.get("command", "")
                            cmd_start = time.monotonic()
                            output = await self._run_blocking(
                                self._execute_command, session, command
                            )
                            logger.debug(
                                f"[Agent] {request_id} command executed in "
                                f"{time.monotonic() - cmd_start:.2f}s: {command[:50]}..."
                            )
                            output = self._limit_output(output)

                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": output,
                                }
                            )
                            trajectory.append(
                                {
                                    "type": "observation",
                                    "command": command,
                                    "content": output,
                                }
                            )
                            num_user_turns += 1

                    reward, parser_results = await self._run_blocking(
                        self._run_tests_and_compute_reward,
                        terminal,
                        task_path,
                    )

                    logger.info(
                        f"[Docker] Task completed: {task_name} (id={request_id}, "
                        f"reward={reward:.2f}, turns={num_assistant_turns})"
                    )

                    if last_response_id:
                        client.set_reward(last_response_id, reward)

                    if self.config.save_trajectory:
                        await self._run_blocking(
                            self._save_trajectory,
                            trajectory=trajectory,
                            task_path=task_path,
                            request_id=request_id,
                            reward_score=reward,
                            parser_results=parser_results,
                            num_user_turns=num_user_turns,
                            num_assistant_turns=num_assistant_turns,
                        )

                    return reward
                finally:
                    stop_start = time.monotonic()
                    if terminal_started and terminal is not None:
                        try:
                            await self._run_blocking(terminal.stop)
                            logger.info(
                                f"[Docker] Container stopped: {request_id} in {time.monotonic() - stop_start:.2f}s"
                            )
                        except Exception as e:
                            logger.warning(f"Error stopping terminal {request_id}: {e}")
        except Exception as e:
            logger.warning(f"[Agent] {request_id} failed with error: {e}")
            if self.config.save_trajectory:
                error_message = error_context or str(e)
                await self._run_blocking(
                    self._save_trajectory,
                    trajectory=trajectory,
                    task_path=task_path,
                    request_id=request_id,
                    reward_score=0.0,
                    parser_results={"error": error_message},
                    num_user_turns=num_user_turns,
                    num_assistant_turns=num_assistant_turns,
                )
            raise

    def _should_use_remote(self) -> bool:
        """Determine whether to use remote mode.

        Returns:
            True if remote mode should be used, False for local Docker.
        """
        if self._use_remote is not None:
            return self._use_remote
        if self.config.use_remote is not None:
            self._use_remote = self.config.use_remote
            return self._use_remote

        # Auto-detect: check DEV_NODE_URL environment variable
        dev_node_url = os.environ.get("DEV_NODE_URL")
        if dev_node_url:
            logger.info(f"Auto-detected remote mode from DEV_NODE_URL: {dev_node_url}")
            self._use_remote = True
            return self._use_remote

        logger.info("Using local Docker mode (no DEV_NODE_URL set)")
        self._use_remote = False
        return self._use_remote

    def _create_terminal(self, task_path: Path, instance_id: str) -> TerminalType:
        """Create terminal for task (local or remote based on config).

        Args:
            task_path: Path to task directory.
            instance_id: Unique instance identifier.

        Returns:
            Terminal or RemoteTerminal instance.
        """
        task_name = task_path.name
        # Sanitize names for Docker (replace problematic characters)
        safe_task_name = task_name.replace(".", "-").replace("/", "-")
        safe_instance_id = instance_id[:8]

        container_name = f"tb__{safe_task_name}__client__{safe_instance_id}"
        # Limit container name to 63 characters
        if len(container_name) > 63:
            container_name = container_name[:63]

        # Auto-generate per-instance log directories to ensure
        # T_BENCH_TASK_LOGS_PATH / T_BENCH_TASK_AGENT_LOGS_PATH are always set
        logs_root = os.environ.get("T_BENCH_TASK_LOGS_ROOT")
        if logs_root is not None:
            base_logs_dir = Path(logs_root)
        else:
            base_logs_dir = task_path / ".tbench" / "logs"
        sessions_logs_path = base_logs_dir / instance_id

        agent_logs_root = os.environ.get("T_BENCH_TASK_AGENT_LOGS_ROOT")
        if agent_logs_root is not None:
            base_agent_logs_dir = Path(agent_logs_root)
        else:
            base_agent_logs_dir = task_path / ".tbench" / "agent-logs"
        agent_logs_path = base_agent_logs_dir / instance_id

        use_remote = self._should_use_remote()

        if use_remote:
            # Use remote dev node
            dev_node_url = os.environ.get("DEV_NODE_URL") or self.config.dev_node_url
            logger.info(
                f"[Docker] Creating remote container: {container_name} "
                f"(task={task_name}, remote={dev_node_url})"
            )
            terminal = RemoteTerminal(
                client_container_name=container_name,
                client_image_name=f"tb_{safe_task_name}_client",
                docker_compose_path=task_path / "docker-compose.yaml",
                no_rebuild=self.config.no_rebuild,
                cleanup=self.config.cleanup,
                dev_node_url=dev_node_url,
                sessions_logs_path=sessions_logs_path,
                agent_logs_path=agent_logs_path,
            )
        else:
            # Use local Docker
            logger.info(f"[Docker] Creating local container: {container_name} (task={task_name})")
            # Create log directories
            sessions_logs_path.mkdir(parents=True, exist_ok=True)
            agent_logs_path.mkdir(parents=True, exist_ok=True)
            terminal = Terminal(
                client_container_name=container_name,
                client_image_name=f"tb_{safe_task_name}_client",
                docker_compose_path=task_path / "docker-compose.yaml",
                no_rebuild=self.config.no_rebuild,
                cleanup=self.config.cleanup,
                sessions_logs_path=sessions_logs_path,
                agent_logs_path=agent_logs_path,
            )

        return terminal

    def _execute_command(self, session: TmuxSessionType, command: str) -> str:
        """Execute command and get output.

        Args:
            session: Tmux session to execute command in (local or remote).
            command: Command to execute.

        Returns:
            Command output string.
        """
        session.send_keys(
            [command, "Enter"],
            block=False,
            min_timeout_sec=self.config.command_timeout_sec,
        )
        return session.get_incremental_output()

    def _run_tests_and_compute_reward(
        self,
        terminal: TerminalType,
        task_path: Path,
    ) -> tuple[float, dict[str, UnitTestStatus] | None]:
        """Run tests and compute reward.

        Args:
            terminal: Terminal instance.
            task_path: Path to task directory.

        Returns:
            Tuple of (reward_score, parser_results).
        """
        try:
            # Copy test files to container
            test_dir = task_path / "tests"
            run_tests_sh = task_path / "run-tests.sh"

            paths_to_copy = []
            if run_tests_sh.exists():
                paths_to_copy.append(run_tests_sh)
            if test_dir.exists():
                paths_to_copy.append(test_dir)

            if paths_to_copy:
                terminal.copy_to_container(
                    paths=paths_to_copy, container_dir="/tests"
                )

            # Create test session and run tests
            test_session = terminal.create_session("tests", as_configured_user=False)
            test_session.send_keys(
                ["bash /tests/run-tests.sh", "Enter"],
                block=True,
                max_timeout_sec=self.config.test_timeout_sec,
            )

            # Parse results
            test_output = test_session.capture_pane(capture_entire=True)
            parser_results = parse_pytest_results(test_output)

            if parser_results:
                reward = compute_reward(parser_results)
                logger.info(f"Test results: {reward:.2%} passed")
                return reward, parser_results

            logger.warning("Failed to parse test results")
            return 0.0, None

        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return 0.0, None

    def _limit_output(self, output: str) -> str:
        """Limit output to max bytes.

        Args:
            output: Output string to limit.

        Returns:
            Limited output string.
        """
        max_bytes = self.config.max_output_bytes
        encoded = output.encode()
        if len(encoded) <= max_bytes:
            return output

        portion = max_bytes // 2
        first = encoded[:portion].decode(errors="ignore")
        last = encoded[-portion:].decode(errors="ignore")
        omitted = len(encoded) - 2 * portion
        return f"{first}\n[...{omitted} bytes omitted...]\n{last}"

    def _save_trajectory(
        self,
        trajectory: list[dict[str, Any]],
        task_path: Path,
        request_id: str,
        reward_score: float | None = None,
        parser_results: dict | None = None,
        num_user_turns: int = 0,
        num_assistant_turns: int = 0,
    ) -> None:
        """Save agent trajectory to JSON file for debugging.

        Args:
            trajectory: List of trajectory steps.
            task_path: Path to task directory.
            request_id: Unique request ID.
            reward_score: Final reward score.
            parser_results: Test results dict.
            num_user_turns: Number of user/observation turns.
            num_assistant_turns: Number of assistant turns.
        """
        try:
            # Use configured trajectory_dir or fallback to running_logs
            if self.config.trajectory_dir:
                running_logs_dir = Path(self.config.trajectory_dir)
            else:
                running_logs_dir = Path("running_logs")
            running_logs_dir.mkdir(parents=True, exist_ok=True)

            # Get task name from task_path
            task_name = task_path.name

            # Generate filename: task_name + random_seq
            filename = f"{task_name}_{request_id}.json"
            filepath = running_logs_dir / filename

            # Convert parser_results enum values to strings if present
            parser_results_serializable = None
            if parser_results:
                parser_results_serializable = {
                    k: v.value if hasattr(v, "value") else v
                    for k, v in parser_results.items()
                }

            # Prepare trajectory data
            trajectory_data = {
                "task_name": task_name,
                "task_path": str(task_path),
                "request_id": request_id,
                "reward_score": reward_score,
                "parser_results": parser_results_serializable,
                "num_user_turns": num_user_turns,
                "num_assistant_turns": num_assistant_turns,
                "trajectory": trajectory,
            }

            # Save to JSON file
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(trajectory_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved trajectory to {filepath}")

        except Exception as e:
            logger.error(f"Error saving trajectory: {e}")


class TerminalBenchWorkflow(RolloutWorkflow):
    """Workflow for terminal bench training."""

    def __init__(
        self,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        agent_config: dict | TerminalAgentConfig,
        export_style: str = "concat",
        turn_discount: float = 0.9,
        samples_per_task: int = 1,
    ):
        """Initialize Terminal Bench workflow.

        Args:
            gconfig: Generation hyperparameters.
            tokenizer: Tokenizer or path to tokenizer.
            agent_config: Agent configuration dict or TerminalAgentConfig.
            export_style: Export style for interactions ("concat" or "individual").
            turn_discount: Discount factor for multi-turn rewards.
            samples_per_task: Number of independent rollouts per task.
        """
        if isinstance(tokenizer, str):
            tokenizer = load_hf_tokenizer(tokenizer)

        self.tokenizer = tokenizer
        self.export_style = export_style
        self.turn_discount = turn_discount
        self.chat_template_type = "concat" if export_style == "concat" else "hf"
        if samples_per_task < 1:
            raise ValueError(
                f"samples_per_task must be >= 1, got {samples_per_task}"
            )
        self.samples_per_task = samples_per_task

        # Create agent config
        if isinstance(agent_config, dict):
            agent_config = TerminalAgentConfig(**agent_config)

        self.gconfig = gconfig.new(n_samples=1)
        self.agent = TerminalBenchAgent(
            gconfig=self.gconfig,
            config=agent_config,
        )

    async def _run_sample(self, engine, data):
        client = ArealOpenAI(
            engine=engine,
            tokenizer=self.tokenizer,
            chat_template_type=self.chat_template_type,
            tool_call_parser="qwen25",  # Use OpenAI function calling format
            reasoning_parser="qwen3",
        )

        reward = await self.agent.run_agent(data=data, client=client)
        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=reward)

        client.apply_reward_discount(turn_discount=self.turn_discount)
        return client.export_interactions(style=self.export_style)

    async def arun_episode(self, engine, data):
        """Run single episode of terminal bench task.

        Args:
            engine: Inference engine.
            data: Task data dict.

        Returns:
            Dict of interactions with rewards.
        """
        if self.samples_per_task == 1:
            return await self._run_sample(engine, data)

        results = await asyncio.gather(
            *[self._run_sample(engine, data) for _ in range(self.samples_per_task)]
        )
        merged: dict[str, Any] = {}
        for result in results:
            merged.update(result)
        return merged if merged else None


@dataclass
class TerminalBenchGRPOConfig(GRPOConfig):
    """Config for terminal bench training."""

    agent_config: dict = field(
        default_factory=lambda: {
            "dev_node_url": "http://localhost:8080",
            "tasks_base_path": "/path/to/tasks",
            "max_turns": 10,
            "command_timeout_sec": 30.0,
            "test_timeout_sec": 300.0,
            "max_output_bytes": 10000,
            "no_rebuild": False,
            "cleanup": True,
            "save_trajectory": True,
            "use_remote": None,  # None=auto-detect, True=remote, False=local
            "max_containers": None,  # None=unlimited, or set to limit Docker containers
        },
        metadata={"help": "Configuration for the terminal bench agent."},
    )
    export_style: str = field(
        default="concat",
        metadata={
            "help": "Export style for completions. 'concat' or 'individual'."
        },
    )
    turn_discount: float = field(
        default=0.9,
        metadata={"help": "Discount factor for multi-turn reward propagation."},
    )
    dataset_path: str = field(
        default="",
        metadata={"help": "Path to the terminal bench dataset (Parquet file)."},
    )
    valid_dataset_path: str = field(
        default="",
        metadata={"help": "Path to the validation dataset (Parquet file)."},
    )


def main(args):
    """Main entry point for terminal bench training."""
    config, _ = load_expr_config(args, TerminalBenchGRPOConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_samples_per_task = max(1, config.gconfig.n_samples)
    eval_samples_per_task = max(1, config.eval_gconfig.n_samples)
    if train_samples_per_task != 1 or eval_samples_per_task != 1:
        logger.info(
            "Using samples_per_task for terminal bench rollouts "
            "(train=%s, eval=%s); forcing gconfig.n_samples=1 for single-sample "
            "generation inside the workflow.",
            train_samples_per_task,
            eval_samples_per_task,
        )
    config.gconfig = config.gconfig.new(n_samples=1)
    config.eval_gconfig = config.eval_gconfig.new(temperature=0.6, n_samples=1)

    # Get tasks_base_path from agent_config
    tasks_base_path = config.agent_config.get("tasks_base_path", "")

    # Set trajectory directory based on experiment config
    trajectory_dir = (
        f"{config.stats_logger.fileroot}/{config.stats_logger.experiment_name}/"
        f"{config.stats_logger.trial_name}/trajectories"
    )
    config.agent_config["trajectory_dir"] = trajectory_dir
    logger.info(f"Trajectories will be saved to: {trajectory_dir}")

    # Load terminal bench dataset
    train_dataset = load_terminal_bench_dataset(
        parquet_path=config.dataset_path,
        tasks_base_path=tasks_base_path,
        split="train",
    )

    # Load validation dataset if provided
    valid_dataset = None
    if config.valid_dataset_path:
        valid_dataset = load_terminal_bench_dataset(
            parquet_path=config.valid_dataset_path,
            tasks_base_path=tasks_base_path,
            split="test",
        )

    workflow_kwargs = dict(
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        agent_config=config.agent_config,
        export_style=config.export_style,
        turn_discount=config.turn_discount,
        samples_per_task=train_samples_per_task,
    )

    # Eval workflow with lower temperature
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.eval_gconfig
    eval_workflow_kwargs["samples_per_task"] = eval_samples_per_task

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="examples.terminal_bench.terminal_bench_rl.TerminalBenchWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="examples.terminal_bench.terminal_bench_rl.TerminalBenchWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
