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
import subprocess
import sys
import uuid
from dataclasses import dataclass, field

import requests
from enum import Enum
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
    from .reward import UnitTestStatus, compute_reward, parse_pytest_results
except ImportError:
    # When running directly with torchrun, use absolute imports
    from examples.terminal_bench.dataset import load_terminal_bench_dataset
    from examples.terminal_bench.remote_terminal import RemoteTerminal, RemoteTmuxSession, Terminal, TmuxSession
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


class AgentState(Enum):
    """State machine states for terminal agent."""

    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_COMMANDS = "processing_commands"
    TERMINATED = "terminated"


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

    # Class-level lock file path for cross-process synchronization
    _LOCK_FILE = "/tmp/areal_docker_container.lock"
    _PENDING_FILE = "/tmp/areal_docker_pending.count"

    def _get_docker_container_count(self) -> int:
        """Get current count of running Docker containers.

        In remote mode, queries the dev node's /health endpoint.
        In local mode, counts local Docker containers with tb__ prefix.

        Returns:
            Number of running containers.
        """
        if self._should_use_remote():
            # Remote mode: query dev node for active container count
            try:
                dev_node_url = os.environ.get("DEV_NODE_URL") or self.config.dev_node_url
                response = requests.get(f"{dev_node_url}/health", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("active_containers", 0)
                return 0
            except Exception:
                return 0
        else:
            # Local mode: count local Docker containers with tb__ prefix
            try:
                result = subprocess.run(
                    ["docker", "ps", "--filter", "name=tb__", "-q"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    # Count non-empty lines
                    lines = [l for l in result.stdout.strip().split("\n") if l]
                    return len(lines)
                return 0
            except Exception:
                return 0

    @staticmethod
    def _read_pending_count() -> int:
        """Read the pending container count from file."""
        try:
            with open(TerminalBenchAgent._PENDING_FILE, "r") as f:
                return int(f.read().strip() or "0")
        except (FileNotFoundError, ValueError):
            return 0

    @staticmethod
    def _write_pending_count(count: int) -> None:
        """Write the pending container count to file."""
        with open(TerminalBenchAgent._PENDING_FILE, "w") as f:
            f.write(str(max(0, count)))

    @staticmethod
    def reset_pending_counter() -> None:
        """Reset pending counter to 0. Call at training start to clear stale state."""
        try:
            with open(TerminalBenchAgent._PENDING_FILE, "w") as f:
                f.write("0")
            logger.info("[Docker] Reset pending container counter to 0")
        except Exception as e:
            logger.warning(f"[Docker] Failed to reset pending counter: {e}")

    def _acquire_container_slot(self, request_id: str) -> bool:
        """Try to acquire a container slot atomically.

        Uses file locking to prevent race conditions. Increments pending counter
        if a slot is available.

        Args:
            request_id: Request ID for logging.

        Returns:
            True if slot acquired, False otherwise.
        """
        import fcntl

        try:
            with open(self._LOCK_FILE, "w") as lock_file:
                # Acquire exclusive lock (blocking)
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    running = self._get_docker_container_count()
                    pending = self._read_pending_count()
                    total = running + pending

                    if total < self.config.max_containers:
                        # Increment pending count to claim the slot
                        self._write_pending_count(pending + 1)
                        logger.info(
                            f"[Docker] {request_id} acquired slot "
                            f"(running={running}, pending={pending+1}, total={total+1}, max={self.config.max_containers})"
                        )
                        return True
                    # Log rejection reason periodically
                    logger.debug(
                        f"[Docker] {request_id} slot denied "
                        f"(running={running}, pending={pending}, total={total}, max={self.config.max_containers})"
                    )
                    return False
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            logger.warning(f"[Docker] Lock error for {request_id}: {e}")
            return False

    def _release_pending_slot(self, request_id: str) -> None:
        """Decrement the pending counter after container started.

        Should be called after Docker container has started successfully.

        Args:
            request_id: Request ID for logging.
        """
        import fcntl

        try:
            with open(self._LOCK_FILE, "w") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    pending = self._read_pending_count()
                    running = self._get_docker_container_count()
                    self._write_pending_count(pending - 1)
                    logger.info(
                        f"[Docker] {request_id} released pending slot "
                        f"(running={running}, pending={max(0, pending-1)}, total={running + max(0, pending-1)})"
                    )
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except Exception as e:
            logger.warning(f"[Docker] Failed to release pending slot for {request_id}: {e}")

    async def _wait_for_container_slot(self, request_id: str) -> None:
        """Wait until Docker container count is below max_containers limit.

        Uses file locking and pending counter to prevent race conditions
        across multiple workers.

        Args:
            request_id: Request ID for logging.
        """
        if self.config.max_containers is None:
            return

        poll_interval = 0.5  # seconds
        logged_waiting = False
        wait_count = 0

        while True:
            if self._acquire_container_slot(request_id):
                if logged_waiting:
                    running = self._get_docker_container_count()
                    pending = self._read_pending_count()
                    logger.info(
                        f"[Docker] Slot finally acquired for {request_id} after {wait_count} polls "
                        f"(running={running}, pending={pending})"
                    )
                return

            wait_count += 1
            # Log waiting status initially and then every 20 polls (10 seconds)
            if not logged_waiting or wait_count % 20 == 0:
                running = self._get_docker_container_count()
                pending = self._read_pending_count()
                logger.info(
                    f"[Docker] Waiting for container slot {request_id} "
                    f"(running={running}, pending={pending}, total={running+pending}, "
                    f"max={self.config.max_containers}, waited={wait_count * poll_interval:.1f}s)"
                )
                logged_waiting = True

            await asyncio.sleep(poll_interval)

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
        messages = [dict(m) for m in data["messages"]]  # Deep copy
        task_path = Path(data["task_path"])
        task_name = data.get("task_name", task_path.name)
        request_id = uuid.uuid4().hex[:8]

        import time as _time
        run_start = _time.time()

        # Trajectory for logging
        trajectory: list[dict[str, Any]] = []
        trajectory.append({"type": "user_input", "content": messages[-1]["content"]})

        # Wait for container slot (cross-process limit via Docker count)
        wait_start = _time.time()
        await self._wait_for_container_slot(request_id)
        wait_time = _time.time() - wait_start
        if wait_time > 1.0:
            logger.info(f"[Docker] {request_id} waited {wait_time:.2f}s for slot")

        # Create and start terminal
        create_start = _time.time()
        terminal = self._create_terminal(task_path, request_id)
        logger.debug(f"[Docker] {request_id} terminal created in {_time.time() - create_start:.2f}s")

        start_start = _time.time()
        try:
            terminal.start()
        except Exception as e:
            # Release pending slot if terminal start fails
            if self.config.max_containers is not None:
                self._release_pending_slot(request_id)
            # Save trajectory even on container start failure
            logger.warning(f"[Docker] {request_id} terminal.start() failed: {e}")
            if self.config.save_trajectory:
                self._save_trajectory(
                    trajectory=trajectory,
                    task_path=task_path,
                    request_id=request_id,
                    reward_score=0.0,
                    parser_results={"error": f"Container start failed: {e}"},
                    num_user_turns=1,
                    num_assistant_turns=0,
                )
            raise
        logger.info(f"[Docker] {request_id} terminal.start() took {_time.time() - start_start:.2f}s")

        # Release pending slot now that container is running
        if self.config.max_containers is not None:
            self._release_pending_slot(request_id)

        try:
            session_start = _time.time()
            session = terminal.create_session("agent", as_configured_user=True)
            logger.info(f"[Docker] {request_id} create_session took {_time.time() - session_start:.2f}s")

            state = AgentState.PENDING
            turn = 0
            last_response_id = None
            num_user_turns = 1  # Initial prompt counts as 1
            num_assistant_turns = 0

            while state != AgentState.TERMINATED and turn < self.config.max_turns:
                if state == AgentState.PENDING:
                    state = AgentState.GENERATING

                elif state == AgentState.GENERATING:
                    # Generate model response with bash tool
                    inference_start = _time.time()
                    logger.debug(f"[Agent] {request_id} turn {turn}: calling inference...")
                    response: ChatCompletion = await client.chat.completions.create(
                        messages=messages,
                        tools=[BASH_TOOL],
                        tool_choice="auto",
                        **self.gconfig.to_openai_args_dict(),
                    )
                    logger.info(
                        f"[Agent] {request_id} turn {turn}: inference took "
                        f"{_time.time() - inference_start:.2f}s"
                    )

                    message = response.choices[0].message
                    last_response_id = response.id
                    messages.append(message)
                    turn += 1
                    num_assistant_turns += 1

                    # Log assistant response
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

                    # Check for tool calls
                    if message.tool_calls:
                        state = AgentState.PROCESSING_COMMANDS
                    else:
                        # No tool call means agent is done
                        state = AgentState.TERMINATED

                elif state == AgentState.PROCESSING_COMMANDS:
                    # Execute bash commands
                    for tool_call in messages[-1].tool_calls:
                        if tool_call.function.name == "bash":
                            try:
                                args = json.loads(tool_call.function.arguments)
                            except json.JSONDecodeError:
                                args = {"command": "echo 'Invalid JSON arguments'"}

                            command = args.get("command", "")
                            cmd_start = _time.time()
                            output = self._execute_command(session, command)
                            logger.debug(
                                f"[Agent] {request_id} command executed in "
                                f"{_time.time() - cmd_start:.2f}s: {command[:50]}..."
                            )
                            output = self._limit_output(output)

                            # Add tool response to messages
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": output,
                                }
                            )

                            # Log observation
                            trajectory.append(
                                {
                                    "type": "observation",
                                    "command": command,
                                    "content": output,
                                }
                            )
                            num_user_turns += 1

                    state = AgentState.GENERATING

            # Run tests and compute reward
            reward, parser_results = self._run_tests_and_compute_reward(
                terminal, task_path
            )
            logger.info(
                f"[Docker] Task completed: {task_name} (id={request_id}, "
                f"reward={reward:.2f}, turns={num_assistant_turns})"
            )

            # Set reward on last response
            if last_response_id:
                client.set_reward(last_response_id, reward)

            # Save trajectory for debugging
            if self.config.save_trajectory:
                self._save_trajectory(
                    trajectory=trajectory,
                    task_path=task_path,
                    request_id=request_id,
                    reward_score=reward,
                    parser_results=parser_results,
                    num_user_turns=num_user_turns,
                    num_assistant_turns=num_assistant_turns,
                )

            return reward

        except Exception as e:
            # Save trajectory even on failure for debugging
            logger.warning(f"[Agent] {request_id} failed with error: {e}")
            if self.config.save_trajectory:
                self._save_trajectory(
                    trajectory=trajectory,
                    task_path=task_path,
                    request_id=request_id,
                    reward_score=0.0,
                    parser_results={"error": str(e)},
                    num_user_turns=num_user_turns,
                    num_assistant_turns=num_assistant_turns,
                )
            raise

        finally:
            # Cleanup - container count will naturally decrease when container stops
            import time as _time
            stop_start = _time.time()
            try:
                terminal.stop()
                running = self._get_docker_container_count()
                pending = self._read_pending_count()
                logger.info(
                    f"[Docker] Container stopped: {request_id} in {_time.time() - stop_start:.2f}s "
                    f"(running={running}, pending={pending}, total={running+pending})"
                )
            except Exception as e:
                logger.warning(f"Error stopping terminal {request_id}: {e}")

    def _should_use_remote(self) -> bool:
        """Determine whether to use remote mode.

        Returns:
            True if remote mode should be used, False for local Docker.
        """
        if self.config.use_remote is not None:
            return self.config.use_remote

        # Auto-detect: check DEV_NODE_URL environment variable
        dev_node_url = os.environ.get("DEV_NODE_URL")
        if dev_node_url:
            logger.info(f"Auto-detected remote mode from DEV_NODE_URL: {dev_node_url}")
            return True

        logger.info("Using local Docker mode (no DEV_NODE_URL set)")
        return False

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
                    paths=paths_to_copy, container_dir="/tmp/tests"
                )

            # Create test session and run tests
            test_session = terminal.create_session("tests", as_configured_user=False)
            test_session.send_keys(
                ["bash /tmp/tests/run-tests.sh", "Enter"],
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
    ):
        """Initialize Terminal Bench workflow.

        Args:
            gconfig: Generation hyperparameters.
            tokenizer: Tokenizer or path to tokenizer.
            agent_config: Agent configuration dict or TerminalAgentConfig.
            export_style: Export style for interactions ("concat" or "individual").
            turn_discount: Discount factor for multi-turn rewards.
        """
        if isinstance(tokenizer, str):
            tokenizer = load_hf_tokenizer(tokenizer)

        self.tokenizer = tokenizer
        self.export_style = export_style
        self.turn_discount = turn_discount
        self.chat_template_type = "concat" if export_style == "concat" else "hf"

        # Create agent config
        if isinstance(agent_config, dict):
            agent_config = TerminalAgentConfig(**agent_config)

        self.agent = TerminalBenchAgent(
            gconfig=gconfig.new(n_samples=1),
            config=agent_config,
        )

        # Reset pending counter to clear stale state from previous runs
        TerminalBenchAgent.reset_pending_counter()

    async def arun_episode(self, engine, data):
        """Run single episode of terminal bench task.

        Args:
            engine: Inference engine.
            data: Task data dict.

        Returns:
            Dict of interactions with rewards.
        """
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
        completions_with_reward = client.export_interactions(style=self.export_style)
        return completions_with_reward


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
    )

    # Eval workflow with lower temperature
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6, n_samples=1)

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
