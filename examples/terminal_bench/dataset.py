"""Dataset loading utilities for Terminal Bench.

This module provides functions to load terminal bench datasets
from Parquet files in verl format.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Default system prompt for terminal bench tasks
DEFAULT_SYSTEM_PROMPT = """You are an expert technical assistant with access to bash tools. You can execute commands in a terminal environment to solve tasks.

When you need to execute a bash command, use the bash function with the command you want to run. The function will return the output of the command.

Focus on solving the task efficiently and correctly. After completing the task, the tests will be run automatically to verify your solution."""


class TerminalBenchDataset(Dataset):
    """Dataset for Terminal Bench tasks loaded from Parquet files."""

    def __init__(
        self,
        parquet_path: str,
        tasks_base_path: str,
        system_prompt: str | None = None,
        split: str = "train",
    ):
        """Initialize Terminal Bench dataset.

        Args:
            parquet_path: Path to Parquet file containing task data.
            tasks_base_path: Base directory containing task directories.
            system_prompt: Custom system prompt to use. If None, uses default.
            split: Dataset split (not used, for API compatibility).
        """
        self.parquet_path = parquet_path
        self.tasks_base_path = Path(tasks_base_path)
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self.split = split

        # Load data from Parquet
        self.data = self._load_parquet(parquet_path)
        logger.info(f"Loaded {len(self.data)} tasks from {parquet_path}")

    def _load_parquet(self, parquet_path: str) -> list[dict[str, Any]]:
        """Load and process Parquet file.

        Args:
            parquet_path: Path to Parquet file.

        Returns:
            List of processed data dicts.
        """
        df = pd.read_parquet(parquet_path)

        data = []
        for _, row in df.iterrows():
            task_name = row.get("task_name", "")
            task_path = self.tasks_base_path / task_name

            # Get prompt from the row
            # verl format: prompt is a list of message dicts (may be numpy array)
            prompt = row.get("prompt", [])

            # Convert numpy array to list if needed
            if isinstance(prompt, np.ndarray):
                prompt = prompt.tolist()

            if isinstance(prompt, str):
                # If prompt is a string (instruction), wrap it in messages
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ]
            elif isinstance(prompt, list):
                # If prompt is already a list of messages, use it
                messages = prompt
                # Ensure system prompt is present
                if not messages or messages[0].get("role") != "system":
                    messages = [
                        {"role": "system", "content": self.system_prompt}
                    ] + messages
            else:
                logger.warning(f"Unexpected prompt format for task {task_name}: {type(prompt)}")
                continue

            # Get extra_info
            extra_info = row.get("extra_info", {})
            if isinstance(extra_info, str):
                try:
                    extra_info = json.loads(extra_info)
                except json.JSONDecodeError:
                    extra_info = {}

            data.append(
                {
                    "messages": messages,
                    "task_name": task_name,
                    "task_path": str(task_path),
                    "extra_info": extra_info,
                    "answer": None,  # Terminal bench uses test results for reward
                }
            )

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.data[idx]


def load_terminal_bench_dataset(
    parquet_path: str,
    tasks_base_path: str,
    system_prompt: str | None = None,
    split: str = "train",
) -> TerminalBenchDataset:
    """Load terminal bench dataset from Parquet file.

    This is a convenience function that creates a TerminalBenchDataset.

    Args:
        parquet_path: Path to Parquet file containing task data.
        tasks_base_path: Base directory containing task directories.
        system_prompt: Custom system prompt to use. If None, uses default.
        split: Dataset split ("train" or "test").

    Returns:
        TerminalBenchDataset instance.
    """
    return TerminalBenchDataset(
        parquet_path=parquet_path,
        tasks_base_path=tasks_base_path,
        system_prompt=system_prompt,
        split=split,
    )
