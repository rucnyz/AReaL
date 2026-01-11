"""Pytest result parser and reward computation for Terminal Bench.

This module provides functions to parse pytest output and compute
reward scores based on test pass/fail rates.
"""

import logging
import re
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class UnitTestStatus(Enum):
    """Status of a unit test."""

    PASSED = "passed"
    FAILED = "failed"


def parse_pytest_results(test_output: str) -> Optional[dict[str, UnitTestStatus]]:
    """Parse pytest test output and return parser_results dict.

    Args:
        test_output: The test output string from pytest.

    Returns:
        dict[str, UnitTestStatus] with test names as keys and status as values,
        or None if parsing fails.
    """
    try:
        # Look for pytest's "short test summary info" section
        # Pattern: ===== short test summary info =====
        pattern = r"=+\s*short test summary info\s*=+"
        parts = re.split(pattern, test_output, flags=re.IGNORECASE, maxsplit=1)

        if len(parts) < 2:
            # Try to parse without short summary (fallback to counting PASSED/FAILED)
            logger.debug("No short test summary found, trying fallback parsing")
            return _parse_test_results_fallback(test_output)

        short_summary = parts[1]

        # Parse test results from short summary
        # Format: PASSED test_name::test_function
        #         FAILED test_name::test_function
        parser_results = {}
        for line in short_summary.splitlines():
            line = line.strip()
            if not line:
                continue

            # Match patterns like "PASSED test_name::test_function" or "FAILED test_name::test_function"
            match = re.match(
                r"^(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)\s+(.+)", line
            )
            if match:
                status_str = match.group(1)
                test_path = match.group(2).strip()

                # Extract test name (test_name = test_path.split("::")[-1])
                test_name = test_path.split("::")[-1] if "::" in test_path else test_path

                if not test_name:
                    continue

                # Map pytest statuses to UnitTestStatus
                # PASSED, SKIPPED, XFAIL -> PASSED
                # FAILED, ERROR, XPASS -> FAILED
                if status_str in ["PASSED", "SKIPPED", "XFAIL"]:
                    parser_results[test_name] = UnitTestStatus.PASSED
                elif status_str in ["FAILED", "ERROR", "XPASS"]:
                    parser_results[test_name] = UnitTestStatus.FAILED

        if not parser_results:
            logger.warning("No test results found in output")
            return None

        return parser_results

    except Exception as e:
        logger.error(f"Error parsing test results: {e}", exc_info=True)
        return None


def _parse_test_results_fallback(
    test_output: str,
) -> Optional[dict[str, UnitTestStatus]]:
    """Fallback method to parse test results when short summary is not available.

    This method looks for patterns like "X passed", "Y failed" in the output.
    Since we can't get individual test names, we create a synthetic result.

    Args:
        test_output: The test output string.

    Returns:
        dict[str, UnitTestStatus] or None if parsing fails.
    """
    try:
        # Look for patterns like "X passed", "Y failed" in the output
        # Common pytest output: "5 passed, 2 failed in 1.23s"
        pattern_passed = r"(\d+)\s+passed"
        passed_match = re.search(pattern_passed, test_output, re.IGNORECASE)

        pattern_failed = r"(\d+)\s+failed"
        failed_match = re.search(pattern_failed, test_output, re.IGNORECASE)

        passed_count = int(passed_match.group(1)) if passed_match else 0
        failed_count = int(failed_match.group(1)) if failed_match else 0

        total_tests = passed_count + failed_count

        if total_tests == 0:
            logger.warning("No tests found in output")
            return None

        # Create synthetic parser_results (can't get individual test names from summary)
        # Use generic names like "test_1", "test_2", etc.
        parser_results = {}
        for i in range(passed_count):
            parser_results[f"test_{i + 1}"] = UnitTestStatus.PASSED
        for i in range(failed_count):
            parser_results[f"test_{passed_count + i + 1}"] = UnitTestStatus.FAILED

        logger.debug(
            f"Fallback parsing: {passed_count}/{total_tests} passed (using synthetic test names)"
        )

        return parser_results

    except Exception as e:
        logger.error(f"Error in fallback parsing: {e}", exc_info=True)
        return None


def compute_reward(parser_results: Optional[dict[str, UnitTestStatus]]) -> float:
    """Compute reward from test results.

    Args:
        parser_results: Dict mapping test names to status, or None if parsing failed.

    Returns:
        Float reward in [0, 1]: passed_tests / total_tests, or 0.0 if no results.
    """
    if not parser_results:
        return 0.0

    total = len(parser_results)
    passed = sum(1 for s in parser_results.values() if s == UnitTestStatus.PASSED)
    return passed / total if total > 0 else 0.0
