"""Core Operations for GPA-Benchmark Driver.

This module provides functions for building and running applications.
"""

import os
import subprocess
from enum import Enum

from gpa_bench_driver.driver_src.driver_utils import (
    SubprocessRunner,
    get_bin_path,
    get_build_path,
    get_run_path,
)


def build_app(
    app: dict, sm_version: int, no_clean: bool, runner: SubprocessRunner, temp_dir: os.PathLike,
) -> tuple[bool, subprocess.CompletedProcess]:
    """Build the application.

    Cleans the application (unless no_clean is True), then builds it with the
    specified SM version.

    Args:
        app: Application configuration dictionary
        sm_version: SM version to build for (e.g., 90 for 9.0)
        no_clean: If True, skip the clean step
        runner: Configured subprocess runner
        temp_dir: Temporary directory where working copy of application directory is located

    Returns:
        Tuple of (success: bool, result: CompletedProcess)
    """
    build_path = get_build_path(app, temp_dir)

    # Clean step
    if not no_clean:
        clean_command = (
            app["clean_command"].split() if "clean_command" in app else ["make", "clean"]
        )
        clean_result = runner.run(clean_command, build_path, quiet=True)

        if clean_result.returncode != 0:
            # Directly remove executable if make clean fails
            bin_path = get_bin_path(app, temp_dir)
            if os.path.exists(bin_path):
                os.remove(bin_path)

    # Build step
    build_command = ["make", "-B", "-j", "8"]
    if "build_command" in app:
        build_command = app["build_command"].split()
    build_command.append(f"SM_VERSION={sm_version}")

    result = runner.run(build_command, build_path)
    return result.returncode == 0, result


def run_app(
    app: dict, runner: SubprocessRunner, temp_dir: os.PathLike,
) -> tuple[bool, subprocess.CompletedProcess]:
    """Run the application.

    Removes the test output file if it exists, then runs the application.

    Args:
        app: Application configuration dictionary
        runner: Configured subprocess runner
        temp_dir: Temporary directory where working copy of application directory is located

    Returns:
        Tuple of (success: bool, result: CompletedProcess)
    """
    # Remove test output file if it exists
    if "test_output" in app:
        test_output_path = os.path.join(temp_dir, app["test_output"])
        if os.path.exists(test_output_path):
            os.remove(test_output_path)

    run_path = get_run_path(app, temp_dir)
    run_command = app["run_command"].split()

    result = runner.run(run_command, run_path)

    return result.returncode == 0, result


class SanitizeTool(Enum):
    """Available sanitizer tools."""
    MEMCHECK = "memcheck"
    INITCHECK = "initcheck"
    RACECHECK = "racecheck"
    SYNCCHECK = "synccheck"

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return self.value


def sanitize_app(
    app: dict,
    runner: SubprocessRunner,
    temp_dir: os.PathLike,
    tool: SanitizeTool,
) -> tuple[bool, subprocess.CompletedProcess]:
    """Sanitize the application with the specified tool.

    Runs sanitizer check on the application with the specified tool.

    Args:
        app: Application configuration dictionary
        runner: Configured subprocess runner
        temp_dir: Temporary directory where working copy of application directory is located
        tool: compute-sanitizer tool to use (MEMCHECK, INITCHECK, RACECHECK, SYNCCHECK)

    Returns:
        Tuple of (success: bool, result: CompletedProcess)
    """
    run_path = get_run_path(app, temp_dir)
    run_command = app["run_command"].split()
    sani_command = ["compute-sanitizer", "--tool", str(tool), *run_command]
    sani_result = runner.run(sani_command, run_path)
    return sani_result.returncode == 0, sani_result
