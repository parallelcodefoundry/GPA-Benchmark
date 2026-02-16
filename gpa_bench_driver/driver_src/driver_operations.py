"""
Core Operations for GPA-Benchmark Driver

This module provides functions for building and running applications.
"""
import os
import subprocess

from gpa_bench_driver.driver_src.driver_utils import subprocess_wrapper, get_bin_path, \
    get_run_path, get_build_path


def build_app(app: dict, sm_version: int, no_clean: bool, env: dict, temp_dir: str,
              log_level: str = "WARNING") -> tuple[bool, subprocess.CompletedProcess]:
    """Build the application.

    Cleans the application (unless no_clean is True), then builds it with the
    specified SM version.

    Args:
        app: Application configuration dictionary
        sm_version: SM version to build for (e.g., 90 for 9.0)
        no_clean: If True, skip the clean step
        env: Environment variables dictionary
        temp_dir: Temporary directory where working copy of application directory is located
        log_level: Logging level (default: "WARNING")

    Returns:
        Tuple of (success: bool, result: CompletedProcess)
    """
    build_path = get_build_path(app, temp_dir)

    # Clean step
    if not no_clean:
        clean_command = app["clean_command"].split() if "clean_command" in app \
            else ["make", "clean"]
        clean_result = subprocess_wrapper(clean_command, build_path, env, quiet=True,
                                          log_level=log_level)

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

    result = subprocess_wrapper(build_command, build_path, env, log_level=log_level)
    return result.returncode == 0, result


def run_app(app: dict, env: dict, temp_dir: str,
            log_level: str = "WARNING") -> tuple[bool, subprocess.CompletedProcess]:
    """Run the application.

    Removes the test output file if it exists, then runs the application.

    Args:
        app: Application configuration dictionary
        env: Environment variables dictionary
        temp_dir: Temporary directory where working copy of application directory is located
        log_level: Logging level (default: "WARNING")

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
    result = subprocess_wrapper(run_command, run_path, env, log_level=log_level)

    return result.returncode == 0, result
