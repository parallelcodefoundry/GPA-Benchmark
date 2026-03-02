"""
Core Operations for GPA-Benchmark Driver

This module provides functions for building and running applications.
"""
import os
import subprocess

from gpa_bench_driver.driver_src.driver_utils import SubprocessRunner, get_bin_path, \
    get_run_path, get_build_path, stdout_uses_file, get_stdout_redirect_path


def build_app(app: dict, sm_version: int, no_clean: bool,
              runner: SubprocessRunner,
              temp_dir: str) -> tuple[bool, subprocess.CompletedProcess]:
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
        clean_command = app["clean_command"].split() if "clean_command" in app \
            else ["make", "clean"]
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


def run_app(app: dict, runner: SubprocessRunner,
            temp_dir: str) -> tuple[bool, subprocess.CompletedProcess]:
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
