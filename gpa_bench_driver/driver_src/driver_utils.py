"""
Utility Functions for GPA-Benchmark Driver

This module provides utility functions for subprocess execution, path resolution,
directory setup, and system detection.
"""
import logging
import os
import subprocess

logger = logging.getLogger("GPA-Benchmark")


def subprocess_wrapper(command: list[str], cwd: str, env: dict, quiet: bool = False,
                       log_level: str = "WARNING",
                       timeout: int | None = None) -> subprocess.CompletedProcess:
    """Wrapper for subprocess.run to capture stdout and stderr, log command before running.

    Args:
        command: Command to run as a list of strings
        cwd: Working directory for the command
        env: Environment variables dictionary
        quiet: If True, suppress default output
        log_level: Logging level (default: "WARNING")
            - DEBUG: Always log stdout and stderr
            - INFO: Log stdout and stderr on failure (except when quiet=True)
            - WARNING/ERROR/CRITICAL: Never log stdout or stderr
        timeout: Timeout in seconds, if None, no timeout enforced (default: None)
    Returns:
        CompletedProcess object with returncode, stdout, and stderr attributes
    """
    logger.debug("Running command %s in directory %s", ' '.join(command), cwd)
    try:
        result = subprocess.run(command, cwd=cwd, env=env, check=False, capture_output=True,
                                timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.error("Command %s timed out after %d seconds", ' '.join(command), timeout)
        return subprocess.CompletedProcess(args=command, returncode=-1, stdout=None, stderr=None)

    if log_level == "DEBUG":
        # DEBUG: Always log stdout and stderr, even with quiet=True
        logger.debug("Command stdout:\n%s", result.stdout.decode('utf-8'))
        logger.debug("Command stderr:\n%s", result.stderr.decode('utf-8'))
    elif log_level == "INFO":
        # INFO: Log stdout and stderr on failure, unless quiet=True (then don't log anything)
        if not quiet and result.returncode != 0:
            if result.stdout:
                logger.info("Command stdout:\n%s", result.stdout.decode('utf-8'))
            if result.stderr:
                logger.info("Command stderr:\n%s", result.stderr.decode('utf-8'))
        # If quiet=True, don't log anything even with INFO
    # else: Default behavior (WARNING/ERROR/CRITICAL): never log stdout or stderr

    return result


def get_bin_path(app: dict, temp_dir: str) -> str:
    """Get the binary path for the application.

    Args:
        app: Application configuration dictionary
        temp_dir: Temporary directory where working copy of application directory is located

    Returns:
        Absolute path to the application binary
    """
    if "run_path" in app:
        return os.path.join(temp_dir, app["run_path"], app["run_command"].split()[0])
    return os.path.join(temp_dir, app["path"], app["run_command"].split()[0])


def get_build_path(app: dict, temp_dir: str) -> str:
    """Get the build path for the application.

    Args:
        app: Application configuration dictionary
        temp_dir: Temporary directory where working copy of application directory is located

    Returns:
        Path where the application should be built
    """
    if "build_path" in app:
        return os.path.join(temp_dir, app["build_path"])
    else:
        return os.path.join(temp_dir, app["path"])


def get_run_path(app: dict, temp_dir: str) -> str:
    """Get the run path for the application.

    Args:
        app: Application configuration dictionary
        temp_dir: Temporary directory where working copy of application directory is located

    Returns:
        Path where the application should be run from
    """
    if "run_path" in app:
        return os.path.join(temp_dir, app["run_path"])
    elif "build_path" in app:
        return os.path.join(temp_dir, app["build_path"])
    else:
        return os.path.join(temp_dir, app["path"])


def setup_profile_dir() -> str:
    """Setup the profile directory for storing profiling output files in the current working
       directory.

    Creates the directory if it doesn't exist.

    Returns:
        Path to the profile directory
    """
    profile_dir: str = os.path.join(os.getcwd(), "profiles")
    if not os.path.exists(profile_dir):
        os.makedirs(profile_dir)
    return profile_dir


def detect_sm_version() -> int:
    """Detect SM version from nvidia-smi.

    Returns the SM version as a two-digit integer (e.g., 9.0 -> 90).

    Returns:
        The SM version as a two-digit integer

    Raises:
        No exceptions raised. If detection fails, prints a warning and returns 90 as default.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        # Get the first line and remove whitespace
        compute_cap = result.stdout.strip().split('\n')[0].strip()
        # Remove decimal point (e.g., "9.0" -> "90")
        sm_version_str = compute_cap.replace('.', '')
        sm_version = int(sm_version_str)
        return sm_version
    except (subprocess.CalledProcessError, ValueError, IndexError, FileNotFoundError) as e:
        logger.warning("Could not detect SM version from nvidia-smi (%s). Defaulting to 90.", e)
        return 90
