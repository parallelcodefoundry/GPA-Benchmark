#!/usr/bin/env python3
"""
Utility Functions for GPA-Benchmark Driver

This module provides utility functions for subprocess execution, path resolution,
directory setup, and system detection.
"""
import os
import subprocess


def subprocess_wrapper(command: list[str], cwd: str, env: dict,
                       quiet: bool = False, verbose: int = 0) -> subprocess.CompletedProcess:
    """Wrapper for subprocess.run to capture stdout and stderr, print command before running.

    Args:
        command: Command to run as a list of strings
        cwd: Working directory for the command
        env: Environment variables dictionary
        quiet: If True, suppress default output
        verbose: Verbosity level (0=default, 1=-v, 2=-vv)
            - 0: Default behavior (never print stdout or stderr)
            - 1: Print stdout and stderr on failure (except when quiet=True)
            - 2: Always print stdout and stderr (even when quiet=True)

    Returns:
        CompletedProcess object with returncode, stdout, and stderr attributes
    """
    print(f"Running command {' '.join(command)} in directory {cwd}")
    result = subprocess.run(command, cwd=cwd, env=env, check=False, capture_output=True)

    if verbose >= 2:
        # -vv: Always print stdout and stderr, even with quiet=True
        print(result.stdout.decode("utf-8"))
        print(result.stderr.decode("utf-8"))
    elif verbose >= 1:
        # -v: Print stdout and stderr on failure, unless quiet=True (then don't print anything)
        if not quiet:
            print(result.stdout.decode("utf-8"))
            if result.returncode != 0:
                print(result.stderr.decode("utf-8"))
        # If quiet=True, don't print anything even with -v
    # else: Default behavior (verbose=0): never print stdout or stderr

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
        print(f"Warning: Could not detect SM version from nvidia-smi ({e}). Defaulting to 90.")
        return 90
