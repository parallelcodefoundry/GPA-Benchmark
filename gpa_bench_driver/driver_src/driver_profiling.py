"""Profiling Functions for GPA-Benchmark Driver.

This module provides functions for profiling applications with Nsight Systems
and Nsight Compute, and for postprocessing profiling data.
"""

import logging
import os
import sqlite3
from collections.abc import Callable, Hashable
from pathlib import Path
from typing import Any

import pandas as pd

from gpa_bench_driver.driver_src.driver_models import SwapConfig
from gpa_bench_driver.driver_src.driver_utils import (
    SubprocessRunner,
    get_run_path,
    setup_profile_dir,
)

logger = logging.getLogger("GPA-Benchmark")


# Default NCU arguments for profiling
NCU_ARGS = [
    "--metrics",
    "regex:sm__inst_executed_pipe_[^.]*.avg.pct_of_peak_sustained_active$,"
    "regex:sm__sass_thread_inst_executed_op.*sum$,"
    "regex:l1tex__t_set_.*_pipe_lsu_mem_global_op_ld.sum$,"
    "regex:l1tex__t_set_accesses.sum$,"
    "regex:l1tex__t_requests.sum$,"
    "regex:l1tex__m_xbar2l1tex_read_sectors.sum$,"
    "sm__average_thread_inst_executed_pred_on_per_inst_executed_realtime,"
    "regex:sm__sass_inst_executed.*sum$,"
    "regex:sm__inst_issued.avg.per_cycle_active$,"
    "regex:.*throughput.avg.pct_of_peak_sustained_active$,"
    "regex:.*throughput.avg.pct_of_peak_sustained_elapsed$",
    "--set",
    "full",
    "--import-source",
    "yes",
    "--target-processes",
    "all",
]


class ProfilingError(Exception):
    """Exception raised when profiling fails."""

    def __init__(self, message: str) -> None:
        """Initialize a ProfilingError."""
        self.message = message
        super().__init__(self.message)


def _update_pbar(
    pbar: Callable[[], None] | None,
    num_samples_finished: int,
    num_samples: int,
) -> None:
    """Update the progress bar.

    Args:
        pbar: Progress bar to update
        num_samples_finished: Number of samples finished
        num_samples: Number of samples to collect

    """
    if pbar is not None:
        for _ in range(num_samples - num_samples_finished):
            pbar()


def _get_profile_filename(app: dict, swap_config: SwapConfig | None, i: int) -> str:
    """Get the profile filename.

    Args:
        app: Application configuration dictionary
        swap_config: Swap configuration containing the code to swap in
        i: Index of the sample

    Returns:
        Profile filename (includes ``_pid_<pid>_`` for uniqueness across parallel runs).

    """
    profile_filename = app["name"]
    if swap_config:
        swap_filename = swap_config.file_swaps[0].swap_file_src_path.with_suffix("").name
        profile_filename += f"_{swap_filename}"
    profile_filename += f"_pid_{os.getpid()}_sample_{i}"
    return profile_filename


def nsys_profile_app(
    app: dict,
    runner: SubprocessRunner,
    temp_dir: Path,
    num_samples: int,
    swap_config: SwapConfig | None = None,
    pbar: Callable[[], None] | None = None,
) -> bool:
    """Profile the application with Nsight Systems.

    Args:
        app: Application configuration dictionary
        runner: Configured subprocess runner
        temp_dir: Temporary directory where working copy of application directory is located
        num_samples: Number of times to collect profiles
        swap_config: Swap configuration containing the code to swap in
        pbar: Progress bar to update

    Returns:
        True if profiling succeeded and output file exists, False otherwise

    """
    profile_dir = setup_profile_dir()

    for i in range(num_samples):
        profile_name = _get_profile_filename(app, swap_config, i)

        profile_path = Path(profile_dir / profile_name)

        nsys_command = ["nsys", "profile", "-o", str(profile_path), "-f", "true"]
        nsys_command.extend(app["run_command"].split())

        run_path = get_run_path(app, temp_dir)
        result = runner.run(nsys_command, run_path)

        if result.returncode != 0:
            msg = f"Nsight Systems profile failed with return code {result.returncode}"
            raise ProfilingError(msg)

        profile_file = Path(f"{profile_path!s}.nsys-rep")
        if not profile_file.exists():
            msg = f"Could not find Nsight Systems profile file {profile_file}"
            raise ProfilingError(msg)

        if pbar is not None:
            pbar()

    return True


def ncu_profile_app(
    app: dict,
    runner: SubprocessRunner,
    temp_dir: Path,
    num_samples: int,
    swap_config: SwapConfig | None = None,
    pbar: Callable[[], None] | None = None,
) -> bool:
    """Profile the application with Nsight Compute.

    Args:
        app: Application configuration dictionary
        runner: Configured subprocess runner
        temp_dir: Temporary directory where working copy of application directory is located
        num_samples: Number of times to collect profiles
        swap_config: Swap configuration containing the code to swap in
        pbar: Progress bar to update

    Returns:
        True if profiling succeeded and output file exists, False otherwise

    """
    profile_dir = setup_profile_dir()

    for i in range(num_samples):
        profile_name = _get_profile_filename(app, swap_config, i)
        profile_path = Path(profile_dir / profile_name)

        ncu_command = ["ncu", "-o", str(profile_path), "-f"]
        if "ncu_args" in app:
            ncu_command.extend(app["ncu_args"].split())
        ncu_command.extend(NCU_ARGS)
        ncu_command.extend(app["run_command"].split())

        run_path = get_run_path(app, temp_dir)
        result = runner.run(ncu_command, run_path)

        if result.returncode != 0:
            msg = f"Nsight Compute profile failed with return code {result.returncode}"
            raise ProfilingError(msg)

        profile_file = Path(f"{profile_path!s}.ncu-rep")
        if not profile_file.exists():
            msg = f"Could not find Nsight Compute profile file {profile_file}"
            raise ProfilingError(msg)

        if pbar is not None:
            pbar()

    return True


def _parse_ncu_args(app: dict) -> tuple[str, int]:
    """Parse kernel name and launch skip from ncu_args.

    Args:
        app: Application configuration dictionary

    Returns:
        Tuple of (kernel_name, launch_skip)

    """
    kernel_name = app.get("kernel_name", "")
    launch_skip = 0

    if "ncu_args" in app:
        ncu_args = app["ncu_args"].split()
        for i, arg in enumerate(ncu_args):
            if i + 1 < len(ncu_args):
                if arg in ("-k", "--kernel-name"):
                    kernel_name = ncu_args[i + 1]
                if arg == "--launch-skip":
                    launch_skip = int(ncu_args[i + 1])

    return kernel_name, launch_skip


def postprocess_nsys_app(
    app: dict,
    runner: SubprocessRunner,
    num_samples: int,
    swap_config: SwapConfig | None = None,
    pbar: Callable[[], None] | None = None,
    *,
    retain_nsys_profiles: bool = False,
) -> list[dict[Hashable, Any]] | None:
    """Postprocess the Nsight Systems profile.

    Converts the nsys-rep file to SQLite format, extracts kernel data, and
    returns data for the kernel of interest. By default, deletes the .nsys-rep
    and .sqlite profile files after postprocessing; set retain_nsys_profiles
    to True to keep them.

    Args:
        app: Application configuration dictionary
        runner: Configured subprocess runner
        num_samples: Number of times to collect profiles
        swap_config: Swap configuration containing the code to swap in
        pbar: Progress bar to update
        retain_nsys_profiles: If True, keep .nsys-rep and .sqlite files after
            postprocessing; if False (default), delete them after extraction.

    Returns:
        List of dictionaries of kernel data if successful, None otherwise

    """
    profile_dir = setup_profile_dir()
    kernel_rows = []

    for i in range(num_samples):
        nsys_name = _get_profile_filename(app, swap_config, i)
        nsys_rep_file = Path(profile_dir / f"{nsys_name!s}.nsys-rep")

        if not nsys_rep_file.exists():
            msg = f"Could not find Nsight Systems profile file {nsys_rep_file}"
            raise ProfilingError(msg)

        # Convert nsys-rep to sqlite
        postprocess_command = ["nsys", "export", "-f", "true", "-t", "sqlite", str(nsys_rep_file)]
        if runner.run(postprocess_command, profile_dir).returncode != 0:
            logger.warning("Could not postprocess Nsight Systems profile file %s", nsys_rep_file)
            _update_pbar(pbar, i, num_samples)
            return None

        sqlite_file = Path(profile_dir / f"{nsys_name!s}.sqlite")
        if not sqlite_file.exists():
            msg = f"Could not find sqlite file {sqlite_file}"
            raise ProfilingError(msg)

        # Read sqlite file into pandas dataframes
        conn = sqlite3.connect(sqlite_file)
        df = pd.read_sql_query("SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL", conn)
        string_ids = pd.read_sql_query("SELECT * FROM StringIds", conn)
        conn.close()

        # Stringify demangledName, shortName, and mangledName columns by looking up string_ids
        string_id_map = string_ids.set_index("id")["value"]
        df["demangledName"] = df["demangledName"].map(string_id_map)
        df["shortName"] = df["shortName"].map(string_id_map)
        df["mangledName"] = df["mangledName"].map(string_id_map)

        # Create exec_time column
        df["exec_time"] = df["end"] - df["start"]

        # Find kernel of interest
        kernel_name, launch_skip = _parse_ncu_args(app)

        # Get the row of interest: launch_skip-th invocation of kernel_name
        try:
            kernel_row = df[df["shortName"] == kernel_name].iloc[[launch_skip]]
        except IndexError:
            logger.warning(
                "Could not find kernel %s in Nsight Systems profile for %s at launch skip %s",
                kernel_name,
                app["name"],
                launch_skip,
            )
            _update_pbar(pbar, i, num_samples)
            return None

        # Return as dict (matching original behavior: to_dict() on DataFrame)
        # This returns a dict where keys are column names and values are Series
        # For a single row, each Series contains one value
        kernel_rows.append(kernel_row.to_dict("records")[0])

        # Delete profile files after extraction unless retaining them
        if not retain_nsys_profiles:
            try:
                nsys_rep_file.unlink()
            except OSError as e:
                logger.warning("Could not remove nsys-rep file %s: %s", nsys_rep_file, e)
            try:
                sqlite_file.unlink()
            except OSError as e:
                logger.warning("Could not remove sqlite file %s: %s", sqlite_file, e)

        if pbar is not None:
            pbar()

    return kernel_rows
