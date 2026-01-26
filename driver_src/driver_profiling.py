#!/usr/bin/env python3
"""
Profiling Functions for GPA-Benchmark Driver

This module provides functions for profiling applications with Nsight Systems
and Nsight Compute, and for postprocessing profiling data.
"""
import os
from typing import Any
from collections.abc import Hashable
import sqlite3
import pandas as pd

from driver_src.driver_models import SwapConfig
from driver_src.driver_utils import subprocess_wrapper, setup_profile_dir, get_run_path


# Default NCU arguments for profiling
NCU_ARGS = [
    "--metrics",
    "regex:sm__inst_executed_pipe_[^.]*.avg.pct_of_peak_sustained_active$," \
        + "regex:sm__sass_thread_inst_executed_op.*sum$," \
        + "regex:l1tex__t_set_.*_pipe_lsu_mem_global_op_ld.sum$," \
        + "regex:l1tex__t_set_accesses.sum$," \
        + "regex:l1tex__t_requests.sum$," \
        + "regex:l1tex__m_xbar2l1tex_read_sectors.sum$," \
        + "sm__average_thread_inst_executed_pred_on_per_inst_executed_realtime," \
        + "regex:sm__sass_inst_executed.*sum$," \
        + "regex:sm__inst_issued.avg.per_cycle_active$," \
        + "regex:.*throughput.avg.pct_of_peak_sustained_active$," \
        + "regex:.*throughput.avg.pct_of_peak_sustained_elapsed$",
    "--set", "full", "--import-source", "yes", "--target-processes", "all"
]


def _update_pbar(pbar: Any | None, num_samples_finished: int, num_samples: int) -> None:
    """Update the progress bar.

    Args:
        pbar: Progress bar to update
        num_samples_finished: Number of samples finished
        num_samples: Number of samples to collect
    """
    if pbar is not None:
        for _ in range(num_samples - num_samples_finished):
            pbar()


def nsys_profile_app(app: dict, env: dict, temp_dir: str, num_samples: int, verbose: int = 0,
                     swap_config: SwapConfig | None = None, pbar: Any | None = None) -> bool:
    """Profile the application with Nsight Systems.

    Args:
        app: Application configuration dictionary
        env: Environment variables dictionary
        temp_dir: Temporary directory where working copy of application directory is located
        num_samples: Number of times to collect profiles
        verbose: Verbosity level (0=default, 1=-v, 2=-vv)
        swap_config: Swap configuration containing the code to swap in
        pbar: Progress bar to update

    Returns:
        True if profiling succeeded and output file exists, False otherwise
    """
    profile_dir = setup_profile_dir()
    num_samples_finished = 0

    for i in range(num_samples):
        profile_output = os.path.join(profile_dir, app["name"])
        if swap_config:
            swap_filename = swap_config.file_swaps[0].swap_file_src_path.split('/')[-1]\
                .replace('.cu', '')
            profile_output += f"_{swap_filename}"
        profile_output += f"_sample_{i}"

        nsys_command = ["nsys", "profile", "-o", profile_output, "-f", "true"]
        nsys_command.extend(app["run_command"].split())

        run_path = get_run_path(app, temp_dir)
        result = subprocess_wrapper(nsys_command, run_path, env, verbose=verbose)

        profile_file = profile_output + ".nsys-rep"
        if not (result.returncode == 0 and os.path.exists(profile_file)):
            print(f"Warning: could not find Nsight Systems profile file {profile_file}")
            _update_pbar(pbar, num_samples_finished, num_samples)
            return False

        num_samples_finished += 1
        if pbar is not None:
            pbar()

    return True


def ncu_profile_app(app: dict, env: dict, temp_dir: str, num_samples: int, verbose: int = 0,
                    swap_config: SwapConfig | None = None, pbar: Any | None = None) -> bool:
    """Profile the application with Nsight Compute.

    Args:
        app: Application configuration dictionary
        env: Environment variables dictionary
        temp_dir: Temporary directory where working copy of application directory is located
        num_samples: Number of times to collect profiles
        verbose: Verbosity level (0=default, 1=-v, 2=-vv)
        swap_config: Swap configuration containing the code to swap in
        pbar: Progress bar to update

    Returns:
        True if profiling succeeded and output file exists, False otherwise
    """
    profile_dir = setup_profile_dir()
    num_samples_finished = 0

    for i in range(num_samples):
        profile_output = os.path.join(profile_dir, app["name"])
        if swap_config:
            swap_filename = swap_config.file_swaps[0].swap_file_src_path.split('/')[-1]\
                .replace('.cu', '')
            profile_output += f"_{swap_filename}"
        profile_output += f"_sample_{i}"

        ncu_command = ["ncu", "-o", profile_output, "-f"]
        if "ncu_args" in app:
            ncu_command.extend(app["ncu_args"].split())
        ncu_command.extend(NCU_ARGS)
        ncu_command.extend(app["run_command"].split())

        run_path = get_run_path(app, temp_dir)
        result = subprocess_wrapper(ncu_command, run_path, env, verbose=verbose)

        profile_file = profile_output + ".ncu-rep"
        if not (result.returncode == 0 and os.path.exists(profile_file)):
            print(f"Warning: could not find Nsight Compute profile file {profile_file}")
            _update_pbar(pbar, num_samples_finished, num_samples)
            return False

        num_samples_finished += 1
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
            if arg == "-k" or arg == "--kernel-name":
                if i + 1 < len(ncu_args):
                    kernel_name = ncu_args[i + 1]
            if arg == "--launch-skip":
                if i + 1 < len(ncu_args):
                    launch_skip = int(ncu_args[i + 1])

    return kernel_name, launch_skip


def postprocess_nsys_app(app: dict, env: dict, num_samples: int, verbose: int = 0,
                         swap_config: SwapConfig | None = None,
                         pbar: Any | None = None) -> list[dict[Hashable, Any]] | None:
    """Postprocess the Nsight Systems profile.

    Converts the nsys-rep file to SQLite format, extracts kernel data, and
    returns data for the kernel of interest.

    Args:
        app: Application configuration dictionary
        env: Environment variables dictionary
        temp_dir: Temporary directory where working copy of application directory is located
        num_samples: Number of times to collect profiles
        verbose: Verbosity level (0=default, 1=-v, 2=-vv)
        swap_config: Swap configuration containing the code to swap in
        pbar: Progress bar to update

    Returns:
        List of dictionaries of kernel data if successful, None otherwise
    """
    profile_dir = setup_profile_dir()
    kernel_rows = []
    num_samples_finished = 0

    for i in range(num_samples):
        nsys_name = app["name"]
        if swap_config:
            swap_filename = swap_config.file_swaps[0].swap_file_src_path.split('/')[-1]\
                .replace('.cu', '')
            nsys_name += f"_{swap_filename}"
        nsys_name += f"_sample_{i}"
        nsys_rep_file = os.path.join(profile_dir, nsys_name + ".nsys-rep")

        if not os.path.exists(nsys_rep_file):
            print(f"Warning: could not find Nsight Systems profile file {nsys_rep_file}")
            _update_pbar(pbar, num_samples_finished, num_samples)
            return None

        # Convert nsys-rep to sqlite
        postprocess_command = ["nsys", "export", "-f", "true", "-t", "sqlite", nsys_rep_file]
        if subprocess_wrapper(postprocess_command, profile_dir, env,
                              verbose=verbose).returncode != 0:
            print(f"Warning: could not postprocess Nsight Systems profile file {nsys_rep_file}")
            _update_pbar(pbar, num_samples_finished, num_samples)
            return None

        sqlite_file = os.path.join(profile_dir, nsys_name + ".sqlite")
        if not os.path.exists(sqlite_file):
            print(f"Warning: could not find sqlite file {sqlite_file}")
            _update_pbar(pbar, num_samples_finished, num_samples)
            return None

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
            print(f"Warning: could not find kernel {kernel_name} in Nsight Systems profile for " \
                + f"{app['name']} at launch skip {launch_skip}")
            _update_pbar(pbar, num_samples_finished, num_samples)
            return None

        # Return as dict (matching original behavior: to_dict() on DataFrame)
        # This returns a dict where keys are column names and values are Series
        # For a single row, each Series contains one value
        kernel_rows.append(kernel_row.to_dict('records')[0])

        num_samples_finished += 1
        if pbar is not None:
            pbar()

    return kernel_rows
