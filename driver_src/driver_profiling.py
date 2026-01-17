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


def nsys_profile_app(app: dict, env: dict, verbose: int = 0) -> bool:
    """Profile the application with Nsight Systems.

    Args:
        app: Application configuration dictionary
        env: Environment variables dictionary
        verbose: Verbosity level (0=default, 1=-v, 2=-vv)

    Returns:
        True if profiling succeeded and output file exists, False otherwise
    """
    profile_dir = setup_profile_dir()
    profile_output = os.path.join(profile_dir, app["name"])

    nsys_command = ["nsys", "profile", "-o", profile_output, "-f", "true"]
    nsys_command.extend(app["run_command"].split())

    run_path = get_run_path(app)
    result = subprocess_wrapper(nsys_command, run_path, env, verbose=verbose)

    profile_file = profile_output + ".nsys-rep"
    return result.returncode == 0 and os.path.exists(profile_file)


def ncu_profile_app(app: dict, env: dict, verbose: int = 0) -> bool:
    """Profile the application with Nsight Compute.

    Args:
        app: Application configuration dictionary
        env: Environment variables dictionary
        verbose: Verbosity level (0=default, 1=-v, 2=-vv)

    Returns:
        True if profiling succeeded and output file exists, False otherwise
    """
    profile_dir = setup_profile_dir()
    profile_output = os.path.join(profile_dir, app["name"])

    ncu_command = ["ncu", "-o", profile_output, "-f"]
    if "ncu_args" in app:
        ncu_command.extend(app["ncu_args"].split())
    ncu_command.extend(NCU_ARGS)
    ncu_command.extend(app["run_command"].split())

    run_path = get_run_path(app)
    result = subprocess_wrapper(ncu_command, run_path, env, verbose=verbose)

    profile_file = profile_output + ".ncu-rep"
    return result.returncode == 0 and os.path.exists(profile_file)


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


def postprocess_nsys_app(app: dict, env: dict, verbose: int = 0) -> dict[Hashable, Any] | None:
    """Postprocess the Nsight Systems profile.

    Converts the nsys-rep file to SQLite format, extracts kernel data, and
    returns data for the kernel of interest.

    Args:
        app: Application configuration dictionary
        env: Environment variables dictionary
        verbose: Verbosity level (0=default, 1=-v, 2=-vv)

    Returns:
        Dictionary of kernel data if successful, None otherwise
    """
    profile_dir = setup_profile_dir()
    nsys_rep_file = os.path.join(profile_dir, app["name"] + ".nsys-rep")

    if not os.path.exists(nsys_rep_file):
        print(f"Warning: could not find Nsight Systems profile file {nsys_rep_file} for " \
            + f"{app['name']}")
        return None

    # Convert nsys-rep to sqlite
    postprocess_command = ["nsys", "export", "-f", "true", "-t", "sqlite", nsys_rep_file]
    if subprocess_wrapper(postprocess_command, profile_dir, env, verbose=verbose).returncode != 0:
        print(f"Warning: could not postprocess Nsight Systems profile file {nsys_rep_file} for " \
            + f"{app['name']}")
        return None

    sqlite_file = os.path.join(profile_dir, app["name"] + ".sqlite")
    if not os.path.exists(sqlite_file):
        print(f"Warning: could not find sqlite file {sqlite_file} for {app['name']}")
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

    # Find kernel of interest
    kernel_name, launch_skip = _parse_ncu_args(app)

    # Get the row of interest: launch_skip-th invocation of kernel_name
    try:
        kernel_row = df[df["shortName"] == kernel_name].iloc[[launch_skip]]
    except IndexError:
        print(f"Warning: could not find kernel {kernel_name} in Nsight Systems profile for " \
            + f"{app['name']} at launch skip {launch_skip}")
        return None

    # Return as dict (matching original behavior: to_dict() on DataFrame)
    # This returns a dict where keys are column names and values are Series
    # For a single row, each Series contains one value
    return kernel_row.to_dict('records')[0]
