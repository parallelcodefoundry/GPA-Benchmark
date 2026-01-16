#!/usr/bin/env python3
"""
GPA-Benchmark Driver
This script holds the main driver for compiling, running, validating, profiling, and testing
optimizations for GPA-Benchmark.
"""
# stl imports
import os
import re
import shutil
import argparse
import subprocess
import json
from enum import Enum
from dataclasses import dataclass
from typing import Optional

# tpl imports
import sqlite3
import yaml
import pandas as pd


NCU_ARGS = ["--metrics",
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
            "--set", "full", "--import-source", "yes", "--target-processes", "all"]

class Operation(Enum):
    """Represents a type of driver operation that can be performed on an application."""
    BUILD = "Build"
    RUN = "Run"
    VALIDATE = "Validate"
    NSYS_PROFILE = "NSYS Profile"
    NCU_PROFILE = "NCU Profile"
    NSYS_POST = "NSYS Post"
    SWAP_BUILDS = "Swap Builds"
    SWAP_RUNS = "Swap Runs"
    SWAP_VALID = "Swap Valid"
    SWAP_NSYS = "Swap NSYS"
    SWAP_NCU = "Swap NCU"
    SWAP_NSYS_POST = "Swap NSYS Post"

@dataclass
class SwapConfig:
    """Represents a swap configuration for replacing code in an application."""
    app_name: str
    swap_file_src_path: str
    swap_file_dest_name: str
    run_num: str
    optimized_code_num: str
    code: str

@dataclass
class DriverPassResult:
    """Represents the results of a single driver pass."""
    app_name: str | None = None
    run_num: str | None = None
    swap_num: str | None = None
    swap_file_src_path: str | None = None
    build: bool | None = None
    run: bool | None = None
    validate: bool | None = None
    nsys_profile: bool | None = None
    ncu_profile: bool | None = None
    nsys_post: bool | None = None
    nsys_data: dict[str, str | int | float | None] | None = None

    def to_dict(self) -> dict[str, str | bool | dict[str, str | int | float | None] | None]:
        """Convert the results to a dictionary."""
        return {
            "app_name": self.app_name,
            "run_num": self.run_num,
            "swap_num": self.swap_num,
            "swap_file_src_path": self.swap_file_src_path,
            "build": self.build,
            "run": self.run,
            "validate": self.validate,
            "nsys_profile": self.nsys_profile,
            "ncu_profile": self.ncu_profile,
            "nsys_post": self.nsys_post,
            "nsys_data": self.nsys_data,
        }



class AppResults:
    """Represents results for all passes of a given application."""
    def __init__(self) -> None:
        # Results for single non-swap pass
        self.build: Optional[bool] = None
        self.run: Optional[bool] = None
        self.validate: Optional[bool] = None
        self.nsys_profile: Optional[bool] = None
        self.ncu_profile: Optional[bool] = None
        self.nsys_post: Optional[bool] = None

        # Results for all swap passes
        self.swap_builds_numerator: int = 0
        self.swap_builds_denominator: int = 0
        self.swap_runs_numerator: int = 0
        self.swap_runs_denominator: int = 0
        self.swap_valid_numerator: int = 0
        self.swap_valid_denominator: int = 0
        self.swap_nsys_numerator: int = 0
        self.swap_nsys_denominator: int = 0
        self.swap_ncu_numerator: int = 0
        self.swap_ncu_denominator: int = 0
        self.swap_nsys_post_numerator: int = 0
        self.swap_nsys_post_denominator: int = 0


    def update_from_pass_result(self, pass_result: DriverPassResult, is_swap: bool) -> None:
        """Update results from a driver pass result."""
        if pass_result.build is not None:
            if not is_swap:
                self.build = pass_result.build
            else:
                # Update swap builds fraction
                self.swap_builds_denominator += 1
                if pass_result.build:
                    self.swap_builds_numerator += 1

        if pass_result.run is not None:
            if not is_swap:
                self.run = pass_result.run
            else:
                # Update swap runs fraction
                self.swap_runs_denominator += 1
                if pass_result.run:
                    self.swap_runs_numerator += 1

        if pass_result.validate is not None:
            if not is_swap:
                self.validate = pass_result.validate
            else:
                # Update swap valid fraction
                self.swap_valid_denominator += 1
                if pass_result.validate:
                    self.swap_valid_numerator += 1

        if pass_result.nsys_profile is not None:
            if not is_swap:
                self.nsys_profile = pass_result.nsys_profile
            else:
                # Update swap nsys profile fraction
                self.swap_nsys_denominator += 1
                if pass_result.nsys_profile:
                    self.swap_nsys_numerator += 1

        if pass_result.ncu_profile is not None:
            if not is_swap:
                self.ncu_profile = pass_result.ncu_profile
            else:
                # Update swap ncu profile fraction
                self.swap_ncu_denominator += 1
                if pass_result.ncu_profile:
                    self.swap_ncu_numerator += 1

        if pass_result.nsys_post is not None:
            if not is_swap:
                self.nsys_post = pass_result.nsys_post
            else:
                # Update swap nsys post fraction
                self.swap_nsys_post_denominator += 1
                if pass_result.nsys_post:
                    self.swap_nsys_post_numerator += 1


    def get(self, key: Operation) -> Optional[bool | str]:
        """Get a result value by key (for backward compatibility with dict interface)."""
        if key == Operation.BUILD:
            return self.build
        elif key == Operation.RUN:
            return self.run
        elif key == Operation.VALIDATE:
            return self.validate
        elif key == Operation.NSYS_PROFILE:
            return self.nsys_profile
        elif key == Operation.NCU_PROFILE:
            return self.ncu_profile
        elif key == Operation.NSYS_POST:
            return self.nsys_post
        elif key == Operation.SWAP_BUILDS:
            if self.swap_builds_denominator > 0:
                return f"{self.swap_builds_numerator}/{self.swap_builds_denominator}"
            return None
        elif key == Operation.SWAP_RUNS:
            if self.swap_runs_denominator > 0:
                return f"{self.swap_runs_numerator}/{self.swap_runs_denominator}"
            return None
        elif key == Operation.SWAP_VALID:
            if self.swap_valid_denominator > 0:
                return f"{self.swap_valid_numerator}/{self.swap_valid_denominator}"
            return None
        elif key == Operation.SWAP_NSYS:
            if self.swap_nsys_denominator > 0:
                return f"{self.swap_nsys_numerator}/{self.swap_nsys_denominator}"
            return None
        elif key == Operation.SWAP_NCU:
            if self.swap_ncu_denominator > 0:
                return f"{self.swap_ncu_numerator}/{self.swap_ncu_denominator}"
            return None
        elif key == Operation.SWAP_NSYS_POST:
            if self.swap_nsys_post_denominator > 0:
                return f"{self.swap_nsys_post_numerator}/{self.swap_nsys_post_denominator}"
            return None



def swap_file_in_app(app: dict, swap_config: SwapConfig) -> None:
    """Swap the file in the application directory on disk. Back up original file. If the file
       contains ">>> START EDITABLE REGION" and "<<< END EDITABLE REGION", then replace
       only that region with the contents of the swap file.
    """
    dest_path = app["replace_file"]
    backup_path = dest_path + ".bak"
    shutil.copy(dest_path, backup_path)
    with open(dest_path, "r", encoding="utf-8") as dest_file:
        dest_text = dest_file.read()
    if ">>> START EDITABLE REGION" not in dest_text or "<<< END EDITABLE REGION" not in dest_text:
        # Replace entire file with swap file code
        with open(dest_path, "w", encoding="utf-8") as dest_file:
            dest_file.write(swap_config.code)
        return
    start_index = dest_text.find(">>> START EDITABLE REGION")
    end_index = dest_text.find("<<< END EDITABLE REGION")
    dest_text = dest_text[:start_index] + swap_config.code + dest_text[end_index:]
    with open(dest_path, "w", encoding="utf-8") as dest_file:
        dest_file.write(dest_text)


def swap_file_out_app(app: dict) -> None:
    """Swap the file out of the application directory on disk. Restore original file."""
    dest_path = app["replace_file"]
    backup_path = dest_path + ".bak"
    shutil.copy(backup_path, dest_path)
    os.remove(backup_path)


def subprocess_wrapper(command: list[str], cwd: str, env: dict,
                       quiet: bool = False) -> subprocess.CompletedProcess:
    """Wrapper for subprocess.run to capture stdout and stderr, print command before running."""
    print(f"Running command {' '.join(command)} in directory {cwd}")
    result = subprocess.run(command, cwd=cwd, env=env, check=False, capture_output=True)
    if not quiet:
        print(result.stdout.decode("utf-8"))
    print(result.stderr.decode("utf-8"))
    return result


def build_app(app: dict, sm_version: int, no_clean: bool,
              env: dict) -> tuple[bool, subprocess.CompletedProcess]:
    """Build the application. Returns True if successful, False otherwise."""
    build_path = app["path"] if "build_path" not in app else app["build_path"]
    if not no_clean:
        clean_result = subprocess_wrapper(app["clean_command"].split() if "clean_command" in app \
            else ["make", "clean"], build_path, env, quiet=True)
        if clean_result.returncode != 0:
            # Directly remove executable if make clean fails
            exe_path = os.path.join(build_path, app["run_command"].split()[0])
            if os.path.exists(exe_path):
                os.remove(exe_path)
    build_command = ["make", "-j", "8"]
    if "build_command" in app:
        build_command = app["build_command"].split()
    build_command.append(f"SM_VERSION={sm_version}")
    result = subprocess_wrapper(build_command, build_path, env)
    return result.returncode == 0, result


def validate_app(app: dict, result: subprocess.CompletedProcess) -> bool:
    """Validate the application. Returns True if successful, False otherwise.
       Types of  :
       - fail_check_text: if the stdout contains the text in fail check text, return False
       - pass_check_text: if the stdout contains the text in pass check text, return True
       - reference_output: if the stdout (or file specified by test_output) matches the reference
                           output, return True
       - float_grep: locate the float in the stdout (or file specified by test_output) and compare
                     it to reference_output, return True if the difference is within float_tolerance
       - output_window: the number of lines to compare at the beginning (positive) or end (negative)
                        of the output, default is 0 (compare all lines)
    """
    if "fail_check_text" in app:
        if app["fail_check_text"] in result.stdout.decode("utf-8"):
            return False
        else:
            return True # Fail check text not found
    if "pass_check_text" in app:
        if app["pass_check_text"] in result.stdout.decode("utf-8"):
            return True
        else:
            return False # Pass check text not found
    if "reference_output" in app:
        with open(app["reference_output"], "r", encoding="utf-8") as ref_file:
            ref_output = ref_file.read()
        if "test_output" in app:
            if not os.path.exists(app["test_output"]):
                print(f"Warning: could not find test output file {app['test_output']} for " \
                    + f"{app['name']}, trying stdout instead")
                test_output = result.stdout.decode("utf-8")
            else:
                with open(app["test_output"], "r", encoding="utf-8") as test_file:
                    test_output = test_file.read()
        else:
            test_output = result.stdout.decode("utf-8")
        if test_output == ref_output:
            return True
        elif "output_window" in app and app["output_window"] != 0:
            return validate_output_window(test_output, app, ref_output)
        elif "float_grep" in app:
            return validate_float(test_output, app, ref_output)
        else:
            return False # Test output does not match and not a float grep validation
    else:
        raise ValueError(f"No validation type specified for {app['name']}")


def validate_output_window(test_output: str, app: dict, ref_output: str) -> bool:
    """Validate the output window. Returns True if successful, False otherwise."""
    window_sizes = app["output_window"]
    if len(window_sizes) != 2:
        raise ValueError(f"Output window must be a list of two integers ({app['name']})")
    test_output = "\n".join(test_output.splitlines()[window_sizes[0]:window_sizes[1]])
    ref_output = "\n".join(ref_output.splitlines()[window_sizes[0]:window_sizes[1]])
    return test_output == ref_output


def validate_float(test_output: str, app: dict, ref_output: str) -> bool:
    """Validate the float in the test output. Returns True if successful, False otherwise."""
    float_str = re.search(r"(\d+\.\d+)", test_output.split(app["float_grep"])[-1])
    ref_str = re.search(r"(\d+\.\d+)", ref_output.split(app["float_grep"])[-1])
    if not ref_str:
        raise ValueError(f"Reference output {app['reference_output']} does not contain " \
                            + f"float {app['float_grep']}")
    if float_str:
        float_value = float(float_str.group(1))
        ref_value = float(ref_str.group(1))
        if abs(float_value - ref_value) <= app["float_tolerance"]:
            return True
        else:
            return False
    else:
        print(f"Warning: No float found in test output {test_output} for {app['name']}, " \
                + f"looking for {app['float_grep']}")
        return False


def run_app(app: dict, env: dict) -> tuple[bool, subprocess.CompletedProcess]:
    """Run the application. Returns True if successful, False otherwise, and the result of the run.
       If the application has a test output file, remove it before running.
    """
    if "test_output" in app:
        if os.path.exists(app["test_output"]):
            os.remove(app["test_output"])
    if "run_path" in app:
        run_path = app["run_path"]
    elif "build_path" in app:
        run_path = app["build_path"]
    else:
        run_path = app["path"]
    run_command = app["run_command"].split()
    result = subprocess_wrapper(run_command, run_path, env)
    return result.returncode == 0, result


def nsys_profile_app(app: dict, env: dict) -> bool:
    """Profile the application with Nsight Systems. Returns True if successful, False otherwise."""
    profile_dir = setup_profile_dir()
    nsys_command = ["nsys", "profile", "-o", os.path.join(profile_dir, app["name"]), "-f", "true"]
    nsys_command.extend(app["run_command"].split())
    if "run_path" in app:
        run_path = app["run_path"]
    elif "build_path" in app:
        run_path = app["build_path"]
    else:
        run_path = app["path"]
    return subprocess_wrapper(nsys_command, run_path, env).returncode == 0 \
        and os.path.exists(os.path.join(profile_dir, app["name"] + ".nsys-rep"))


def postprocess_nsys_app(app: dict, env: dict) -> dict[str, str | int | float | None] | None:
    """Postprocess the Nsight Systems profile. Returns True if successful, False otherwise."""
    profile_dir = setup_profile_dir()

    # Convert nsys-rep file to sqlite file
    nsys_rep_file = os.path.join(profile_dir, app["name"] + ".nsys-rep")
    if not os.path.exists(nsys_rep_file):
        print(f"Warning: could not find Nsight Systems profile file {nsys_rep_file} for " \
            + f"{app['name']}")
        return None
    postprocess_command = ["nsys", "export","-f", "true", "-t", "sqlite", nsys_rep_file]
    if subprocess_wrapper(postprocess_command, profile_dir, env).returncode != 0:
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
    df["demangledName"] = df["demangledName"].map(string_ids.set_index("id")["value"])
    df["shortName"] = df["shortName"].map(string_ids.set_index("id")["value"])
    df["mangledName"] = df["mangledName"].map(string_ids.set_index("id")["value"])

    # Find kernel of interest using ncu_args
    kernel_name = app["kernel_name"]
    launch_skip = 0
    ncu_args = app["ncu_args"].split()
    for i,arg in enumerate(ncu_args):
        if arg == "-k" or arg == "--kernel-name":
            kernel_name = ncu_args[i+1]
        if arg == "--launch-skip":
            launch_skip = int(ncu_args[i+1])

    # Get the row of interest: launch_skip-th invocation of kernel_name
    try:
        kernel_row = df[df["shortName"] == kernel_name].iloc[[launch_skip]]
    except IndexError:
        print(f"Warning: could not find kernel {kernel_name} in Nsight Systems profile for " \
            + f"{app['name']} at launch skip {launch_skip}")
        return None

    # Export the row of interest to a CSV file with header row from column name of df
    #if swaps:
    #    csv_name = os.path.join(swaps, app["name"] + ".csv")
    #else:
    #    csv_name = os.path.join(profile_dir, app["name"] + ".csv")
    #kernel_row.to_csv(csv_name, index=False)

    return kernel_row.to_dict()


def ncu_profile_app(app: dict, env: dict) -> bool:
    """Profile the application with Nsight Compute. Returns True if successful, False otherwise."""
    profile_dir = setup_profile_dir()
    ncu_command = ["ncu", "-o", os.path.join(profile_dir, app["name"]), "-f"]
    if "ncu_args" in app:
        ncu_command.extend(app["ncu_args"].split())
    ncu_command.extend(NCU_ARGS)
    ncu_command.extend(app["run_command"].split())
    if "run_path" in app:
        run_path = app["run_path"]
    elif "build_path" in app:
        run_path = app["build_path"]
    else:
        run_path = app["path"]
    return subprocess_wrapper(ncu_command, run_path, env).returncode == 0 \
        and os.path.exists(os.path.join(profile_dir, app["name"] + ".ncu-rep"))


def setup_profile_dir() -> str:
    """Setup the profile directory."""
    profile_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")
    if not os.path.exists(profile_dir):
        os.makedirs(profile_dir)
    return profile_dir


def print_report_table(results: dict[str, AppResults], operations: list[Operation]) -> None:
    """Print a formatted table showing the status of all operations for each application."""
    if not results:
        return

    # Determine column widths
    app_name_width = max(len(app_name) for app_name in results.keys())
    app_name_width = max(app_name_width, len("Application"))
    col_width = max(len(op.value) for op in operations) if operations else 10
    col_width = max(col_width, 8)

    # Print header
    header = f"{'Application':<{app_name_width}}"
    for op in operations:
        header += f" | {op.value:<{col_width}}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    # Print rows
    for app_name, app_results in results.items():
        row = f"{app_name:<{app_name_width}}"
        for op in operations:
            status = app_results.get(op)
            if status is True:
                symbol = "✓"
            elif status is False:
                symbol = "✗"
            elif isinstance(status, str):
                symbol = status
            else:
                symbol = "-"
            row += f" | {symbol:<{col_width}}"
        print(row)

    print("=" * len(header))
    print()


def setup_app_config(args: argparse.Namespace) -> tuple[dict, dict[str, SwapConfig] | None, dict]:
    """Setup the application configuration."""
    if args.postprocess_nsys and (args.nsys or args.swaps or args.ncu or args.build):
        raise ValueError("Cannot postprocess Nsight Systems profiles only if other operations " \
            + "are specified.")
    if args.build and (args.nsys or args.ncu or args.swaps or args.postprocess_nsys):
        raise ValueError("Cannot build applications only if other operations are specified.")
    app_config: dict = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    if args.app != "all" and args.app not in [app["name"] for app in app_config["apps"]]:
        raise ValueError(f"Application {args.app} not found in config file {args.config}")
    cuda_home = (args.cuda_home or
                 os.getenv("CUDA_HOME") or
                 os.getenv("CUDA_PATH") or
                 os.getenv("CUDA_ROOT") or
                 "/usr/local/cuda")
    env = os.environ.copy()
    env["CUDA_HOME"] = cuda_home
    if args.swaps:
        swaps_dict = build_swaps_dict(args.swaps)
        return app_config, swaps_dict, env
    else:
        return app_config, None, env


def build_swaps_dict(swaps: str) -> dict[str, SwapConfig]:
    """Build the swaps dictionary."""
    swaps_dict: dict[str, SwapConfig] = {}
    for root, _, files in os.walk(swaps):
        for file in files:
            if match := re.match(r"run_(\d+)_optimized_code_(\d+)\.cu", file):
                run_num = match.group(1)
                optimized_code_num = match.group(2)
                # App name is everything before the last _ in the directory three levels up
                app_name = "_".join(root.split("/")[-3].split("_")[:-1])
                full_path = os.path.join(root, file)
                with open(full_path, "r", encoding="utf-8") as f:
                    code = f.read()
                # First line is automatically generated, ends with file name
                swap_file_name = code.splitlines()[0].split(" ")[-1]
                rest_of_code = "\n".join([line for line in code.splitlines()[1:]
                                          if not line.startswith("```")])
                swaps_dict[full_path] = SwapConfig(
                    app_name=app_name,
                    swap_file_src_path=full_path,
                    swap_file_dest_name=swap_file_name,
                    run_num=run_num,
                    optimized_code_num=optimized_code_num,
                    code=rest_of_code
                )
    return swaps_dict


def determine_operations(args: argparse.Namespace) -> list[Operation]:
    """Determine which operations will be performed."""
    operations: list[Operation] = []
    if not args.postprocess_nsys:
        operations.append(Operation.BUILD)
        if not args.build:
            operations.append(Operation.RUN)
            operations.append(Operation.VALIDATE)
    if args.nsys:
        operations.append(Operation.NSYS_PROFILE)
    if args.nsys or args.postprocess_nsys:
        operations.append(Operation.NSYS_POST)
    if args.ncu:
        operations.append(Operation.NCU_PROFILE)
    if args.swaps:
        operations.append(Operation.SWAP_BUILDS)
        if not args.build:
            operations.append(Operation.SWAP_RUNS)
            operations.append(Operation.SWAP_VALID)
            if args.nsys:
                operations.append(Operation.SWAP_NSYS)
            if args.ncu:
                operations.append(Operation.SWAP_NCU)
            if args.postprocess_nsys:
                operations.append(Operation.SWAP_NSYS_POST)

    return operations


def run_all(app_config: dict, swaps_dict: dict[str, SwapConfig] | None, env: dict,
            args: argparse.Namespace) -> tuple[dict[str, AppResults], list[Operation],
                                               dict[str, list[DriverPassResult]]]:
    """Run the applications."""
    results: dict[str, AppResults] = {}
    long_results: dict[str, list[DriverPassResult]] = {}
    operations = determine_operations(args)

    for app in app_config["apps"]:
        if args.app != "all" and app["name"] != args.app:
            continue

        app_name = app["name"]
        results[app_name] = AppResults()
        long_results[app_name] = []

        driver_passes: list[SwapConfig | None] = [None]
        if swaps_dict:
            driver_passes.extend([swap for swap in swaps_dict.values()
                            if swap.app_name == app["name"]])
        for driver_pass in driver_passes:
            pass_results = run_driver_pass(app, env, args, swap_config=driver_pass)
            results[app_name].update_from_pass_result(pass_results, driver_pass is not None)
            long_results[app_name].append(pass_results)
    return results, operations, long_results


def run_driver_pass(app: dict, env: dict, args: argparse.Namespace,
                    swap_config: SwapConfig | None = None) -> DriverPassResult:
    """Run a driver pass. Returns the results."""
    result = DriverPassResult()
    result.app_name = app["name"]
    result.run_num = swap_config.run_num if swap_config else None
    result.swap_num = swap_config.optimized_code_num if swap_config else None
    result.swap_file_src_path = swap_config.swap_file_src_path if swap_config else None
    if not args.postprocess_nsys:
        if swap_config:
            swap_file_in_app(app, swap_config)

        try:
            build_success, build_result = build_app(app, args.sm_version, args.no_clean, env)
            bin_path = os.path.join(app["path"], app["run_command"].split()[0])
            result.build = build_success and os.path.exists(bin_path) \
                and os.access(bin_path, os.X_OK)

            if args.build or result.build is False:
                return result

            # Run
            run_success, run_result = run_app(app, env)
            result.run = run_success

            # Validate
            validate_success = validate_app(app, run_result)
            print(f"Validate success: {validate_success}")
            result.validate = validate_success

            # NSYS Profile
            if args.nsys:
                nsys_success = nsys_profile_app(app, env)
                result.nsys_profile = nsys_success

            # NCU Profile
            if args.ncu:
                ncu_success = ncu_profile_app(app, env)
                result.ncu_profile = ncu_success

        finally:
            if swap_config:
                swap_file_out_app(app)

    if args.postprocess_nsys or args.nsys:
        postprocess_nsys_result = postprocess_nsys_app(app, env)
        result.nsys_post = postprocess_nsys_result is not None
        result.nsys_data = postprocess_nsys_result

    return result


def save_results(long_results: dict[str, list[DriverPassResult]], output_file: str) -> None:
    """Save the long results to a JSON file."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([result.to_dict() for results in long_results.values() for result in results],
                  f, indent=4)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", type=str, default="all", help="The application to run")
    parser.add_argument("--sm-version", type=int, default=90, help="The SM version to use")
    parser.add_argument("--cuda-home", type=str, default=None,
                        help="Path to the CUDA installation to use")
    parser.add_argument("--no-clean", action="store_true",
                        help="Do not clean the application before building")
    parser.add_argument("--build", action="store_true", help="Only build the application")
    parser.add_argument("--nsys", action="store_true",
                        help="Profile the application with Nsight Systems, then export and " \
                            + "process the sqlite database into a CSV file")
    parser.add_argument("--ncu", action="store_true",
                        help="Profile the application with Nsight Compute")
    parser.add_argument("--config", type=str, default="driver_apps.yaml",
                        help="The app config file to use")
    parser.add_argument("--swaps", type=str, default=None,
                        help="The path to the directory containing code files to swap in for the " \
                            + "kernel, with the filename being app_name.swap, app file path to " \
                            + "swap for is set in the config file")
    parser.add_argument("--postprocess-nsys", action="store_true",
                        help="Postprocess the nsys-rep file(s) only, do not run the application")
    parser.add_argument("--output-file", type=str, default="driver_results.json",
                        help="The file to save the long results to")
    return parser.parse_args()


def main() -> None:
    """Main function for driver."""
    print("Start driver.py")

    args = parse_args()

    app_config, swaps_dict, env = setup_app_config(args)

    results, operations, long_results = run_all(app_config, swaps_dict, env, args)

    print_report_table(results, operations)

    save_results(long_results, args.output_file)


if __name__ == "__main__":
    main()
