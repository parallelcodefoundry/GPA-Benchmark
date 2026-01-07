#!/usr/bin/env python3
"""
GPA-Benchmark Driver
This script holds the main driver for compiling, running, validating, profiling, and testing
optimizations for GPA-Benchmark.
"""
import os
import shutil
import argparse
import subprocess
import yaml

NCU_ARGS = ["--metrics",
            "regex:sm__inst_executed_pipe_[^.]*.avg.pct_of_peak_sustained_active$,regex:sm__sass_thread_inst_executed_op.*sum$,regex:l1tex__t_set_.*_pipe_lsu_mem_global_op_ld.sum$,regex:l1tex__t_set_accesses.sum$,regex:l1tex__t_requests.sum$,regex:l1tex__m_xbar2l1tex_read_sectors.sum$,sm__average_thread_inst_executed_pred_on_per_inst_executed_realtime,regex:sm__sass_inst_executed.*sum$,regex:sm__inst_issued.avg.per_cycle_active$,regex:.*throughput.avg.pct_of_peak_sustained_active$,regex:.*throughput.avg.pct_of_peak_sustained_elapsed$",
            "--set", "full", "--import-source", "yes", "--target-processes", "all"]


def swap_file_in_app(app: dict, swap_file_path: str) -> None:
    """Swap the file in the application directory on disk. Back up original file. If the file
       contains ">>> START EDITABLE REGION" and "<<< END EDITABLE REGION", then replace
       only that region with the contents of the swap file.
    """
    dest_path = app["replace_file"]
    src_path = os.path.join(swap_file_path, app["name"] + ".swap")

    backup_path = dest_path + ".bak"
    shutil.copy(dest_path, backup_path)

    with open(dest_path, "r", encoding="utf-8") as dest_file:
        dest_text = dest_file.read()
    if ">>> START EDITABLE REGION" not in dest_text or "<<< END EDITABLE REGION" not in dest_text:
        # Replace entire file with swap file
        shutil.copy(src_path, dest_path)
        return
    with open(src_path, "r", encoding="utf-8") as src_file:
        src_text = src_file.read()
    start_index = dest_text.find(">>> START EDITABLE REGION")
    end_index = dest_text.find("<<< END EDITABLE REGION")
    dest_text = dest_text[:start_index] + src_text + dest_text[end_index:]
    with open(dest_path, "w", encoding="utf-8") as dest_file:
        dest_file.write(dest_text)


def swap_file_out_app(app: dict) -> None:
    """Swap the file out of the application directory on disk. Restore original file."""
    dest_path = app["replace_file"]
    backup_path = dest_path + ".bak"
    swap_save_path = dest_path + ".swap"
    shutil.copy(dest_path, swap_save_path)
    shutil.copy(backup_path, dest_path)


def subprocess_wrapper(command: list[str], cwd: str, env: dict) -> subprocess.CompletedProcess:
    """Wrapper for subprocess.run to capture stdout and stderr, print command before running."""
    print(f"Running command {' '.join(command)} in directory {cwd}")
    result = subprocess.run(command, cwd=cwd, env=env, check=False, capture_output=True)
    print(result.stdout.decode("utf-8"))
    print(result.stderr.decode("utf-8"))
    return result


def build_app(app: dict, sm_version: int, no_clean: bool, env: dict) -> bool:
    """Build the application. Returns True if successful, False otherwise."""
    build_path = app["path"] if "build_path" not in app else app["build_path"]
    if "rodinia" in build_path:
        build_path = "rodinia"
    if not no_clean and "clean_command" in app:
        clean_result = subprocess_wrapper(app["clean_command"].split(), build_path, env)
        if clean_result.returncode != 0:
            return False
    build_command = ["make", "-j", "8"]
    if "build_command" in app:
        build_command = app["build_command"].split()
    build_command.append(f"SM_VERSION={sm_version}")
    return subprocess_wrapper(build_command, build_path, env).returncode == 0


def run_app(app: dict, env: dict) -> bool:
    """Run the application. Returns True if successful, False otherwise."""
    if "run_path" in app:
        run_path = app["run_path"]
    elif "build_path" in app:
        run_path = app["build_path"]
    else:
        run_path = app["path"]
    return subprocess_wrapper(app["run_command"].split(), run_path, env).returncode == 0


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


def print_report_table(results: dict, operations: list) -> None:
    """Print a formatted table showing the status of all operations for each application."""
    if not results:
        return

    # Determine column widths
    app_name_width = max(len(app_name) for app_name in results.keys())
    app_name_width = max(app_name_width, len("Application"))
    col_width = max(len(op) for op in operations) if operations else 10
    col_width = max(col_width, 8)

    # Print header
    header = f"{'Application':<{app_name_width}}"
    for op in operations:
        header += f" | {op:<{col_width}}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    # Print rows
    for app_name, app_results in results.items():
        row = f"{app_name:<{app_name_width}}"
        for op in operations:
            status = app_results.get(op, None)
            if status is True:
                symbol = "✓"
            elif status is False:
                symbol = "✗"
            else:
                symbol = "-"
            row += f" | {symbol:<{col_width}}"
        print(row)

    print("=" * len(header))
    print()


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
                        help="Profile the application with Nsight Systems")
    parser.add_argument("--ncu", action="store_true",
                        help="Profile the application with Nsight Compute")
    parser.add_argument("--config", type=str, default="driver_apps.yaml",
                        help="The app config file to use")
    parser.add_argument("--swaps", type=str, default=None,
                        help="The path to the directory containing code files to swap in for the " \
                            + "kernel, with the filename being app_name.swap, app file path to " \
                            + "swap for is set in the config file")
    return parser.parse_args()


def setup_app_config(args: argparse.Namespace) -> tuple[dict, list[str] | None, dict]:
    """Setup the application configuration."""
    app_config: dict = yaml.load(open(args.config, "r", encoding="utf-8"),
                                 Loader=yaml.FullLoader)
    if args.app != "all" and args.app not in [app["name"] for app in app_config["apps"]]:
        raise ValueError(f"Application {args.app} not found in config file {args.config}")
    if args.swaps:
        swap_files = os.listdir(args.swaps)
    else:
        swap_files = None
    cuda_home = (args.cuda_home or
                 os.getenv("CUDA_HOME") or
                 os.getenv("CUDA_PATH") or
                 os.getenv("CUDA_ROOT") or
                 "/usr/local/cuda")
    env = os.environ.copy()
    env["CUDA_HOME"] = cuda_home
    return app_config, swap_files, env


def determine_operations(args: argparse.Namespace) -> list[str]:
    """Determine which operations will be performed."""
    operations: list[str] = []
    operations.append("Build")
    if not args.build:
        operations.append("Run")
    if args.nsys:
        operations.append("NSYS Profile")
    if args.ncu:
        operations.append("NCU Profile")
    return operations


def run_apps(app_config: dict, swap_files: list[str] | None, env: dict,
             args: argparse.Namespace) -> tuple[dict[str, dict[str, bool]], list[str]]:
    """Run the applications."""
    results: dict[str, dict[str, bool]] = {}
    operations = determine_operations(args)
    rodinia_built = False

    for app in app_config["apps"]:
        if args.app != "all" and app["name"] != args.app:
            continue

        app_name = app["name"]
        results[app_name] = {}

        if swap_files and app["name"] in swap_files:
            swap_file_in_app(app, args.swaps)

        # Build
        if "rodinia" not in app["path"] or not rodinia_built:
            build_success = build_app(app, args.sm_version, args.no_clean, env)
            results[app_name]["Build"] = build_success
            if "rodinia" in app["path"]:
                rodinia_built = build_success
        else:
            results[app_name]["Build"] = True

        if args.build:
            if swap_files and app["name"] in swap_files:
                swap_file_out_app(app)
            continue

        # Run
        run_success = run_app(app, env)
        results[app_name]["Run"] = run_success

        # NSYS Profile
        if args.nsys:
            nsys_success = nsys_profile_app(app, env)
            results[app_name]["NSYS Profile"] = nsys_success

        # NCU Profile
        if args.ncu:
            ncu_success = ncu_profile_app(app, env)
            results[app_name]["NCU Profile"] = ncu_success

        if swap_files and app["name"] in swap_files:
            swap_file_out_app(app)

    return results, operations


def main() -> None:
    """Main function for driver."""
    print("Start driver.py")

    args = parse_args()

    app_config, swap_files, env = setup_app_config(args)

    results, operations = run_apps(app_config, swap_files, env, args)

    print_report_table(results, operations)


if __name__ == "__main__":
    main()
