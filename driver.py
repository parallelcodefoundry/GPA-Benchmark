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
            "regex:sm__inst_executed_pipe_[^.]*.avg.pct_of_peak_sustained_active$,regex:sm__sass_thread_inst_executed_op.*sum$,regex:l1tex__t_set_.*_pipe_lsu_mem_global_op_ld.sum$,regex:l1tex__t_set_accesses.sum$,regex:l1tex__t_requests.sum$,regex:l1tex__m_xbar2l1tex_read_sectors.sum$,sm__average_thread_inst_executed_pred_on_per_inst_executed_realtime,regex:sm__sass_inst_executed.*sum$,regex:sm__inst_issued.avg.per_cycle_active$,regex:.*throughput.avg.pct_of_peak_sustained_active$,regex:.*throughput.avg.pct_of_peak_sustained_elapsed$]",
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


def build_app(app: dict, sm_version: int, no_clean: bool, env: dict) -> None:
    """Build the application."""
    build_path = app["path"] if "build_path" not in app else app["build_path"]
    if "rodinia" in build_path:
        build_path = "rodinia"
    if not no_clean and "clean_command" in app:
        subprocess.run(app["clean_command"].split(), cwd=build_path, check=True)
    build_command = "make -j 8" if "build_command" not in app else app["build_command"]
    build_command += f" SM_VERSION={sm_version}"
    build_command = build_command.split()
    print(" ".join(build_command))
    subprocess.run(build_command, cwd=build_path, env=env, check=True)


def run_app(app: dict, env: dict) -> None:
    """Run the application."""
    if "run_path" in app:
        run_path = app["run_path"]
    elif "build_path" in app:
        run_path = app["build_path"]
    else:
        run_path = app["path"]
    run_command = app["run_command"].split()
    print(" ".join(run_command))
    subprocess.run(run_command, cwd=run_path, env=env, check=True)


def nsys_profile_app(app: dict, env: dict) -> None:
    """Profile the application with Nsight Systems."""
    profile_dir = setup_profile_dir()
    nsys_command = ["nsys", "profile", "-o", os.path.join(profile_dir, app["name"]), "-f", "true"]
    nsys_command.extend(app["run_command"].split())
    if "run_path" in app:
        run_path = app["run_path"]
    elif "build_path" in app:
        run_path = app["build_path"]
    else:
        run_path = app["path"]
    print(" ".join(nsys_command))
    subprocess.run(nsys_command, cwd=run_path, env=env, check=True)


def ncu_profile_app(app: dict, env: dict) -> None:
    """Profile the application with Nsight Compute."""
    profile_dir = setup_profile_dir()
    ncu_command = ["ncu", "-o", os.path.join(profile_dir, app["name"]), "-f", "true"]
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
    print(" ".join(ncu_command))
    subprocess.run(ncu_command, cwd=run_path, env=env, check=True)


def setup_profile_dir() -> str:
    """Setup the profile directory."""
    profile_dir: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles")
    if not os.path.exists(profile_dir):
        os.makedirs(profile_dir)
    return profile_dir


def main() -> None:
    """Main function for driver."""
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
    args = parser.parse_args()
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
    for app in app_config["apps"]:
        if args.app != "all" and app["name"] != args.app:
            continue
        if swap_files and app["name"] in swap_files:
            swap_file_in_app(app, args.swaps)
        build_app(app, args.sm_version, args.no_clean, env)
        if args.build:
            continue
        run_app(app, env)
        if args.nsys:
            nsys_profile_app(app, env)
        if args.ncu:
            ncu_profile_app(app, env)
        if swap_files and app["name"] in swap_files:
            swap_file_out_app(app)


if __name__ == "__main__":
    main()
