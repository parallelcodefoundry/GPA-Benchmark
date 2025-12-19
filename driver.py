"""
GPA-Benchmark Driver
This script holds the main driver for compiling, running, validating, profiling, and testing
optimizations for GPA-Benchmark.
"""
import argparse
import subprocess
import yaml

def build_app(app: dict, sm_version: int, no_clean: bool, cuda_home: str | None) -> None:
    """Build the application."""
    build_path = app["path"] if "build_path" not in app else app["build_path"]
    if "rodinia" in build_path:
        build_path = "rodinia"
    if not no_clean:
        subprocess.run(app["clean_command"].split() if "clean_command" in app
                       else ["make", "clean"],
                       cwd=build_path, check=True)
    build_command = app["build_command"] if "build_command" not in app else app["build_command"]
    build_command.append(f" SM_VERSION={sm_version}")
    env = {"CUDA_HOME": cuda_home} if cuda_home else {}
    subprocess.run(build_command.split(), cwd=build_path, env=env, check=True)


def run_app(app: dict, cuda_home: str | None) -> None:
    """Run the application."""
    if "run_path" in app:
        run_path = app["run_path"]
    elif "build_path" in app:
        run_path = app["build_path"]
    else:
        run_path = app["path"]
    run_command = app["run_command"] if "run_command" not in app else app["run_command"]
    env = {"CUDA_HOME": cuda_home} if cuda_home else {}
    subprocess.run(run_command.split(), cwd=run_path, env=env, check=True)


def main() -> None:
    """Main function for driver."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", type=str, default="all", help="The application to run")
    parser.add_argument("--sm_version", type=int, default=90, help="The SM version to use")
    parser.add_argument("--cuda-home", type=str, default=None,
                        help="Path to the CUDA installation to use")
    parser.add_argument("--no-clean", action="store_true",
                        help="Do not clean the application before building")
    parser.add_argument("--build", action="store_true", help="Only build the application")
    parser.add_argument("--nsys_profile", action="store_true",
                        help="Profile the application with Nsight Systems")
    parser.add_argument("--ncu_profile", action="store_true",
                        help="Profile the application with Nsight Compute")
    parser.add_argument("--config_file", type=str, default="driver_apps.yaml",
                        help="The app config file to use")
    args = parser.parse_args()
    app_config: list[dict] = yaml.load(open(args.config_file, "r", encoding="utf-8"),
                                       Loader=yaml.FullLoader)
    for app in app_config:
        if args.app != "all" and app["name"] != args.app:
            continue
        build_app(app, args.sm_version, args.no_clean, args.cuda_home)
        if args.build:
            continue
        run_app(app, args.cuda_home)
        if args.nsys_profile:
            nsys_profile_app(app)
        if args.ncu_profile:
            ncu_profile_app(app)


if __name__ == "__main__":
    main()
