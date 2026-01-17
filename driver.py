#!/usr/bin/env python3
"""
GPA-Benchmark Driver

This script is the main driver for compiling, running, validating, profiling, and testing
optimizations for GPA-Benchmark applications. It orchestrates the execution of multiple
operations across one or more applications, optionally with code swapping for testing
optimizations.

The driver supports:
- Building applications with specified SM versions
- Running applications and capturing output
- Validating output against reference outputs
- Profiling with Nsight Systems and Nsight Compute
- Testing multiple code variants via file swapping
"""
import argparse
import os
from contextlib import nullcontext

from alive_progress import alive_bar

from driver_src.driver_models import Operation, SwapConfig, DriverPassResult, AppResults
from driver_src.driver_utils import get_bin_path, detect_sm_version
from driver_src.driver_file_swapping import swap_file_in_app, swap_file_out_app
from driver_src.driver_validation import validate_app
from driver_src.driver_operations import build_app, run_app
from driver_src.driver_profiling import nsys_profile_app, ncu_profile_app, postprocess_nsys_app
from driver_src.driver_config import setup_app_config, determine_operations
from driver_src.driver_reporting import print_report_table, save_results


def run_driver_pass(app: dict, env: dict, args: argparse.Namespace,
                    swap_config: SwapConfig | None = None) -> DriverPassResult:
    """Run a single driver pass for an application.

    A driver pass consists of building, running, validating, and optionally
    profiling an application. If swap_config is provided, the application's code
    is swapped before building and restored after completion.

    Args:
        app: Application configuration dictionary
        env: Environment variables dictionary
        args: Parsed command line arguments
        swap_config: Optional swap configuration for testing optimized code

    Returns:
        DriverPassResult object containing all results from this pass

    Raises:
        ValueError: If baseline (non-swap) build, run, or validation fails
    """
    result = DriverPassResult()
    result.app_name = app["name"]
    result.run_num = swap_config.run_num if swap_config else None
    result.swap_num = swap_config.optimized_code_num if swap_config else None
    result.swap_file_src_path = swap_config.swap_file_src_path if swap_config else None
    verbose = getattr(args, 'verbose', 0)

    # Skip build/run/validate if only postprocessing
    if not args.postprocess_nsys:
        # Swap file in if this is a swap pass
        if swap_config:
            swap_file_in_app(app, swap_config)

        try:
            # Build
            build_success, build_result = build_app(
                app, args.sm_version, args.no_clean, env, verbose
            )
            result.build_stdout = build_result.stdout.decode("utf-8")
            result.build_stderr = build_result.stderr.decode("utf-8")

            bin_path = get_bin_path(app)
            result.build = (build_success and
                          os.path.exists(bin_path) and
                          os.access(bin_path, os.X_OK))

            # Early return if build-only mode or build failed
            if args.build or result.build is False:
                if swap_config is None:
                    raise ValueError(f"Build failed for baseline ({app['name']})")
                return result

            # Run
            run_success, run_result = run_app(app, env, verbose)
            result.run_stdout = run_result.stdout.decode("utf-8")
            result.run_stderr = run_result.stderr.decode("utf-8")
            result.run = run_success

            if result.run is False:
                if swap_config is None:
                    raise ValueError(f"Run failed for baseline ({app['name']})")
                return result

            # Validate
            validate_success = validate_app(app, run_result)
            print(f"Validate success: {validate_success}")
            result.validate = validate_success

            if result.validate is False:
                if swap_config is None:
                    raise ValueError(f"Validation failed for baseline ({app['name']})")
                return result

            # NSYS Profile
            if args.nsys:
                nsys_success = nsys_profile_app(app, env, verbose)
                result.nsys_profile = nsys_success

            # NCU Profile
            if args.ncu:
                ncu_success = ncu_profile_app(app, env, verbose)
                result.ncu_profile = ncu_success

        finally:
            # Always restore original file if we swapped
            if swap_config:
                swap_file_out_app(app)

    # Postprocess NSYS (either standalone or after profiling)
    if args.postprocess_nsys or (args.nsys and result.nsys_profile):
        postprocess_nsys_result = postprocess_nsys_app(app, env, verbose)
        result.nsys_post = postprocess_nsys_result is not None
        result.nsys_data = postprocess_nsys_result

    return result


def run_all(app_config: dict, swaps_dict: dict[str, SwapConfig] | None, env: dict,
            args: argparse.Namespace) -> tuple[dict[str, AppResults], list[Operation],
                                               dict[str, list[DriverPassResult]]]:
    """Run all applications with their configured operations.

    Processes each application in the configuration, running baseline passes
    and optionally swap passes for each. Results are aggregated and returned.

    Args:
        app_config: Application configuration dictionary
        swaps_dict: Optional dictionary of swap configurations
        env: Environment variables dictionary
        args: Parsed command line arguments

    Returns:
        Tuple of:
        - results: Dictionary mapping app names to AppResults
        - operations: List of operations that were performed
        - long_results: Dictionary mapping app names to lists of DriverPassResult objects
    """
    results: dict[str, AppResults] = {}
    long_results: dict[str, list[DriverPassResult]] = {}
    operations = determine_operations(args)

    # Calculate total number of passes for progress bar
    num_apps = len(app_config["apps"]) if args.app == "all" else 1
    if swaps_dict:
        num_apps = len(set([swap.app_name for swap in swaps_dict.values()]))
    num_runs = num_apps + (len(swaps_dict) if swaps_dict else 0)

    # Use progress bar unless disabled
    progress_context = alive_bar(num_runs) if not args.no_progress else None
    if progress_context is None:
        # Create a no-op context manager for when progress is disabled
        progress_context = nullcontext()

    with progress_context as pbar:
        for app in app_config["apps"]:
            # Filter by app name if specified
            if args.app != "all" and app["name"] != args.app:
                continue

            app_name = app["name"]
            results[app_name] = AppResults()
            long_results[app_name] = []

            # Build list of passes: baseline first, then swaps
            driver_passes: list[SwapConfig | None] = [None]  # None = baseline
            if swaps_dict:
                driver_passes.extend([
                    swap for swap in swaps_dict.values()
                    if swap.app_name == app["name"]
                ])

            # Run each pass
            for driver_pass in driver_passes:
                pass_results = run_driver_pass(app, env, args, swap_config=driver_pass)
                is_swap = driver_pass is not None
                results[app_name].update_from_pass_result(pass_results, is_swap)
                long_results[app_name].append(pass_results)

                if pbar is not None:
                    pbar()  # pylint: disable=not-callable

    return results, operations, long_results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="GPA-Benchmark Driver: Build, run, validate, and profile applications"
    )
    parser.add_argument(
        "--app", type=str, default="all",
        help="The application to run (default: all)"
    )
    parser.add_argument(
        "--sm-version", type=int, default=None,
        help="The SM version to use (default: auto-detect from nvidia-smi)"
    )
    parser.add_argument(
        "--cuda-home", type=str, default=None,
        help="Path to the CUDA installation to use"
    )
    parser.add_argument(
        "--no-clean", action="store_true",
        help="Do not clean the application before building"
    )
    parser.add_argument(
        "--build", action="store_true",
        help="Only build the application (skip run and validate)"
    )
    parser.add_argument(
        "--nsys", action="store_true",
        help="Profile the application with Nsight Systems, then export and "
             "process the sqlite database (profiles stored under profiles/)"
    )
    parser.add_argument(
        "--ncu", action="store_true",
        help="Profile the application with Nsight Compute (profiles stored under profiles/)"
    )
    parser.add_argument(
        "--config", type=str, default="driver_apps.yaml",
        help="The app config file to use (default: driver_apps.yaml)"
    )
    parser.add_argument(
        "--swaps", type=str, default=None,
        help="The path to the directory containing code files to swap in for the "
             "kernel, with the filename being run_<num>_optimized_code_<num>.cu"
    )
    parser.add_argument(
        "--postprocess-nsys", action="store_true",
        help="Only postprocess nsys-rep file(s) found under profiles/, do not run the application"
    )
    parser.add_argument(
        "--output-file", type=str, default="driver_results.json",
        help="The file to save the long results to (default: driver_results.json)"
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0,
        help="Increase verbosity: -v outputs stdout/stderr on failure, "
             "-vv always outputs stdout and stderr"
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Do not display a progress bar"
    )
    return parser.parse_args()


def main() -> None:
    """Main function for driver.

    Orchestrates the entire driver workflow: argument parsing, configuration
    setup, execution, and result reporting.

    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If required files don't exist
    """
    print("Start driver.py")

    args = parse_args()

    # Auto-detect SM version if not provided
    if args.sm_version is None:
        args.sm_version = detect_sm_version()

    # Setup configuration
    app_config, swaps_dict, env = setup_app_config(args)

    # Run all applications
    results, operations, long_results = run_all(app_config, swaps_dict, env, args)

    # Display results
    print_report_table(results, operations)

    # Save results
    save_results(long_results, args.output_file)


if __name__ == "__main__":
    main()
