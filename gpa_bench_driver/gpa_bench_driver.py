#!/usr/bin/env python3
"""
GPA-Benchmark Driver

This script is the main driver for compiling, running, validating, profiling, and testing
optimizations for GPA-Benchmark applications. It orchestrates the execution of multiple
operations across one or more applications, optionally with code swapping for testing
optimizations.

The driver supports:
- Building applications with specified or auto-detected SM versions
- Running applications and capturing output
- Validating output against reference outputs
- Profiling with Nsight Systems and Nsight Compute
- Testing multiple code variants via file swapping

API Usage:
    The driver can be used programmatically via the run_driver() function:

    from driver import run_driver

    results, operations, long_results = run_driver(
        app="XSBench",
        nsys=True,
        num_samples=5
    )
"""
import argparse
import logging
import os
import tempfile
import shutil
from typing import Any
from alive_progress import alive_bar

from gpa_bench_driver.driver_src.driver_models import Operation, SwapConfig, DriverPassResult, \
    AppResults, DriverConfig
from gpa_bench_driver.driver_src.driver_utils import get_bin_path
from gpa_bench_driver.driver_src.driver_file_swapping import swap_file_in_app, swap_file_out_app
from gpa_bench_driver.driver_src.driver_validation import validate_app
from gpa_bench_driver.driver_src.driver_operations import build_app, run_app
from gpa_bench_driver.driver_src.driver_profiling import nsys_profile_app, ncu_profile_app, \
    postprocess_nsys_app
from gpa_bench_driver.driver_src.driver_config import setup_app_config, determine_operations
from gpa_bench_driver.driver_src.driver_reporting import print_report_table, save_results

logger = logging.getLogger("GPA-Benchmark")
#TODO: Rename logger to gpa_bench_driver
#TODO: Update project to src/ layout

APP_DIRS = ["Castro", "darknet", "ExaTENSOR", "LULESH", "PeleC", "Quicksilver", "rodinia",
            "XSBench"]


def count_operations_per_pass(config: DriverConfig) -> int:
    """Count the number of operations that will be performed in a single driver pass.

    Args:
        config: Driver configuration object

    Returns:
        Number of operations per pass
    """
    count = 0

    if not config.postprocess_nsys:
        count += 1 # BUILD always runs (unless only postprocessing)
        if not config.build_only:
            count += 2 # RUN and VALIDATE run if not build-only
            if config.nsys:
                count += config.num_samples
            if config.ncu:
                count += config.num_samples

    if config.postprocess_nsys or config.nsys:
        count += config.num_samples

    return count


def update_progress_for_skipped_operations(config: DriverConfig, pbar: Any,
                                           failure_stage: str) -> None:
    """Update progress bar for operations that will be skipped due to a failure.

    Args:
        config: Driver configuration object
        pbar: Progress bar to update
        failure_stage: Stage where failure occurred: 'build', 'run', 'validate', or 'nsys_profile'
    """
    if pbar is None:
        return

    skipped_ops = 0

    if config.build_only:
        return
    if failure_stage == 'build':
        skipped_ops += 1 # RUN
    if failure_stage in ['run', 'build']:
        skipped_ops += 1 # VALIDATE
    if failure_stage in ['build', 'run', 'validate']:
        if config.nsys:
            skipped_ops += config.num_samples # NSYS_PROFILE
        if config.ncu:
            skipped_ops += config.num_samples # NCU_PROFILE
    if failure_stage in ['build', 'run', 'validate', 'nsys_profile']:
        if config.postprocess_nsys or config.nsys:
            skipped_ops += config.num_samples # NSYS_POST

    for _ in range(skipped_ops):
        pbar()


def run_driver_pass(app: dict, env: dict, config: DriverConfig, temp_dir: str,
                    swap_config: SwapConfig | None = None,
                    pbar: Any = None) -> DriverPassResult:
    """Run a single driver pass for an application.

    A driver pass consists of building, running, validating, and optionally
    profiling an application. If swap_config is provided, the application's code
    is swapped before building and restored after completion.

    Args:
        app: Application configuration dictionary
        env: Environment variables dictionary
        config: Driver configuration object
        temp_dir: Temporary directory where working copy of application directory is located
        swap_config: Optional swap configuration for testing optimized code
        pbar: Optional progress bar to update after each operation

    Returns:
        DriverPassResult object containing all results from this pass

    Raises:
        ValueError: If baseline (non-swap) build, run, or validation fails
    """
    result = DriverPassResult()
    result.app_name = app["name"]
    result.run_num = swap_config.run_num if swap_config else None
    result.metadata = swap_config.metadata if swap_config else None
    result.swap_num = swap_config.optimized_code_num if swap_config else None
    if swap_config and swap_config.file_swaps:
        result.swap_file_src_path = ",".join([fs.swap_file_src_path
                                              for fs in swap_config.file_swaps])
    else:
        result.swap_file_src_path = None
    log_level = config.log_level

    # Skip build/run/validate if only postprocessing
    if not config.postprocess_nsys:
        # Swap file in if this is a swap pass
        if swap_config:
            swap_file_in_app(swap_config, temp_dir, config.detect_regions)

        try:
            # Build
            build_success, build_result = build_app(
                app, config.sm_version, config.no_clean, env, temp_dir, log_level
            )
            result.build_stdout = build_result.stdout.decode("utf-8")
            result.build_stderr = build_result.stderr.decode("utf-8")

            bin_path = get_bin_path(app, temp_dir)
            result.build = (build_success and
                          os.path.exists(bin_path) and
                          os.access(bin_path, os.X_OK))

            if pbar is not None:
                pbar()  # Update progress for BUILD operation

            # Early return if build-only mode or build failed
            if config.build_only or result.build is False:
                if swap_config is None:
                    raise ValueError(f"Build failed for baseline ({app['name']})")
                # Update progress for skipped operations due to build failure
                if not config.build_only:
                    update_progress_for_skipped_operations(config, pbar, 'build')
                return result

            # Run
            run_success, run_result = run_app(app, env, temp_dir, log_level)
            result.run_stdout = run_result.stdout.decode("utf-8")
            result.run_stderr = run_result.stderr.decode("utf-8")
            result.run = run_success

            if pbar is not None:
                pbar()  # Update progress for RUN operation

            if result.run is False:
                if swap_config is None:
                    raise ValueError(f"Run failed for baseline ({app['name']})")
                # Update progress for skipped operations due to run failure
                update_progress_for_skipped_operations(config, pbar, 'run')
                return result

            # Validate
            validate_success, validation_output = validate_app(app, run_result, temp_dir)
            logger.debug("Validate success: %s", validate_success)
            result.validate = validate_success
            if not validate_success and validation_output is not None:
                result.validation_output = validation_output

            if pbar is not None:
                pbar()  # Update progress for VALIDATE operation

            if result.validate is False:
                if swap_config is None:
                    raise ValueError(f"Validation failed for baseline ({app['name']})")
                # Update progress for skipped operations due to validation failure
                update_progress_for_skipped_operations(config, pbar, 'validate')
                return result

            # NSYS Profile
            if config.nsys:
                nsys_success = nsys_profile_app(app, env, temp_dir, config.num_samples, log_level,
                                                swap_config=swap_config or None, pbar=pbar)
                result.nsys_profile = nsys_success

            # NCU Profile
            if config.ncu:
                ncu_success = ncu_profile_app(app, env, temp_dir, config.num_samples, log_level,
                                              swap_config=swap_config or None, pbar=pbar)
                result.ncu_profile = ncu_success

        finally:
            # Always restore original files if we swapped
            if swap_config:
                swap_file_out_app(app, temp_dir, swap_config)

    # Postprocess NSYS (either standalone or after profiling)
    if config.postprocess_nsys or (config.nsys and result.nsys_profile):
        postprocess_nsys_result = postprocess_nsys_app(app, env, config.num_samples, log_level,
                                                       swap_config=swap_config or None, pbar=pbar)
        result.nsys_post = postprocess_nsys_result is not None
        result.nsys_data = postprocess_nsys_result
    elif config.nsys:
        # NSYS_POST was expected but didn't run because nsys_profile failed
        # Still update progress bar for this skipped operation
        update_progress_for_skipped_operations(config, pbar, 'nsys_profile')

    return result


def run_all(app_config: dict, swaps_dict: dict[str, SwapConfig] | None, env: dict,
            config: DriverConfig) -> tuple[dict[str, AppResults], list[Operation],
                                               dict[str, list[DriverPassResult]]]:
    """Run all applications with their configured operations.

    Processes each application in the configuration, running baseline passes
    and optionally swap passes for each. Results are aggregated and returned.

    Args:
        app_config: Application configuration dictionary
        swaps_dict: Optional dictionary of swap configurations
        env: Environment variables dictionary
        config: Driver configuration object

    Returns:
        Tuple of:
        - results: Dictionary mapping app names to AppResults
        - operations: List of operations that were performed
        - long_results: Dictionary mapping app names to lists of DriverPassResult objects
    """
    results: dict[str, AppResults] = {}
    long_results: dict[str, list[DriverPassResult]] = {}
    operations = determine_operations(config)

    # Calculate total number of operations for progress bar
    # Count operations per pass
    ops_per_pass = count_operations_per_pass(config)

    # Count number of passes (baseline + swaps) for each app
    num_passes = 0
    for app in app_config["apps"]:
        # Filter by app name if specified
        if config.app != "all" and app["name"] != config.app:
            continue
        # Baseline pass
        num_passes += 1
        # Swap passes for this app
        if swaps_dict:
            num_passes += sum(1 for swap in swaps_dict.values() if swap.app_name == app["name"])
    if num_passes == 0:
        raise ValueError(f"No passes generated, could not find {config.app} in app_config!")

    # Total operations = operations per pass * number of passes
    total_operations = ops_per_pass * num_passes

    with alive_bar(total_operations, disable=config.no_progress) as pbar:
        for app in app_config["apps"]:
            # Filter by app name if specified
            if config.app != "all" and app["name"] != config.app:
                continue

            with tempfile.TemporaryDirectory(dir=config.temp_dir) as temp_dir:
                app_dir = next(app_dir for app_dir in APP_DIRS if app_dir in app["path"])
                shutil.copytree(os.path.join(os.path.dirname(__file__), "..", app_dir),
                                os.path.join(temp_dir, app_dir))

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

                logger.debug("Driver is running %d passes for %s", len(driver_passes), app_name)

                # Run each pass
                for pass_num, driver_pass in enumerate(driver_passes):
                    pass_results = run_driver_pass(app, env, config, temp_dir,
                                                   swap_config=driver_pass, pbar=pbar)
                    is_swap = driver_pass is not None
                    results[app_name].update_from_pass_result(pass_results, is_swap)
                    long_results[app_name].append(pass_results)
                    logger.debug("Driver pass %d results:", pass_num)
                    logger.debug("  Build: %s", pass_results.build)
                    logger.debug("  Run: %s", pass_results.run)
                    logger.debug("  Validate: %s", pass_results.validate)
                    logger.debug("  NSYS Profile: %s", pass_results.nsys_profile)
                    logger.debug("  NCU Profile: %s", pass_results.ncu_profile)
                    logger.debug("  NSYS Post: %s", pass_results.nsys_post)
                    logger.debug("  NSYS Data: %s", pass_results.nsys_data)

    return results, operations, long_results


def run_driver(
    app: str = "all",
    sm_version: int | None = None,
    cuda_home: str | None = None,
    no_clean: bool = False,
    build_only: bool = False,
    nsys: bool = False,
    ncu: bool = False,
    config: str | None = None,
    swaps: str | None = None,
    detect_regions: bool = False,
    postprocess_nsys: bool = False,
    num_samples: int = 3,
    output_file: str | None= None,
    temp_dir: str | None = None,
    log_level: str = "WARNING",
    no_progress: bool = True,
    swaps_override: dict[str, str] | None = None,
    timeout: int | None = 300
) -> tuple[dict[str, AppResults], list[Operation], dict[str, list[DriverPassResult]]]:
    """Run the driver programmatically with the same interface as the CLI.

    This function provides a programmatic API that accepts the same parameters
    as the command-line interface. It can be called from other Python packages
    to run the driver functionality.

    Args:
        app: The application to run (default: "all")
        sm_version: The SM version to use (default: None, will auto-detect from nvidia-smi)
        cuda_home: Path to the CUDA installation to use (default: None)
        no_clean: Do not clean the application before building (default: False)
        build_only: Only build the application (skip run and validate) (default: False)
        nsys: Profile the application with Nsight Systems (default: False)
        ncu: Profile the application with Nsight Compute (default: False)
        config: The app config file to use (default: None, will use "driver_apps.yaml")
        swaps: Path to the directory containing code files to swap in (default: None)
        detect_regions: Detect editable region markers in swap files (default: False)
        postprocess_nsys: Only postprocess nsys-rep file(s) (default: False)
        num_samples: Number of times to collect ncu/nsys profiles (default: 3)
        output_file: File to save the long results to (default: None, no output file will be saved)
        temp_dir: Temporary directory to use (default: None, uses /tmp)
        log_level: Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: WARNING)
        no_progress: Do not display a progress bar (default: True)
        swaps_override: Override the swaps dictionary with a custom one for a single app, where keys
                        are filenames and values are code contents (default: None)
        timeout: The timeout in seconds for the driver to run, if negative, not timeout enforced
                 (default: 300)
    Returns:
        Tuple of:
        - results: Dictionary mapping app names to AppResults
        - operations: List of operations that were performed
        - long_results: Dictionary mapping app names to lists of DriverPassResult objects

    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If required files don't exist
    """
    # Create DriverConfig from parameters
    driver_config = DriverConfig(
        app=app,
        sm_version=sm_version,
        cuda_home=cuda_home,
        no_clean=no_clean,
        build_only=build_only,
        nsys=nsys,
        ncu=ncu,
        config=config or os.path.join(os.path.dirname(__file__), "..", "driver_apps.yaml"),
        swaps=swaps,
        detect_regions=detect_regions,
        postprocess_nsys=postprocess_nsys,
        num_samples=num_samples,
        output_file=output_file,
        temp_dir=temp_dir,
        log_level=log_level,
        no_progress=no_progress,
        swaps_override=swaps_override,
        timeout=timeout
    )

    return run_driver_config(driver_config)


def run_driver_config(config: DriverConfig) -> tuple[dict[str, AppResults], list[Operation],
                                                     dict[str, list[DriverPassResult]]]:
    """Run the driver with a DriverConfig object.

    Args:
        config: Driver configuration object

    Returns:
        Tuple of:
        - results: Dictionary mapping app names to AppResults
        - operations: List of operations that were performed
        - long_results: Dictionary mapping app names to lists of DriverPassResult objects
    """
    # Setup configuration
    app_config, swaps_dict, env = setup_app_config(config)

    # Run all applications
    results, operations, long_results = run_all(app_config, swaps_dict, env, config)

    # Display results
    print_report_table(results, operations)

    # Save results
    if config.output_file:
        save_results(long_results, config.output_file)
    else:
        logger.debug("Output writing disabled, results not saved")

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
        help="The SM version to use (default: None, will auto-detect from nvidia-smi)"
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
        "--build-only", action="store_true",
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
        "--detect-regions", action="store_true",
        help="Detect if the kernel file to swap into contains editable region markers and "
             "substitute into them rather than replacing the entire file"
    )
    parser.add_argument(
        "--postprocess-nsys", action="store_true",
        help="Only postprocess nsys-rep file(s) found under profiles/, do not run the application"
    )
    parser.add_argument(
        "-n", "--num-samples", type=int, default=5,
        help="The number of times to collect ncu/nsys profiles for each application and swap"
    )
    parser.add_argument(
        "-o", "--output-file", type=str, default="driver_results.json",
        help="The file to save the long results to (default: driver_results.json)"
    )
    parser.add_argument(
        "-t", "--temp-dir", type=str, default=None,
        help="The temporary directory to use for the driver, must exist (default: /tmp)"
    )
    parser.add_argument(
        "-l", "--log-level", type=str, default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: WARNING)"
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Do not display a progress bar"
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="The timeout in seconds for the driver to run, if negative, not timeout enforced"
             "(default: 300)"
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
    args = parse_args()

    # Configure logging
    log_level = getattr(logging, args.log_level.upper(), logging.WARNING)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] - %(message)s'
    )

    logger.info("Start gpa_bench_driver.py")

    # Convert argparse.Namespace to DriverConfig
    driver_config = DriverConfig.from_args(args)

    # Call run_driver_config with the driver_config
    run_driver_config(driver_config)


if __name__ == "__main__":
    main()
