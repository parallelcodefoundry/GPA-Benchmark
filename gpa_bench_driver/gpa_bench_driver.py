#!/usr/bin/env python3
"""GPA-Benchmark Driver.

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
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess

from alive_progress import alive_bar

from gpa_bench_driver.driver_src.driver_config import (
    AppNameNotFoundError,
    determine_operations,
    get_canonical_app_name,
    setup_app_config,
)
from gpa_bench_driver.driver_src.driver_file_swapping import swap_file_in_app, swap_file_out_app
from gpa_bench_driver.driver_src.driver_models import (
    AppResults,
    DriverConfig,
    DriverPassResult,
    Operation,
    SwapConfig,
)
from gpa_bench_driver.driver_src.driver_operations import (
    SanitizeTool,
    build_app,
    run_app,
    sanitize_app,
)
from gpa_bench_driver.driver_src.driver_profiling import (
    ncu_profile_app,
    nsys_profile_app,
    postprocess_nsys_app,
)
from gpa_bench_driver.driver_src.driver_reporting import print_report_table, save_results
from gpa_bench_driver.driver_src.driver_utils import (
    SubprocessRunner,
    SubprocessRunnerConfig,
    get_bin_path,
)
from gpa_bench_driver.driver_src.driver_validation import validate_app


@dataclass
class DriverPassContext:
    """Context for a single driver pass (one app, optional swap).

    Attributes:
        app: Application configuration dictionary
        env: Environment variables dictionary
        config: Driver configuration object
        temp_dir: Temporary directory for the working copy
        swap_config: Optional swap configuration for this pass
        pbar: Optional progress bar callback

    """

    app: dict
    env: dict
    config: DriverConfig
    temp_dir: Path
    swap_config: SwapConfig | None
    pbar: Callable[[], None] | None


logger = logging.getLogger("GPA-Benchmark")
# TODO(jhdavis): Rename logger to gpa_bench_driver
# TODO(jhdavis): Update project to src/ layout

APP_DIRS = [
    "Castro",
    "darknet",
    "ExaTENSOR",
    "LULESH",
    "PeleC",
    "Quicksilver",
    "rodinia",
    "XSBench",
]


class BaselineError(Exception):
    """Exception raised for errors in the baseline pass.

    Attributes:
        message: explanation of the error

    """

    def __init__(self, message: str) -> None:
        """Initialize the BaselineError.

        Args:
            message: explanation of the error

        """
        self.message = message
        super().__init__(self.message)


def count_operations_per_pass(config: DriverConfig) -> int:
    """Count the number of operations that will be performed in a single driver pass.

    Args:
        config: Driver configuration object

    Returns:
        Number of operations per pass

    """
    count = 0

    if not config.postprocess_nsys:
        count += 1  # BUILD always runs (unless only postprocessing)
        if not config.build_only:
            count += 2  # RUN and VALIDATE run if not build-only
            if not config.no_sanitize:
                count += 4  # SANITIZE with all 4 tools
            if config.nsys:
                count += config.num_samples
            if config.ncu:
                count += config.num_samples

    if config.postprocess_nsys or config.nsys:
        count += config.num_samples

    return count


def update_progress_for_skipped_operations(
    config: DriverConfig,
    pbar: Callable | None,
    failure_stage: str,
) -> None:
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
    if failure_stage == "build" and not config.no_sanitize:
        skipped_ops += 4  # SANITIZE with all 4 tools
    if failure_stage in ["build", "sanitize"]:
        skipped_ops += 1  # RUN
    if failure_stage in ["build", "sanitize", "run"]:
        skipped_ops += 1  # VALIDATE
    if failure_stage in ["build", "sanitize", "run", "validate"]:
        # NSYS_PROFILE and NCU_PROFILE
        skipped_ops += config.num_samples * (1 if config.nsys else 0 + 1 if config.ncu else 0)
    if failure_stage in [
        "build",
        "sanitize",
        "run",
        "validate",
        "nsys_profile",
        "ncu_profile",
    ] and (config.postprocess_nsys or config.nsys):
        skipped_ops += config.num_samples  # NSYS_POST

    for _ in range(skipped_ops):
        pbar()


def _init_pass_result(ctx: DriverPassContext) -> DriverPassResult:
    """Initialize a DriverPassResult from pass context."""
    result = DriverPassResult()
    result.app_name = ctx.app["name"]
    result.run_num = ctx.swap_config.run_num if ctx.swap_config else None
    result.metadata = ctx.swap_config.metadata if ctx.swap_config else None
    result.swap_num = ctx.swap_config.optimized_code_num if ctx.swap_config else None
    if ctx.swap_config and ctx.swap_config.file_swaps:
        result.swap_file_src_path = ",".join(
            [str(fs.swap_file_src_path) for fs in ctx.swap_config.file_swaps],
        )
    else:
        result.swap_file_src_path = None
    return result


def _run_build_phase(
    ctx: DriverPassContext,
    result: DriverPassResult,
    runner: SubprocessRunner,
) -> bool:
    """Run build phase; return True if build succeeded and binary exists."""
    build_success, build_result = build_app(
        ctx.app,
        ctx.config.sm_version,
        runner,
        ctx.temp_dir,
        no_clean=ctx.config.no_clean,
    )
    result.build_stdout = (
        build_result.stdout.decode("utf-8") if build_result.stdout is not None else ""
    )
    result.build_stderr = (
        build_result.stderr.decode("utf-8") if build_result.stderr is not None else ""
    )
    bin_path = Path(get_bin_path(ctx.app, ctx.temp_dir))
    result.build = (
        build_success and bin_path.exists() and bin_path.is_file() and os.access(bin_path, os.X_OK)
    )
    if ctx.pbar is not None:
        ctx.pbar()
    return result.build


def _run_sanitize_phase(
    ctx: DriverPassContext,
    result: DriverPassResult,
    runner: SubprocessRunner,
) -> bool:
    """Run sanitize phase; return True if all sanitizers passed."""
    tools = SanitizeTool.__members__.values()
    result.sanitize_stdouts = dict.fromkeys(tools, "")
    result.sanitize_stderrs = dict.fromkeys(tools, "")
    result.sanitize_details = dict.fromkeys(tools, False)
    for i, tool in enumerate(tools):
        if tool == SanitizeTool.RACECHECK and get_canonical_app_name(ctx.app["name"]) == "LULESH":
            logger.debug("Skipping racecheck for LULESH")
            if ctx.pbar is not None:
                ctx.pbar()
            continue
        sanitize_success, sanitize_result = sanitize_app(
            ctx.app,
            runner,
            ctx.temp_dir,
            tool,
        )
        result.sanitize_stdouts[tool] = (
            runner.decode_and_limit(sanitize_result.stdout)
            if sanitize_result.stdout is not None
            else ""
        )
        result.sanitize_stderrs[tool] = (
            runner.decode_and_limit(sanitize_result.stderr)
            if sanitize_result.stderr is not None
            else ""
        )
        result.sanitize_details[tool] = sanitize_success
        if ctx.pbar is not None:
            ctx.pbar()
        if not sanitize_success:
            stderr_raw = sanitize_result.stderr or b""
            timeout_marker = b"TIME LIMIT" if runner.use_srun else b"TIMEOUT"
            if timeout_marker in stderr_raw:
                timeout_label = "TIME LIMIT" if runner.use_srun else "TIMEOUT"
                logger.error(
                    "Sanitize failed due to timeout (%s); skipping remaining sanitizers.",
                    timeout_label,
                )
                # Advance progress bar by number of remaining sanitizers
                if ctx.pbar is not None and i < len(tools) - 1:
                    for _ in range(len(tools) - i - 1):
                        ctx.pbar()
                return False
        if not sanitize_success and ctx.swap_config is None:
            logger.error(
                "Baseline sanitize stdout: %s",
                result.sanitize_stdouts[tool],
            )
            logger.error(
                "Baseline sanitize stderr: %s",
                result.sanitize_stderrs[tool],
            )
            msg = f"Sanitize failed for baseline ({ctx.app['name']})"
            raise BaselineError(msg)
    result.sanitize = all(result.sanitize_details.values())
    return result.sanitize


def _run_run_phase(
    ctx: DriverPassContext,
    result: DriverPassResult,
    runner: SubprocessRunner,
) -> tuple[bool, CompletedProcess]:
    """Run app; return (success, run_result for validation)."""
    run_success, run_result = run_app(ctx.app, runner, ctx.temp_dir)
    result.run_stdout = (
        runner.decode_and_limit(run_result.stdout) if run_result.stdout is not None else ""
    )
    result.run_stderr = (
        runner.decode_and_limit(run_result.stderr) if run_result.stderr is not None else ""
    )
    result.run = run_success
    if ctx.pbar is not None:
        ctx.pbar()
    return run_success, run_result


def _run_validate_phase(
    ctx: DriverPassContext,
    result: DriverPassResult,
    run_result: CompletedProcess,
) -> bool:
    """Run validation; return True if validation passed."""
    validate_success, validation_output = validate_app(
        ctx.app,
        run_result,
        ctx.temp_dir,
    )
    logger.debug("Validate success: %s", validate_success)
    result.validate = validate_success
    if not validate_success and validation_output is not None:
        result.validation_output = validation_output
    if ctx.pbar is not None:
        ctx.pbar()
    return result.validate


def _handle_early_exit(
    ctx: DriverPassContext,
    result: DriverPassResult,
    stage: str,
    failed_tool: SanitizeTool | None = None,
) -> DriverPassResult:
    """Either raise BaselineError (baseline pass) or update progress and return result."""
    if ctx.swap_config is None:
        if stage == "build":
            logger.error("Baseline build stdout: %s", result.build_stdout)
            logger.error("Baseline build stderr: %s", result.build_stderr)
        elif stage == "sanitize":
            if failed_tool is None:
                raise ValueError("Sanitize phase failure reported but no failed tool provided")
            if result.sanitize_stdouts is None or result.sanitize_stderrs is None:
                raise ValueError(
                    "Sanitize phase failure reported but no sanitize stdouts or stderrs were set"
                )
            logger.error("Baseline sanitize stdout: %s", result.sanitize_stdouts[failed_tool])
            logger.error("Baseline sanitize stderr: %s", result.sanitize_stderrs[failed_tool])
        elif stage == "run":
            logger.error("Baseline run output: %s", result.run_stdout)
            logger.error("Baseline run stderr: %s", result.run_stderr)
        elif stage == "validate":
            logger.error("Baseline validation output: %s", result.validation_output)
        msg = f"{stage.capitalize()} failed for baseline ({ctx.app['name']})"
        raise BaselineError(msg)
    update_progress_for_skipped_operations(ctx.config, ctx.pbar, stage)
    return result


def _run_profiling_phase(
    ctx: DriverPassContext,
    result: DriverPassResult,
    runner: SubprocessRunner,
) -> None:
    """Run NSYS and NCU profiling if enabled."""
    if ctx.config.nsys:
        result.nsys_profile = nsys_profile_app(
            ctx.app,
            runner,
            ctx.temp_dir,
            ctx.config.num_samples,
            swap_config=ctx.swap_config,
            pbar=ctx.pbar,
        )
    if ctx.config.ncu:
        result.ncu_profile = ncu_profile_app(
            ctx.app,
            runner,
            ctx.temp_dir,
            ctx.config.num_samples,
            swap_config=ctx.swap_config,
            pbar=ctx.pbar,
        )


def _run_nsys_post_phase(
    ctx: DriverPassContext,
    result: DriverPassResult,
    runner: SubprocessRunner,
) -> None:
    """Postprocess NSYS (standalone or after profiling)."""
    if ctx.config.postprocess_nsys or (ctx.config.nsys and result.nsys_profile):
        postprocess_nsys_result = postprocess_nsys_app(
            ctx.app,
            runner,
            ctx.config.num_samples,
            ctx.swap_config,
            ctx.pbar,
            retain_nsys_profiles=ctx.config.retain_nsys_profiles,
        )
        result.nsys_post = postprocess_nsys_result is not None
        result.nsys_data = postprocess_nsys_result
    elif ctx.config.nsys:
        update_progress_for_skipped_operations(
            ctx.config,
            ctx.pbar,
            "nsys_profile",
        )


def run_driver_pass(ctx: DriverPassContext) -> DriverPassResult:
    """Run a single driver pass for an application.

    A driver pass consists of building, running, validating, and optionally
    profiling an application. If ctx.swap_config is provided, the application's code
    is swapped before building and restored after completion.

    Args:
        ctx: Driver pass context (app, env, config, temp_dir, swap_config, pbar)

    Returns:
        DriverPassResult object containing all results from this pass

    Raises:
        BaselineError: If baseline (non-swap) build, run, or validation fails

    """
    result = _init_pass_result(ctx)
    runner_config = SubprocessRunnerConfig(
        log_level=ctx.config.log_level,
        timeout=ctx.config.timeout,
        output_char_limit=ctx.config.subprocess_output_char_limit,
        suppress_command_stdout=ctx.config.suppress_command_stdout,
        use_srun=ctx.config.srun,
    )
    runner = SubprocessRunner(env=ctx.env, config=runner_config)

    if not ctx.config.postprocess_nsys:
        if ctx.swap_config:
            swap_file_in_app(
                swap_config=ctx.swap_config,
                temp_dir=ctx.temp_dir,
                detect_regions=ctx.config.detect_regions,
            )
        try:
            if not _run_build_phase(ctx, result, runner):
                return _handle_early_exit(ctx, result, "build")
            if ctx.config.build_only:
                return result
            if not ctx.config.no_sanitize and not _run_sanitize_phase(ctx, result, runner):
                if not result.sanitize_details:
                    raise ValueError("Ran sanitize phase but no sanitize details were set")
                return _handle_early_exit(
                    ctx,
                    result,
                    "sanitize",
                    failed_tool=next(
                        (
                            t
                            for t in SanitizeTool.__members__.values()
                            if not result.sanitize_details[t]
                        ),
                        None,
                    ),
                )
            run_ok, run_result = _run_run_phase(ctx, result, runner)
            if not run_ok:
                return _handle_early_exit(ctx, result, "run")
            if not _run_validate_phase(ctx, result, run_result):
                return _handle_early_exit(ctx, result, "validate")
            _run_profiling_phase(ctx, result, runner)
        finally:
            if ctx.swap_config:
                swap_file_out_app(ctx.app, ctx.temp_dir, ctx.swap_config)

    _run_nsys_post_phase(ctx, result, runner)
    return result


def run_all(
    app_config: dict,
    swaps_dict: dict[str, SwapConfig] | None,
    env: dict,
    config: DriverConfig,
) -> tuple[dict[str, AppResults], list[Operation], dict[str, list[DriverPassResult]]]:
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
        raise AppNameNotFoundError(config.app)

    # Total operations = operations per pass * number of passes
    total_operations = ops_per_pass * num_passes

    with alive_bar(total_operations, disable=config.no_progress) as pbar:
        for app in app_config["apps"]:
            # Filter by app name if specified
            if config.app != "all" and app["name"] != config.app:
                continue

            with tempfile.TemporaryDirectory(dir=config.temp_dir) as temp_dir_raw:
                temp_dir = Path(temp_dir_raw)
                app_dir = next(app_dir for app_dir in APP_DIRS if app_dir in app["path"])
                shutil.copytree(
                    Path(__file__).parent.parent / app_dir,
                    temp_dir / app_dir,
                )

                app_name = app["name"]
                results[app_name] = AppResults()
                long_results[app_name] = []

                # Build list of passes: baseline first, then swaps
                driver_passes: list[SwapConfig | None] = [None]  # None = baseline
                if swaps_dict:
                    driver_passes.extend(
                        [swap for swap in swaps_dict.values() if swap.app_name == app["name"]],
                    )

                logger.debug("Driver is running %d passes for %s", len(driver_passes), app_name)

                # Run each pass
                for pass_num, driver_pass in enumerate(driver_passes):
                    pass_ctx = DriverPassContext(
                        app=app,
                        env=env,
                        config=config,
                        temp_dir=temp_dir,
                        swap_config=driver_pass,
                        pbar=pbar,
                    )
                    pass_results = run_driver_pass(pass_ctx)
                    is_swap = driver_pass is not None
                    results[app_name].update_from_pass_result(
                        pass_result=pass_results,
                        is_swap=is_swap,
                    )
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
    config: DriverConfig,
) -> tuple[dict[str, AppResults], list[Operation], dict[str, list[DriverPassResult]]]:
    """Run the driver with a DriverConfig object.

    Args:
        config: Driver configuration object

    Returns:
        Tuple of:
        - results: Dictionary mapping app names to AppResults
        - operations: List of operations that were performed
        - long_results: Dictionary mapping app names to lists of DriverPassResult objects

    """
    logger.debug("Entering run_driver")

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
        description="GPA-Benchmark Driver: Build, run, validate, and profile applications",
    )
    parser.add_argument(
        "--app",
        type=str,
        default="all",
        help="The application to run (default: all)",
    )
    parser.add_argument(
        "--sm-version",
        type=int,
        default=None,
        help="The SM version to use (default: None, will auto-detect from nvidia-smi, then 90)",
    )
    parser.add_argument(
        "--cuda-home",
        type=Path,
        default=None,
        help="Path to the CUDA installation to use (default: None, will auto-detect from "
        "environment PATH, then /usr/local/cuda)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not clean the application before building",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only build the application (skip run and validate)",
    )
    parser.add_argument(
        "--nsys",
        action="store_true",
        help="Profile the application with Nsight Systems, then export and "
        "process the sqlite database (profiles stored under profiles/)",
    )
    parser.add_argument(
        "--ncu",
        action="store_true",
        help="Profile the application with Nsight Compute (profiles stored under profiles/)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="The app config file to use (default: driver_apps.yaml in the same directory as "
        "this script)",
    )
    parser.add_argument(
        "--swaps",
        type=Path,
        default=None,
        help="The path to the directory containing code files to swap in for the "
        "kernel, with the filename being run_<num>_optimized_code_<num>.cu",
    )
    parser.add_argument(
        "--detect-regions",
        action="store_true",
        help="Detect if the kernel file to swap into contains editable region markers and "
        "substitute into them rather than replacing the entire file",
    )
    parser.add_argument(
        "--postprocess-nsys",
        action="store_true",
        help="Only postprocess nsys-rep file(s) found under profiles/, do not run the application",
    )
    parser.add_argument(
        "--retain-nsys-profiles",
        action="store_true",
        help="Keep .nsys-rep and .sqlite profile files after postprocessing (default: delete "
        "them)",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=3,
        help="The number of times to collect ncu/nsys profiles for each application and swap",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=Path,
        default="driver_results.json",
        help="The file to save the long results to (default: driver_results.json)",
    )
    parser.add_argument(
        "-t",
        "--temp-dir",
        type=Path,
        default=None,
        help="The temporary directory to use for the driver, must exist (default: /tmp)",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: WARNING)",
    )
    parser.add_argument("--no-progress", action="store_true", help="Do not display a progress bar")
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="The timeout in seconds for the driver to run, if negative, not timeout enforced"
        "(default: 300)",
    )
    parser.add_argument(
        "--subprocess-output-char-limit",
        type=int,
        default=25000,
        help="Maximum characters to log for subprocess stdout/stderr. Characters are removed "
        "from the middle of the output to stay within the limit. Set to <= 0 to disable "
        "truncation. (default: 25000)",
    )
    parser.add_argument(
        "--suppress-command-stdout",
        action="store_true",
        help="Never log stdout/stderr from command runs (build, run, profile, etc.), regardless "
        "of log level or failure. Driver logging is unchanged.",
    )
    parser.add_argument(
        "--no-sanitize",
        action="store_true",
        help="Do not run compute sanitizer checks before running the application",
    )
    parser.add_argument(
        "--srun",
        action="store_true",
        help="Prepend Slurm srun to all commands and enforce timeout via srun --time=00:n; "
        "bypasses multiprocessing-based timeout handling (for use inside sbatch/salloc).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the GPA-Benchmark driver.

    Orchestrates the entire driver workflow: argument parsing, configuration
    setup, execution, and result reporting.

    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If required files don't exist

    """
    args = parse_args()

    # Configure logging
    log_level = getattr(logging, args.log_level.upper(), logging.WARNING)
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] - %(message)s")

    logger.info("Start gpa_bench_driver.py")

    # Convert argparse.Namespace to DriverConfig
    driver_config = DriverConfig.from_args(args)

    # Run the driver with the created DriverConfig object
    run_driver(driver_config)


if __name__ == "__main__":
    main()
