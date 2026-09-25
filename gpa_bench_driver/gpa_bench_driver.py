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
import contextlib
import logging
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess

try:
    from alive_progress import alive_bar
except ImportError:  # optional: not installed in every venv (e.g. APPEB's Frontier venv)

    @contextlib.contextmanager
    def alive_bar(*_args: object, **_kwargs: object) -> Iterator[Callable[..., None]]:
        """No-op stand-in for alive_progress.alive_bar."""
        yield lambda *_a, **_k: None

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
from gpa_bench_driver.driver_src.driver_check import (
    check_output,
    load_reference,
    produced_output,
    reference_from_output,
)
from gpa_bench_driver.driver_src.driver_gate import check_kernel_source
from gpa_bench_driver.driver_src.driver_reporting import print_report_table, save_results
from gpa_bench_driver.driver_src.driver_rocprof import (
    RocprofError,
    get_score_regex,
    profile_once,
    rocprof_time_app,
)
from gpa_bench_driver.driver_src.driver_utils import (
    DriverInfraError,
    SubprocessRunner,
    SubprocessRunnerConfig,
    get_bin_path,
    get_run_path,
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
    hip_state: dict | None = None  # hip only: {"gpa_root": Path, "ref": Reference | None}
    skip_timing: bool = False  # hip interleaved flow: timing happens after all builds
    keep_swapped: bool = False  # hip interleaved flow: the swap copy keeps the swapped sources


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
    "XSBench-hip",  # before "XSBench": the lookup below is a substring match
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
    result.generating_llm = ctx.swap_config.generating_llm if ctx.swap_config else None
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
    if _backend(ctx.config) == "hip":
        build_success, build_result = build_app(
            ctx.app,
            ctx.config.sm_version,
            runner,
            ctx.temp_dir,
            no_clean=ctx.config.no_clean,
            gpu_backend="hip",
            offload_arch=ctx.config.offload_arch,
        )
    else:
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
                logger.warning(
                    "Sanitize (%s) failed due to timeout (%s); skipping remaining sanitizers.",
                    tool,
                    timeout_label,
                )
            else:
                logger.debug("Sanitize (%s) failed,, skipping remaining sanitizers.", tool)
            # Advance progress bar by number of remaining sanitizers
            if ctx.pbar is not None and i < len(tools) - 1:
                for _ in range(len(tools) - i - 1):
                    ctx.pbar()
            return False
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
) -> DriverPassResult:
    """Either raise BaselineError (baseline pass) or update progress and return result."""
    if ctx.swap_config is None:
        if stage == "build":
            logger.error("Baseline build stdout: %s", result.build_stdout)
            logger.error("Baseline build stderr: %s", result.build_stderr)
        elif stage == "sanitize":
            if result.sanitize_details is None:
                raise ValueError(
                    "Sanitize phase failure reported but no sanitize details were set"
                )
            failed_tool = next(
                (t for t in SanitizeTool.__members__.values() if not result.sanitize_details[t]),
                None,
            )
            if failed_tool is None:
                raise ValueError("Sanitize phase failure reported but no failed tool was found")
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


def _backend(config: DriverConfig) -> str:
    """GPU backend of a config ("cuda" for configs that predate the hip backend)."""
    return getattr(config, "gpu_backend", "cuda")


def _run_rocprof_phase(
    ctx: DriverPassContext,
    result: DriverPassResult,
    runner: SubprocessRunner,
) -> None:
    """hip backend: kernel timing with rocprofv3 (selected by config.nsys).

    Fills result.nsys_profile/nsys_post/nsys_data (one timing dict per sample, see
    driver_rocprof) so consumers of the cuda backend's nsys_data keep working.
    """
    if ctx.config.ncu:
        logger.warning("Nsight Compute is not available on the hip backend; skipping it.")
    if not ctx.config.nsys:
        return
    state = ctx.hip_state or {}

    def _validate_sample(run_result: CompletedProcess, run_dir: Path) -> tuple[bool, str | None]:
        return check_output(
            ctx.app, produced_output(ctx.app, run_result.stdout, run_dir), state["ref"],
        )

    data = rocprof_time_app(
        ctx.app,
        runner,
        ctx.temp_dir,
        ctx.config.num_samples,
        swap_config=ctx.swap_config,
        pbar=ctx.pbar,
        rocm_path=getattr(ctx.config, "rocm_path", None),
        retain_profiles=ctx.config.retain_nsys_profiles,
        validate=_validate_sample if state.get("ref") is not None else None,
    )
    result.nsys_profile = data is not None
    result.nsys_post = data is not None
    result.nsys_data = data
    # R4: every timed run's output is checked; one failing run fails the pass
    bad = [(i, s) for i, s in enumerate(data or []) if s.get("valid") is False]
    if bad:
        i, sample = bad[0]
        result.validate = False
        result.validation_output = (
            f"TIMED RUN {i}: {sample.get('validation_output')}"
            + (f" ({len(bad)} of {len(data)} timed runs failed)" if len(bad) > 1 else "")
        )
        if ctx.swap_config is None:
            msg = f"Validate failed for baseline timed run ({ctx.app['name']}): " + (
                result.validation_output
            )
            raise BaselineError(msg)


def _run_profiling_phase(
    ctx: DriverPassContext,
    result: DriverPassResult,
    runner: SubprocessRunner,
) -> None:
    """Run NSYS and NCU profiling if enabled."""
    if _backend(ctx.config) == "hip":
        _run_rocprof_phase(ctx, result, runner)
        return
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
    if _backend(ctx.config) == "hip":
        return  # rocprofv3 timing is parsed inline by _run_rocprof_phase
    if ctx.config.postprocess_nsys or (ctx.config.nsys and result.nsys_profile):
        postprocess_nsys_result = postprocess_nsys_app(
            ctx.app,
            runner,
            ctx.temp_dir,
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


def _run_hip_validate_phase(
    ctx: DriverPassContext,
    result: DriverPassResult,
    run_result: CompletedProcess,
) -> bool:
    """hip: check the run's output against the harness-side reference (driver_check).

    With reference_from_baseline the baseline pass's output defines the reference instead.
    """
    state = ctx.hip_state if ctx.hip_state is not None else {}
    produced = produced_output(ctx.app, run_result.stdout, get_run_path(ctx.app, ctx.temp_dir))
    if ctx.swap_config is None and getattr(ctx.config, "reference_from_baseline", False):
        try:
            state["ref"] = reference_from_output(ctx.app, produced)
            ok, message = True, None
        except ValueError as exc:
            ok, message = False, str(exc)
    else:
        if state.get("ref") is None:
            state["ref"] = load_reference(ctx.app, state["gpa_root"])
        ok, message = check_output(ctx.app, produced, state["ref"])
    result.validate = ok
    if not ok:
        result.validation_output = message
    if ctx.pbar is not None:
        ctx.pbar()
    return ok


def _gate_swap(ctx: DriverPassContext, result: DriverPassResult) -> bool:
    """hip: run the kernel-file gate on every swapped file (after swap-in, before building)."""
    state = ctx.hip_state or {}
    gpa_root = state.get("gpa_root")
    if ctx.swap_config is None or gpa_root is None:
        return True
    violations: list[str] = []
    messages: list[str] = []
    for file_swap in ctx.swap_config.file_swaps:
        rel = str(file_swap.swap_file_dest_name)
        candidate = (ctx.temp_dir / rel).read_text(encoding="utf-8", errors="replace")
        gate = check_kernel_source({**ctx.app, "kernel_file": rel}, candidate, gpa_root=gpa_root)
        if not gate.ok:
            violations.extend(gate.violations)
            messages.append(gate.message())
    if not violations:
        return True
    result.gate_violations = violations
    result.build = False
    result.build_stdout = ""
    result.build_stderr = "\n".join(messages)
    if ctx.pbar is not None:
        ctx.pbar()
    return False


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
    hip = _backend(ctx.config) == "hip"
    if hip:
        runner_config.stdin_devnull = True  # R4: apps never read the driver's stdin
        runner_config.raise_infra = True  # F5: harness failures are DriverInfraError
    runner = SubprocessRunner(env=ctx.env, config=runner_config)

    if not ctx.config.postprocess_nsys:
        if ctx.swap_config:
            swap_file_in_app(
                swap_config=ctx.swap_config,
                temp_dir=ctx.temp_dir,
                detect_regions=ctx.config.detect_regions,
            )
        try:
            if hip and getattr(ctx.config, "kernel_gate", True) and not _gate_swap(ctx, result):
                return _handle_early_exit(ctx, result, "build")
            if not _run_build_phase(ctx, result, runner):
                return _handle_early_exit(ctx, result, "build")
            if ctx.config.build_only:
                return result
            if not ctx.config.no_sanitize and not _run_sanitize_phase(ctx, result, runner):
                return _handle_early_exit(ctx, result, "sanitize")
            run_ok, run_result = _run_run_phase(ctx, result, runner)
            if not run_ok:
                return _handle_early_exit(ctx, result, "run")
            validate_phase = _run_hip_validate_phase if hip else _run_validate_phase
            if not validate_phase(ctx, result, run_result):
                return _handle_early_exit(ctx, result, "validate")
            if not ctx.skip_timing:
                _run_profiling_phase(ctx, result, runner)
        finally:
            if ctx.swap_config and not ctx.keep_swapped:
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
        if config.app != "all" and app["name"].lower() != config.app.lower():
            continue
        # Baseline pass
        num_passes += 1
        # Swap passes for this app
        if swaps_dict:
            num_passes += sum(1 for swap in swaps_dict.values() if swap.app_name.lower() == app["name"].lower())
    if num_passes == 0:
        raise AppNameNotFoundError(config.app)

    # Total operations = operations per pass * number of passes
    total_operations = ops_per_pass * num_passes

    with alive_bar(total_operations, disable=config.no_progress) as pbar:
        for app in app_config["apps"]:
            # Filter by app name if specified
            if config.app != "all" and app["name"].lower() != config.app.lower():
                continue

            with tempfile.TemporaryDirectory(
                dir=config.temp_dir, **({"ignore_cleanup_errors": True} if _backend(config) == "hip" else {}),
            ) as temp_dir_raw:
                temp_dir = Path(temp_dir_raw)
                if _backend(config) == "hip":
                    app_dir = _stage_hip_app(app, Path(__file__).parent.parent, temp_dir)
                else:
                    app_dir = next(app_dir for app_dir in APP_DIRS if app_dir in app["path"])
                    shutil.copytree(
                        Path(__file__).parent.parent / app_dir,
                        temp_dir / app_dir,
                    )
                # Ensure all copied files are writable so builds can
                # overwrite stale binaries and object files.
                for root, dirs, files in os.walk(temp_dir / app_dir):
                    for d in dirs:
                        os.chmod(os.path.join(root, d), 0o755)
                    for f in files:
                        fp = os.path.join(root, f)
                        os.chmod(fp, os.stat(fp).st_mode | 0o644)

                app_name = app["name"]
                results[app_name] = AppResults()
                long_results[app_name] = []

                # Build list of passes: baseline first, then swaps
                driver_passes: list[SwapConfig | None] = [None]  # None = baseline
                if swaps_dict:
                    driver_passes.extend(
                        [swap for swap in swaps_dict.values() if swap.app_name.lower() == app["name"].lower()],
                    )

                logger.debug("Driver is running %d passes for %s", len(driver_passes), app_name)

                # hip: per-app state shared by the passes (the reference to check against)
                hip_state = (
                    {"gpa_root": Path(__file__).parent.parent, "ref": None}
                    if _backend(config) == "hip"
                    else None
                )

                if hip_state is not None and getattr(config, "interleave", True):
                    app_passes = _run_app_hip_interleaved(
                        app, driver_passes, env, config, pbar, temp_dir, hip_state,
                    )
                    for pass_num, pass_results in enumerate(app_passes):
                        results[app_name].update_from_pass_result(
                            pass_result=pass_results, is_swap=pass_num > 0,
                        )
                        long_results[app_name].append(pass_results)
                    continue

                # Run each pass
                for pass_num, driver_pass in enumerate(driver_passes):
                    pass_ctx = DriverPassContext(
                        app=app,
                        env=env,
                        config=config,
                        temp_dir=temp_dir,
                        swap_config=driver_pass,
                        pbar=pbar,
                        hip_state=hip_state,
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


def _hip_runner(config: DriverConfig, env: dict) -> SubprocessRunner:
    return SubprocessRunner(env=env, config=SubprocessRunnerConfig(
        log_level=config.log_level,
        timeout=config.timeout,
        output_char_limit=config.subprocess_output_char_limit,
        suppress_command_stdout=config.suppress_command_stdout,
        use_srun=config.srun,
        stdin_devnull=True,
        raise_infra=True,
    ))


def _chmod_tree(root: Path) -> None:
    for dirpath, dirs, files in os.walk(root):
        for d in dirs:
            os.chmod(os.path.join(dirpath, d), 0o755)
        for f in files:
            fp = os.path.join(dirpath, f)
            os.chmod(fp, os.stat(fp).st_mode | 0o644)


class _SideError(Exception):
    """Wraps a timing-run failure with the side index (0 = original, 1 = swap) so the caller can
    attribute a side-0 (j=0) failure to infra and a side-1 failure to the agent (J2)."""

    def __init__(self, side: int, exc: Exception) -> None:
        self.side = side
        self.exc = exc
        super().__init__(str(exc))


def _interleaved_series(
    app: dict,
    runner: SubprocessRunner,
    config: DriverConfig,
    hip_state: dict,
    roots: list[Path],
) -> tuple[list[list[dict]], list[str | None]]:
    """F1 timing (H1): 1 discarded profiled warm-up per side, then n_samples alternating profiled
    samples (side 0, side 1, ...). Each profiled sample carries cpu_s (G-cpu is measured in the
    same scored runs). Every run's output is checked against hip_state["ref"] (R4).

    Returns:
        (profiled samples per side, warm-up failure message per side)

    """
    ref = hip_state.get("ref")

    def validate(run_result: CompletedProcess, run_dir: Path) -> tuple[bool, str | None]:
        if ref is None:
            return True, None
        return check_output(app, produced_output(app, run_result.stdout, run_dir), ref)

    score_regex = get_score_regex(app)
    rocm = getattr(config, "rocm_path", None)
    paths = [get_run_path(app, root) for root in roots]
    profile_dirs = [root / "profiles" for root in roots]
    for d in profile_dirs:
        d.mkdir(parents=True, exist_ok=True)
    samples: list[list[dict]] = [[] for _ in roots]
    warm_fail: list[str | None] = [None for _ in roots]
    retain = config.retain_nsys_profiles
    n_samples = getattr(config, "final_samples", None) or config.num_samples
    def _once(j, path, outdir):
        # J2: a side-0 (original) failure is tagged so the caller treats it as infra, not the agent
        try:
            return profile_once(app, runner, path, outdir, score_regex=score_regex,
                                rocm_path=rocm, validate=validate, retain_profiles=retain)
        except (RocprofError, DriverInfraError) as exc:
            raise _SideError(j, exc) from exc

    for j, path in enumerate(paths):  # warm-up, discarded
        w = _once(j, path, profile_dirs[j] / "rocprof_warmup")
        if w.get("valid") is False:
            warm_fail[j] = w.get("validation_output")
    order = 0
    for i in range(n_samples):
        for j, path in enumerate(paths):
            s = _once(j, path, profile_dirs[j] / f"rocprof_sample_{i}")
            s["order"] = order
            s["warmup"] = False
            order += 1
            samples[j].append(s)
    return samples, warm_fail


def _first_invalid(label: str, samples: list[dict], warm: str | None) -> str | None:
    bad = [(i, s) for i, s in enumerate(samples) if s.get("valid") is False]
    if not bad:
        return None if warm is None else f"WARM-UP RUN: {warm}"
    i, s = bad[0]
    more = f" ({len(bad)} of {len(samples)} {label} runs failed)" if len(bad) > 1 else ""
    return f"{label.upper()} RUN {i}: {s.get('validation_output')}{more}"


def _run_app_hip_interleaved(
    app: dict,
    driver_passes: list,
    env: dict,
    config: DriverConfig,
    pbar: Callable | None,
    temp_dir: Path,
    hip_state: dict,
) -> list[DriverPassResult]:
    """hip flow (fix round 2 F1): build and check every copy first, then time interleaved."""
    try:
        base_ctx = DriverPassContext(app=app, env=env, config=config, temp_dir=temp_dir,
                                     swap_config=None, pbar=pbar, hip_state=hip_state,
                                     skip_timing=True)
        base = run_driver_pass(base_ctx)
        passes = [base]
        if config.build_only:
            return passes
        ready: list[tuple[Path, DriverPassResult]] = []
        gpa_root = Path(__file__).parent.parent
        for i, swap in enumerate(driver_passes[1:]):
            root = temp_dir / f"swap{i}"
            _stage_hip_app(app, gpa_root, root)
            _chmod_tree(root)
            ctx = DriverPassContext(app=app, env=env, config=config, temp_dir=root,
                                    swap_config=swap, pbar=pbar, hip_state=hip_state,
                                    skip_timing=True, keep_swapped=True)
            res = run_driver_pass(ctx)
            passes.append(res)
            if res.build and res.run and res.validate:
                ready.append((root, res))
        if not config.nsys:
            return passes
        runner = _hip_runner(config, env)
        pairs = ready or [(None, None)]
        for k, (root, res) in enumerate(pairs):
            roots = [temp_dir] if root is None else [temp_dir, root]
            try:
                samples, warm = _interleaved_series(app, runner, config, hip_state, roots)
            except _SideError as se:
                if se.side == 0:  # J2: the ORIGINAL failed during timing -> infra (retry once)
                    if isinstance(se.exc, DriverInfraError):
                        raise se.exc
                    msg = (f"the original {app['name']} failed during the timing loop "
                           f"(j=0): {se.exc}")
                    raise DriverInfraError(msg) from se.exc
                # H3: the swap's own crash is an agent failure on that pass
                if res is None:
                    raise se.exc
                res.run = False
                res.validate = False
                res.validation_output = f"the program failed during timing: {se.exc}"
                continue
            base_problem = _first_invalid("baseline timed", samples[0], warm[0])
            if base_problem is not None:
                msg = f"Validate failed for baseline timed run ({app['name']}): {base_problem}"
                raise BaselineError(msg)
            if any(s["target_dispatches"] == 0 for s in samples[0]):
                logger.warning("No dispatch of the target kernel (score_regex %r) in the %s "
                               "baseline trace", get_score_regex(app), app["name"])
                b_nsys = None
            else:
                b_nsys = samples[0]
            if k == 0:
                base.nsys_data = b_nsys
                base.nsys_profile = base.nsys_post = b_nsys is not None
            if res is None:
                continue
            res.nsys_data = samples[1]
            res.baseline_nsys_data = b_nsys
            res.nsys_profile = res.nsys_post = True
            problem = _first_invalid("timed", samples[1], warm[1])
            if problem is not None:
                res.validate = False
                res.validation_output = problem
        return passes
    except (FileNotFoundError, NotADirectoryError) as exc:
        if not Path(temp_dir).is_dir():
            msg = f"the driver's working directory {temp_dir} vanished: {exc}"
            raise DriverInfraError(msg) from exc
        raise


def _stage_hip_app(app: dict, gpa_root: Path, temp_dir: Path) -> str:
    """Copy one hip app into temp_dir and return the top-level directory to chmod.

    The cuda path copies the whole top-level app tree (all of rodinia/, including ~2 GB of
    input data). The hip ports are self-contained: only the app's own directory is copied
    (without object files and setup build logs; the driver cleans and rebuilds anyway), and
    the read-only Rodinia inputs are symlinked as rodinia/data.
    """
    top = Path(app["path"]).parts[0]
    ignore = shutil.ignore_patterns("*.o", ".frontier_build.log", ".rocprofv3")
    if top == "rodinia":
        app_rel = Path(app["path"])
        shutil.copytree(gpa_root / app_rel, temp_dir / app_rel, ignore=ignore, symlinks=True)
        data = gpa_root / "rodinia" / "data"
        if data.exists():
            (temp_dir / "rodinia" / "data").symlink_to(data.resolve(), target_is_directory=True)
        return str(app_rel)
    shutil.copytree(gpa_root / top, temp_dir / top, ignore=ignore, symlinks=True)
    return top


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
    parser.add_argument(
        "--gpu-backend",
        type=str,
        choices=["cuda", "hip"],
        default=None,
        help="GPU backend (default: auto-detect; hip = AMD/Frontier: hipcc builds, rocprofv3 "
        "timing via --nsys, driver_apps.frontier.yaml, no compute-sanitizer)",
    )
    parser.add_argument(
        "--offload-arch",
        type=str,
        default=None,
        help="AMD GPU ISA for hip builds (default: auto-detect from rocminfo, then gfx90a)",
    )
    parser.add_argument(
        "--gpu-device",
        type=int,
        default=None,
        help="Run the app on this GPU only (ROCR_VISIBLE_DEVICES on hip, CUDA_VISIBLE_DEVICES "
        "on cuda)",
    )
    parser.add_argument(
        "--small-problem",
        action="store_true",
        help="Use the app's small_run_command (if present in driver_apps.yaml) instead of "
        "run_command.",
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
