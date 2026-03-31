"""Configuration Management for GPA-Benchmark Driver.

This module handles loading and validating application configuration, determining
which operations to perform, and setting up the environment.
"""

import logging
import os
from pathlib import Path

import yaml

from gpa_bench_driver.driver_src.driver_file_swapping import build_swaps_dict
from gpa_bench_driver.driver_src.driver_models import DriverConfig, Operation, SwapConfig
from gpa_bench_driver.driver_src.driver_utils import get_default_apps_config_path

logger = logging.getLogger("GPA-Benchmark")


class AppNameNotFoundError(Exception):
    """Exception raised for errors in the app name.

    Attributes:
        message: explanation of the error

    """

    def __init__(self, app_name: str) -> None:
        """Initialize the AppNameNotFoundError.

        Args:
            app_name: the name of the app that was not found

        """
        self.message = f"Could not find {app_name} in app_config!"
        super().__init__(self.message)


class OperationCombinationError(ValueError):
    """Exception raised for invalid operation combinations."""

    def __init__(self, message: str) -> None:
        """Initialize the OperationCombinationError.

        Args:
            message: The error message

        """
        super().__init__(message)


def setup_app_config(config: DriverConfig) -> tuple[dict, dict[str, SwapConfig] | None, dict]:
    """Set up the application configuration.

    Loads the YAML configuration file, validates arguments, sets up environment
    variables, and optionally loads swap configurations.

    Args:
        config: Driver configuration object

    Returns:
        Tuple of (app_config, swaps_dict, env)
        - app_config: Application configuration dictionary
        - swaps_dict: Dictionary of swap configurations or None
        - env: Environment variables dictionary

    Raises:
        ValueError: If argument combinations are invalid or app not found in config
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML

    """
    logger.debug("Entering setup_app_config")
    # Validate argument combinations
    if config.postprocess_nsys and (
        config.nsys or config.swaps or config.ncu or config.build_only or config.swaps_override
    ):
        msg = "Cannot postprocess Nsight Systems profiles only if other operations are specified."
        raise OperationCombinationError(msg)
    if config.build_only and (
        config.nsys
        or config.ncu
        or config.swaps
        or config.postprocess_nsys
        or config.swaps_override
    ):
        msg = "Cannot build applications only if other operations are specified."
        raise OperationCombinationError(msg)
    if config.swaps_override and config.swaps:
        msg = "Cannot specify both swaps and swaps_override."
        raise OperationCombinationError(msg)
    if config.swaps_override and config.app == "all":
        msg = "Cannot specify swaps_override for apps = all."
        raise OperationCombinationError(msg)

    # Load config file
    with config.config.open("r", encoding="utf-8") as f:
        app_config: dict = yaml.safe_load(f)

    logger.debug("App config loaded")

    # Validate app name
    if config.app != "all":
        config.app = get_canonical_app_name(config.app, app_config)

    # Optionally run smaller problem sizes (when the app provides small_run_command).
    # This is applied to the selected app(s) only.
    if config.small_problem:
        for app in app_config.get("apps", []):
            if (
                config.app == "all" or app.get("name") == config.app
            ) and "small_run_command" in app:
                app["run_command"] = app["small_run_command"]

    # Setup CUDA environment
    cuda_home = Path(
        config.cuda_home
        or os.getenv("CUDA_HOME")
        or os.getenv("CUDA_PATH")
        or os.getenv("CUDA_ROOT")
        or "/usr/local/cuda",
    )
    env = os.environ.copy()
    env["CUDA_HOME"] = str(cuda_home)
    cuda_lib64 = cuda_home / "lib64"
    existing_ld_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = str(
        f"{cuda_lib64}:{existing_ld_path}" if existing_ld_path else cuda_lib64,
    )

    logger.debug("CUDA environment set")

    # Load swaps or override swaps if specified
    if config.swaps_override:
        formatted_swaps: dict[
            tuple[str, int, str | None, str | None, int], list[tuple[Path, str]],
        ] = {
            (config.app, 0, None, None, 0): list(config.swaps_override.items()),
        }
        swaps_dict = build_swaps_dict(formatted_swaps, config.app, app_config)
        return app_config, swaps_dict, env

    if config.swaps:
        swaps_dict = build_swaps_dict(config.swaps, config.app, app_config)
        return app_config, swaps_dict, env

    return app_config, None, env


def get_canonical_app_name(
    app_name: str,
    config: Path | dict | None = None,
) -> str:
    """Get the canonical name for an application, converting aliases to the primary name.

    Args:
        app_name: The name of the application
        config: The application configuration dictionary or YAML file path

    Returns:
        The canonical name for the application

    """
    if config is None:
        config = get_default_apps_config_path()
    if isinstance(config, Path):
        with config.open("r", encoding="utf-8") as f:
            app_config = yaml.safe_load(f)
    else:
        app_config = config
    all_names = [app["name"] for app in app_config["apps"]]
    if app_name in all_names:
        return app_name
    for app in app_config["apps"]:
        if "aliases" in app and app_name in app["aliases"]:
            return app["name"]
    raise AppNameNotFoundError(app_name)


def determine_operations(config: DriverConfig) -> list[Operation]:
    """Determine which operations will be performed based on driver configuration.

    Args:
        config: Driver configuration object

    Returns:
        List of Operation enums that will be performed

    """
    operations: list[Operation] = []

    if not config.postprocess_nsys:
        operations.append(Operation.BUILD)
        if not config.build_only:
            if not config.no_sanitize:
                operations.append(Operation.SANITIZE)
            operations.append(Operation.RUN)
            operations.append(Operation.VALIDATE)

    if config.nsys:
        operations.append(Operation.NSYS_PROFILE)

    if config.nsys or config.postprocess_nsys:
        operations.append(Operation.NSYS_POST)

    if config.ncu:
        operations.append(Operation.NCU_PROFILE)

    if config.swaps or config.swaps_override:
        operations.append(Operation.SWAP_BUILDS)
        if not config.build_only:
            if not config.no_sanitize:
                operations.append(Operation.SWAP_SANITIZES)
            operations.append(Operation.SWAP_RUNS)
            operations.append(Operation.SWAP_VALID)
            if config.nsys:
                operations.append(Operation.SWAP_NSYS)
            if config.ncu:
                operations.append(Operation.SWAP_NCU)
            if config.postprocess_nsys:
                operations.append(Operation.SWAP_NSYS_POST)

    return operations
