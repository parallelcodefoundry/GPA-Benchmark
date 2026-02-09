"""
Configuration Management for GPA-Benchmark Driver

This module handles loading and validating application configuration, determining
which operations to perform, and setting up the environment.
"""
import os
import yaml

from gpa_bench_driver.driver_src.driver_models import Operation, SwapConfig, DriverConfig
from gpa_bench_driver.driver_src.driver_file_swapping import build_swaps_dict


def setup_app_config(config: DriverConfig) -> tuple[dict, dict[str, SwapConfig] | None, dict]:
    """Setup the application configuration.

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
    # Validate argument combinations
    if config.postprocess_nsys and (config.nsys or config.swaps or config.ncu or config.build_only \
        or config.swaps_override):
        raise ValueError("Cannot postprocess Nsight Systems profiles only if other operations " \
            + "are specified.")
    if config.build_only and (config.nsys or config.ncu or config.swaps or config.postprocess_nsys \
        or config.swaps_override):
        raise ValueError("Cannot build applications only if other operations are specified.")
    if config.swaps_override and config.swaps:
        raise ValueError("Cannot specify both swaps and swaps_override.")
    if config.swaps_override and config.app == "all":
        raise ValueError("Cannot specify swaps_override for apps = all.")

    # Load config file
    with open(config.config, "r", encoding="utf-8") as f:
        app_config: dict = yaml.safe_load(f)

    # Validate app name
    if config.app != "all":
        all_names = [app["name"].lower() for app in app_config["apps"]]
        if config.app not in all_names:
            # Try to convert app name alias to canonical name
            for app in app_config["apps"]:
                if config.app in app["aliases"]:
                    config.app = app["name"]
                    break
            if config.app not in all_names:
                raise ValueError(f"Application {config.app} not found in config file " \
                                 + f"{config.config}")

    # Setup CUDA environment
    cuda_home = (config.cuda_home or
                 os.getenv("CUDA_HOME") or
                 os.getenv("CUDA_PATH") or
                 os.getenv("CUDA_ROOT") or
                 "/usr/local/cuda")
    env = os.environ.copy()
    env["CUDA_HOME"] = cuda_home

    # Load swaps or override swaps if specified
    if config.swaps_override:
        formatted_swaps: dict[tuple[str, str | None, int, int], list[tuple[str, str]]] = \
            {(config.app, None, 0, 0): list(config.swaps_override.items())}
        swaps_dict = build_swaps_dict(formatted_swaps, config.app, app_config)
        return app_config, swaps_dict, env

    if config.swaps:
        swaps_dict = build_swaps_dict(config.swaps, config.app, app_config)
        return app_config, swaps_dict, env

    return app_config, None, env


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
            operations.append(Operation.SWAP_RUNS)
            operations.append(Operation.SWAP_VALID)
            if config.nsys:
                operations.append(Operation.SWAP_NSYS)
            if config.ncu:
                operations.append(Operation.SWAP_NCU)
            if config.postprocess_nsys:
                operations.append(Operation.SWAP_NSYS_POST)

    return operations
