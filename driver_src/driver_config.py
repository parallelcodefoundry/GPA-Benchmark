#!/usr/bin/env python3
"""
Configuration Management for GPA-Benchmark Driver

This module handles loading and validating application configuration, determining
which operations to perform, and setting up the environment.
"""
import os
import argparse
import yaml

from driver_src.driver_models import Operation, SwapConfig
from driver_src.driver_file_swapping import build_swaps_dict


def setup_app_config(args: argparse.Namespace) -> tuple[dict, dict[str, SwapConfig] | None, dict]:
    """Setup the application configuration.

    Loads the YAML configuration file, validates arguments, sets up environment
    variables, and optionally loads swap configurations.

    Args:
        args: Parsed command line arguments

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
    if args.postprocess_nsys and (args.nsys or args.swaps or args.ncu or args.build):
        raise ValueError("Cannot postprocess Nsight Systems profiles only if other operations " \
            + "are specified.")
    if args.build and (args.nsys or args.ncu or args.swaps or args.postprocess_nsys):
        raise ValueError("Cannot build applications only if other operations are specified.")

    # Load config file
    with open(args.config, "r", encoding="utf-8") as f:
        app_config: dict = yaml.safe_load(f)

    # Validate app name
    if args.app != "all" and args.app not in [app["name"] for app in app_config["apps"]]:
        raise ValueError(f"Application {args.app} not found in config file {args.config}")

    # Setup CUDA environment
    cuda_home = (args.cuda_home or
                 os.getenv("CUDA_HOME") or
                 os.getenv("CUDA_PATH") or
                 os.getenv("CUDA_ROOT") or
                 "/usr/local/cuda")
    env = os.environ.copy()
    env["CUDA_HOME"] = cuda_home

    # Load swaps if specified
    if args.swaps:
        swaps_dict = build_swaps_dict(args.swaps, args.app, app_config)
        return app_config, swaps_dict, env
    else:
        return app_config, None, env


def determine_operations(args: argparse.Namespace) -> list[Operation]:
    """Determine which operations will be performed based on command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        List of Operation enums that will be performed
    """
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
