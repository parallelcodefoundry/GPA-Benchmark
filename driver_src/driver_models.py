#!/usr/bin/env python3
"""
Data Models for GPA-Benchmark Driver

This module defines the data classes, enums, and result structures used throughout
the driver system.
"""
from enum import Enum
from dataclasses import dataclass
from typing import Any
from collections.abc import Hashable
import argparse

from numpy import mean

from driver_src.driver_utils import detect_sm_version


@dataclass
class DriverConfig:
    """Configuration for the driver, mirroring all CLI arguments.

    This class represents all the parameters that can be passed to the driver,
    either via CLI or programmatically via the API.

    Attributes:
        app: The application to run (default: "all")
        sm_version: The SM version to use (default: None, auto-detect)
        cuda_home: Path to the CUDA installation to use (default: None)
        no_clean: Do not clean the application before building (default: False)
        build: Only build the application (skip run and validate) (default: False)
        nsys: Profile the application with Nsight Systems (default: False)
        ncu: Profile the application with Nsight Compute (default: False)
        config: The app config file to use (default: "driver_apps.yaml")
        swaps: Path to the directory containing code files to swap in (default: None)
        detect_regions: Detect editable region markers in swap files (default: False)
        postprocess_nsys: Only postprocess nsys-rep file(s) (default: False)
        num_samples: Number of times to collect ncu/nsys profiles (default: 3)
        output_file: File to save the long results to (default: "driver_results.json")
        temp_dir: Temporary directory to use (default: None, uses /tmp)
        log_level: Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: "WARNING")
        no_progress: Do not display a progress bar (default: False)
    """
    sm_version: int
    app: str = "all"
    cuda_home: str | None = None
    no_clean: bool = False
    build: bool = False
    nsys: bool = False
    ncu: bool = False
    config: str = "driver_apps.yaml"
    swaps: str | None = None
    detect_regions: bool = False
    postprocess_nsys: bool = False
    num_samples: int = 3
    output_file: str = "driver_results.json"
    temp_dir: str | None = None
    log_level: str = "WARNING"
    no_progress: bool = False

    def __init__(self, app: str,
                 sm_version: int | None,
                 cuda_home: str | None,
                 no_clean: bool,
                 build: bool,
                 nsys: bool,
                 ncu: bool,
                 config: str,
                 swaps: str | None,
                 detect_regions: bool,
                 postprocess_nsys: bool,
                 num_samples: int,
                 output_file: str,
                 temp_dir: str | None,
                 log_level: str,
                 no_progress: bool):
        self.app = app
        self.sm_version = sm_version if sm_version is not None else detect_sm_version()
        self.cuda_home = cuda_home
        self.no_clean = no_clean
        self.build = build
        self.nsys = nsys
        self.ncu = ncu
        self.config = config
        self.swaps = swaps
        self.detect_regions = detect_regions
        self.postprocess_nsys = postprocess_nsys
        self.num_samples = num_samples
        self.output_file = output_file
        self.temp_dir = temp_dir
        self.log_level = log_level
        self.no_progress = no_progress

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'DriverConfig':
        """Create a DriverConfig from an argparse.Namespace.

        Args:
            args: argparse.Namespace containing the command line arguments

        Returns:
            DriverConfig object
        """
        return cls(
            app=args.app,
            sm_version=args.sm_version,
            cuda_home=args.cuda_home,
            no_clean=args.no_clean,
            build=args.build,
            nsys=args.nsys,
            ncu=args.ncu,
            config=args.config,
            swaps=args.swaps,
            detect_regions=args.detect_regions,
            postprocess_nsys=args.postprocess_nsys,
            num_samples=args.num_samples,
            output_file=args.output_file,
            temp_dir=args.temp_dir,
            log_level=args.log_level,
            no_progress=args.no_progress
        )


class Operation(Enum):
    """Represents a type of driver operation that can be performed on an application.

    Attributes:
        BUILD: Build the application
        RUN: Run the application
        VALIDATE: Validate the application output
        NSYS_PROFILE: Profile with Nsight Systems
        NCU_PROFILE: Profile with Nsight Compute
        NSYS_POST: Postprocess Nsight Systems profile
        SWAP_BUILDS: Build operations for swapped code
        SWAP_RUNS: Run operations for swapped code
        SWAP_VALID: Validation operations for swapped code
        SWAP_NSYS: Nsight Systems profiling for swapped code
        SWAP_NCU: Nsight Compute profiling for swapped code
        SWAP_NSYS_POST: Nsight Systems postprocessing for swapped code
    """
    BUILD = "Build"
    RUN = "Run"
    VALIDATE = "Validate"
    NSYS_PROFILE = "NSYS Profile"
    NCU_PROFILE = "NCU Profile"
    NSYS_POST = "NSYS Post"
    SWAP_BUILDS = "Swap Builds"
    SWAP_RUNS = "Swap Runs"
    SWAP_VALID = "Swap Valid"
    SWAP_NSYS = "Swap NSYS"
    SWAP_NCU = "Swap NCU"
    SWAP_NSYS_POST = "Swap NSYS Post"


@dataclass
class FileSwap:
    """Represents a single file swap operation.

    Attributes:
        swap_file_src_path: Path to the source swap file
        swap_file_dest_name: Destination filename for the swap
        code: The code content to swap in
    """
    swap_file_src_path: str
    swap_file_dest_name: str
    code: str


@dataclass
class SwapConfig:
    """Represents a swap configuration for replacing code in an application.

    Attributes:
        app_name: Name of the application
        file_swaps: List of file swaps to perform
        run_num: Run number identifier
        optimized_code_num: Optimized code number identifier
    """
    app_name: str
    file_swaps: list[FileSwap]
    run_num: str
    optimized_code_num: str


@dataclass
class DriverPassResult:
    """Represents the results of a single driver pass.

    Attributes:
        app_name: Name of the application
        run_num: Run number identifier (if from swap)
        swap_num: Swap number identifier (if from swap)
        swap_file_src_path: Path to the swap file used (if from swap)
        build: Whether the build succeeded
        run: Whether the run succeeded
        validate: Whether validation succeeded
        nsys_profile: Whether Nsight Systems profiling succeeded
        ncu_profile: Whether Nsight Compute profiling succeeded
        nsys_post: Whether Nsight Systems postprocessing succeeded
        nsys_data: Data extracted from Nsight Systems profile
        build_stdout: Standard output from build process
        build_stderr: Standard error from build process
        run_stdout: Standard output from run process
        run_stderr: Standard error from run process
    """
    app_name: str | None = None
    run_num: str | None = None
    swap_num: str | None = None
    swap_file_src_path: str | None = None
    build: bool | None = None
    run: bool | None = None
    validate: bool | None = None
    nsys_profile: bool | None = None
    ncu_profile: bool | None = None
    nsys_post: bool | None = None
    nsys_data: list[dict[Hashable, Any]] | None = None
    build_stdout: str | None = None
    build_stderr: str | None = None
    run_stdout: str | None = None
    run_stderr: str | None = None

    def to_dict(self) -> dict[str, str | bool | float | int | list[dict[Hashable, Any]] | None]:
        """Convert the results to a dictionary.

        Returns:
            Dictionary representation of the driver pass result
        """
        exec_times = None
        if self.nsys_data is not None:
            exec_times = [data["exec_time"] for data in self.nsys_data]
        results = {
            "app_name": self.app_name,
            "run_num": self.run_num,
            "swap_num": self.swap_num,
            "swap_file_src_path": self.swap_file_src_path,
            "build": self.build,
            "run": self.run,
            "validate": self.validate,
            "nsys_profile": self.nsys_profile,
            "ncu_profile": self.ncu_profile,
            "nsys_post": self.nsys_post,
            "nsys_data": self.nsys_data,
            "exec_time.mean": mean(exec_times) if exec_times else None,
            "exec_time.min": min(exec_times) if exec_times else None,
            "exec_time.max": max(exec_times) if exec_times else None,
            "build_stdout": self.build_stdout,
            "build_stderr": self.build_stderr,
            "run_stdout": self.run_stdout,
            "run_stderr": self.run_stderr,
        }
        return results


class AppResults:
    """Represents results for all passes of a given application.

    Tracks both baseline (non-swap) results and aggregated swap results
    as fractions (numerator/denominator).
    """
    def __init__(self) -> None:
        """Initialize AppResults with all fields set to None or zero."""
        # Results for single non-swap pass
        self.build: bool | None = None
        self.run: bool | None = None
        self.validate: bool | None = None
        self.nsys_profile: bool | None = None
        self.ncu_profile: bool | None = None
        self.nsys_post: bool | None = None

        # Results for all swap passes (tracked as fractions)
        self.swap_builds_numerator: int = 0
        self.swap_builds_denominator: int = 0
        self.swap_runs_numerator: int = 0
        self.swap_runs_denominator: int = 0
        self.swap_valid_numerator: int = 0
        self.swap_valid_denominator: int = 0
        self.swap_nsys_numerator: int = 0
        self.swap_nsys_denominator: int = 0
        self.swap_ncu_numerator: int = 0
        self.swap_ncu_denominator: int = 0
        self.swap_nsys_post_numerator: int = 0
        self.swap_nsys_post_denominator: int = 0

    def update_from_pass_result(self, pass_result: DriverPassResult, is_swap: bool) -> None:
        """Update results from a driver pass result.

        Args:
            pass_result: The driver pass result to incorporate
            is_swap: Whether this result is from a swap pass (True) or baseline (False)
        """
        if pass_result.build is not None:
            if not is_swap:
                self.build = pass_result.build
            else:
                self.swap_builds_denominator += 1
                if pass_result.build:
                    self.swap_builds_numerator += 1

        if pass_result.run is not None:
            if not is_swap:
                self.run = pass_result.run
            else:
                self.swap_runs_denominator += 1
                if pass_result.run:
                    self.swap_runs_numerator += 1

        if pass_result.validate is not None:
            if not is_swap:
                self.validate = pass_result.validate
            else:
                self.swap_valid_denominator += 1
                if pass_result.validate:
                    self.swap_valid_numerator += 1

        if pass_result.nsys_profile is not None:
            if not is_swap:
                self.nsys_profile = pass_result.nsys_profile
            else:
                self.swap_nsys_denominator += 1
                if pass_result.nsys_profile:
                    self.swap_nsys_numerator += 1

        if pass_result.ncu_profile is not None:
            if not is_swap:
                self.ncu_profile = pass_result.ncu_profile
            else:
                self.swap_ncu_denominator += 1
                if pass_result.ncu_profile:
                    self.swap_ncu_numerator += 1

        if pass_result.nsys_post is not None:
            if not is_swap:
                self.nsys_post = pass_result.nsys_post
            else:
                self.swap_nsys_post_denominator += 1
                if pass_result.nsys_post:
                    self.swap_nsys_post_numerator += 1

    def get(self, key: Operation) -> bool | str | None:
        """Get a result value by key (for backward compatibility with dict interface).

        Args:
            key: The operation to get results for

        Returns:
            The result value (bool for baseline, "n/d" string for swaps, None if not available)
        """
        if key == Operation.BUILD:
            return self.build
        elif key == Operation.RUN:
            return self.run
        elif key == Operation.VALIDATE:
            return self.validate
        elif key == Operation.NSYS_PROFILE:
            return self.nsys_profile
        elif key == Operation.NCU_PROFILE:
            return self.ncu_profile
        elif key == Operation.NSYS_POST:
            return self.nsys_post
        elif key == Operation.SWAP_BUILDS:
            if self.swap_builds_denominator > 0:
                return f"{self.swap_builds_numerator}/{self.swap_builds_denominator}"
            return None
        elif key == Operation.SWAP_RUNS:
            if self.swap_runs_denominator > 0:
                return f"{self.swap_runs_numerator}/{self.swap_runs_denominator}"
            return None
        elif key == Operation.SWAP_VALID:
            if self.swap_valid_denominator > 0:
                return f"{self.swap_valid_numerator}/{self.swap_valid_denominator}"
            return None
        elif key == Operation.SWAP_NSYS:
            if self.swap_nsys_denominator > 0:
                return f"{self.swap_nsys_numerator}/{self.swap_nsys_denominator}"
            return None
        elif key == Operation.SWAP_NCU:
            if self.swap_ncu_denominator > 0:
                return f"{self.swap_ncu_numerator}/{self.swap_ncu_denominator}"
            return None
        elif key == Operation.SWAP_NSYS_POST:
            if self.swap_nsys_post_denominator > 0:
                return f"{self.swap_nsys_post_numerator}/{self.swap_nsys_post_denominator}"
            return None
        else:
            return None
