#!/usr/bin/env python3
"""
File Swapping Operations for GPA-Benchmark Driver

This module handles swapping code files in and out of applications for testing
optimizations.
"""
import os
import re
import shutil

from driver_src.driver_models import SwapConfig


def swap_file_in_app(app: dict, swap_config: SwapConfig, temp_dir: str) -> None:
    """Swap the file in the application directory on disk.

    Backs up the original file. If the file contains ">>> START EDITABLE REGION"
    and "<<< END EDITABLE REGION", then replaces only that region with the
    contents of the swap file. Otherwise, replaces the entire file.

    Args:
        app: Application configuration dictionary
        swap_config: Swap configuration containing the code to swap in
        temp_dir: Temporary directory where working copy of application directory is located

    Raises:
        FileNotFoundError: If the destination file doesn't exist
        IOError: If file operations fail
    """
    dest_path = os.path.join(temp_dir, app["kernel_file"])
    backup_path = dest_path + ".bak"
    shutil.copy(dest_path, backup_path)

    with open(dest_path, "r", encoding="utf-8") as dest_file:
        dest_text = dest_file.read()

    if ">>> START EDITABLE REGION" not in dest_text or "<<< END EDITABLE REGION" not in dest_text:
        # Replace entire file with swap file code
        with open(dest_path, "w", encoding="utf-8") as dest_file:
            dest_file.write(swap_config.code)
        return

    # Replace only the editable region
    start_index = dest_text.find(">>> START EDITABLE REGION")
    end_index = dest_text.find("<<< END EDITABLE REGION")
    dest_text = dest_text[:start_index] + swap_config.code + dest_text[end_index:]

    with open(dest_path, "w", encoding="utf-8") as dest_file:
        dest_file.write(dest_text)


def swap_file_out_app(app: dict, temp_dir: str) -> None:
    """Swap the file out of the application directory on disk.

    Restores the original file from backup and removes the backup file.

    Args:
        app: Application configuration dictionary
        temp_dir: Temporary directory where working copy of application directory is located

    Raises:
        FileNotFoundError: If the backup file doesn't exist
        IOError: If file operations fail
    """
    dest_path = os.path.join(temp_dir, app["kernel_file"])
    backup_path = dest_path + ".bak"
    shutil.copy(backup_path, dest_path)
    os.remove(backup_path)


def build_swaps_dict(swaps: str, app: str) -> dict[str, SwapConfig]:
    """Build the swaps dictionary from a directory of swap files.

    Scans the directory for files matching the pattern "run_<num>_optimized_code_<num>.cu"
    and creates SwapConfig objects for each.

    Args:
        swaps: Path to the directory containing swap files
        app: Name of the application to build swaps for (or "all")

    Returns:
        Dictionary mapping file paths to SwapConfig objects

    Raises:
        FileNotFoundError: If the swaps directory doesn't exist
        IOError: If file reading fails
    """
    swaps_dict: dict[str, SwapConfig] = {}

    for root, _, files in os.walk(swaps):
        for file in files:
            if match := re.match(r"run_(\d+)_optimized_code_(\d+)\.cu", file):
                run_num = match.group(1)
                optimized_code_num = match.group(2)
                # App name is everything before the last _ in the directory three levels up
                app_name = "_".join(root.split("/")[-3].split("_")[:-1])
                full_path = os.path.join(root, file)

                with open(full_path, "r", encoding="utf-8") as f:
                    code = f.read()

                # First line is automatically generated, ends with file name
                swap_file_name = code.splitlines()[0].split(" ")[-1]
                rest_of_code = "\n".join([line for line in code.splitlines()[1:]
                                          if not line.startswith("```")])

                swaps_dict[full_path] = SwapConfig(
                    app_name=app_name,
                    swap_file_src_path=full_path,
                    swap_file_dest_name=swap_file_name,
                    run_num=run_num,
                    optimized_code_num=optimized_code_num,
                    code=rest_of_code
                )

    if app != "all":
        return {key: value for key, value in swaps_dict.items() if value.app_name == app}
    else:
        return swaps_dict
