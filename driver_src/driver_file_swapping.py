#!/usr/bin/env python3
"""
File Swapping Operations for GPA-Benchmark Driver

This module handles swapping code files in and out of applications for testing
optimizations.
"""
import os
import re
import shutil

from driver_src.driver_models import SwapConfig, FileSwap


def swap_file_in_app(app: dict, swap_config: SwapConfig, temp_dir: str,
                     detect_regions: bool) -> None:
    """Swap files in the application directory on disk.

    For each file in the swap configuration, backs up the original file. If the file
    contains ">>> START EDITABLE REGION" and "<<< END EDITABLE REGION", then replaces
    only that region with the contents of the swap file. Otherwise, replaces the
    entire file.

    Args:
        app: Application configuration dictionary
        swap_config: Swap configuration containing the files to swap in
        temp_dir: Temporary directory where working copy of application directory is located
        detect_regions: Whether to detect if the kernel file to swap into contains editable region
                        markers and substitute into them rather than replacing the entire file
    Raises:
        FileNotFoundError: If a destination file doesn't exist
        IOError: If file operations fail
    """
    for file_swap in swap_config.file_swaps:
        dest_path = os.path.join(temp_dir, file_swap.swap_file_dest_name)
        backup_path = dest_path + ".bak"

        # Skip if file doesn't exist (as per requirement: if no replacement file found, don't swap)
        if not os.path.exists(dest_path):
            continue

        shutil.copy(dest_path, backup_path)

        with open(dest_path, "r", encoding="utf-8") as dest_file:
            dest_text = dest_file.read()

        if not detect_regions or ">>> START EDITABLE REGION" not in dest_text \
            or "<<< END EDITABLE REGION" not in dest_text:
            # Replace entire file with swap file code
            with open(dest_path, "w", encoding="utf-8") as dest_file:
                dest_file.write(file_swap.code)
            continue

        # Replace only the editable region
        start_index = dest_text.find(">>> START EDITABLE REGION")
        end_index = dest_text.find("<<< END EDITABLE REGION")
        dest_text = dest_text[:start_index] + file_swap.code + dest_text[end_index:]

        with open(dest_path, "w", encoding="utf-8") as dest_file:
            dest_file.write(dest_text)


def swap_file_out_app(app: dict, temp_dir: str, swap_config: SwapConfig | None = None) -> None:
    """Swap files out of the application directory on disk.

    Restores the original files from backup and removes the backup files.
    If swap_config is provided, restores only the files specified in the swap config.
    Otherwise, finds all .bak files in the temp_dir and restores them.

    Args:
        app: Application configuration dictionary
        temp_dir: Temporary directory where working copy of application directory is located
        swap_config: Optional swap configuration to know which files to restore

    Raises:
        FileNotFoundError: If a backup file doesn't exist
        IOError: If file operations fail
    """
    if swap_config:
        # Restore only files that were swapped according to the swap config
        for file_swap in swap_config.file_swaps:
            dest_path = os.path.join(temp_dir, file_swap.swap_file_dest_name)
            backup_path = dest_path + ".bak"

            if os.path.exists(backup_path):
                shutil.copy(backup_path, dest_path)
                os.remove(backup_path)
    else:
        # Fallback: restore all .bak files found in temp_dir (for backward compatibility)
        # This handles the case where swap_config is not provided
        dest_path = os.path.join(temp_dir, app.get("kernel_file", ""))
        if dest_path:
            backup_path = dest_path + ".bak"
            if os.path.exists(backup_path):
                shutil.copy(backup_path, dest_path)
                os.remove(backup_path)


def build_swaps_dict(swaps: str, app: str, app_config: dict) -> dict[str, SwapConfig]:
    """Build the swaps dictionary from a directory of swap files.

    Scans the directory for files matching the pattern "*_file_<n>.cu" where n is an index,
    grouped by run_num and optimized_code_num. Each file has a comment on the top line
    indicating which original file it replaces. Files are matched against kernel_file and
    extra_files from the app config.

    Args:
        swaps: Path to the directory containing swap files
        app: Name of the application to build swaps for (or "all")
        app_config: Application configuration dictionary

    Returns:
        Dictionary mapping a unique key to SwapConfig objects

    Raises:
        FileNotFoundError: If the swaps directory doesn't exist
        IOError: If file reading fails
    """
    swaps_dict: dict[str, SwapConfig] = {}

    # Group files by (app_name, run_num, optimized_code_num)
    grouped_files: dict[tuple[str, str, str], list[tuple[str, str]]] = {}

    for root, _, files in os.walk(swaps):
        for file in files:
            run_num = None
            optimized_code_num = None
            app_name = None

            # Try pattern: run_<num>_optimized_code_<num>_file_<n>.cu (new format)
            if match := re.match(r"run_(\d+)_optimized_code_(\d+)_file_(\d+)\.cu$", file):
                run_num = match.group(1)
                optimized_code_num = match.group(2)
                # App name is everything before the last _ in the directory three levels up
                path_parts = root.split("/")
                if len(path_parts) >= 3:
                    app_name = "_".join(path_parts[-3].split("_")[:-1])
                else:
                    continue
            # Try pattern: *_file_<n>.cu (new format, run_num/optimized_code_num in directory)
            elif match := re.match(r".*_file_(\d+)\.cu$", file):
                # Extract run_num and optimized_code_num from directory structure
                path_parts = root.split("/")
                if len(path_parts) >= 3:
                    app_name = "_".join(path_parts[-3].split("_")[:-1])
                else:
                    continue

                # Look for pattern run_<num>_optimized_code_<num> in the path
                for part in path_parts:
                    if match_run := re.search(r"run_(\d+)_optimized_code_(\d+)", part):
                        run_num = match_run.group(1)
                        optimized_code_num = match_run.group(2)
                        break
            # Try old pattern: run_<num>_optimized_code_<num>.cu (backward compatibility)
            elif match := re.match(r"run_(\d+)_optimized_code_(\d+)\.cu$", file):
                run_num = match.group(1)
                optimized_code_num = match.group(2)
                # App name is everything before the last _ in the directory three levels up
                path_parts = root.split("/")
                if len(path_parts) >= 3:
                    app_name = "_".join(path_parts[-3].split("_")[:-1])
                else:
                    continue

            if run_num is None or optimized_code_num is None or app_name is None:
                continue

            full_path = os.path.join(root, file)
            key = (app_name, run_num, optimized_code_num)

            if key not in grouped_files:
                grouped_files[key] = []
            grouped_files[key].append((full_path, file))

    # Process each group to create SwapConfig objects
    for (app_name, run_num, optimized_code_num), file_list in grouped_files.items():
        # Filter by app name if specified
        if app != "all" and app_name != app:
            continue

        # Find the app config for this app
        app_dict = None
        for app_entry in app_config.get("apps", []):
            if app_entry["name"] == app_name:
                app_dict = app_entry
                break

        if app_dict is None:
            continue

        # Get list of files that can be swapped (kernel_file + extra_files)
        swappable_files = [app_dict.get("kernel_file")]
        if "extra_files" in app_dict:
            swappable_files.extend(app_dict["extra_files"])
        swappable_files = [f for f in swappable_files if f]  # Remove None values

        # Process each file in this group
        file_swaps: list[FileSwap] = []
        for full_path, file in file_list:
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    code = f.read()

                # First line contains comment indicating target filename
                first_line = code.splitlines()[0] if code.splitlines() else ""
                # Extract filename from first line (typically at the end)
                # Try different patterns: could be "// filename" or "# filename" or just ends with filename
                target_filename = None

                # Try to extract from comment
                if "//" in first_line:
                    parts = first_line.split("//")
                    if len(parts) > 1:
                        target_filename = parts[-1].strip().split()[-1]
                elif "#" in first_line:
                    parts = first_line.split("#")
                    if len(parts) > 1:
                        target_filename = parts[-1].strip().split()[-1]
                else:
                    # Fallback: last word in first line
                    words = first_line.strip().split()
                    if words:
                        target_filename = words[-1]

                if target_filename:
                    # Check if this target filename is in the swappable files list
                    # Match by basename or full path
                    target_basename = os.path.basename(target_filename)
                    matched = False
                    for swappable_file in swappable_files:
                        swappable_basename = os.path.basename(swappable_file)
                        if target_filename == swappable_file or target_basename == swappable_basename:
                            # Extract code (skip first line and any markdown code fences)
                            rest_of_code = "\n".join([line for line in code.splitlines()[1:]
                                                      if not line.startswith("```")])

                            file_swaps.append(FileSwap(
                                swap_file_src_path=full_path,
                                swap_file_dest_name=swappable_file,
                                code=rest_of_code
                            ))
                            matched = True
                            break
            except (IOError, UnicodeDecodeError) as e:
                # Skip files that can't be read
                continue

        # Only create SwapConfig if we have at least one file swap
        if file_swaps:
            # Use a unique key for this swap config
            config_key = f"{app_name}_run_{run_num}_opt_{optimized_code_num}"
            swaps_dict[config_key] = SwapConfig(
                app_name=app_name,
                file_swaps=file_swaps,
                run_num=run_num,
                optimized_code_num=optimized_code_num
            )

    return swaps_dict
