"""
File Swapping Operations for GPA-Benchmark Driver

This module handles swapping code files in and out of applications for testing
optimizations.
"""
import logging
import os
import re
import shutil

from gpa_bench_driver.driver_src.driver_models import SwapConfig, FileSwap

logger = logging.getLogger("GPA-Benchmark")


def swap_file_in_app(swap_config: SwapConfig, temp_dir: str, detect_regions: bool) -> None:
    """Swap files in the application directory on disk.

    For each file in the swap configuration, backs up the original file. If the file
    contains ">>> START EDITABLE REGION" and "<<< END EDITABLE REGION", then replaces
    only that region with the contents of the swap file. Otherwise, replaces the
    entire file.

    Args:
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
            logger.debug("Replaced entire file %s with swap file %s", dest_path,
                         file_swap.swap_file_src_path)
            continue

        # Replace only the editable region
        start_index = dest_text.find(">>> START EDITABLE REGION")
        end_index = dest_text.find("<<< END EDITABLE REGION")
        dest_text = dest_text[:start_index] + file_swap.code + dest_text[end_index:]

        with open(dest_path, "w", encoding="utf-8") as dest_file:
            dest_file.write(dest_text)
        logger.debug("Replaced editable region in %s with swap file %s", dest_path,
                     file_swap.swap_file_src_path)


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
                raise FileNotFoundError(f"No backup file ({backup_path}) found for {dest_path}")
            logger.debug("Restored file %s", dest_path)
    else:
        # Fallback: restore all .bak files found in temp_dir (for backward compatibility)
        # This handles the case where swap_config is not provided
        dest_path = os.path.join(temp_dir, app.get("kernel_file", ""))
        if dest_path:
            backup_path = dest_path + ".bak"
            if os.path.exists(backup_path):
                shutil.copy(backup_path, dest_path)
                os.remove(backup_path)
            else:
                raise FileNotFoundError(f"No backup file ({backup_path}) found for {dest_path}")
            logger.debug("Restored file %s", dest_path)
        else:
            raise FileNotFoundError(f"No kernel file ({dest_path}) found for {app.get('name')}")


def _try_match_filename(target_filename: str, curr_filename: str, root: str,
                        partial_target_filename: str | None = None) -> tuple[str | None, str | None,
                                                                             int | None,
                                                                             int | None]:
    """Try to match the filename to the target filename.

    Args:
        target_filename: The target filename to match
        curr_filename: The current filename to match
        root: The root directory of the current filename
        partial_target_filename: The partial target filename to match, if provided it will be used
                                 to match the run_num and optimized_code_num in the directory
                                 structure

    Returns:
        The app name, metadata string, run number, and optimized code number if the filename matches
        the target
        None, None, None, None if the filename does not match the target filename(s) specified
    """
    if match := re.match(target_filename, curr_filename):
        path_parts = root.split("/")
        if len(path_parts) >= 3:
            app_name = "_".join(path_parts[-3].split("_")[:-1])
        else:
            return None, None, None, None

        if partial_target_filename:
            for part in path_parts:
                if match_run := re.search(partial_target_filename, part):
                    return app_name, match_run.group(1), int(match_run.group(2)), \
                        int(match_run.group(3))
            return None, None, None, None
        return app_name, match.group(1), int(match.group(2)), int(match.group(3))
    return None, None, None, None


def _find_grouped_files(swaps: str) -> dict[tuple[str, str | None, int, int],
                                            list[tuple[str, str]]]:
    """Find the grouped files in the swaps directory.

    Args:
        swaps: Path to the directory containing swap files

    Returns:
        Dictionary mapping a unique key to a list of tuples containing the full path and code
        of the swap files
    """
    grouped_files: dict[tuple[str, str | None, int, int], list[tuple[str, str]]] = {}

    for root, _, files in os.walk(swaps):
        for file in files:
            run_num = None
            optimized_code_num = None
            app_name = None

            # Try pattern: run_<num>_optimized_code_<metadata><num>_file_<n>.cu (new format)
            app_name, metadata_str, run_num, optimized_code_num = \
                _try_match_filename(r"run_(\d+)_optimized_code_([a-z|_]*)(\d+)_file_(\d+)\.cu$",
                                    file, root)
            if app_name is None or run_num is None or optimized_code_num is None:
                # Try pattern: *_file_<n>.cu (new format, run_num/optimized_code_num in directory)
                app_name, metadata_str, run_num, optimized_code_num = \
                    _try_match_filename(r".*optimized_code_([a-z|_]*)(\d+)_file_(\d+)\.cu$", file,
                                        root,
                                        partial_target_filename=r"run_(\d+)_optimized_code_(\d+)")
                if app_name is None or run_num is None or optimized_code_num is None:
                    # Try old pattern: run_<num>_optimized_code_<num>.cu (backward compatibility)
                    app_name, metadata_str, run_num, optimized_code_num = \
                        _try_match_filename(r"run_(\d+)_optimized_code_([a-z|_]*)(\d+)\.cu$", file,
                                            root)
                    if app_name is None or run_num is None or optimized_code_num is None:
                        continue

            full_path = os.path.join(root, file)
            if metadata_str == "":
                metadata_str = None
            elif metadata_str is not None:
                metadata_str = metadata_str.replace("_", " ").strip().lower()
            key = (app_name, metadata_str, run_num, optimized_code_num)

            with open(full_path, "r", encoding="utf-8") as f:
                code = f.read()

            if key not in grouped_files:
                grouped_files[key] = []
            grouped_files[key].append((full_path, code))

    return grouped_files


def _get_swappable_files(app_config: dict, app_name: str) -> list[str]:
    """Get the list of files that can be swapped for an app.

    Args:
        app_config: The application configuration dictionary
        app_name: The name of the application

    Returns:
        The list of files that can be swapped for the app
    """
    # Find the app config for this app
    app_dict = None
    for app_entry in app_config.get("apps", []):
        if app_entry["name"] == app_name:
            app_dict = app_entry
            break

    if app_dict is None:
        return []

    # Get list of files that can be swapped (kernel_file + extra_files)
    swappable_files = [app_dict.get("kernel_file")]
    if "extra_files" in app_dict:
        swappable_files.extend(app_dict["extra_files"])
    swappable_files = [f for f in swappable_files if f]  # Remove None values

    return swappable_files


def build_swaps_dict(swaps: str | dict[tuple[str, str | None, int, int], list[tuple[str, str]]],
                     app: str, app_config: dict) -> dict[str, SwapConfig]:
    """Build the swaps dictionary from a directory of swap files.

    Scans the directory for files matching the pattern "*_file_<n>.cu" where n is an index,
    grouped by run_num and optimized_code_num. Each file has a comment on the top line
    indicating which original file it replaces. Files are matched against kernel_file and
    extra_files from the app config.

    Args:
        swaps: Path to the directory containing swap files or a dictionary of files grouped by
               (app_name, metadata_str, run_num, optimized_code_num)
        app: Name of the application to build swaps for (or "all")
        app_config: Application configuration dictionary

    Returns:
        Dictionary mapping a unique key to SwapConfig objects

    Raises:
        FileNotFoundError: If the swaps directory doesn't exist
        IOError: If file reading fails
    """
    if isinstance(swaps, str):
        grouped_files = _find_grouped_files(swaps)
    else:
        grouped_files = swaps
    return _build_swaps_dict_from_grouped_files(grouped_files, app_config, app)


def _extract_target_filename(first_line: str) -> str | None:
    """Try to extract the target filename from the first line of the code.

    Args:
        first_line: The first line of the code

    Returns:
        The target filename if it is found
    """
    # Try to extract from comment
    if "//" in first_line:
        parts = first_line.split("//")
        if len(parts) > 1:
            return parts[-1].strip().split()[-1]
    elif "#" in first_line:
        parts = first_line.split("#")
        if len(parts) > 1:
            return parts[-1].strip().split()[-1]
    else:
        # Fallback: last word in first line
        words = first_line.strip().split()
        if words:
            return words[-1]
    return None


def _find_file_swap(target_filename: str | None, swappable_files: list[str], code: str,
                    full_path: str) -> FileSwap | None:
    """Find the file swap for the target filename.

    Args:
        target_filename: The target filename to match
        swappable_files: The list of files that can be swapped
        code: The code of the file to swap
        full_path: The full path to the file to swap

    Returns:
        The FileSwap object if it is found
    """
    if target_filename is None:
        return None

    target_basename = os.path.basename(target_filename)
    for swappable_file in swappable_files:
        swappable_basename = os.path.basename(swappable_file)
        if target_filename == swappable_file or target_basename == swappable_basename:
            # Extract code (skip first line and any markdown code fences)
            rest_of_code = "\n".join([line for line in code.splitlines()[1:]
                                      if not line.startswith("```")])

            return FileSwap(
                swap_file_src_path=full_path,
                swap_file_dest_name=swappable_file,
                code=rest_of_code
            )

    return None


def _build_swaps_dict_from_grouped_files(grouped_files: dict[tuple[str, str | None, int, int],
                                                             list[tuple[str, str]]],
                                        app_config: dict, app: str) -> dict[str, SwapConfig]:
    """Build the swaps dictionary from a dictionary of grouped files.

    Args:
        grouped_files: The dictionary of grouped files
        app_config: The application configuration dictionary
        app: The name of the application to build swaps for (or "all")

    Returns:
        Dictionary mapping a unique key to SwapConfig objects
    """
    swaps_dict: dict[str, SwapConfig] = {}

    # Process each group to create SwapConfig objects
    for (app_name, metadata, run_num, optimized_code_num), file_list in grouped_files.items():
        # Filter by app name if specified
        if app != "all" and app_name != app:
            continue

        # Get the list of files that can be swapped for this app
        swappable_files = _get_swappable_files(app_config, app_name)
        if not swappable_files:
            continue

        # Process each file in this group
        file_swaps: list[FileSwap] = []
        for full_path, code in file_list:
            # Extract the target filename from the first line of the code
            target_filename = _extract_target_filename(code.splitlines()[0]
                                                       if code.splitlines() else "")

            # Find the file swap for the target filename
            file_swap = _find_file_swap(target_filename, swappable_files, code, full_path)
            if file_swap:
                file_swaps.append(file_swap)
            else:
                logger.warning("No file swap found for %s in %s", target_filename, full_path)

        # Only create SwapConfig if we have at least one file swap
        if file_swaps:
            # Use a unique key for this swap config
            metadata_str = metadata if metadata is not None else "none"
            config_key = f"{app_name}_run_{run_num}_opt_{optimized_code_num}_meta_{metadata_str}"
            swaps_dict[config_key] = SwapConfig(
                app_name=app_name,
                file_swaps=file_swaps,
                run_num=str(run_num),
                optimized_code_num=str(optimized_code_num),
                metadata=metadata
            )
        else:
            logger.warning("No files to swap for %s run %s optimized code %s in %s",
                           app_name, run_num, optimized_code_num,
                           [full_path for full_path, _ in file_list])

    return swaps_dict
