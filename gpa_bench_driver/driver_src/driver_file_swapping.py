"""File Swapping Operations for GPA-Benchmark Driver.

This module handles swapping code files in and out of applications for testing optimizations.
"""

import logging
import os
import re
import shutil
from pathlib import Path

from gpa_bench_driver.driver_src.driver_models import FileSwap, SwapConfig

logger = logging.getLogger("GPA-Benchmark")


def swap_file_in_app(swap_config: SwapConfig, temp_dir: Path, *, detect_regions: bool) -> None:
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
        dest_path = temp_dir / file_swap.swap_file_dest_name
        backup_path = Path(str(dest_path) + ".bak")

        # Always create a backup just in case, swap out function will always expect it to exist
        shutil.copy(dest_path, backup_path)

        # Skip if file doesn't exist (as per requirement: if no replacement file found, don't swap)
        if not dest_path.exists():
            continue

        with dest_path.open("r", encoding="utf-8") as dest_file:
            dest_text = dest_file.read()

        if (
            not detect_regions
            or ">>> START EDITABLE REGION" not in dest_text
            or "<<< END EDITABLE REGION" not in dest_text
        ):
            # Replace entire file with swap file code
            with dest_path.open("w", encoding="utf-8") as dest_file:
                dest_file.write(file_swap.code)
            logger.debug(
                "Replaced entire file %s with swap file %s",
                dest_path,
                file_swap.swap_file_src_path,
            )
            continue

        # Replace only the editable region
        start_index = dest_text.find(">>> START EDITABLE REGION")
        end_index = dest_text.find("<<< END EDITABLE REGION")
        dest_text = dest_text[:start_index] + file_swap.code + dest_text[end_index:]

        with dest_path.open("w", encoding="utf-8") as dest_file:
            dest_file.write(dest_text)
        logger.debug(
            "Replaced editable region in %s with swap file %s",
            dest_path,
            file_swap.swap_file_src_path,
        )


def swap_file_out_app(
    app: dict,
    temp_dir: Path,
    swap_config: SwapConfig | None = None,
) -> None:
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
            dest_path = temp_dir / file_swap.swap_file_dest_name
            backup_path = Path(str(dest_path) + ".bak")
            if backup_path.exists():
                shutil.copy(backup_path, dest_path)
                backup_path.unlink()
                logger.debug("Restored file %s", dest_path)
    else:
        # Fallback: restore all .bak files found in temp_dir (for backward compatibility)
        # This handles the case where swap_config is not provided
        dest_path = temp_dir / app.get("kernel_file", "")
        if dest_path:
            backup_path = Path(str(dest_path) + ".bak")
            if backup_path.exists():
                shutil.copy(backup_path, dest_path)
                backup_path.unlink()
                logger.debug("Restored file %s", dest_path)
        else:
            msg = f"No kernel file ({dest_path}) found for {app.get('name')}"
            raise FileNotFoundError(msg)


def _try_match_filename(
    target_filename: str,
    curr_filename: str,
    root: Path,
) -> tuple[str | None, int | None, str | None, int | None]:
    """Try to match the filename to the target filename.

    Args:
        target_filename: The target filename to match
        curr_filename: The current filename to match
        root: The root directory of the current filename

    Returns:
        The app name, run number, metadata, and optimized code number if the filename matches the
        target, otherwise None, None, None, None

    """
    if match := re.match(target_filename, curr_filename):
        path_parts = root.parts
        if len(path_parts) >= 3:  # noqa: PLR2004
            app_name = "_".join(path_parts[-3].split("_")[:-1])
        else:
            return None, None, None, None

        return app_name, int(match.group(1)), match.group(2), int(match.group(3))
    return None, None, None, None


def _extract_path_metadata(root: Path, app_name: str) -> str | None:
    """Extract metadata from path.

    Uses folder names from root (inclusive) up to AgenticAnalyzer (exclusive). Drops app name if
    found in any path part, as well as "." and ".." parts.

    Args:
        root: Full path to the current directory (e.g.
              .../gpa-bench-nodr/backprop_gpt-oss-120b/AgenticAnalyzer/run_0)
        app_name: The name of the application to drop if found in any path part

    Returns:
        Joined folder names as a string, or None if there are no such folders

    """
    parts = [p for p in root.parts if p and p not in [".", ".."]]
    try:
        agentic_idx = parts.index("AgenticAnalyzer")
    except ValueError:
        agentic_idx = len(parts)
    path_meta_parts = parts[:agentic_idx]
    for i, part in enumerate(path_meta_parts):
        if app_name in part:
            path_meta_parts[i] = part.replace(app_name, "").strip("_-.")
    if not path_meta_parts:
        return None
    return "_".join(path_meta_parts).strip().lower()


def _find_grouped_files(
    swaps: Path,
) -> dict[tuple[str, int, str | None, int], list[tuple[Path, str]]]:
    """Find the grouped files in the swaps directory.

    Args:
        swaps: Path to the directory containing swap files

    Returns:
        Dictionary mapping a unique key to a list of tuples containing the full path and code
        of the swap files

    """
    logger.debug("Entering _find_grouped_files")
    grouped_files: dict[tuple[str, int, str | None, int], list[tuple[Path, str]]] = {}

    for root, _, files in swaps.walk():
        for file in files:
            run_num = None
            optimized_code_num = None
            app_name = None

            # Try pattern: run_<num>_optimized_code_<metadata><num>_file_<n>.cu (new format)
            app_name, run_num, metadata, optimized_code_num = _try_match_filename(
                r"run_(\d+)_optimized_code_([a-z|_]*)(\d+)_file_(\d+)\.cu$",
                file,
                root,
            )

            if app_name is None or run_num is None or optimized_code_num is None:
                continue

            full_path = root / file
            # Path-based metadata: folder names from swaps root up to AgenticAnalyzer (exclusive)
            path_metadata = _extract_path_metadata(root, app_name)
            if metadata == "":
                metadata = "keet"  # TODO(jhdavis): Update KEET code to put this in for us
            elif metadata is not None:
                metadata = metadata.strip().lower()
            # Combine path metadata with filename-derived metadata for keying
            if path_metadata is not None:
                metadata = f"{path_metadata}_{metadata}".strip() if metadata else path_metadata
            if metadata == "":
                metadata = None
            key = (app_name, run_num, metadata, optimized_code_num)

            with full_path.open("r", encoding="utf-8") as f:
                code = f.read()

            if key not in grouped_files:
                grouped_files[key] = []
            grouped_files[key].append((full_path, code))

    return grouped_files


def _get_swappable_files(app_config: dict, app_name: str) -> list[Path]:
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
    return [Path(f) for f in swappable_files if f]  # Remove None values


def build_swaps_dict(
    swaps: Path | dict[tuple[str, int, str | None, int], list[tuple[Path, str]]],
    app: str,
    app_config: dict,
) -> dict[str, SwapConfig]:
    """Build the swaps dictionary from a directory of swap files.

    Scans the directory for files matching the pattern "*_file_<n>.cu" where n is an index,
    grouped by run_num and optimized_code_num. Each file has a comment on the top line
    indicating which original file it replaces. Files are matched against kernel_file and
    extra_files from the app config.

    Args:
        swaps: Path to the directory containing swap files or a dictionary of files grouped by
               (app_name, run_num, metadata_str, optimized_code_num)
        app: Name of the application to build swaps for (or "all")
        app_config: Application configuration dictionary

    Returns:
        Dictionary mapping a unique key to SwapConfig objects

    Raises:
        FileNotFoundError: If the swaps directory doesn't exist
        IOError: If file reading fails

    """
    logger.debug("Entering build_swaps_dict")
    grouped_files = _find_grouped_files(swaps) if isinstance(swaps, os.PathLike) else swaps
    return _build_swaps_dict_from_grouped_files(grouped_files, app_config, app)


def _extract_target_filename(first_line: str) -> Path | None:
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
            return Path(parts[-1].strip().split()[-1])
    elif "#" in first_line:
        parts = first_line.split("#")
        if len(parts) > 1:
            return Path(parts[-1].strip().split()[-1])
    else:
        # Fallback: last word in first line
        words = first_line.strip().split()
        if words:
            return Path(words[-1])
    return None


def _find_file_swap(
    target_filename: Path | None,
    swappable_files: list[Path],
    code: str,
    full_path: Path,
) -> FileSwap | None:
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

    target_basename = target_filename.name
    for swappable_file in swappable_files:
        swappable_basename = swappable_file.name
        if target_filename == swappable_file or target_basename == swappable_basename:
            # Extract code (skip first line and any markdown code fences)
            rest_of_code = "\n".join(
                [line for line in code.splitlines()[1:] if not line.startswith("```")],
            )

            return FileSwap(
                swap_file_src_path=full_path,
                swap_file_dest_name=swappable_file,
                code=rest_of_code,
            )

    return None


def _build_swaps_dict_from_grouped_files(
    grouped_files: dict[tuple[str, int, str | None, int], list[tuple[Path, str]]],
    app_config: dict,
    app: str,
) -> dict[str, SwapConfig]:
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
    for (app_name, run_num, metadata, optimized_code_num), file_list in grouped_files.items():
        # Filter by app name if specified
        if app not in ("all", app_name):
            continue

        # Get the list of files that can be swapped for this app
        swappable_files = _get_swappable_files(app_config, app_name)
        if not swappable_files:
            continue

        # Process each file in this group
        file_swaps: list[FileSwap] = []
        for full_path, code in file_list:
            # Extract the target filename from the first line of the code
            target_filename = _extract_target_filename(
                code.splitlines()[0] if code.splitlines() else "",
            )

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
                metadata=metadata,
            )
        else:
            logger.warning(
                "No files to swap for %s run %s optimized code %s in %s",
                app_name,
                run_num,
                optimized_code_num,
                [full_path for full_path, _ in file_list],
            )

    return swaps_dict
