"""Reporting Functions for GPA-Benchmark Driver.

This module provides functions for displaying results and saving output to files.
"""
import json
import logging
import os
from pathlib import Path

from gpa_bench_driver.driver_src.driver_models import AppResults, DriverPassResult, Operation

logger = logging.getLogger("GPA-Benchmark")


def print_report_table(results: dict[str, AppResults], operations: list[Operation]) -> None:
    """Print a formatted table showing the status of all operations for each application.

    Args:
        results: Dictionary mapping application names to AppResults objects
        operations: List of operations to display in the table

    """
    if not results:
        return

    # Determine column widths
    app_name_width = max(len(app_name) for app_name in results)
    app_name_width = max(app_name_width, len("Application"))
    col_width = max(len(op.value) for op in operations) if operations else 10
    col_width = max(col_width, 8)

    # Print header
    header = f"{'Application':<{app_name_width}}"
    for op in operations:
        header += f" | {op.value:<{col_width}}"
    header_str = "\n" + "=" * len(header) + "\n" + header + "\n" + "=" * len(header)
    for line in header_str.split("\n"):
        logger.info(line)

    # Print rows
    for app_name, app_results in results.items():
        row = f"{app_name:<{app_name_width}}"
        for op in operations:
            status = app_results.get(op)
            if status is True:
                symbol = "✓"
            elif status is False:
                symbol = "✗"
            elif isinstance(status, str):
                symbol = status
            else:
                symbol = "-"
            row += f" | {symbol:<{col_width}}"
        logger.info(row)

    logger.info("=" * len(header))
    logger.info("")


def save_results(
    long_results: dict[str, list[DriverPassResult]], output_file: os.PathLike,
) -> None:
    """Save the long results to a JSON file.

    Args:
        long_results: Dictionary mapping application names to lists of DriverPassResult objects
        output_file: Path to the output JSON file

    """
    # Flatten the results into a single list
    all_results = []
    for results in long_results.values():
        all_results.extend([r.to_dict() for r in results])

    with Path(output_file).open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
