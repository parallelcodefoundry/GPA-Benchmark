#!/usr/bin/env python3
"""
Reporting Functions for GPA-Benchmark Driver

This module provides functions for displaying results and saving output to files.
"""
import json

from driver_models import AppResults, Operation, DriverPassResult


def print_report_table(results: dict[str, AppResults], operations: list[Operation]) -> None:
    """Print a formatted table showing the status of all operations for each application.

    Args:
        results: Dictionary mapping application names to AppResults objects
        operations: List of operations to display in the table
    """
    if not results:
        return

    # Determine column widths
    app_name_width = max(len(app_name) for app_name in results.keys())
    app_name_width = max(app_name_width, len("Application"))
    col_width = max(len(op.value) for op in operations) if operations else 10
    col_width = max(col_width, 8)

    # Print header
    header = f"{'Application':<{app_name_width}}"
    for op in operations:
        header += f" | {op.value:<{col_width}}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

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
        print(row)

    print("=" * len(header))
    print()


def save_results(long_results: dict[str, list[DriverPassResult]], output_file: str) -> None:
    """Save the long results to a JSON file.

    Args:
        long_results: Dictionary mapping application names to lists of DriverPassResult objects
        output_file: Path to the output JSON file

    Raises:
        IOError: If the file cannot be written
    """
    # Flatten the results into a single list
    all_results = []
    for results in long_results.values():
        for result in results:
            all_results.append(result.to_dict())

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
