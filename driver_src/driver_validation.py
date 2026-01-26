#!/usr/bin/env python3
"""
Validation Functions for GPA-Benchmark Driver

This module provides functions for validating application output against
reference outputs using various validation strategies.
"""
import logging
import os
import re
import subprocess

logger = logging.getLogger("GPA-Benchmark")


def _get_test_output(app: dict, result: subprocess.CompletedProcess, temp_dir: str) -> str:
    """Get the test output from file or stdout.

    Args:
        app: Application configuration dictionary
        result: Completed process result from running the application
        temp_dir: Temporary directory where working copy of application directory is located

    Returns:
        The test output as a string
    """
    if "test_output" in app:
        test_output_path = os.path.join(temp_dir, app["test_output"])
        if not os.path.exists(test_output_path):
            logger.warning("Could not find test output file %s for %s, trying stdout instead",
                           app['test_output'], app['name'])
            return result.stdout.decode("utf-8")
        else:
            with open(test_output_path, "r", encoding="utf-8") as test_file:
                return test_file.read()
    else:
        return result.stdout.decode("utf-8")


def validate_app(app: dict, result: subprocess.CompletedProcess, temp_dir: str) -> bool:
    """Validate the application output.

    Supports multiple validation types:
    - fail_check_text: Returns False if stdout contains the fail check text
    - pass_check_text: Returns True if stdout contains the pass check text
    - reference_output: Compares stdout (or test_output file) to reference output
    - float_grep: Locates a float in output and compares to reference within tolerance
    - output_window: Compares a window of lines from the output

    Args:
        app: Application configuration dictionary
        result: Completed process result from running the application
        temp_dir: Temporary directory where working copy of application directory is located

    Returns:
        True if validation succeeds, False otherwise

    Raises:
        ValueError: If no validation type is specified or if validation configuration is invalid
    """
    stdout_text = result.stdout.decode("utf-8")

    # Check for fail text
    if "fail_check_text" in app:
        return app["fail_check_text"] not in stdout_text

    # Check for pass text
    if "pass_check_text" in app:
        return app["pass_check_text"] in stdout_text

    # Compare to reference output
    if "reference_output" in app:
        with open(os.path.join(temp_dir, app["reference_output"]), "r",
                  encoding="utf-8") as ref_file:
            ref_output = ref_file.read()

        test_output = _get_test_output(app, result, temp_dir)

        # Exact match
        if test_output == ref_output:
            return True

        # Window comparison
        if "output_window" in app and app["output_window"] != 0:
            return validate_output_window(test_output, app, ref_output)

        # Float comparison
        if "float_grep" in app:
            return validate_float(test_output, app, ref_output)

        # No match and no special validation
        return False

    raise ValueError(f"No validation type specified for {app['name']}")


def validate_output_window(test_output: str, app: dict, ref_output: str) -> bool:
    """Validate the output using a window of lines.

    Compares a specific range of lines from the test output to the reference output.

    Args:
        test_output: The test output string
        app: Application configuration dictionary containing "output_window" key
        ref_output: The reference output string

    Returns:
        True if the window matches, False otherwise

    Raises:
        ValueError: If output_window is not a list of two integers
    """
    window_sizes = app["output_window"]
    if len(window_sizes) != 2:
        raise ValueError(f"Output window must be a list of two integers ({app['name']})")

    test_lines = test_output.splitlines()
    ref_lines = ref_output.splitlines()

    test_window = "\n".join(test_lines[window_sizes[0]:window_sizes[1]])
    ref_window = "\n".join(ref_lines[window_sizes[0]:window_sizes[1]])

    return test_window == ref_window


def validate_float(test_output: str, app: dict, ref_output: str) -> bool:
    """Validate a float value in the test output.

    Locates a float value after a specific search string and compares it to the
    reference output within a specified tolerance.

    Args:
        test_output: The test output string
        app: Application configuration dictionary containing "float_grep" and "float_tolerance" keys
        ref_output: The reference output string

    Returns:
        True if the float values match within tolerance, False otherwise

    Raises:
        ValueError: If the reference output doesn't contain the expected float pattern
    """
    float_grep = app["float_grep"]
    tolerance = app["float_tolerance"]

    # Find float in test output
    test_portion = test_output.split(float_grep)[-1] if float_grep in test_output else ""
    float_match = re.search(r"(\d+\.\d+)", test_portion)

    # Find float in reference output
    ref_portion = ref_output.split(float_grep)[-1] if float_grep in ref_output else ""
    ref_match = re.search(r"(\d+\.\d+)", ref_portion)

    if not ref_match:
        raise ValueError(f"Reference output {app['reference_output']} does not contain " \
                        + f"float pattern after '{float_grep}'")

    if not float_match:
        logger.warning("No float found in test output for %s, looking for pattern after '%s'",
                       app['name'], float_grep)
        return False

    float_value = float(float_match.group(1))
    ref_value = float(ref_match.group(1))

    return abs(float_value - ref_value) <= tolerance
