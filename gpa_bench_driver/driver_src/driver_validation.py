"""
Validation Functions for GPA-Benchmark Driver

This module provides functions for validating application output against
reference outputs using various validation strategies.
"""
import difflib
import logging
import os
import re
import subprocess

logger = logging.getLogger("GPA-Benchmark")


def _get_test_output(app: dict, result: subprocess.CompletedProcess, temp_dir: os.PathLike) -> str:
    """Get the test output from file or stdout.

    For apps that output to stdout (reference_output present, no test_output), the output is read
    from the result stdout field.  For apps with an explicit test_output file the named file is
    used.  Stdout capture is used as a fallback when a named test_output file is expected but
    missing.

    Args:
        app: Application configuration dictionary
        result: Completed process result from running the application
        temp_dir: Temporary directory where working copy of application directory is located

    Returns:
        The test output as a string
    """
    # TODO: Evaluate if this check should be replaced with calling stdout_uses_file
    if "test_output" in app:
        test_output_path = os.path.join(temp_dir, app["test_output"])
        if not os.path.exists(test_output_path):
            logger.warning("Could not find test output file %s for %s, trying stdout instead",
                           app['test_output'], app['name'])
        else:
            with open(test_output_path, "r", encoding="utf-8") as test_file:
                return test_file.read()
    return result.stdout.decode("utf-8") if result.stdout is not None else ""


def validate_app(app: dict, result: subprocess.CompletedProcess,
                 temp_dir: os.PathLike) -> tuple[bool, str | None]:
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
        Tuple of (success, validation_output). success is True if validation passes.
        validation_output is a diagnostic string on failure (e.g. diff, found float); None on
        success.

    Raises:
        ValueError: If no validation type is specified or if validation configuration is invalid
    """
    stdout_text = result.stdout.decode("utf-8") if result.stdout is not None else ""

    # Check for fail text
    if "fail_check_text" in app:
        fail_text = app["fail_check_text"]
        if fail_text in stdout_text:
            return False, f"Fail check text '{fail_text}' was found in output."
        return True, None

    # Check for pass text
    if "pass_check_text" in app:
        pass_text = app["pass_check_text"]
        if pass_text not in stdout_text:
            return False, f"Pass check text '{pass_text}' was not found in output."
        return True, None

    # Compare to reference output
    if "reference_output" in app:
        with open(os.path.join(temp_dir, app["reference_output"]), "r",
                  encoding="utf-8") as ref_file:
            ref_output = ref_file.read()

        test_output = _get_test_output(app, result, temp_dir)

        # Exact match
        if test_output == ref_output:
            return True, None

        # Window comparison
        if "output_window" in app and app["output_window"] != 0:
            return validate_output_window(test_output, app, ref_output)

        # Float comparison
        if "float_grep" in app:
            return validate_float(test_output, app, ref_output)

        # No match and no special validation: produce diff
        diff_lines = difflib.unified_diff(
            ref_output.splitlines(keepends=True),
            test_output.splitlines(keepends=True),
            fromfile="reference",
            tofile="test",
            lineterm=""
        )
        diff_text = "".join(diff_lines)
        return False, f"Output did not match reference (exact match). Diff:\n{diff_text}"

    raise ValueError(f"No validation type specified for {app['name']}")


def validate_output_window(test_output: str, app: dict, ref_output: str) -> tuple[bool, str | None]:
    """Validate the output using a window of lines.

    Compares a specific range of lines from the test output to the reference output.

    Args:
        test_output: The test output string
        app: Application configuration dictionary containing "output_window" key
        ref_output: The reference output string

    Returns:
        Tuple of (success, validation_output). validation_output is diagnostic text on failure.

    Raises:
        ValueError: If output_window is not a list of two integers
    """
    window_sizes = app["output_window"]
    if len(window_sizes) != 2:
        raise ValueError(f"Output window must be a list of two integers ({app['name']})")

    test_lines = test_output.splitlines()
    ref_lines = ref_output.splitlines()

    start, end = window_sizes[0], window_sizes[1]
    test_window = "\n".join(test_lines[start:end])
    ref_window = "\n".join(ref_lines[start:end])

    if test_window == ref_window:
        return True, None

    diff_lines = difflib.unified_diff(
        ref_window.splitlines(keepends=True),
        test_window.splitlines(keepends=True),
        fromfile=f"reference (lines {start}:{end})",
        tofile=f"test (lines {start}:{end})",
        lineterm=""
    )
    diff_text = "".join(diff_lines)
    return False, f"Output window [{start}:{end}] did not match.\n{diff_text}"


def validate_float(test_output: str, app: dict, ref_output: str) -> tuple[bool, str | None]:
    """Validate a float value in the test output.

    Locates a float value after a specific search string and compares it to the
    reference output within a specified tolerance.

    Args:
        test_output: The test output string
        app: Application configuration dictionary containing "float_grep" and "float_tolerance" keys
        ref_output: The reference output string

    Returns:
        Tuple of (success, validation_output). validation_output is diagnostic text on failure.

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

    ref_value = float(ref_match.group(1))

    if not float_match:
        logger.warning("No float found in test output for %s, looking for pattern after '%s'",
                       app['name'], float_grep)
        return False, (
            f"No float found in test output after '{float_grep}'. "
            f"Reference value (from reference output): {ref_value}."
        )

    float_value = float(float_match.group(1))
    if abs(float_value - ref_value) <= tolerance:
        return True, None

    return False, (
        f"Float comparison failed. Expected (reference): {ref_value}, got (test): {float_value}, "
        f"tolerance: {tolerance}, difference: {abs(float_value - ref_value)}."
    )
