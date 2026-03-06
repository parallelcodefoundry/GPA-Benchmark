"""Utility Functions for GPA-Benchmark Driver.

This module provides utility functions for subprocess execution, path resolution,
directory setup, and system detection. Subprocess results are transferred via
disk (files written by the worker, read by the parent); no multiprocessing
queues are used.
"""

import faulthandler
import logging
import multiprocessing
import os
import signal
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger("GPA-Benchmark")

# Filenames used inside the per-run result directory for disk IPC
_RESULT_RETURNCODE_FILE = "returncode.txt"
_RESULT_STDOUT_FILE = "stdout.bin"
_RESULT_STDERR_FILE = "stderr.bin"

MAX_OUTPUT_CHAR_LIMIT = 25000


class SubprocessRunner:
    """Configured subprocess executor.

    Bundles the execution environment and logging/timeout/truncation settings so they are specified
    once at construction and do not need to be threaded through every call site.

    Attributes:
        env: Environment variables dictionary passed to every subprocess
        log_level: Logging level controlling when stdout/stderr are emitted
            - DEBUG: Always log stdout and stderr
            - INFO: Log stdout and stderr on failure (except when quiet=True)
            - WARNING/ERROR/CRITICAL: Never log stdout or stderr
        quiet: Default quiet flag; suppresses INFO-level output on failure when True
        timeout: Default timeout in seconds; None means no limit
        output_char_limit: Maximum characters logged for stdout/stderr; characters are removed from
            the middle to stay within the limit.  Set to <= 0 to disable truncation.
        suppress_command_stdout: When True, never log stdout/stderr from commands (overrides
            log_level and quiet). Driver logging is unchanged.
        use_srun: When True, prepend Slurm srun to all commands and enforce timeout via
            srun --time=00:n, bypassing multiprocessing-based timeout handling.

    """

    def __init__(
        self,
        *,
        env: dict,
        log_level: str = "WARNING",
        quiet: bool = False,
        timeout: int | None = None,
        output_char_limit: int = MAX_OUTPUT_CHAR_LIMIT,
        suppress_command_stdout: bool = False,
        use_srun: bool = False,
    ) -> None:
        """Initialize the SubprocessRunner.

        Args:
            env: Environment variables dictionary passed to every subprocess
            log_level: Logging level controlling when stdout/stderr are emitted
            quiet: Default quiet flag; suppresses INFO-level output on failure when True
            timeout: Default timeout in seconds; None means no limit
            output_char_limit: Maximum characters logged for stdout/stderr; characters are removed
                from the middle to stay within the limit.  Set to <= 0 to disable truncation.
            suppress_command_stdout: When True, never log stdout/stderr from commands (overrides
                log_level and quiet). Driver logging is unchanged.
            use_srun: When True, prepend Slurm srun to all commands and enforce timeout via
                srun --time=00:n, bypassing multiprocessing-based timeout handling.

        """
        self.env = env
        self.log_level = log_level
        self.quiet = quiet
        self.timeout = timeout
        self.output_char_limit = output_char_limit
        self.suppress_command_stdout = suppress_command_stdout
        self.use_srun = use_srun

    def run(
        self,
        command: list[str],
        cwd: os.PathLike,
        *,
        quiet: bool | None = None,
    ) -> subprocess.CompletedProcess:
        """Execute a command and return the completed process.

        Stdout and stderr are always buffered through temporary files in the worker process to
        avoid deadlocking on the pipe buffer in subprocess.run.  The content read back (up to
        output_char_limit bytes) is returned as bytes in the CompletedProcess attributes, so
        callers are unaffected.

        When a timeout is configured, the command runs in a dedicated child process so the driver
        cannot hang if subprocess.run itself blocks.  The child is forcibly terminated (SIGKILL) if
        it does not finish in time.

        Args:
            command: Command to run as a list of strings
            cwd: Working directory for the command
            quiet: Per-call quiet override.  When provided, takes precedence over the instance
                -level quiet setting.  Useful for suppressing output on a single call (e.g. the
                clean step) without changing the runner.

        Returns:
            CompletedProcess object with returncode, stdout, and stderr attributes.

        """
        faulthandler.enable()
        effective_quiet = self.quiet if quiet is None else quiet
        log_level = self.log_level
        timeout = self.timeout

        if self.use_srun:
            # Prepend srun and use --time=00:n for timeout; run directly (no multiprocessing).
            srun_cmd = ["srun"]
            if timeout is not None:
                srun_cmd.append(f"--time=00:{timeout}")
            full_command = srun_cmd + command
            logger.debug("Running command %s in directory %s", " ".join(full_command), cwd)

            result_dir = Path(tempfile.mkdtemp(prefix="gpa_bench_result_"))
            try:
                stdout_path = result_dir / _RESULT_STDOUT_FILE
                stderr_path = result_dir / _RESULT_STDERR_FILE
                with stdout_path.open("wb") as out_f, stderr_path.open("wb") as err_f:
                    proc = subprocess.run(  # noqa: S603
                        full_command,
                        cwd=cwd,
                        env=self.env,
                        check=False,
                        stdout=out_f,
                        stderr=err_f,
                    )
                returncode = proc.returncode
                with stdout_path.open("rb") as f:
                    stdout_bytes = f.read()
                with stderr_path.open("rb") as f:
                    stderr_bytes = f.read()
                result = subprocess.CompletedProcess(
                    args=command,
                    returncode=returncode,
                    stdout=stdout_bytes,
                    stderr=stderr_bytes,
                )
            finally:
                try:
                    for name in result_dir.iterdir():
                        if name.is_file():
                            name.unlink()
                    result_dir.rmdir()
                except OSError:
                    pass
        else:
            logger.debug("Running command %s in directory %s", " ".join(command), cwd)

            result_dir = Path(tempfile.mkdtemp(prefix="gpa_bench_result_"))
            try:
                worker = multiprocessing.Process(
                    target=self._run_subprocess,
                    args=(command, cwd, self.env, result_dir),
                    daemon=True,
                )
                worker.start()
                worker.join(timeout=timeout)

                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=30)
                    if worker.is_alive():
                        worker.kill()
                        worker.join()
                    logger.error(
                        "Command %s timed out after %d seconds",
                        " ".join(command),
                        timeout,
                    )
                    timeout_msg = f"TIMEOUT ({timeout} seconds)".encode()
                    return subprocess.CompletedProcess(
                        args=command,
                        returncode=-1,
                        stdout=timeout_msg,
                        stderr=timeout_msg,
                    )

                returncode_path = result_dir / _RESULT_RETURNCODE_FILE
                if not returncode_path.exists():
                    logger.error(
                        "Command %s produced no result (worker exited with code %s)",
                        " ".join(command),
                        worker.exitcode,
                    )
                    return subprocess.CompletedProcess(
                        args=command,
                        returncode=-1,
                        stdout=None,
                        stderr=None,
                    )

                with returncode_path.open("r", encoding="utf-8") as f:
                    returncode = int(f.read().strip())

                stdout_path = result_dir / _RESULT_STDOUT_FILE
                stderr_path = result_dir / _RESULT_STDERR_FILE

                if stdout_path.exists():
                    with stdout_path.open("rb") as f:
                        stdout_bytes = f.read()
                else:
                    stdout_bytes = f"Could not find stdout file {stdout_path}".encode()

                if stderr_path.exists():
                    with stderr_path.open("rb") as f:
                        stderr_bytes = f.read()
                else:
                    stderr_bytes = f"Could not find stderr file {stderr_path}".encode()

                result = subprocess.CompletedProcess(
                    args=command,
                    returncode=returncode,
                    stdout=stdout_bytes,
                    stderr=stderr_bytes,
                )
            finally:
                try:
                    for name in result_dir.iterdir():
                        if name.is_file():
                            name.unlink()
                    result_dir.rmdir()
                except OSError:
                    pass

        if not self.suppress_command_stdout:
            if log_level == "DEBUG":
                # DEBUG: Always log stdout and stderr, even with quiet=True
                logger.debug("Command stdout:\n%s", self.decode_and_limit(result.stdout))
                logger.debug("Command stderr:\n%s", self.decode_and_limit(result.stderr))
            elif log_level == "INFO":
                # INFO: Log stdout and stderr on failure, unless quiet=True
                if not effective_quiet and result.returncode != 0:
                    if result.stdout:
                        logger.info("Command stdout:\n%s", self.decode_and_limit(result.stdout))
                    if result.stderr:
                        logger.info("Command stderr:\n%s", self.decode_and_limit(result.stderr))
        # else suppress_command_stdout or WARNING/ERROR/CRITICAL — don't log command stdout/stderr

        return result

    def _run_subprocess(
        self,
        command: list[str],
        cwd: os.PathLike,
        env: dict,
        result_dir: Path,
    ) -> None:
        """Run a subprocess and write results to result_dir, wraps subprocess.run().

        Target function for a worker process that executes subprocess.run and writes results to
        result_dir for the parent to read. Runs without a timeout; the parent enforces one. Stdout
        and stderr are written to files under result_dir so the parent can read them from disk. The
        worker only runs the subprocess and writes the returncode; it does not read stdout/stderr
        back.

        Args:
            command: Command to run as a list of strings
            cwd: Working directory for the command
            env: Environment variables dictionary passed to the subprocess
            result_dir: Path to the directory where the results will be written

        Returns:
            None

        """
        faulthandler.enable()
        faulthandler.register(signal.SIGTERM)
        stderr_path = result_dir / _RESULT_STDERR_FILE
        stdout_path = result_dir / _RESULT_STDOUT_FILE
        returncode_path = result_dir / _RESULT_RETURNCODE_FILE
        try:
            with stderr_path.open("wb") as err_f, stdout_path.open("wb") as out_f:
                proc = subprocess.run(  # noqa: S603
                    command, cwd=cwd, env=env, check=False, stdout=out_f, stderr=err_f,
                )
            returncode = proc.returncode
            with returncode_path.open("w", encoding="utf-8") as f:
                f.write(str(returncode))
        except Exception as exc:  # pylint: disable=broad-except # noqa: BLE001
            with returncode_path.open("w", encoding="utf-8") as f:
                f.write("-1")
            with stderr_path.open("wb") as f:
                f.write(str(exc).encode())

    def decode_and_limit(self, raw: bytes | None) -> str:
        """Decode a bytes object to a string and truncate.

        Truncates the string if it exceeds the output character limit.

        Args:
            raw: The bytes object to decode and truncate

        Returns:
            The decoded and truncated string

        """
        text = raw.decode("utf-8") if raw else ""
        return (
            self.truncate_middle(text, self.output_char_limit)
            if self.output_char_limit > 0
            else text
        )

    def truncate_middle(self, text: str, char_limit: int) -> str:
        """Truncate a string to char_limit characters by removing characters from the middle.

        If the string is within the limit it is returned unchanged. Otherwise, equal halves of the
        allowed characters are kept from the start and end of the string, with a placeholder
        message inserted in between.

        Args:
            text: The string to truncate
            char_limit: Maximum number of characters to keep (must be > 0)

        Returns:
            Truncated string, or original string if already within limit

        """
        if len(text) <= char_limit:
            return text
        half = char_limit // 2
        omitted = len(text) - char_limit
        placeholder = f"\n... [{omitted} characters omitted] ...\n"
        return text[:half] + placeholder + text[len(text) - half :]


def stdout_uses_file(app: dict) -> bool:
    """Return True if the app's validation output comes from stdout (not a file).

    Apps that have a reference_output but no test_output write their output directly to stdout.

    Args:
        app: Application configuration dictionary

    Returns:
        True if the app's validation output comes from stdout (not a file)

    """
    return "reference_output" in app and "test_output" not in app


def get_stdout_redirect_path(temp_dir: os.PathLike) -> os.PathLike:
    """Get the path for the stdout redirect file in the temporary directory.

    Args:
        temp_dir: Temporary directory where working copy of application directory is located

    Returns:
        Absolute path to the stdout redirect file

    """
    return Path(temp_dir) / _RESULT_STDOUT_FILE


def get_bin_path(app: dict, temp_dir: os.PathLike) -> os.PathLike:
    """Get the binary path for the application.

    Args:
        app: Application configuration dictionary
        temp_dir: Temporary directory where working copy of application directory is located

    Returns:
        Absolute path to the application binary

    """
    if "run_path" in app:
        return Path(temp_dir) / app["run_path"] / app["run_command"].split()[0]
    return Path(temp_dir) / app["path"] / app["run_command"].split()[0]


def get_build_path(app: dict, temp_dir: os.PathLike) -> os.PathLike:
    """Get the build path for the application.

    Args:
        app: Application configuration dictionary
        temp_dir: Temporary directory where working copy of application directory is located

    Returns:
        Path where the application should be built

    """
    if "build_path" in app:
        return Path(temp_dir) / app["build_path"]
    return Path(temp_dir) / app["path"]


def get_run_path(app: dict, temp_dir: os.PathLike) -> os.PathLike:
    """Get the run path for the application.

    Args:
        app: Application configuration dictionary
        temp_dir: Temporary directory where working copy of application directory is located

    Returns:
        Path where the application should be run from

    """
    if "run_path" in app:
        return Path(temp_dir) / app["run_path"]
    if "build_path" in app:
        return Path(temp_dir) / app["build_path"]
    return Path(temp_dir) / app["path"]


def setup_profile_dir() -> os.PathLike:
    """Set up the profile directory.

    Set up the profile directory for storing profiling output files in the current working
    directory ("profiles" subdirectory). Creates the directory if it doesn't exist.

    Returns:
        Path to the profile directory

    """
    profile_dir: Path = Path(Path.cwd()) / "profiles"
    if not profile_dir.exists():
        profile_dir.mkdir(parents=True, exist_ok=True)
    return profile_dir


def detect_sm_version(cuda_home: os.PathLike) -> int:
    """Detect SM version from nvidia-smi.

    Returns the SM version as a two-digit integer (e.g., 9.0 -> 90).

    Returns:
        The SM version as a two-digit integer

    """
    try:
        result = subprocess.run(  # noqa: S603
            [
                f"{cuda_home}/bin/nvidia-smi",
                "--query-gpu=compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        # Get the first line and remove whitespace
        compute_cap = result.stdout.strip().split("\n")[0].strip()
        # Remove decimal point (e.g., "9.0" -> "90")
        sm_version_str = compute_cap.replace(".", "")
        return int(sm_version_str)
    except (subprocess.CalledProcessError, ValueError, IndexError, FileNotFoundError) as e:
        logger.warning("Could not detect SM version from nvidia-smi (%s). Defaulting to 90.", e)
        return 90
