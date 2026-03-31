"""Utility Functions for GPA-Benchmark Driver.

This module provides utility functions for subprocess execution, path resolution,
directory setup, and system detection. Subprocess results are transferred via
disk (files written by the worker, read by the parent); no multiprocessing
queues are used.
"""

import faulthandler
import logging
import multiprocessing
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("GPA-Benchmark")

# Filenames used inside the per-run result directory for disk IPC
_RESULT_RETURNCODE_FILE = "returncode.txt"
_RESULT_STDOUT_FILE = "stdout.bin"
_RESULT_STDERR_FILE = "stderr.bin"

MAX_OUTPUT_CHAR_LIMIT = 25000


@dataclass
class SubprocessRunnerConfig:
    """Configuration for SubprocessRunner (logging, timeout, output limits, srun).

    Attributes:
        log_level: When to log command stdout/stderr (DEBUG, INFO, WARNING, etc.).
        quiet: Default quiet flag; suppresses INFO-level output on failure when True.
        timeout: Default timeout in seconds; None means no limit.
        output_char_limit: Max characters logged for stdout/stderr; <= 0 to disable truncation.
        suppress_command_stdout: When True, never log stdout/stderr from commands.
        use_srun: When True, prepend srun and use srun --time for timeout.

    """

    log_level: str = "WARNING"
    quiet: bool = False
    timeout: int | None = None
    output_char_limit: int = MAX_OUTPUT_CHAR_LIMIT
    suppress_command_stdout: bool = False
    use_srun: bool = False


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

    def __init__(self, *, env: dict, config: SubprocessRunnerConfig) -> None:
        """Initialize the SubprocessRunner.

        Args:
            env: Environment variables dictionary passed to every subprocess.
            config: Runner configuration (logging, timeout, output limits, srun).

        """
        self.env = env
        self.log_level = config.log_level
        self.quiet = config.quiet
        self.timeout = config.timeout
        self.output_char_limit = config.output_char_limit
        self.suppress_command_stdout = config.suppress_command_stdout
        self.use_srun = config.use_srun

    def run(
        self,
        command: list[str],
        cwd: Path,
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

        if self.use_srun:
            result = self._run_with_srun(command, cwd)
        else:
            result = self._run_with_multiprocessing(command, cwd)

        self._log_command_result(result, effective_quiet=effective_quiet)
        return result

    def _run_with_srun(
        self,
        command: list[str],
        cwd: Path,
    ) -> subprocess.CompletedProcess:
        """Run command under srun with optional --time; no multiprocessing."""
        srun_cmd = ["srun"]
        if self.timeout is not None:
            srun_cmd.append(f"--time=00:{self.timeout}")
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
            with stdout_path.open("rb") as f:
                stdout_bytes = f.read()
            with stderr_path.open("rb") as f:
                stderr_bytes = f.read()
            return subprocess.CompletedProcess(
                args=command,
                returncode=proc.returncode,
                stdout=stdout_bytes,
                stderr=stderr_bytes,
            )
        finally:
            self._cleanup_result_dir(result_dir)

    def _run_with_multiprocessing(
        self,
        command: list[str],
        cwd: Path,
    ) -> subprocess.CompletedProcess:
        """Run command in a worker process with multiprocessing-based timeout."""
        logger.debug("Running command %s in directory %s", " ".join(command), cwd)
        result_dir = Path(tempfile.mkdtemp(prefix="gpa_bench_result_"))
        try:
            worker = multiprocessing.Process(
                target=self._run_subprocess,
                args=(command, cwd, self.env, result_dir),
                daemon=True,
            )
            worker.start()
            worker.join(timeout=self.timeout)

            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=30)
                if worker.is_alive():
                    worker.kill()
                    worker.join()
                logger.error(
                    "Command %s timed out after %d seconds",
                    " ".join(command),
                    self.timeout,
                )
                return subprocess.CompletedProcess(
                    args=command,
                    returncode=-1,
                    stdout=f"TIMEOUT ({self.timeout} seconds)".encode(),
                    stderr=f"TIMEOUT ({self.timeout} seconds)".encode(),
                )

            worker_result = self._read_worker_result(result_dir, command)
            if worker_result is not None:
                return worker_result
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
        finally:
            self._cleanup_result_dir(result_dir)

    def _log_command_result(
        self,
        result: subprocess.CompletedProcess,
        *,
        effective_quiet: bool,
    ) -> None:
        """Log command stdout/stderr according to log_level and quiet."""
        if self.suppress_command_stdout:
            return
        if self.log_level == "DEBUG":
            logger.debug("Command stdout:\n%s", self.decode_and_limit(result.stdout))
            logger.debug("Command stderr:\n%s", self.decode_and_limit(result.stderr))
        elif self.log_level == "INFO" and not effective_quiet and result.returncode != 0:
            if result.stdout:
                logger.info("Command stdout:\n%s", self.decode_and_limit(result.stdout))
            if result.stderr:
                logger.info("Command stderr:\n%s", self.decode_and_limit(result.stderr))

    def _run_subprocess(
        self,
        command: list[str],
        cwd: Path,
        env: dict,
        result_dir: Path,
    ) -> None:
        """Run a subprocess and write results to result_dir (worker process target).

        Target for a worker process: runs subprocess.run and writes returncode and
        stdout/stderr to result_dir. No timeout; the parent enforces one.
        """
        faulthandler.enable()
        faulthandler.register(signal.SIGTERM)
        stderr_path = result_dir / _RESULT_STDERR_FILE
        stdout_path = result_dir / _RESULT_STDOUT_FILE
        returncode_path = result_dir / _RESULT_RETURNCODE_FILE
        try:
            with stderr_path.open("wb") as err_f, stdout_path.open("wb") as out_f:
                proc = subprocess.run(  # noqa: S603
                    command,
                    cwd=cwd,
                    env=env,
                    check=False,
                    stdout=out_f,
                    stderr=err_f,
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

    def _cleanup_result_dir(self, result_dir: Path) -> None:
        """Remove temporary result directory and its contents."""
        try:
            for name in result_dir.iterdir():
                if name.is_file():
                    name.unlink()
            result_dir.rmdir()
        except OSError:
            pass

    def _read_worker_result(
        self,
        result_dir: Path,
        command: list[str],
    ) -> subprocess.CompletedProcess | None:
        """Read returncode and stdout/stderr from worker result_dir. Returns None if missing."""
        returncode_path = result_dir / _RESULT_RETURNCODE_FILE
        if not returncode_path.exists():
            return None
        with returncode_path.open("r", encoding="utf-8") as f:
            returncode = int(f.read().strip())

        stdout_path = result_dir / _RESULT_STDOUT_FILE
        stderr_path = result_dir / _RESULT_STDERR_FILE
        stdout_bytes = stdout_path.read_bytes() if stdout_path.exists() else b""
        stderr_bytes = stderr_path.read_bytes() if stderr_path.exists() else b""
        return subprocess.CompletedProcess(
            args=command,
            returncode=returncode,
            stdout=stdout_bytes,
            stderr=stderr_bytes,
        )


def stdout_uses_file(app: dict) -> bool:
    """Return True if the app's validation output comes from stdout (not a file).

    Apps that have a reference_output but no test_output write their output directly to stdout.

    Args:
        app: Application configuration dictionary

    Returns:
        True if the app's validation output comes from stdout (not a file)

    """
    return "reference_output" in app and "test_output" not in app


def get_stdout_redirect_path(temp_dir: Path) -> Path:
    """Get the path for the stdout redirect file in the temporary directory.

    Args:
        temp_dir: Temporary directory where working copy of application directory is located

    Returns:
        Absolute path to the stdout redirect file

    """
    return temp_dir / _RESULT_STDOUT_FILE


def get_bin_path(app: dict, temp_dir: Path) -> Path:
    """Get the binary path for the application.

    Args:
        app: Application configuration dictionary
        temp_dir: Temporary directory where working copy of application directory is located

    Returns:
        Absolute path to the application binary

    """
    if "run_path" in app:
        return temp_dir / app["run_path"] / app["run_command"].split()[0]
    return temp_dir / app["path"] / app["run_command"].split()[0]


def get_build_path(app: dict, temp_dir: Path) -> Path:
    """Get the build path for the application.

    Args:
        app: Application configuration dictionary
        temp_dir: Temporary directory where working copy of application directory is located

    Returns:
        Path where the application should be built

    """
    if "build_path" in app:
        return temp_dir / app["build_path"]
    return temp_dir / app["path"]


def get_run_path(app: dict, temp_dir: Path) -> Path:
    """Get the run path for the application.

    Args:
        app: Application configuration dictionary
        temp_dir: Temporary directory where working copy of application directory is located

    Returns:
        Path where the application should be run from

    """
    if "run_path" in app:
        return temp_dir / app["run_path"]
    if "build_path" in app:
        return temp_dir / app["build_path"]
    return temp_dir / app["path"]


def setup_profile_dir(base_dir: Path | None = None) -> Path:
    """Set up the profile directory.

    Set up the profile directory for storing profiling output files.
    Uses the provided base_dir (e.g. temp working directory) so that
    profiles are created in a writable location rather than the
    potentially read-only GPA-Benchmark root.

    Args:
        base_dir: Base directory for profiles.  Falls back to cwd if None.

    Returns:
        Path to the profile directory

    """
    parent = base_dir if base_dir is not None else Path.cwd()
    profile_dir: Path = parent / "profiles"
    if not profile_dir.exists():
        profile_dir.mkdir(parents=True, exist_ok=True)
    return profile_dir


def detect_cuda_home() -> Path | None:
    """Detect CUDA install directory from the location of nvcc.

    Uses the path two parents up from the executable returned by 'which nvcc'
    (e.g. /usr/local/cuda/bin/nvcc -> /usr/local/cuda).

    Returns:
        The CUDA install directory path, or None if nvcc is not found or path
        cannot be derived.

    """
    nvcc_path = shutil.which("nvcc")
    if not nvcc_path:
        return None
    path = Path(nvcc_path).resolve()
    return path.parent.parent


def detect_sm_version() -> int | None:
    """Detect SM version from nvidia-smi.

    Returns the SM version as a two-digit integer (e.g., 9.0 -> 90).

    Returns:
        The SM version as a two-digit integer

    """
    nvidia_smi_path = shutil.which("nvidia-smi")
    if not nvidia_smi_path:
        return None
    result = subprocess.run(
        [  # noqa: S607
            "nvidia-smi",
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


def get_default_apps_config_path() -> Path:
    """Get the default apps config path.

    Returns:
        The default apps config path

    """
    return Path(__file__).parent.parent.parent / "driver_apps.yaml"
