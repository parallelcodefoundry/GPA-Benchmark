"""
Utility Functions for GPA-Benchmark Driver

This module provides utility functions for subprocess execution, path resolution,
directory setup, and system detection.
"""
import logging
import multiprocessing
import os
import subprocess
import tempfile

logger = logging.getLogger("GPA-Benchmark")


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
    """

    def __init__(
        self,
        env: dict,
        log_level: str = "WARNING",
        quiet: bool = False,
        timeout: int | None = None,
        output_char_limit: int = 25000,
    ) -> None:
        self.env = env
        self.log_level = log_level
        self.quiet = quiet
        self.timeout = timeout
        self.output_char_limit = output_char_limit

    def run(self, command: list[str], cwd: str,
            quiet: bool | None = None,
            stdout_file: str | None = None) -> subprocess.CompletedProcess:
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
            stdout_file: When provided, stdout is written to this persistent named file instead of
                a temporary file, and result.stdout will be None. Used for apps whose stdout is the
                validation output so it can be read back by the validation step.

        Returns:
            CompletedProcess object with returncode, stdout, and stderr attributes. stdout will be
            None when stdout_file is provided.
        """
        effective_quiet = self.quiet if quiet is None else quiet
        log_level = self.log_level
        timeout = self.timeout

        logger.debug("Running command %s in directory %s", ' '.join(command), cwd)

        result_queue: multiprocessing.Queue = multiprocessing.Queue()
        worker = multiprocessing.Process(
            target=self._run_subprocess,
            args=(command, cwd, self.env, result_queue, stdout_file, self.output_char_limit),
            daemon=True,
        )
        worker.start()
        worker.join(timeout=timeout)

        if worker.is_alive():
            worker.kill()
            worker.join()
            logger.error("Command %s timed out after %d seconds", ' '.join(command), timeout)
            timeout_msg = f"TIMEOUT ({timeout} seconds)".encode()
            return subprocess.CompletedProcess(
                args=command,
                returncode=-1,
                stdout=None if stdout_file else timeout_msg,
                stderr=timeout_msg,
            )

        if result_queue.empty():
            logger.error("Command %s produced no result (worker exited with code %s)",
                         ' '.join(command), worker.exitcode)
            return subprocess.CompletedProcess(args=command, returncode=-1,
                                               stdout=None, stderr=None)

        returncode, stdout, stderr = result_queue.get_nowait()
        result: subprocess.CompletedProcess = subprocess.CompletedProcess(
            args=command, returncode=returncode, stdout=stdout, stderr=stderr,
        )

        if log_level == "DEBUG":
            # DEBUG: Always log stdout and stderr, even with quiet=True
            if stdout_file:
                logger.debug("Command stdout written to file: %s", stdout_file)
            else:
                logger.debug("Command stdout:\n%s", self.decode_and_limit(result.stdout))
            logger.debug("Command stderr:\n%s", self.decode_and_limit(result.stderr))
        elif log_level == "INFO":
            # INFO: Log stdout and stderr on failure, unless quiet=True
            if not effective_quiet and result.returncode != 0:
                if stdout_file:
                    logger.info("Command stdout written to file: %s", stdout_file)
                elif result.stdout:
                    logger.info("Command stdout:\n%s", self.decode_and_limit(result.stdout))
                if result.stderr:
                    logger.info("Command stderr:\n%s", self.decode_and_limit(result.stderr))
        # else: WARNING/ERROR/CRITICAL — never log stdout or stderr

        return result

    @staticmethod
    def _read_file_truncated(path: str, char_limit: int) -> bytes:
        """Read a file and return its content as bytes, truncated to char_limit characters.

        Reads only as much of the file as needed to stay within the character limit, keeping the
        first and last halves and discarding the middle.

        Args:
            path: Path to the file to read
            char_limit: Maximum number of characters to return (0 or negative = no limit)

        Returns:
            File content as UTF-8 bytes, possibly with a truncation placeholder in the middle
        """
        size = os.path.getsize(path)
        if char_limit <= 0 or size <= char_limit:
            with open(path, "rb") as f:
                return f.read()

        half = char_limit // 2
        placeholder = f"\n... [{size - char_limit} characters omitted] ...\n".encode()
        with open(path, "rb") as f:
            head = f.read(half)
            f.seek(-half, 2)
            tail = f.read(half)
        return head + placeholder + tail

    def _run_subprocess(
        self, command: list[str], cwd: str, env: dict, result_queue: multiprocessing.Queue,
        stdout_file: str | None = None,
        output_char_limit: int = 0,
    ) -> None:
        """Target function for a worker process that executes subprocess.run and puts the result
        (returncode, stdout, stderr) onto result_queue.  Runs without a timeout so the parent
        process is responsible for enforcing one.

        Stdout and stderr are always written to files to avoid deadlocking on the pipe buffer in
        subprocess.run.  When stdout_file is provided, stdout is written to that persistent path
        and result stdout is None; otherwise a temporary file is used and content is read back and
        put on the queue.  Stderr always uses a temporary file and is read back.
        """
        stderr_path = None
        stdout_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False) as stderr_tmp:
                stderr_path = stderr_tmp.name

            if stdout_file:
                with open(stdout_file, "w", encoding="utf-8") as out_f, \
                     open(stderr_path, "wb") as err_f:
                    proc = subprocess.run(command, cwd=cwd, env=env, check=False,
                                          stdout=out_f, stderr=err_f)
                stderr_bytes = self._read_file_truncated(stderr_path, output_char_limit)
                result_queue.put((proc.returncode, None, stderr_bytes))
            else:
                with tempfile.NamedTemporaryFile(delete=False) as stdout_tmp:
                    stdout_path = stdout_tmp.name
                with open(stdout_path, "wb") as out_f, open(stderr_path, "wb") as err_f:
                    proc = subprocess.run(command, cwd=cwd, env=env, check=False,
                                          stdout=out_f, stderr=err_f)
                stdout_bytes = self._read_file_truncated(stdout_path, output_char_limit)
                stderr_bytes = self._read_file_truncated(stderr_path, output_char_limit)
                result_queue.put((proc.returncode, stdout_bytes, stderr_bytes))
        except Exception as exc:  # pylint: disable=broad-except
            result_queue.put((-1, None, str(exc).encode()))
        finally:
            if stdout_path and os.path.exists(stdout_path):
                os.unlink(stdout_path)
            if stderr_path and os.path.exists(stderr_path):
                os.unlink(stderr_path)

    def decode_and_limit(self, raw: bytes | None) -> str:
        """Decode a bytes object to a string and truncate it if it exceeds the output character
           limit.

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
        return text[:half] + placeholder + text[len(text) - half:]


STDOUT_REDIRECT_FILENAME = "driver-subprocess-stdout.txt"


def stdout_uses_file(app: dict) -> bool:
    """Return True if the app's validation output comes from stdout (not a file).

    Apps that have a reference_output but no test_output write their output directly to stdout.
    For these apps the stdout should be redirected to a file to avoid capturing potentially
    megabytes of output in memory.

    Args:
        app: Application configuration dictionary

    Returns:
        True if stdout should be redirected to STDOUT_REDIRECT_FILENAME
    """
    return "reference_output" in app and "test_output" not in app


def get_stdout_redirect_path(temp_dir: str) -> str:
    """Get the path for the stdout redirect file in the temporary directory.

    Args:
        temp_dir: Temporary directory where working copy of application directory is located

    Returns:
        Absolute path to the stdout redirect file
    """
    return os.path.join(temp_dir, STDOUT_REDIRECT_FILENAME)


def get_bin_path(app: dict, temp_dir: str) -> str:
    """Get the binary path for the application.

    Args:
        app: Application configuration dictionary
        temp_dir: Temporary directory where working copy of application directory is located

    Returns:
        Absolute path to the application binary
    """
    if "run_path" in app:
        return os.path.join(temp_dir, app["run_path"], app["run_command"].split()[0])
    return os.path.join(temp_dir, app["path"], app["run_command"].split()[0])


def get_build_path(app: dict, temp_dir: str) -> str:
    """Get the build path for the application.

    Args:
        app: Application configuration dictionary
        temp_dir: Temporary directory where working copy of application directory is located

    Returns:
        Path where the application should be built
    """
    if "build_path" in app:
        return os.path.join(temp_dir, app["build_path"])
    else:
        return os.path.join(temp_dir, app["path"])


def get_run_path(app: dict, temp_dir: str) -> str:
    """Get the run path for the application.

    Args:
        app: Application configuration dictionary
        temp_dir: Temporary directory where working copy of application directory is located

    Returns:
        Path where the application should be run from
    """
    if "run_path" in app:
        return os.path.join(temp_dir, app["run_path"])
    elif "build_path" in app:
        return os.path.join(temp_dir, app["build_path"])
    else:
        return os.path.join(temp_dir, app["path"])


def setup_profile_dir() -> str:
    """Setup the profile directory for storing profiling output files in the current working
       directory.

    Creates the directory if it doesn't exist.

    Returns:
        Path to the profile directory
    """
    profile_dir: str = os.path.join(os.getcwd(), "profiles")
    if not os.path.exists(profile_dir):
        os.makedirs(profile_dir)
    return profile_dir


def detect_sm_version() -> int:
    """Detect SM version from nvidia-smi.

    Returns the SM version as a two-digit integer (e.g., 9.0 -> 90).

    Returns:
        The SM version as a two-digit integer

    Raises:
        No exceptions raised. If detection fails, prints a warning and returns 90 as default.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        # Get the first line and remove whitespace
        compute_cap = result.stdout.strip().split('\n')[0].strip()
        # Remove decimal point (e.g., "9.0" -> "90")
        sm_version_str = compute_cap.replace('.', '')
        sm_version = int(sm_version_str)
        return sm_version
    except (subprocess.CalledProcessError, ValueError, IndexError, FileNotFoundError) as e:
        logger.warning("Could not detect SM version from nvidia-smi (%s). Defaulting to 90.", e)
        return 90
