"""J0 T0: the VRAM reset tool in the trust zone (GPA-G1 J0-A, fix round 5 K3).

``frontier_tools/vram_reset`` (built by ``scripts/frontier_prepare.sh build`` from the md5-pinned
``scripts/vram_reset.cpp``) runs before every timed process. It is untracked, so git-based R9
cannot see it. Instead:

- the runner takes :func:`snapshot` BEFORE the agent runs and passes its ``sha256`` to the driver
  (``DriverConfig.vram_reset_sha256``); R9 re-checks :func:`snapshot` after the agent;
- :class:`VramResetTool` checks the binary against THAT value (or, only when the caller passes
  :data:`BUILD_RECORD` -- gpa_test -- against the build record
  ``frontier_tools/vram_reset.sha256``; no value is refused), then keeps the verified bytes in a sealed
  in-memory file (memfd, F_SEAL_WRITE) and executes that copy, so nothing on disk can change
  what runs between the check and the uses. Where memfd is unavailable it falls back to a private
  0700 copy whose sha256 is re-checked before every use.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from gpa_bench_driver.driver_src.driver_utils import DriverInfraError

_GPA_ROOT = Path(__file__).resolve().parent.parent.parent
TOOL_REL = "frontier_tools/vram_reset"
RECORD_REL = "frontier_tools/vram_reset.sha256"
# explicit opt-in to checking against the build record in the (agent-writable) tree: gpa_test
# only. Scoring paths must pass the pre-agent sha256; None is refused (fix round 6, K3 closed).
BUILD_RECORD = "build-record"


def _root(gpa_root: Path | None) -> Path:
    return Path(gpa_root) if gpa_root is not None else _GPA_ROOT


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_record(gpa_root: Path | None = None) -> str | None:
    """The build record's sha256 (sha256sum format), or None when the record is missing/empty."""
    rec = _root(gpa_root) / RECORD_REL
    if not rec.is_file():
        return None
    parts = rec.read_text().split()
    return parts[0].lower() if parts else None


def snapshot(gpa_root: Path | None = None) -> dict:
    """{"path", "sha256", "record_sha256"} of the reset binary (the runner's pre-agent snapshot
    and R9's post-agent check).

    Raises:
        DriverInfraError: the binary or its build record is missing, or they disagree

    """
    tool = _root(gpa_root) / TOOL_REL
    if not tool.is_file():
        msg = f"{tool} is missing (run scripts/frontier_prepare.sh build)"
        raise DriverInfraError(msg)
    got = sha256_bytes(tool.read_bytes())
    rec = read_record(gpa_root)
    if rec is None:
        msg = f"{_root(gpa_root) / RECORD_REL} is missing (run scripts/frontier_prepare.sh build)"
        raise DriverInfraError(msg)
    if got != rec:
        msg = f"{tool} sha256 {got} does not match its build record {rec}"
        raise DriverInfraError(msg)
    return {"path": str(tool), "sha256": got, "record_sha256": rec}


class VramResetTool:
    """A verified, immutable copy of the reset binary for one timing series."""

    def __init__(self, expected_sha256: str | None = None, gpa_root: Path | None = None) -> None:
        tool = _root(gpa_root) / TOOL_REL
        if not tool.is_file():
            msg = (f"VRAM reset failed: {tool} is missing (run scripts/frontier_prepare.sh build); "
                   "the J0 protocol needs it before every timed process")
            raise DriverInfraError(msg)
        data = tool.read_bytes()
        got = sha256_bytes(data)
        if not expected_sha256:
            msg = ("VRAM reset refused: no pre-agent sha256 of frontier_tools/vram_reset was "
                   "given (DriverConfig.vram_reset_sha256); a scoring path must pass the runner's "
                   "snapshot (gpa_test passes 'build-record')")
            raise DriverInfraError(msg)
        if expected_sha256 != BUILD_RECORD:
            if got != expected_sha256.lower():
                msg = (f"VRAM reset failed: {tool} sha256 {got} does not match the pre-agent "
                       f"sha256 {expected_sha256.lower()} (the binary changed after the snapshot)")
                raise DriverInfraError(msg)
        else:
            rec = read_record(gpa_root)
            if rec is None:
                msg = (f"VRAM reset failed: its build record {_root(gpa_root) / RECORD_REL} is "
                       "missing (run scripts/frontier_prepare.sh build)")
                raise DriverInfraError(msg)
            if got != rec:
                msg = (f"VRAM reset failed: {tool} sha256 {got} does not match its build record "
                       f"{rec}")
                raise DriverInfraError(msg)
        self.sha256 = got
        self._data = data
        self._fd: int | None = None
        self._copy: Path | None = None
        self._tmpdir: str | None = None
        try:
            import fcntl

            fd = os.memfd_create("vram_reset", os.MFD_ALLOW_SEALING)
            os.write(fd, data)
            fcntl.fcntl(fd, fcntl.F_ADD_SEALS, fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SHRINK
                        | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SEAL)
            self._fd = fd
        except (AttributeError, OSError, ImportError):
            self._tmpdir = tempfile.mkdtemp(prefix="gpa_vram_reset_")
            os.chmod(self._tmpdir, 0o700)
            self._copy = Path(self._tmpdir) / "vram_reset"
            self._copy.write_bytes(data)
            self._copy.chmod(0o700)

    @property
    def sealed(self) -> bool:
        return self._fd is not None

    def _verify(self) -> None:
        if self._fd is not None:
            now = os.pread(self._fd, len(self._data) + 1, 0)
        else:
            now = self._copy.read_bytes() if self._copy is not None else b""
        if sha256_bytes(now) != self.sha256:
            msg = "VRAM reset failed: the verified copy of vram_reset changed before use"
            raise DriverInfraError(msg)

    def run(self, env: dict | None, cwd: Path | str, args: tuple[str, ...] = (),
            timeout: float = 30) -> float:
        """Run the verified copy (T0 reset, or with a GiB argument a perturbing allocation).
        Returns its wall time (s).

        Raises:
            DriverInfraError: the copy changed, timed out, could not start, or exited non-zero

        """
        self._verify()
        if self._fd is not None:
            argv = [f"/proc/self/fd/{self._fd}", *args]
            pass_fds: tuple[int, ...] = (self._fd,)
        else:
            argv = [str(self._copy), *args]
            pass_fds = ()
        start = time.perf_counter()
        try:
            proc = subprocess.run(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,  # noqa: S603
                                  capture_output=True, text=True, timeout=timeout, check=False,
                                  pass_fds=pass_fds)
        except subprocess.TimeoutExpired as exc:
            msg = f"VRAM reset failed: vram_reset timed out after {timeout:.0f} s"
            raise DriverInfraError(msg) from exc
        except OSError as exc:
            msg = f"VRAM reset failed: {exc}"
            raise DriverInfraError(msg) from exc
        if proc.returncode != 0:
            msg = (f"VRAM reset failed (exit {proc.returncode}): "
                   f"{(proc.stderr or proc.stdout)[-400:]}")
            raise DriverInfraError(msg)
        return time.perf_counter() - start

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None

    def __enter__(self) -> VramResetTool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
