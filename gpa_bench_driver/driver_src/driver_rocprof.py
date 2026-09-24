"""rocprofv3 kernel timing for the hip backend (AMD GPUs, e.g. Frontier MI250X).

The hip backend replaces the Nsight Systems timing of the cuda backend. Each sample runs the app
once under ``rocprofv3 --kernel-trace --output-format csv`` and reduces the kernel trace to one
dict (stored in ``DriverPassResult.nsys_data`` so existing consumers keep working):

    {
        "exec_time": int,          # ns; == target_ns (scored time WITHOUT the new-kernel term)
        "target_ns": int,          # sum of the durations of EVERY dispatch of the target kernel
        "target_dispatches": int,  # number of dispatches of the target kernel
        "kernels": {name: int},    # total ns per demangled kernel name, all non-runtime kernels
        "wall_s": float,           # wall time of the profiled run (seconds)
        "backend": "rocprofv3",
    }

Dispatch duration = End_Timestamp - Start_Timestamp (ns). Runtime-internal kernels (names that
start with ``__amd_rocclr_``: copy/fill blits) are ignored everywhere. The target kernel is every
kernel whose demangled name matches the app's ``score_regex`` (re.search, anchored in the YAML).

The Frontier scoring rule (new/renamed kernels count as target, integrity guards) needs the
baseline pass, so consumers compute it from ``kernels``; see ``scored_time_ns`` (pass the
app's ``score_regex``) and ``integrity_guards`` below for the reference implementation.
"""

from __future__ import annotations

import csv
import logging
import re
import shutil
import time
from collections.abc import Callable, Hashable, Iterable
from pathlib import Path
from typing import Any

from gpa_bench_driver.driver_src.driver_models import SwapConfig
from gpa_bench_driver.driver_src.driver_utils import (
    SubprocessRunner,
    get_run_path,
    setup_profile_dir,
)

logger = logging.getLogger("GPA-Benchmark")

RUNTIME_KERNEL_PREFIX = "__amd_rocclr_"
BACKEND_NAME = "rocprofv3"

# Integrity guards of the Frontier scoring rule (GPA-G1 contract section 2.3)
G_OTHER_FACTOR = 1.10
G_OTHER_SLACK_NS = 50_000
G_WALL_FACTOR = 1.5
G_WALL_SLACK_S = 1.0


class RocprofError(Exception):
    """Raised when a rocprofv3 timing run fails."""

    def __init__(self, message: str) -> None:
        """Initialize a RocprofError."""
        self.message = message
        super().__init__(self.message)


def default_score_regex(kernel_name: str) -> str:
    """Anchored regex for a kernel name: matches ``name(...)`` (C++) and ``name`` (extern "C")."""
    return rf"^{re.escape(kernel_name)}(\(|$)"


def get_score_regex(app: dict) -> str:
    """Return the app's score_regex, or the anchored default built from kernel_name."""
    if app.get("score_regex"):
        return str(app["score_regex"])
    return default_score_regex(str(app["kernel_name"]))


def is_runtime_kernel(name: str) -> bool:
    """True for ROCm runtime-internal kernels (copy/fill blits), which are never scored."""
    return name.startswith(RUNTIME_KERNEL_PREFIX)


def read_kernel_trace(csv_paths: Iterable[Path]) -> list[tuple[str, int]]:
    """Read rocprofv3 kernel-trace CSV files into (kernel name, duration ns) rows.

    Columns are looked up by header name (``Kernel_Name``, ``Start_Timestamp``,
    ``End_Timestamp``), so both the ROCm 6.x and 7.x layouts parse.

    Args:
        csv_paths: kernel_trace.csv files (one per profiled process)

    Returns:
        One (name, duration_ns) tuple per dispatch, in file order

    Raises:
        RocprofError: if a file lacks the required columns

    """
    rows: list[tuple[str, int]] = []
    for path in csv_paths:
        with Path(path).open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
            missing = {"Kernel_Name", "Start_Timestamp", "End_Timestamp"} - fields
            if missing:
                msg = f"{path}: not a rocprofv3 kernel trace (missing columns {sorted(missing)})"
                raise RocprofError(msg)
            for rec in reader:
                start = int(rec["Start_Timestamp"])
                end = int(rec["End_Timestamp"])
                rows.append((rec["Kernel_Name"], end - start))
    return rows


def summarize_kernel_trace(
    rows: Iterable[tuple[str, int]],
    score_regex: str,
) -> dict[str, Any]:
    """Reduce kernel-trace rows to the per-sample timing dict (without wall_s/backend).

    Args:
        rows: (name, duration_ns) per dispatch
        score_regex: regex identifying the target kernel (re.search on the demangled name)

    Returns:
        dict with exec_time, target_ns, target_dispatches, kernels

    """
    pattern = re.compile(score_regex)
    kernels: dict[str, int] = {}
    target_ns = 0
    target_dispatches = 0
    for name, duration in rows:
        if is_runtime_kernel(name):
            continue
        kernels[name] = kernels.get(name, 0) + duration
        if pattern.search(name):
            target_ns += duration
            target_dispatches += 1
    return {
        "exec_time": target_ns,
        "target_ns": target_ns,
        "target_dispatches": target_dispatches,
        "kernels": kernels,
    }


def find_kernel_trace_files(outdir: Path) -> list[Path]:
    """Return every ``*kernel_trace.csv`` below a rocprofv3 output directory (sorted)."""
    return sorted(Path(outdir).rglob("*kernel_trace.csv"))


def rocprofv3_command(rocm_path: Path | None, outdir: Path, run_command: list[str]) -> list[str]:
    """Build the rocprofv3 kernel-trace command line for one timing sample."""
    rocprof = "rocprofv3"
    if rocm_path is not None and (Path(rocm_path) / "bin" / "rocprofv3").exists():
        rocprof = str(Path(rocm_path) / "bin" / "rocprofv3")
    return [
        rocprof,
        "--kernel-trace",
        "--output-format", "csv",
        "-d", str(outdir),
        "-o", "trace",
        "--",
        *run_command,
    ]


def _profile_dir_name(app: dict, swap_config: SwapConfig | None, i: int) -> str:
    name = app["name"]
    if swap_config:
        name += "_" + Path(swap_config.file_swaps[0].swap_file_src_path).with_suffix("").name
    return f"rocprof_{name}_sample_{i}".replace("/", "_")


def rocprof_time_app(
    app: dict,
    runner: SubprocessRunner,
    temp_dir: Path,
    num_samples: int,
    swap_config: SwapConfig | None = None,
    pbar: Callable[[], None] | None = None,
    *,
    rocm_path: Path | None = None,
    retain_profiles: bool = False,
) -> list[dict[Hashable, Any]] | None:
    """Time the app with rocprofv3 kernel tracing, one run per sample.

    Args:
        app: Application configuration dictionary (needs run_command and score_regex or
            kernel_name; honours stdout_cap_bytes)
        runner: Configured subprocess runner (its env pins the GPU)
        temp_dir: Temporary directory holding the working copy; traces go to temp_dir/profiles
        num_samples: Number of profiled runs
        swap_config: Swap configuration (None for the baseline pass)
        pbar: Progress bar callback (advanced twice per sample, like nsys profile + post)
        rocm_path: ROCm install whose bin/rocprofv3 is used (else rocprofv3 from PATH)
        retain_profiles: Keep the rocprofv3 output directories after parsing

    Returns:
        One timing dict per sample, or None when the baseline pass has no dispatch of the
        target kernel (a configuration error, reported as a warning like the cuda backend)

    Raises:
        RocprofError: if a profiled run fails or produces no kernel trace

    """
    profile_dir = setup_profile_dir(temp_dir)
    run_path = get_run_path(app, temp_dir)
    score_regex = get_score_regex(app)
    stdout_cap = app.get("stdout_cap_bytes")
    samples: list[dict[Hashable, Any]] = []

    for i in range(num_samples):
        outdir = profile_dir / _profile_dir_name(app, swap_config, i)
        if outdir.exists():
            shutil.rmtree(outdir)
        command = rocprofv3_command(rocm_path, outdir, app["run_command"].split())
        start = time.perf_counter()
        result = runner.run(
            command,
            run_path,
            stdout_cap_bytes=int(stdout_cap) if stdout_cap is not None else None,
        )
        wall_s = time.perf_counter() - start

        if result.returncode != 0:
            stderr_text = (result.stderr or b"").decode("utf-8", errors="replace")[-2000:]
            msg = f"rocprofv3 timing run failed with return code {result.returncode}"
            if stderr_text:
                msg += f": {stderr_text}"
            raise RocprofError(msg)

        trace_files = find_kernel_trace_files(outdir)
        if not trace_files:
            msg = f"rocprofv3 wrote no kernel trace under {outdir}"
            raise RocprofError(msg)

        sample = summarize_kernel_trace(read_kernel_trace(trace_files), score_regex)
        sample["wall_s"] = wall_s
        sample["backend"] = BACKEND_NAME

        if not retain_profiles:
            shutil.rmtree(outdir, ignore_errors=True)

        if swap_config is None and sample["target_dispatches"] == 0:
            logger.warning(
                "No dispatch of the target kernel (score_regex %r) in the rocprofv3 trace of "
                "the %s baseline; kernels seen: %s",
                score_regex,
                app["name"],
                sorted(sample["kernels"]),
            )
            if pbar is not None:
                for _ in range(2 * (num_samples - i)):
                    pbar()
            return None

        samples.append(sample)
        if pbar is not None:
            pbar()
            pbar()

    return samples


# ---------------------------------------------------------------------------------------------
# Reference implementation of the Frontier scoring rule (consumers: gpa_test, the APPEB runner).
# ---------------------------------------------------------------------------------------------


def new_kernel_ns(
    sample: dict,
    baseline_kernel_names: Iterable[str],
    score_regex: str | None = None,
) -> int:
    """Sum of the durations of the kernels in ``sample`` that are new relative to the baseline.

    A kernel is new if its name is absent from ``baseline_kernel_names`` and, when
    ``score_regex`` is given, does not match it: a new name that matches the target regex
    (e.g. a renamed or templated variant of the target) is already part of
    ``sample["target_ns"]`` and must not be counted a second time. Callers that omit
    ``score_regex`` must put the target-matching names into ``baseline_kernel_names``
    themselves.
    """
    known = set(baseline_kernel_names)
    pattern = re.compile(score_regex) if score_regex else None
    return sum(
        ns
        for name, ns in sample["kernels"].items()
        if name not in known and not (pattern is not None and pattern.search(name))
    )


def scored_time_ns(
    sample: dict,
    baseline_kernel_names: Iterable[str],
    score_regex: str | None = None,
) -> int:
    """Scored time of one run: every target dispatch + every kernel not seen in the baseline.

    Pass the app's ``score_regex`` so that a new name matching the target is counted once
    (see ``new_kernel_ns``).
    """
    return int(sample["target_ns"]) + new_kernel_ns(sample, baseline_kernel_names, score_regex)


def baseline_other_ns(sample: dict, baseline_samples: list[dict], score_regex: str) -> int:
    """Time spent in the baseline's NON-target kernels, measured in ``sample``."""
    pattern = re.compile(score_regex)
    other_names = {
        name
        for base in baseline_samples
        for name in base["kernels"]
        if not pattern.search(name)
    }
    return sum(ns for name, ns in sample["kernels"].items() if name in other_names)


def integrity_guards(
    baseline_samples: list[dict],
    optimized_samples: list[dict],
    score_regex: str,
) -> dict[str, Any]:
    """Evaluate the two integrity guards on sample means.

    G-other: mean(non-target baseline kernels, optimized) <= 1.10 * mean(baseline) + 50 us.
    G-wall:  mean(wall, optimized) <= 1.5 * mean(wall, baseline) + 1 s.

    Returns:
        dict with the measured values, limits and pass flags (``ok`` = both pass)

    """
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    base_other = _mean([baseline_other_ns(s, baseline_samples, score_regex) for s in baseline_samples])
    opt_other = _mean([baseline_other_ns(s, baseline_samples, score_regex) for s in optimized_samples])
    other_limit = G_OTHER_FACTOR * base_other + G_OTHER_SLACK_NS
    base_wall = _mean([float(s["wall_s"]) for s in baseline_samples])
    opt_wall = _mean([float(s["wall_s"]) for s in optimized_samples])
    wall_limit = G_WALL_FACTOR * base_wall + G_WALL_SLACK_S
    return {
        "other_ns_baseline": base_other,
        "other_ns_optimized": opt_other,
        "other_ns_limit": other_limit,
        "g_other_ok": opt_other <= other_limit,
        "wall_s_baseline": base_wall,
        "wall_s_optimized": opt_wall,
        "wall_s_limit": wall_limit,
        "g_wall_ok": opt_wall <= wall_limit,
        "ok": opt_other <= other_limit and opt_wall <= wall_limit,
    }
