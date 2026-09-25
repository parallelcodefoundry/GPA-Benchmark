"""rocprofv3 kernel timing for the hip backend (AMD GPUs, e.g. Frontier MI250X).

The hip backend replaces the Nsight Systems timing of the cuda backend. Each sample runs the app
once under ``rocprofv3 --kernel-trace --output-format csv`` and reduces the kernel trace to one
dict (stored in ``DriverPassResult.nsys_data`` so existing consumers keep working):

    {
        "exec_time": int,           # ns; == target_ns
        "target_ns": int,           # sum of the durations of EVERY dispatch of the target kernel
        "target_dispatches": int,   # number of dispatches of the target kernel
        "kernels": {name: int},     # total ns per demangled kernel name, ALL kernels
        "kernel_dispatches": {name: int},
        "wall_s": float,            # wall time of the profiled run (seconds)
        "backend": "rocprofv3",
        "valid": bool | None,       # R4: this run's output passed the app's check (None: unchecked)
        "validation_output": str | None,
    }

Dispatch duration = End_Timestamp - Start_Timestamp (ns). No kernel is excluded by name (GPA-G1
fix round 1, R1: runtime-looking names such as ``__amd_rocclr_*`` are ordinary kernels). The
target kernel is every kernel whose demangled name matches the app's ``score_regex`` (re.search,
anchored in the YAML).

The Frontier scoring rule needs the baseline pass, so consumers compute it from the samples with
:func:`score_frontier` (R1 scored time incl. the other-kernel charge, R2 launch count, R3 G-wall,
R11 fail-closed input checks).
"""

from __future__ import annotations

import csv
import logging
import math
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
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
    """True for names of ROCm runtime-internal kernels (copy/fill blits).

    Informational only: since fix round 1 (R1) no kernel is excluded from scoring by its name.
    """
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
        dict with exec_time, target_ns, target_dispatches, kernels, kernel_dispatches
        (every kernel counts; nothing is dropped by name)

    """
    pattern = re.compile(score_regex)
    kernels: dict[str, int] = {}
    dispatches: dict[str, int] = {}
    target_ns = 0
    target_dispatches = 0
    for name, duration in rows:
        kernels[name] = kernels.get(name, 0) + duration
        dispatches[name] = dispatches.get(name, 0) + 1
        if pattern.search(name):
            target_ns += duration
            target_dispatches += 1
    return {
        "exec_time": target_ns,
        "target_ns": target_ns,
        "target_dispatches": target_dispatches,
        "kernels": kernels,
        "kernel_dispatches": dispatches,
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
    validate: Callable[[subprocess.CompletedProcess, Path], tuple[bool, str | None]]
    | None = None,
) -> list[dict[Hashable, Any]] | None:
    """Time the app with rocprofv3 kernel tracing, one run per sample.

    Before every sample the app's test_output file is deleted; after it, ``validate`` (if given)
    checks that sample's output (R4), and the result lands in the sample's ``valid`` /
    ``validation_output``. The full stdout is kept (the check may need it).

    Args:
        app: Application configuration dictionary (needs run_command and score_regex or
            kernel_name)
        runner: Configured subprocess runner (its env pins the GPU)
        temp_dir: Temporary directory holding the working copy; traces go to temp_dir/profiles
        num_samples: Number of profiled runs
        swap_config: Swap configuration (None for the baseline pass)
        pbar: Progress bar callback (advanced twice per sample, like nsys profile + post)
        rocm_path: ROCm install whose bin/rocprofv3 is used (else rocprofv3 from PATH)
        retain_profiles: Keep the rocprofv3 output directories after parsing
        validate: (completed process, run dir) -> (ok, message) for the sample's output

    Returns:
        One timing dict per sample, or None when the baseline pass has no dispatch of the
        target kernel (a configuration error, reported as a warning like the cuda backend)

    Raises:
        RocprofError: if a profiled run fails or produces no kernel trace

    """
    profile_dir = setup_profile_dir(temp_dir)
    run_path = get_run_path(app, temp_dir)
    score_regex = get_score_regex(app)
    samples: list[dict[Hashable, Any]] = []

    for i in range(num_samples):
        outdir = profile_dir / _profile_dir_name(app, swap_config, i)
        if outdir.exists():
            shutil.rmtree(outdir)
        if "test_output" in app:  # never let a previous run's output validate this one
            (run_path / Path(str(app["test_output"])).name).unlink(missing_ok=True)
        command = rocprofv3_command(rocm_path, outdir, app["run_command"].split())
        start = time.perf_counter()
        result = runner.run(command, run_path)
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
        sample["valid"] = None
        sample["validation_output"] = None
        if validate is not None:
            ok, message = validate(result, run_path)
            sample["valid"] = bool(ok)
            sample["validation_output"] = None if ok else message

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
# The Frontier scoring rule (GPA-G1 fix round 1: R1, R2, R3, R11). Consumers: gpa_test, runner.
# ---------------------------------------------------------------------------------------------


class ScoringError(ValueError):
    """The samples cannot be scored (fail closed, R11): empty or mismatched lists, bad samples."""


_REQUIRED_SAMPLE_KEYS = ("kernels", "target_ns", "target_dispatches", "wall_s")


def _check_samples(samples: list[dict], label: str) -> None:
    if not samples:
        msg = f"no {label} samples to score"
        raise ScoringError(msg)
    for i, sample in enumerate(samples):
        if not isinstance(sample, dict):
            msg = f"{label} sample {i} is not a dict"
            raise ScoringError(msg)
        for key in _REQUIRED_SAMPLE_KEYS:
            if sample.get(key) is None:
                msg = f"{label} sample {i} has no {key!r}"
                raise ScoringError(msg)
        wall = float(sample["wall_s"])
        if not math.isfinite(wall) or wall < 0:
            msg = f"{label} sample {i} has an invalid wall_s {sample['wall_s']!r}"
            raise ScoringError(msg)
        if not isinstance(sample["kernels"], dict):
            msg = f"{label} sample {i}: 'kernels' is not a dict"
            raise ScoringError(msg)


@dataclass(frozen=True)
class BaselineSummary:
    """What scoring needs from the baseline (pristine) samples."""

    score_regex: str
    known_names: frozenset[str]
    other_names: frozenset[str]
    other_mean_ns: float
    target_dispatches: int | None
    wall_mean_s: float


def summarize_baseline(baseline_samples: list[dict], score_regex: str) -> BaselineSummary:
    """Summarize the baseline samples (R1): known/other kernel names, mean other-kernel time.

    Raises:
        ScoringError: empty list or malformed samples

    """
    _check_samples(baseline_samples, "baseline")
    pattern = re.compile(score_regex)
    known = frozenset(name for s in baseline_samples for name in s["kernels"])
    other = frozenset(name for name in known if not pattern.search(name))
    other_sums = [sum(ns for name, ns in s["kernels"].items() if name in other)
                  for s in baseline_samples]
    counts = {int(s["target_dispatches"]) for s in baseline_samples}
    return BaselineSummary(
        score_regex=score_regex,
        known_names=known,
        other_names=other,
        other_mean_ns=sum(other_sums) / len(other_sums),
        target_dispatches=counts.pop() if len(counts) == 1 else None,
        wall_mean_s=sum(float(s["wall_s"]) for s in baseline_samples) / len(baseline_samples),
    )


def score_terms(sample: dict, base: BaselineSummary) -> dict[str, Any]:
    """R1 scored time of one run and its terms.

    scored_ns = target_ns (every dispatch matching score_regex)
              + new_ns (kernels absent from the baseline and not matching score_regex)
              + other_charge_ns = max(0, other_ns - baseline mean other_ns), where other_ns is
                the time of the baseline-known non-target kernels in this run.
    """
    pattern = re.compile(base.score_regex)
    new_kernels = {name: int(ns) for name, ns in sample["kernels"].items()
                   if name not in base.known_names and not pattern.search(name)}
    other_ns = sum(int(ns) for name, ns in sample["kernels"].items() if name in base.other_names)
    charge = max(0.0, other_ns - base.other_mean_ns)
    new_ns = sum(new_kernels.values())
    target_ns = int(sample["target_ns"])
    return {
        "target_ns": target_ns,
        "new_ns": new_ns,
        "other_ns": other_ns,
        "other_charge_ns": charge,
        "scored_ns": target_ns + new_ns + charge,
        "new_kernels": new_kernels,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def score_frontier(
    baseline_samples: list[dict],
    optimized_samples: list[dict],
    score_regex: str,
    *,
    fixed_target_dispatches: bool = False,
) -> dict[str, Any]:
    """Apply the Frontier GPA scoring rule (fix round 1) to baseline and optimized samples.

    R1 scored time per run (see :func:`score_terms`), speedup = mean(baseline scored) /
    mean(optimized scored). Failures (no speedup credited): R2 ``launch-count`` (only with
    fixed_target_dispatches: every optimized run must launch the target exactly as often as the
    baseline), R3 ``G-wall`` (mean optimized wall > 1.5 x mean baseline wall + 1 s), R11
    ``G-zero`` (a mean scored time <= 0). The other-kernel increase is charged, not guarded; its
    numbers are reported under "other".

    Raises:
        ScoringError: empty or differently sized sample lists, malformed samples, or baseline
            runs that disagree on the target launch count when it must stay fixed

    """
    _check_samples(baseline_samples, "baseline")
    _check_samples(optimized_samples, "optimized")
    if len(baseline_samples) != len(optimized_samples):
        msg = (f"{len(baseline_samples)} baseline samples but {len(optimized_samples)} optimized "
               "samples")
        raise ScoringError(msg)
    base = summarize_baseline(baseline_samples, score_regex)
    if fixed_target_dispatches and base.target_dispatches is None:
        msg = "the baseline runs disagree on the target launch count"
        raise ScoringError(msg)

    def side(samples: list[dict]) -> dict[str, Any]:
        terms = [score_terms(s, base) for s in samples]
        out: dict[str, Any] = {
            key: [t[key] for t in terms]
            for key in ("scored_ns", "target_ns", "new_ns", "other_ns", "other_charge_ns")
        }
        out["target_dispatches"] = [int(s["target_dispatches"]) for s in samples]
        out["wall_s"] = [float(s["wall_s"]) for s in samples]
        out["mean_scored_ns"] = _mean(out["scored_ns"])
        out["wall_mean_s"] = _mean(out["wall_s"])
        totals: dict[str, float] = {}
        for t in terms:
            for name, ns in t["new_kernels"].items():
                totals[name] = totals.get(name, 0.0) + ns
        out["new_kernels"] = {name: v / len(samples) for name, v in totals.items()}
        return out

    b = side(baseline_samples)
    o = side(optimized_samples)
    b["other_mean_ns"] = base.other_mean_ns
    failures: list[dict[str, str]] = []

    launch_ok = True
    if fixed_target_dispatches:
        launch_ok = all(n == base.target_dispatches for n in o["target_dispatches"])
        if not launch_ok:
            failures.append({"code": "launch-count", "message": (
                f"LAUNCH COUNT CHANGED: the target kernel must be launched exactly "
                f"{base.target_dispatches} times per run (warmup + iterations, as the original); "
                f"your runs launched it {o['target_dispatches']} times.")})

    wall_limit = G_WALL_FACTOR * base.wall_mean_s + G_WALL_SLACK_S
    wall_ok = o["wall_mean_s"] <= wall_limit
    if not wall_ok:
        failures.append({"code": "G-wall", "message": (
            f"WALL TIME: your program ran {o['wall_mean_s']:.2f} s per run on average, more than "
            f"{G_WALL_FACTOR} x the original's {base.wall_mean_s:.2f} s + {G_WALL_SLACK_S:.0f} s "
            f"= {wall_limit:.2f} s (work moved to the CPU or to an untimed phase).")})

    raw = None
    if b["mean_scored_ns"] > 0 and o["mean_scored_ns"] > 0:
        raw = b["mean_scored_ns"] / o["mean_scored_ns"]
    else:
        side_name = "original" if b["mean_scored_ns"] <= 0 else "optimized"
        failures.append({"code": "G-zero", "message": (
            f"NO GPU TIME: the {side_name} program's scored GPU time is zero, so no speedup can "
            "be computed.")})

    other_opt = _mean(o["other_ns"])
    return {
        "ok": not failures,
        "speedup": raw if not failures else None,
        "raw_speedup": raw,
        "failures": failures,
        "baseline": b,
        "optimized": o,
        "other": {
            "baseline_mean_ns": base.other_mean_ns,
            "optimized_mean_ns": other_opt,
            "charge_mean_ns": _mean(o["other_charge_ns"]),
            "ratio": other_opt / base.other_mean_ns if base.other_mean_ns > 0 else None,
        },
        "g_wall": {"baseline_s": base.wall_mean_s, "optimized_s": o["wall_mean_s"],
                   "limit_s": wall_limit, "ok": wall_ok},
        "launch_count": {"required": bool(fixed_target_dispatches),
                         "baseline": base.target_dispatches,
                         "optimized": o["target_dispatches"], "ok": launch_ok},
        "rule": {"g_wall_factor": G_WALL_FACTOR, "g_wall_slack_s": G_WALL_SLACK_S},
    }


# ---------------------------------------------------------------------------------------------
# Pre-fix-round-1 helpers, kept importable for existing callers. They are NOT the R1 rule (no
# other-kernel charge); score with score_frontier().
# ---------------------------------------------------------------------------------------------


def new_kernel_ns(
    sample: dict,
    baseline_kernel_names: Iterable[str],
    score_regex: str | None = None,
) -> int:
    """Kernels of ``sample`` absent from ``baseline_kernel_names`` (and not matching score_regex).

    Superseded by :func:`score_terms` (``new_ns``).
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
    """target_ns + new_kernel_ns (no other-kernel charge). Superseded by :func:`score_terms`."""
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
    """Pre-fix-round-1 guards (G-other as a pass/fail guard, G-wall); superseded by score_frontier.

    Raises:
        ScoringError: empty sample lists (fail closed)

    """
    _check_samples(baseline_samples, "baseline")
    _check_samples(optimized_samples, "optimized")

    base_other = _mean([baseline_other_ns(s, baseline_samples, score_regex)
                        for s in baseline_samples])
    opt_other = _mean([baseline_other_ns(s, baseline_samples, score_regex)
                       for s in optimized_samples])
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
