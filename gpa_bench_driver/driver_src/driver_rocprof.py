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
        "cpu_s": float | None,      # user+sys CPU of the app process tree during THIS profiled run
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
from gpa_bench_driver.driver_src import driver_j0_rule as _rule
from gpa_bench_driver.driver_src.driver_utils import (
    DriverInfraError,
    SubprocessRunner,
    get_run_path,
    head_tail,
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


class ProfilerOnlyError(DriverInfraError):
    """The profiler never started the app although the same binary runs fine without it (F5/H3).
    Infra on the original's side; on the swap side the driver retries the run once and a repeat
    is the program's own failure (fix round 5 K4)."""


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


def _app_output_path(app: dict, run_path: Path) -> Path | None:
    if "test_output" not in app:
        return None
    return run_path / Path(str(app["test_output"])).name


def profile_once(
    app: dict,
    runner: SubprocessRunner,
    run_path: Path,
    outdir: Path,
    *,
    score_regex: str,
    rocm_path: Path | None = None,
    validate: Callable[[subprocess.CompletedProcess, Path], tuple[bool, str | None]]
    | None = None,
    retain_profiles: bool = False,
) -> dict[Hashable, Any]:
    """One rocprofv3 kernel-trace run of the app in run_path: its sample dict (see module doc).

    The app's test_output file is deleted first; ``validate`` checks this run's output.

    Raises:
        RocprofError: the app fails under the profiler (and also without it)
        DriverInfraError: the profiler failed or wrote no trace although the same binary runs
            fine without it (F5)

    """
    if outdir.exists():
        shutil.rmtree(outdir)
    out_file = _app_output_path(app, run_path)
    if out_file is not None:  # never let a previous run's output validate this one
        out_file.unlink(missing_ok=True)
    command = rocprofv3_command(rocm_path, outdir, app["run_command"].split())
    start = time.perf_counter()
    result = runner.run(command, run_path, measure_cpu=True)
    wall_s = time.perf_counter() - start
    trace_files = find_kernel_trace_files(outdir) if result.returncode == 0 else []
    if result.returncode != 0 or not trace_files:
        what = (f"rocprofv3 timing run failed with return code {result.returncode}"
                if result.returncode != 0 else f"rocprofv3 wrote no kernel trace under {outdir}")
        stderr = head_tail(result.stderr)
        # H3: infra ONLY when the profiler never started the app (its output dir was not even
        # created) AND the plain binary is fine. Otherwise a non-zero exit is the AGENT's program
        # (a lucky plain rerun must not launder an intermittent kernel crash).
        if not outdir.exists():
            plain = runner.run(app["run_command"].split(), run_path)
            if plain.returncode == 0:
                msg = (f"{what} before the app started (output dir never created), but the same "
                       f"binary runs fine without the profiler (profiler/harness failure): {stderr}")
                raise ProfilerOnlyError(msg)
        msg = f"the program failed under the profiler (return code {result.returncode})"
        if stderr:
            msg += f": {stderr}"
        raise RocprofError(msg)
    sample = summarize_kernel_trace(read_kernel_trace(trace_files), score_regex)
    sample["wall_s"] = wall_s
    user = getattr(result, "cpu_user_s", None)
    sys_t = getattr(result, "cpu_sys_s", None)
    sample["cpu_s"] = (user + sys_t) if (user is not None and sys_t is not None) else None
    sample["backend"] = BACKEND_NAME
    sample["valid"] = None
    sample["validation_output"] = None
    if validate is not None:
        ok, message = validate(result, run_path)
        sample["valid"] = bool(ok)
        sample["validation_output"] = None if ok else message
    if not retain_profiles:
        shutil.rmtree(outdir, ignore_errors=True)
    return sample


def cpu_run_once(
    app: dict,
    runner: SubprocessRunner,
    run_path: Path,
    *,
    validate: Callable[[subprocess.CompletedProcess, Path], tuple[bool, str | None]]
    | None = None,
) -> dict[str, Any]:
    """One UNPROFILED run of the app: its CPU time (user + sys, all threads), wall time, check.

    Raises:
        DriverInfraError: the CPU time could not be measured
        RocprofError: the program fails (non-zero exit)

    """
    out_file = _app_output_path(app, run_path)
    if out_file is not None:
        out_file.unlink(missing_ok=True)
    start = time.perf_counter()
    result = runner.run(app["run_command"].split(), run_path, measure_cpu=True)
    wall_s = time.perf_counter() - start
    if result.returncode != 0:
        msg = (f"unprofiled run failed with return code {result.returncode}: "
               f"{head_tail(result.stderr)}")
        raise RocprofError(msg)
    user = getattr(result, "cpu_user_s", None)
    sys_t = getattr(result, "cpu_sys_s", None)
    if user is None or sys_t is None:
        msg = "the program's CPU time could not be measured (runner without rusage support)"
        raise DriverInfraError(msg)
    sample: dict[str, Any] = {"cpu_s": user + sys_t, "user_s": user, "sys_s": sys_t,
                              "wall_s": wall_s, "valid": None, "validation_output": None}
    if validate is not None:
        ok, message = validate(result, run_path)
        sample["valid"] = bool(ok)
        sample["validation_output"] = None if ok else message
    return sample


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
        sample = profile_once(app, runner, run_path, outdir, score_regex=score_regex,
                              rocm_path=rocm_path, validate=validate,
                              retain_profiles=retain_profiles)

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
    """R1 terms of one run.

    target_ns (every dispatch matching score_regex), new_ns (kernels absent from the baseline
    and not matching score_regex), other_ns (the baseline-known non-target kernels in this run).
    scored_ns = target_ns + new_ns. (other_charge_ns = max(0, other_ns - baseline mean) is kept
    for reference; score_frontier charges the increase on MEANS, fix round 2 F1.)
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
        "scored_ns": target_ns + new_ns,
        "new_kernels": new_kernels,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


# J0 protocol (M.md T3): level tolerance for the instability flag and the credit floor
# J0 (M.md T2-T5): the credit rule and estimator live in driver_j0_rule (one rule everywhere, K2)
J0_LEVEL_TOL = _rule.J0_LEVEL_TOL
J0_CREDIT_FLOOR = _rule.J0_CREDIT_FLOOR


def _median(values: list[float]) -> float:
    try:
        return _rule.median(values)
    except ValueError as exc:
        raise ScoringError(str(exc)) from exc


def _robust_spread(values: list[float]) -> float:
    """(2nd largest - 2nd smallest) / median; 0 for fewer than 4 values (M.md T3)."""
    return _rule.robust_spread(values)


def _lower_bound_index(m: int) -> int:
    """Order statistic of the pair ratios that bounds their median from below (binomial, K2)."""
    return _rule.lower_bound_index(m)


def _j0_estimate(b_samples: list[dict], o_samples: list[dict], b_scored: list[float],
                 o_scored: list[float], charge: float, level_tol: float,
                 placement_levels_ms: dict | None) -> dict[str, Any]:
    """M.md T3/T5 via :func:`driver_j0_rule.j0_estimate`, paired by the samples' ABBA ``pair``
    index (by position when the samples carry none)."""
    try:
        return _rule.j0_estimate(
            b_scored, o_scored, charge_ns=charge,
            pairs_b=[s.get("pair", i) for i, s in enumerate(b_samples)],
            pairs_o=[s.get("pair", i) for i, s in enumerate(o_samples)],
            level_tol=level_tol, placement_levels_ms=placement_levels_ms)
    except ValueError as exc:
        raise ScoringError(str(exc)) from exc


def _t0_record(baseline_samples: list[dict], optimized_samples: list[dict]) -> dict[str, Any]:
    """T0 evidence for the sidecar: the per-sample VRAM reset wall times (s)."""
    rb = [s.get("vram_reset_s") for s in baseline_samples]
    ro = [s.get("vram_reset_s") for s in optimized_samples]
    vals = [float(x) for x in rb + ro if x is not None]
    return {"vram_reset_s": {"baseline": rb, "optimized": ro}, "n_resets": len(vals),
            "median_s": _rule.median(vals) if vals else None,
            "max_s": max(vals) if vals else None,
            "all_samples_reset": bool(vals) and len(vals) == len(rb) + len(ro)}


def score_frontier_pooled(
    baseline_series: list[list[dict]],
    optimized_series: list[list[dict]],
    score_regex: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """T4: score several ABBA series of one phase as one pooled series (e.g. the first series
    and its re-measure). Pair indices are offset per series so pairs keep their own ratios;
    the lower bound uses the pooled pair count (m=20 -> 5th smallest)."""
    if len(baseline_series) != len(optimized_series) or not baseline_series:
        msg = "pooled scoring needs the same number (>= 1) of baseline and optimized series"
        raise ScoringError(msg)
    bs: list[dict] = []
    os_: list[dict] = []
    offset = 0
    for b_ser, o_ser in zip(baseline_series, optimized_series):
        n = max([s.get("pair", i) for i, s in enumerate(b_ser)] + [-1]) + 1
        bs += [{**s, "pair": offset + s.get("pair", i)} for i, s in enumerate(b_ser)]
        os_ += [{**s, "pair": offset + s.get("pair", i)} for i, s in enumerate(o_ser)]
        offset += n
    result = score_frontier(bs, os_, score_regex, **kwargs)
    if result.get("j0") is not None:
        result["j0"]["remeasured"] = len(baseline_series) > 1
        result["j0"]["n_series"] = len(baseline_series)
    return result


def _cpu_values(samples: list, label: str) -> list[float]:
    out = []
    for i, v in enumerate(samples):
        cpu = v.get("cpu_s") if isinstance(v, dict) else v
        if cpu is None:
            msg = f"{label} sample {i} has no cpu_s (G-cpu is measured in the profiled runs, H1)"
            raise ScoringError(msg)
        out.append(float(cpu))
    return out


def cpu_guard(
    baseline_samples: list,
    optimized_samples: list,
    cpu_sigma_s: float | None,
    *,
    cpu_k: float = 5.0,
    cpu_floor_s: float = 0.10,
) -> dict[str, Any]:
    """G-cpu (fix round 3 H1): the program's own CPU time (measured in the profiled, scored runs)
    must not grow beyond the noise, so in-process profiler detection cannot separate a scored run
    from a measured one.

    ``cpu_s`` is read from each sample dict (user+sys of the app process tree during that
    profiled run). slack = max(cpu_k * cpu_sigma_s / sqrt(n), cpu_floor_s); FAIL when
    mean(optimized) - mean(baseline) > slack. ``marginal`` is True when it fails by < 2*slack
    (the caller re-measures on a pooled series before deciding, H1).

    Raises:
        ScoringError: empty or differently sized lists, a sample without cpu_s, or no cpu_sigma_s

    """
    base = _cpu_values(list(baseline_samples or []), "baseline")
    opt = _cpu_values(list(optimized_samples or []), "optimized")
    if not base or not opt:
        msg = "G-cpu needs at least one baseline and one optimized profiled sample"
        raise ScoringError(msg)
    if len(base) != len(opt):
        msg = f"{len(base)} baseline samples but {len(opt)} optimized samples for G-cpu"
        raise ScoringError(msg)
    if cpu_sigma_s is None:
        msg = "G-cpu needs the app's calibrated cpu_sigma_s"
        raise ScoringError(msg)
    n = len(opt)
    slack = max(cpu_k * float(cpu_sigma_s) / math.sqrt(n), cpu_floor_s)
    delta = _mean(opt) - _mean(base)
    ok = delta <= slack
    return {"checked": True, "baseline_mean_s": _mean(base), "optimized_mean_s": _mean(opt),
            "delta_s": delta, "sigma_s": float(cpu_sigma_s), "n_pairs": n, "k": cpu_k,
            "floor_s": cpu_floor_s, "slack_s": slack, "ok": ok,
            "marginal": (not ok) and delta <= slack + 2.0 * slack,
            "baseline_s": base, "optimized_s": opt}


def pooled_cpu_ok(baseline_samples: list, optimized_samples: list, cpu_sigma_s: float | None,
                  *, cpu_k: float = 5.0, cpu_floor_s: float = 0.10) -> dict[str, Any]:
    """G-cpu on a pooled (re-measured) series; same return shape as :func:`cpu_guard`."""
    return cpu_guard(baseline_samples, optimized_samples, cpu_sigma_s, cpu_k=cpu_k,
                     cpu_floor_s=cpu_floor_s)


def score_frontier(
    baseline_samples: list[dict],
    optimized_samples: list[dict],
    score_regex: str,
    *,
    fixed_target_dispatches: bool = False,
    cpu_sigma_s: float | None = None,
    cpu_k: float = 5.0,
    cpu_floor_s: float = 0.10,
    protocol: str = "j0",
    level_tol: float = J0_LEVEL_TOL,
    placement_levels_ms: dict | None = None,
) -> dict[str, Any]:
    """Apply the Frontier GPA scoring rule to baseline and optimized samples.

    protocol "j0" (GPA-G1 J0, M.md T2-T5; the default): medians and ABBA pair ratios, see
    :func:`_j0_estimate`. protocol "mean": the fix-round-1..4 ratio of means (kept for
    comparison and old records).

    Scored time per run = target + kernels new to the baseline (:func:`score_terms`); the
    increase of the baseline's other kernels is charged on MEANS: max(0, mean(other_opt) -
    mean(other_base)) is added to the optimized mean. speedup = mean(baseline scored) /
    mean(optimized scored + charge). Failures (no speedup credited): R2 ``launch-count`` (only
    with fixed_target_dispatches), R3 ``G-wall`` (mean optimized wall > 1.5 x baseline + 1 s),
    F4/H1 ``G-cpu`` (CPU read from the profiled samples; see :func:`cpu_guard`), R11 ``G-zero``.

    Raises:
        ScoringError: empty or differently sized sample lists, malformed samples, baseline runs
            that disagree on the launch count when it must stay fixed, bad CPU inputs

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
    other_opt = _mean(o["other_ns"])
    other_j0: dict[str, float] = {}
    if protocol == "j0":  # T2 on medians, minus a noise tolerance (fix round 5)
        other_j0 = _rule.other_charge(b["other_ns"], o["other_ns"])
        charge = other_j0["charge_ns"]
        # the per-sample lists hold the charge actually applied (not legacy per-run values)
        b["other_charge_ns"] = [0.0] * len(b["other_ns"])
        o["other_charge_ns"] = [charge] * len(o["other_ns"])
    else:
        charge = max(0.0, other_opt - base.other_mean_ns)
    o["mean_scored_ns"] = _mean(o["scored_ns"]) + charge
    b["median_scored_ns"] = _median(b["scored_ns"])
    o["median_scored_ns"] = _median(o["scored_ns"]) + charge
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

    # G-cpu (H1): CPU is read from the profiled samples themselves. It is checked whenever the
    # app has a calibrated cpu_sigma_s; a sample missing cpu_s FAILS closed (H4).
    cpu: dict[str, Any] = {"checked": False}
    if cpu_sigma_s is not None:
        cpu = cpu_guard(baseline_samples, optimized_samples, cpu_sigma_s, cpu_k=cpu_k,
                        cpu_floor_s=cpu_floor_s)
        if not cpu["ok"]:
            failures.append({"code": "G-cpu", "message": (
                f"CPU TIME: your program used {cpu['optimized_mean_s']:.3f} s of CPU per run, "
                f"{cpu['delta_s']:.3f} s more than the original's {cpu['baseline_mean_s']:.3f} s "
                f"(allowed: {cpu['slack_s']:.3f} s of noise); work moved from the GPU to the CPU "
                "is not credited." + (" [marginal: re-measure]" if cpu["marginal"] else ""))})

    raw = None
    j0: dict[str, Any] | None = None
    key = "median_scored_ns" if protocol == "j0" else "mean_scored_ns"
    if b[key] > 0 and o[key] > 0:
        if protocol == "j0":
            j0 = _j0_estimate(baseline_samples, optimized_samples, b["scored_ns"], o["scored_ns"],
                              charge, level_tol, placement_levels_ms)
            raw = j0["speedup"]
            # unstable: only the lower bound can carry credit (j0_credited); below it -> failure
            if j0["unstable"] and not j0["lower_bound"] > J0_CREDIT_FLOOR:
                failures.append({"code": "unstable", "message": (
                    f"TIMING UNSTABLE on this GCD (the VRAM placement changed between runs): "
                    f"speedup {raw:.4f} (pair ratios {min(j0['pair_ratios']):.4f}.."
                    f"{max(j0['pair_ratios']):.4f}, lower bound {j0['lower_bound']:.4f} is not "
                    f"above {J0_CREDIT_FLOOR}).")})
        else:
            raw = b["mean_scored_ns"] / o["mean_scored_ns"]
    else:
        side_name = "original" if b[key] <= 0 else "optimized"
        failures.append({"code": "G-zero", "message": (
            f"NO GPU TIME: the {side_name} program's scored GPU time is zero, so no speedup can "
            "be computed.")})

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
            "charge_ns": charge,
            "charge_mean_ns": charge,  # alias (fix round 1 name)
            "ratio": other_opt / base.other_mean_ns if base.other_mean_ns > 0 else None,
            **{k: v for k, v in other_j0.items() if k != "charge_ns"},
        },
        "t0": _t0_record(baseline_samples, optimized_samples),
        "cpu": cpu,
        "g_wall": {"baseline_s": base.wall_mean_s, "optimized_s": o["wall_mean_s"],
                   "limit_s": wall_limit, "ok": wall_ok},
        "launch_count": {"required": bool(fixed_target_dispatches),
                         "baseline": base.target_dispatches,
                         "optimized": o["target_dispatches"], "ok": launch_ok},
        "rule": {"g_wall_factor": G_WALL_FACTOR, "g_wall_slack_s": G_WALL_SLACK_S},
        "protocol": protocol,
        "j0": j0,
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
