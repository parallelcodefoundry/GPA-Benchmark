"""J0 protocol entry points for APPEB (GPA-G1 J0-A, agent M's M.md T0-T6).

``rescore_kernel`` runs the runner-final J0 protocol (T0 VRAM reset before every timed process,
T1 ABBA pairs, T3 estimator, T4 one pooled re-measure when unstable, T5 regime) for ONE kernel
file on ONE GCD, and returns the scoring result. APPEB's post-batch multi-GCD rescoring (T6,
agent C) loops it over GCDs/nodes. Optionally a random-size perturbing allocation runs first so
the GCD's VRAM state is decorrelated from the batch (M.md T6).
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import yaml

from gpa_bench_driver.driver_src.driver_models import DriverConfig
from gpa_bench_driver.driver_src.driver_rocprof import score_frontier, score_frontier_pooled
from gpa_bench_driver.driver_src.driver_t0 import VramResetTool, snapshot
from gpa_bench_driver.driver_src.driver_utils import DriverInfraError

_GPA_ROOT = Path(__file__).resolve().parent.parent.parent


def app_entry(app: str, gpa_root: Path | None = None) -> dict:
    root = Path(gpa_root) if gpa_root is not None else _GPA_ROOT
    with (root / "driver_apps.frontier.yaml").open() as f:
        for entry in yaml.safe_load(f)["apps"]:
            if app in (entry["name"], *entry.get("aliases", [])):
                return entry
    msg = f"unknown Frontier GPA app {app!r}"
    raise ValueError(msg)


def score_kwargs(entry: dict) -> dict[str, Any]:
    """The per-app keyword arguments of score_frontier (J0 protocol)."""
    return {
        "fixed_target_dispatches": bool(entry.get("fixed_target_dispatches")),
        "cpu_sigma_s": entry.get("cpu_sigma_s"),
        "cpu_floor_s": float(entry.get("cpu_floor_s", 0.10)),
        "protocol": "j0",
        "level_tol": float(entry.get("level_tol", 0.025)),
        "placement_levels_ms": entry.get("placement_levels_ms"),
    }


def vram_reset_snapshot(gpa_root: Path | None = None) -> dict:
    """K3: {"path", "sha256", "record_sha256"} of frontier_tools/vram_reset. The runner takes it
    BEFORE the agent runs, passes ``sha256`` as DriverConfig.vram_reset_sha256 and re-checks it
    after the agent (R9). Raises DriverInfraError when the binary or its record is missing, or they differ."""
    return snapshot(gpa_root)


def _perturb(gpu_device: int, gib: float, gpa_root: Path, env: dict | None = None,
             expected_sha256: str | None = None) -> None:
    run_env = dict(env or os.environ)
    for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"):
        run_env.pop(var, None)
    run_env["ROCR_VISIBLE_DEVICES"] = str(gpu_device)
    with VramResetTool(expected_sha256, gpa_root) as tool:
        tool.run(run_env, gpa_root, (f"{gib:.3f}",), timeout=60)


def _one_series(app: str, kernel_name: str, kernel_text: str, gcd: int, pairs: int,
                temp_dir: Path, overrides: dict | None, reference_from_baseline: bool,
                vram_reset_sha256: str | None = None):
    from gpa_bench_driver.gpa_bench_driver import run_driver

    cfg = DriverConfig(app=app, gpu_backend="hip", nsys=True, pairs=pairs,
                       swaps_override={Path(kernel_name): f"// {kernel_name}\n" + kernel_text},
                       temp_dir=temp_dir, gpu_device=gcd, timeout=1500,
                       app_overrides=overrides, reference_from_baseline=reference_from_baseline,
                       vram_reset_sha256=vram_reset_sha256)
    _, _, long = run_driver(cfg)
    passes = long[app]
    if len(passes) < 2:
        msg = f"run_driver returned {len(passes)} pass(es) for {app}; expected baseline + swap"
        raise DriverInfraError(msg)
    return passes[0], passes[1]


def needs_remeasure(result: dict) -> bool:
    """T4 / H1: re-measure once when the timing was unstable, or when the only failure is a
    G-cpu miss by less than 2 x slack (the runner's rule, frameworks' gpa_frontier_score)."""
    if (result.get("j0") or {}).get("unstable"):
        return True
    codes = [f.get("code") for f in result.get("failures", [])]
    return (not result.get("ok") and codes == ["G-cpu"]
            and bool((result.get("cpu") or {}).get("marginal")))


def _summary(result: dict) -> dict[str, Any]:
    j = result.get("j0") or {}
    cpu = result.get("cpu") or {}
    return {"ok": result.get("ok"), "speedup": j.get("speedup"), "unstable": j.get("unstable"),
            "lower_bound": j.get("lower_bound"), "credited": j.get("credited"),
            "failures": [f.get("code") for f in result.get("failures", [])],
            "cpu_delta_s": cpu.get("delta_s"), "cpu_slack_s": cpu.get("slack_s")}


def rescore_kernel(
    app: str,
    kernel_text: str,
    gcd: int,
    *,
    pairs: int | None = None,
    gpa_root: Path | None = None,
    temp_dir: Path,
    perturb_gib: float | None = None,
    overrides: dict | None = None,
    reference_from_baseline: bool = False,
    seed: int | None = None,
    vram_reset_sha256: str | None = None,
) -> dict[str, Any]:
    """Score one kernel file against the pristine baseline on one GCD with the J0 protocol.

    Args:
        app: Frontier GPA app name
        kernel_text: full text of the kernel file to score (the agent's final kernel)
        gcd: GCD index (ROCR_VISIBLE_DEVICES)
        pairs: ABBA pairs (default: the app's yaml final_pairs, 10)
        gpa_root: GPA-Benchmark root (default: this checkout)
        temp_dir: scratch dir for the working copies (node-local; created if missing)
        perturb_gib: if given (> 0), allocate this many GiB on the GCD and free it before the
            series (T6 decorrelation); if None and ``seed`` is given, a random 0-8 GiB is drawn
        overrides / reference_from_baseline: for an input variant (R7), as in DriverConfig
        seed: seed for the random perturb size
        vram_reset_sha256: the pre-agent/pre-batch sha256 of frontier_tools/vram_reset (K3);
            required (None is refused by the driver: fail closed)

    Returns:
        the score_frontier result, plus "gcd", "perturb_gib", "pairs", "remeasured", the pass
        validation state ("validate", "validation_output") of the kernel's pass and, when
        re-measured (:func:`needs_remeasure`), "first": a summary of the first series' score

    """
    root = Path(gpa_root) if gpa_root is not None else _GPA_ROOT
    if not vram_reset_sha256:  # fix round 6: fail closed before building anything
        msg = ("rescore_kernel needs vram_reset_sha256 (the pre-batch snapshot of "
               "frontier_tools/vram_reset, driver_j0.vram_reset_snapshot())")
        raise DriverInfraError(msg)
    entry = app_entry(app, root)
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    m = int(pairs or entry.get("final_pairs", 10))
    kernel_name = Path(entry["kernel_file"]).name
    if perturb_gib is None and seed is not None:
        perturb_gib = random.Random(seed).uniform(0.0, 8.0)  # noqa: S311
    if perturb_gib:
        _perturb(gcd, perturb_gib, root, expected_sha256=vram_reset_sha256)
    kw = score_kwargs(entry)
    base, swap = _one_series(entry["name"], kernel_name, kernel_text, gcd, m, Path(temp_dir),
                             overrides, reference_from_baseline, vram_reset_sha256)
    out: dict[str, Any] = {"gcd": gcd, "perturb_gib": perturb_gib, "pairs": m,
                           "validate": swap.validate,
                           "validation_output": swap.validation_output}
    if not swap.validate or not swap.nsys_data or not swap.baseline_nsys_data:
        out.update({"ok": False, "speedup": None, "remeasured": False,
                    "failures": [{"code": "correctness",
                                  "message": swap.validation_output or "no timing data"}]})
        return out
    result = score_frontier(swap.baseline_nsys_data, swap.nsys_data, entry["score_regex"], **kw)
    if needs_remeasure(result):  # T4 / H1: one pooled re-measure
        out["first"] = _summary(result)
        base2, swap2 = _one_series(entry["name"], kernel_name, kernel_text, gcd, m,
                                   Path(temp_dir), overrides, reference_from_baseline,
                                   vram_reset_sha256)
        if not swap2.validate:  # R4: every timed run is validated, the re-measure too
            out.update({"ok": False, "speedup": None, "remeasured": True,
                        "failures": [{"code": "correctness",
                                      "message": swap2.validation_output or "no timing data"}]})
            return out
        if swap2.nsys_data and swap2.baseline_nsys_data:
            result = score_frontier_pooled([swap.baseline_nsys_data, swap2.baseline_nsys_data],
                                           [swap.nsys_data, swap2.nsys_data],
                                           entry["score_regex"], **kw)
    out.update(result)
    out["remeasured"] = bool(result.get("j0") and result["j0"].get("remeasured"))
    return out
