"""J0 measurement protocol (GPA-G1 J0-A; agent M's M.md T0-T5 and its tests a-d).

Series are built from a per-process time function over the GLOBAL run order, so a GCD's
placement-state history (e.g. period-2 alternation) is applied to whichever arm ran at that
position. "abba" = the J0 order; "alt" = the old strict B,O,B,O order (mutation check: under the
old order + mean estimator the same no-op IS falsely credited).
"""

from __future__ import annotations

import json
import statistics
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from gpa_bench_driver.driver_src.driver_rocprof import (
    _lower_bound_index,
    score_frontier,
    score_frontier_pooled,
)

GPA_ROOT = Path(__file__).resolve().parent.parent
T = "k(int)"
RX = r"^k\("


def _series(times_by_order, m: int, order: str = "abba", o_scale: float = 1.0):
    """Build (baseline, optimized) sample lists of m pairs. times_by_order(pos) -> ms for the
    process at global position pos; the optimized arm's time is divided by o_scale (a real
    speedup of o_scale)."""
    b, o = [], []
    pos = 0
    for k in range(m):
        if order == "abba":
            sides = (0, 1) if k % 2 == 0 else (1, 0)
        else:  # strict alternation B,O,B,O
            sides = (0, 1)
        for p_in, side in enumerate(sides):
            ms = times_by_order(pos)
            if side == 1:
                ms = ms / o_scale
            ns = int(ms * 1e6)
            s = {"target_ns": ns, "target_dispatches": 1, "kernels": {T: ns}, "wall_s": 1.0,
                 "cpu_s": 1.0, "pair": k, "pos_in_pair": p_in}
            (b if side == 0 else o).append(s)
            pos += 1
    return b, o


def _credited_j0(b, o, **kw):
    r = score_frontier(b, o, RX, protocol="j0", **kw)
    return bool(r["ok"] and r["j0"]["credited"]), r


def _credited_mean(b, o):
    r = score_frontier(b, o, RX, protocol="mean")
    return bool(r["ok"] and r["raw_speedup"] > 1.005), r


# ---------------------------------------------------------------- (a) period-2 placement state

def _period2(pos):
    return 15.66 if pos % 2 == 0 else 13.90


@pytest.mark.parametrize("m", [6, 10])
def test_a_period2_noop_never_credited_under_j0(m):
    b, o = _series(_period2, m, "abba")
    credited, r = _credited_j0(b, o)
    assert not credited, r["j0"]
    assert r["j0"]["unstable"]  # the placement flip is detected


def test_a_mutation_old_alternation_and_mean_falsely_credit_the_same_noop():
    # the regression's rr_pristine4 case: strict alternation puts every baseline on one state
    b, o = _series(_period2, 5, "alt")
    credited, r = _credited_mean(b, o)
    assert credited and r["raw_speedup"] > 1.10  # 15.66/13.90 = 1.127 -> the old false credit


# ---------------------------------------------------------------- (b) single spike

def test_b_one_plus27_spike_in_one_arm_not_credited():
    base = 0.559

    def spike(pos):
        return base * 1.27 if pos == 0 else base  # first process (a baseline) spikes +27%

    b, o = _series(spike, 6, "abba")
    credited, r = _credited_j0(b, o)
    assert not credited
    assert r["j0"]["speedup"] == pytest.approx(1.0, abs=1e-6)
    # mutation: the mean estimator is pulled up by the spike and falsely credits it
    cm, rm = _credited_mean(b, o)
    assert cm and rm["raw_speedup"] > 1.005


# ---------------------------------------------------------------- (c) step (flip halfway)

def test_c_step_noop_not_credited():
    def step(pos):
        return 15.66 if pos < 10 else 13.90  # the GCD flips state halfway through

    b, o = _series(step, 10, "abba")
    credited, r = _credited_j0(b, o)
    assert not credited, r["j0"]


def test_c_real_107_with_one_split_pair_still_credited_at_m10():
    def step(pos):
        return 15.66 if pos < 9 else 13.90  # the flip lands inside one pair

    b, o = _series(step, 10, "abba", o_scale=1.07)
    credited, r = _credited_j0(b, o)
    assert credited, r["j0"]
    assert r["j0"]["speedup"] == pytest.approx(1.07, rel=0.005)


def test_real_speedup_on_quiet_gcd_credited_at_right_size():
    b, o = _series(lambda pos: 3.043, 6, "abba", o_scale=1.117)
    credited, r = _credited_j0(b, o)
    assert credited and not r["j0"]["unstable"]
    assert r["j0"]["speedup"] == pytest.approx(1.117, rel=1e-3)


# ---------------------------------------------------------------- T3 details, T4, T5

# K2 (fix round 5): binomial quantile, coverage >= 98% for any m >= 6; M.md's j at 6/10/12/20
@pytest.mark.parametrize(("m", "j"), [(4, 0), (6, 0), (7, 0), (8, 0), (10, 1), (11, 1), (12, 2),
                                      (15, 3), (16, 3), (20, 4), (22, 5), (40, 13)])
def test_lower_bound_order_statistic(m, j):
    from gpa_bench_driver.driver_src.driver_j0_rule import lower_bound_coverage

    assert _lower_bound_index(m) == j
    if m >= 6:
        assert lower_bound_coverage(m) >= 0.98


def test_other_kernel_charge_on_medians():
    b, o = _series(lambda pos: 1.0, 6, "abba")
    for s in b:
        s["kernels"]["other(int)"] = 100_000
    for i, s in enumerate(o):  # the optimized run's other kernel is 50 us slower (median)
        s["kernels"]["other(int)"] = 150_000 if i != 0 else 900_000  # one spike
    r = score_frontier(b, o, RX, protocol="j0")
    # median, not pulled by the spike; minus the 1 us tolerance floor (no noise: MADs are 0)
    assert r["other"]["charge_ns"] == pytest.approx(50_000 - 1_000)
    assert r["other"]["tol_ns"] == 1_000
    assert r["optimized"]["other_charge_ns"] == [r["other"]["charge_ns"]] * 6
    assert r["baseline"]["other_charge_ns"] == [0.0] * 6


def _noisy_other(seed, n, shift_ns=0.0, rel=0.02, base_ns=440_000):
    import random

    rng = random.Random(seed)
    return [base_ns * (1 + rng.uniform(-rel, rel)) + shift_ns for _ in range(n)]


def test_other_kernel_charge_is_noise_tolerant_on_a_noop():
    """fix round 5: the one-sided median charge is reduced by a noise tolerance; fix round 6 (L1)
    caps that tolerance at half the floor margin (backprop: 1.1 us), so some bias remains."""
    charged_old = charged_new = 0
    sum_old = sum_new = 0.0
    for seed in range(200):
        b, o = _series(lambda pos: 0.44, 10, "abba")
        for s, v in zip(b, _noisy_other(seed, 10)):
            s["kernels"]["other(int)"] = int(v)
        for s, v in zip(o, _noisy_other(seed + 1000, 10)):
            s["kernels"]["other(int)"] = int(v)
        r = score_frontier(b, o, RX, protocol="j0")
        med = r["other"]["optimized_median_ns"] - r["other"]["baseline_median_ns"]
        assert r["other"]["tol_ns"] == pytest.approx(0.5 * 0.005 * 440_000)  # the L1 cap
        assert r["other"]["charge_ns"] <= max(0.0, med)
        charged_old += med > 0
        charged_new += r["other"]["charge_ns"] > 0
        sum_old += max(0.0, med)
        sum_new += r["other"]["charge_ns"]
    assert charged_new < charged_old and sum_new < 0.85 * sum_old


def test_other_kernel_charge_still_charges_a_real_increase_under_noise():
    b, o = _series(lambda pos: 0.44, 10, "abba")
    for s, v in zip(b, _noisy_other(1, 10)):
        s["kernels"]["other(int)"] = int(v)
    for s, v in zip(o, _noisy_other(2, 10, shift_ns=60_000)):  # +60 us per run, moved work
        s["kernels"]["other(int)"] = int(v)
    r = score_frontier(b, o, RX, protocol="j0")
    tol = r["other"]["tol_ns"]
    assert 60_000 - tol - 6_000 < r["other"]["charge_ns"] < 60_000 + 6_000  # medians' own noise
    assert tol <= 0.5 * 0.005 * 440_000 + 1e-6


def test_T4_pooled_series_uses_pooled_lower_bound():
    b1, o1 = _series(_period2, 10, "abba")
    b2, o2 = _series(lambda pos: 15.66, 10, "abba")
    r = score_frontier_pooled([b1, b2], [o1, o2], RX, protocol="j0")
    assert r["j0"]["n_pairs"] == 20 and r["j0"]["remeasured"] is True
    assert r["j0"]["lower_bound"] == sorted(r["j0"]["pair_ratios"])[4]


def test_T5_regime_is_the_nearest_placement_level():
    b, o = _series(lambda pos: 142.2, 6, "abba", o_scale=1.073)
    r = score_frontier(b, o, RX, protocol="j0",
                       placement_levels_ms={"fast": 142.1, "slow": 156.7})
    assert r["j0"]["regime"] == "fast"
    assert r["j0"]["baseline_level_ms"] == pytest.approx(142.2, rel=1e-3)
    assert score_frontier(b, o, RX, protocol="j0")["j0"]["regime"] is None


def test_unstable_and_low_bound_is_a_failure_with_a_clear_message():
    b, o = _series(_period2, 6, "abba", o_scale=1.02)
    r = score_frontier(b, o, RX, protocol="j0")
    assert r["j0"]["unstable"] and not r["ok"]
    msg = next(f["message"] for f in r["failures"] if f["code"] == "unstable")
    assert "TIMING UNSTABLE" in msg and "lower bound" in msg


def test_yaml_j0_fields():
    with (GPA_ROOT / "driver_apps.frontier.yaml").open() as f:
        apps = {a["name"]: a for a in yaml.safe_load(f)["apps"]}
    for name, a in apps.items():
        assert a["test_pairs"] % 2 == 0 and a["final_pairs"] % 2 == 0, name
        assert a["level_tol"] == 0.025, name
    assert set(apps["bfs"]["placement_levels_ms"]) == {"F", "M", "S"}
    assert set(apps["nw"]["placement_levels_ms"]) == {"fast", "slow"}
    assert all("placement_levels_ms" not in apps[n] for n in apps if n not in ("bfs", "nw"))


# ---------------------------------------------------------------- (d) reset failure -> infra

_TOY_RUN = textwrap.dedent('''\
    #!/bin/bash
    . ./kernel.cu
    echo "result 42" > output.txt
    echo "toy_kernel(int),1000" >> kernels.csv
    ''')
_TOY_MAKE = "toy: kernel.cu run.sh\n\tcp run.sh toy && chmod +x toy\nclean:\n\trm -f toy\n"
_FAKE_ROCPROF = textwrap.dedent('''\
    #!/bin/bash
    while [ "$1" != "--" ]; do [ "$1" = -d ] && out=$2; shift; done; shift
    rm -f kernels.csv
    "$@" || exit $?
    mkdir -p "$out"
    { echo '"Kind","Kernel_Name","Start_Timestamp","End_Timestamp"'
      t=100; while IFS=, read -r n d; do echo "\\"KERNEL_DISPATCH\\",\\"$n\\",$t,$((t+d))"; t=$((t+d+1)); done < kernels.csv
    } > "$out/trace_kernel_trace.csv"
    ''')
_DRIVE = textwrap.dedent('''\
    import json, sys
    from pathlib import Path
    root = Path(sys.argv[1]); sys.path.insert(0, str(root))
    from gpa_bench_driver.gpa_bench_driver import run_driver
    from gpa_bench_driver.driver_src.driver_models import DriverConfig
    from gpa_bench_driver.driver_src.driver_utils import DriverInfraError
    cfg = DriverConfig(app="toy", gpu_backend="hip", rocm_path=root / "rocm", offload_arch="gfx90a",
                       config=root / "apps.yaml", nsys=True, pairs=2,
                       swaps_override={Path("kernel.cu"): "// kernel.cu\\nMODE=good"},
                       temp_dir=root / "tmp", kernel_gate=False, vram_reset_sha256="build-record")
    try:
        _, _, long = run_driver(cfg)
    except DriverInfraError as e:
        print(json.dumps({"infra": str(e)[:200]})); sys.exit(0)
    b, s = long["toy"]
    print(json.dumps({"reset_s": [x.get("vram_reset_s") for x in s.nsys_data],
                      "protocol": [x.get("protocol") for x in s.nsys_data]}))
    ''')


def _toy(tmp_path: Path, reset_body: str | None, record: bool = True) -> Path:
    import hashlib

    root = tmp_path / "gpa"
    shutil.copytree(GPA_ROOT / "gpa_bench_driver", root / "gpa_bench_driver",
                    ignore=shutil.ignore_patterns("__pycache__"))
    if reset_body is not None:
        (root / "frontier_tools").mkdir()
        r = root / "frontier_tools" / "vram_reset"
        r.write_text(reset_body)
        r.chmod(0o755)
        if record:
            (root / "frontier_tools" / "vram_reset.sha256").write_text(
                hashlib.sha256(r.read_bytes()).hexdigest() + "  vram_reset\n")
    app = root / "rodinia" / "toy-hip"
    app.mkdir(parents=True)
    (app / "kernel.cu").write_text("MODE=good\n")
    (app / "run.sh").write_text(_TOY_RUN)
    (app / "Makefile").write_text(_TOY_MAKE)
    (root / "rocm" / "bin").mkdir(parents=True)
    fake = root / "rocm" / "bin" / "rocprofv3"
    fake.write_text(_FAKE_ROCPROF)
    fake.chmod(0o755)
    ref = b"result 42\n"
    (root / "frontier_refs" / "toy").mkdir(parents=True)
    (root / "frontier_refs" / "toy" / "ref.txt").write_bytes(ref)
    (root / "frontier_refs.md5").write_text(
        f"{hashlib.md5(ref).hexdigest()}  frontier_refs/toy/ref.txt\n")  # noqa: S324
    (root / "apps.yaml").write_text(yaml.safe_dump({"apps": [{
        "name": "toy", "kernel_name": "toy_kernel", "score_regex": r"^toy_kernel\(",
        "path": "rodinia/toy-hip", "run_command": "./toy", "kernel_file": "rodinia/toy-hip/kernel.cu",
        "reference_output": "frontier_refs/toy/ref.txt", "test_output": "rodinia/toy-hip/output.txt",
    }]}))
    (root / "tmp").mkdir()
    return root


def _drive(root: Path) -> dict:
    proc = subprocess.run([sys.executable, "-c", _DRIVE, str(root)], capture_output=True,
                          text=True, check=False, timeout=600)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_d_vram_reset_failure_is_infra(tmp_path):
    out = _drive(_toy(tmp_path, "#!/bin/bash\necho out-of-memory >&2\nexit 3\n"))
    assert "VRAM reset failed" in out["infra"]


def test_d_missing_vram_reset_tool_is_infra(tmp_path):
    out = _drive(_toy(tmp_path, None))
    assert "VRAM reset failed" in out["infra"] and "missing" in out["infra"]


def test_T0_reset_runs_before_every_timed_sample(tmp_path):
    out = _drive(_toy(tmp_path, "#!/bin/bash\necho 'vram_reset allocs=1 GiB=0.00'\n"))
    assert len(out["reset_s"]) == 2 and all(r is not None and r >= 0 for r in out["reset_s"])
    assert out["protocol"] == ["j0", "j0"]


# ---------------------------------------------------------------- T6 entry point (driver_j0)

def test_T6_rescore_kernel_creates_temp_dir_and_remeasures_when_unstable(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from gpa_bench_driver.driver_src import driver_j0

    calls = []

    def fake_series(app, kernel_name, kernel_text, gcd, m, temp_dir, overrides, rfb, sha=None):
        assert temp_dir.is_dir()  # created by rescore_kernel (the caller may pass a fresh path)
        calls.append(m)
        fn = _period2 if len(calls) == 1 else (lambda pos: 15.66)
        b, o = _series(fn, m, "abba")
        return None, SimpleNamespace(validate=True, validation_output="", nsys_data=o,
                                     baseline_nsys_data=b)

    monkeypatch.setattr(driver_j0, "_one_series", fake_series)
    r = driver_j0.rescore_kernel("bfs", "// k\n", 3, temp_dir=tmp_path / "fresh" / "dir",
                                 vram_reset_sha256="f" * 64)
    assert calls == [10, 10]  # yaml final_pairs, then one T4 pooled re-measure
    assert r["gcd"] == 3 and r["pairs"] == 10 and r["remeasured"] is True
    assert r["j0"]["n_pairs"] == 20 and not r["j0"]["credited"]


def test_T6_rescore_kernel_remeasures_a_marginal_g_cpu_miss_like_the_runner(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from gpa_bench_driver.driver_src import driver_j0

    calls = []

    def fake_series(app, kernel_name, kernel_text, gcd, m, temp_dir, overrides, rfb, sha=None):
        calls.append(m)
        b, o = _series(lambda pos: 15.66, m, "abba")
        if len(calls) == 1:  # first series: one +2.5 s CPU spike in the optimized arm
            o[0]["cpu_s"] = 3.5  # nw: +0.25 s mean > slack 0.188 (marginal); pooled 0.125 < 0.133
        return None, SimpleNamespace(validate=True, validation_output="", nsys_data=o,
                                     baseline_nsys_data=b)

    monkeypatch.setattr(driver_j0, "_one_series", fake_series)
    r = driver_j0.rescore_kernel("nw", "// k\n", 0, temp_dir=tmp_path, vram_reset_sha256="f" * 64)
    assert calls == [10, 10]
    assert r["first"]["failures"] == ["G-cpu"] and r["remeasured"] is True
    assert r["cpu"]["ok"] and r["cpu"]["n_pairs"] == 20  # decided on the pooled series


def test_needs_remeasure_rule():
    from gpa_bench_driver.driver_src.driver_j0 import needs_remeasure

    g = {"code": "G-cpu"}
    assert needs_remeasure({"ok": False, "j0": {"unstable": True}, "failures": []})
    assert needs_remeasure({"ok": False, "failures": [g], "cpu": {"marginal": True}})
    assert not needs_remeasure({"ok": False, "failures": [g], "cpu": {"marginal": False}})
    assert not needs_remeasure({"ok": False, "failures": [g, {"code": "G-wall"}],
                                "cpu": {"marginal": True}})
    assert not needs_remeasure({"ok": True, "j0": {"unstable": False}, "failures": []})


# ---------------------------------------------------------------- K2: one credit rule everywhere

NW2 = json.loads((GPA_ROOT / "tests/fixtures/j0/nw_two_regime_pooled.json").read_text())


def test_K2_rule_module_is_stdlib_only():
    import ast

    src = (GPA_ROOT / "gpa_bench_driver/driver_src/driver_j0_rule.py").read_text()
    mods = {n.module if isinstance(n, ast.ImportFrom) else a.name
            for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.Import, ast.ImportFrom))
            for a in (n.names if isinstance(n, ast.Import) else [None])}
    assert mods <= {"__future__", "math", "typing"}


@pytest.mark.parametrize(("s", "u", "lb", "want"), [
    (1.02, False, 0.9, True), (1.02, True, 1.006, True), (1.02, True, 1.005, False),
    (1.005, False, 1.1, False), (None, False, None, False), (1.3, True, None, False)])
def test_K2_j0_credited(s, u, lb, want):
    from gpa_bench_driver.driver_src.driver_j0_rule import j0_credited

    assert j0_credited(s, u, lb) is want


def test_K2_nw_two_regime_pooled_series_is_credited_by_the_rule_and_by_score_frontier():
    from gpa_bench_driver.driver_src.driver_j0_rule import j0_decision, j0_estimate

    j = j0_estimate(NW2["baseline_scored_ns"], NW2["optimized_scored_ns"])
    assert j["unstable"] and j["n_pairs"] == 20
    assert j["lower_bound"] > 1.02 and j["credited"]
    assert j["speedup"] == pytest.approx(NW2["recorded_j0"]["speedup"], rel=1e-9)
    assert j0_decision(NW2["recorded_j0"])[0] is True
    # the same runs as driver samples (two series of 10 ABBA pairs), pooled -> same verdict
    def samples(scored, other):
        return [{"target_ns": int(x), "target_dispatches": 2048,
                 "kernels": {"needle_cuda_shared_1(int)": int(x), "needle_cuda_shared_2(int)": int(y)},
                 "wall_s": 4.5, "cpu_s": 4.5, "pair": i % 10}
                for i, (x, y) in enumerate(zip(scored, other))]
    b = samples(NW2["baseline_scored_ns"], NW2["baseline_other_ns"])
    o = samples(NW2["optimized_scored_ns"], NW2["optimized_other_ns"])
    r = score_frontier_pooled([b[:10], b[10:]], [o[:10], o[10:]], r"needle_cuda_shared_1\(",
                              protocol="j0", cpu_sigma_s=0.119)
    assert r["ok"] and r["j0"]["credited"] and r["j0"]["remeasured"]
    assert r["j0"]["speedup"] == pytest.approx(j["speedup"], rel=1e-9)


def test_K2_j0_decision_recomputes_the_lower_bound_from_the_pair_ratios():
    from gpa_bench_driver.driver_src.driver_j0_rule import j0_decision

    ratios = [1.001, 1.002] + [1.03] * 14  # m=16: old fixed j=4 -> 1.03, binomial j=3 -> 1.03
    rec = {"speedup": 1.03, "unstable": True, "pair_ratios": ratios, "lower_bound": 0.5}
    assert j0_decision(rec)[0] is True
    ratios = [1.001, 1.002, 1.003, 1.004] + [1.03] * 12  # j=3 -> 1.004: not credited
    assert j0_decision({**rec, "pair_ratios": ratios})[0] is False


# ---------------------------------------------------------------- L1: tolerance capped (round 6)

BP = json.loads((GPA_ROOT / "tests/fixtures/j0/backprop_noop_series.json").read_text())


@pytest.mark.parametrize("key", sorted(BP["series"]))
def test_L1_reviewer_scenario_shift_of_tol_is_never_credited(key):
    """verify6/reviewer/charge_tol_backprop.txt: moving exactly `tol` of target work into the
    unchanged other kernel goes uncharged; the shift-only speedup is T / (T - tol). Capped at half
    the floor margin it is <= 1 / (1 - 0.0025) = 1.0025 and never credited; the round-5 (uncapped) tolerance gave
    1.006-1.018 on these same replays."""
    from gpa_bench_driver.driver_src.driver_j0_rule import j0_credited, other_charge

    d = BP["series"][key]
    t = statistics.median(d["b_scored"])
    capped = other_charge(d["b_other"], d["o_other"], d["b_scored"])["tol_ns"]
    uncapped = other_charge(d["b_other"], d["o_other"])["tol_ns"]
    shift_only = t / (t - capped)
    assert capped <= 0.5 * 0.005 * t + 1e-6
    assert shift_only <= 1 / (1 - 0.0025) + 1e-9 and not j0_credited(shift_only, False, None)
    assert t / (t - uncapped) > 1.005  # mutation: the round-5 tolerance credits the shift


@pytest.mark.parametrize("key", sorted(BP["series"]))
def test_L1_measured_shift_gain_is_bounded_by_tol_plus_favourable_other_noise(key):
    """On the real series a shift can also absorb the other kernel's favourable median noise
    g < 0 (the one-sided charge ignores it): the optimized arm's effective time drops by at most
    tol + max(0, -g). Documented residual (A.md, fix round 6)."""
    from gpa_bench_driver.driver_src.driver_j0_rule import j0_estimate, median, other_charge

    d = BP["series"][key]
    t = median(d["b_scored"])
    c0 = other_charge(d["b_other"], d["o_other"], d["b_scored"])
    g = median(d["o_other"]) - median(d["b_other"])
    base = j0_estimate(d["b_scored"], d["o_scored"])  # the no-op's own target-only reading
    bound = max(base["speedup_median_ratio"], base["speedup_pair_median"]) * t / (
        t - c0["tol_ns"] - max(0.0, -g))
    for shift in [x * 250.0 for x in range(0, 61)]:  # 0 .. 15 us
        oo = [v + shift for v in d["o_other"]]
        os_ = [v - shift for v in d["o_scored"]]
        c = other_charge(d["b_other"], oo, d["b_scored"])
        j = j0_estimate(d["b_scored"], os_, charge_ns=c["charge_ns"])
        assert j["speedup"] <= bound * 1.0005, (shift, j["speedup"], bound)


def test_L1_score_frontier_reports_the_cap():
    d = BP["series"]["k5_replay_honest/variant"]
    def samples(scored, other):
        return [{"target_ns": int(x), "target_dispatches": 1,
                 "kernels": {"k(int)": int(x), "other(int)": int(y)}, "wall_s": 1.0, "cpu_s": 1.0,
                 "pair": i} for i, (x, y) in enumerate(zip(scored, other))]
    r = score_frontier(samples(d["b_scored"], d["b_other"]), samples(d["o_scored"], d["o_other"]),
                       RX, protocol="j0")
    o = r["other"]
    assert o["tol_cap_ns"] == pytest.approx(0.5 * 0.005 * statistics.median(d["b_scored"]))
    assert o["tol_ns"] == min(o["tol_noise_ns"], o["tol_cap_ns"])
