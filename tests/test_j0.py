"""J0 measurement protocol (GPA-G1 J0-A; agent M's M.md T0-T5 and its tests a-d).

Series are built from a per-process time function over the GLOBAL run order, so a GCD's
placement-state history (e.g. period-2 alternation) is applied to whichever arm ran at that
position. "abba" = the J0 order; "alt" = the old strict B,O,B,O order (mutation check: under the
old order + mean estimator the same no-op IS falsely credited).
"""

from __future__ import annotations

import json
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

@pytest.mark.parametrize(("m", "j"), [(4, 0), (6, 0), (7, 0), (8, 1), (10, 1), (11, 1), (12, 2),
                                      (15, 2), (16, 4), (20, 4)])
def test_lower_bound_order_statistic(m, j):
    assert _lower_bound_index(m) == j


def test_other_kernel_charge_on_medians():
    b, o = _series(lambda pos: 1.0, 6, "abba")
    for s in b:
        s["kernels"]["other(int)"] = 100_000
    for i, s in enumerate(o):  # the optimized run's other kernel is 50 us slower (median)
        s["kernels"]["other(int)"] = 150_000 if i != 0 else 900_000  # one spike
    r = score_frontier(b, o, RX, protocol="j0")
    assert r["other"]["charge_ns"] == pytest.approx(50_000)  # median, not pulled by the spike


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
                       temp_dir=root / "tmp", kernel_gate=False)
    try:
        _, _, long = run_driver(cfg)
    except DriverInfraError as e:
        print(json.dumps({"infra": str(e)[:200]})); sys.exit(0)
    b, s = long["toy"]
    print(json.dumps({"reset_s": [x.get("vram_reset_s") for x in s.nsys_data],
                      "protocol": [x.get("protocol") for x in s.nsys_data]}))
    ''')


def _toy(tmp_path: Path, reset_body: str | None) -> Path:
    import hashlib

    root = tmp_path / "gpa"
    shutil.copytree(GPA_ROOT / "gpa_bench_driver", root / "gpa_bench_driver",
                    ignore=shutil.ignore_patterns("__pycache__"))
    if reset_body is not None:
        (root / "frontier_tools").mkdir()
        r = root / "frontier_tools" / "vram_reset"
        r.write_text(reset_body)
        r.chmod(0o755)
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

    def fake_series(app, kernel_name, kernel_text, gcd, m, temp_dir, overrides, rfb):
        assert temp_dir.is_dir()  # created by rescore_kernel (the caller may pass a fresh path)
        calls.append(m)
        fn = _period2 if len(calls) == 1 else (lambda pos: 15.66)
        b, o = _series(fn, m, "abba")
        return None, SimpleNamespace(validate=True, validation_output="", nsys_data=o,
                                     baseline_nsys_data=b)

    monkeypatch.setattr(driver_j0, "_one_series", fake_series)
    r = driver_j0.rescore_kernel("bfs", "// k\n", 3, temp_dir=tmp_path / "fresh" / "dir")
    assert calls == [10, 10]  # yaml final_pairs, then one T4 pooled re-measure
    assert r["gcd"] == 3 and r["pairs"] == 10 and r["remeasured"] is True
    assert r["j0"]["n_pairs"] == 20 and not r["j0"]["credited"]
