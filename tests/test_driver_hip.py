"""Tests for the GPA driver's hip backend (Frontier) and for the unchanged cuda backend.

Covers: rocprofv3 kernel-trace parsing (ROCm 6 and 7 CSV layouts, real traces), target
regex / summing / runtime-kernel exclusion, the per-sample timing dict, the Frontier scoring
helpers, the expected_checksum check (incl. a doctored "(Valid)" line), app_overrides,
gpu_device pinning, hip staging, driver_apps.frontier.yaml consistency, and the cuda path.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from gpa_bench_driver import gpa_bench_driver as gbd
from gpa_bench_driver.driver_src import driver_config, driver_rocprof, driver_utils
from gpa_bench_driver.driver_src.driver_config import OperationCombinationError, setup_app_config
from gpa_bench_driver.driver_src.driver_models import DriverConfig
from gpa_bench_driver.driver_src.driver_operations import build_app
from gpa_bench_driver.driver_src.driver_rocprof import (
    RocprofError,
    integrity_guards,
    new_kernel_ns,
    read_kernel_trace,
    rocprof_time_app,
    scored_time_ns,
    summarize_kernel_trace,
)
from gpa_bench_driver.driver_src.driver_utils import SubprocessRunner, SubprocessRunnerConfig
from gpa_bench_driver.driver_src.driver_validation import validate_app

GPA_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "rocprofv3"
FRONTIER_YAML = GPA_ROOT / "driver_apps.frontier.yaml"
CUDA_YAML = GPA_ROOT / "driver_apps.yaml"

BFS_KERNEL = "Kernel(Node*, int*, bool*, bool*, bool*, int*, int)"
BFS_KERNEL2 = "Kernel2(bool*, bool*, bool*, bool*, int)"


def _frontier_apps() -> dict[str, dict]:
    with FRONTIER_YAML.open() as f:
        return {a["name"]: a for a in yaml.safe_load(f)["apps"]}


def _completed(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout.encode())


# --------------------------------------------------------------------------- rocprofv3 parsing


def test_read_kernel_trace_rocm7_layout_bfs():
    rows = read_kernel_trace([FIXTURES / "rocm7_bfs_kernel_trace.csv"])
    assert len(rows) == 12
    kernel = [d for n, d in rows if n == BFS_KERNEL]
    assert kernel == [11520, 12160, 15841]
    assert [d for n, d in rows if n == BFS_KERNEL2] == [4320, 4320, 4960]


def test_read_kernel_trace_rocm6_layout_backprop():
    rows = read_kernel_trace([FIXTURES / "rocm6_backprop_kernel_trace.csv"])
    assert rows[0] == ("bpnn_layerforward_CUDA(float*, float*, float*, float*, int, int)", 458882)
    assert rows[1] == ("__amd_rocclr_copyBuffer", 6560)


def test_read_kernel_trace_rejects_non_trace(tmp_path):
    bad = tmp_path / "x_kernel_trace.csv"
    bad.write_text('"Kind","Name"\n"KERNEL_DISPATCH","k"\n')
    with pytest.raises(RocprofError, match="missing columns"):
        read_kernel_trace([bad])


def test_summarize_anchored_regex_sums_every_dispatch_keeps_runtime_kernels():
    rows = read_kernel_trace([FIXTURES / "rocm7_bfs_kernel_trace.csv"])
    s = summarize_kernel_trace(rows, _frontier_apps()["bfs"]["score_regex"])
    # ^Kernel\( must not match Kernel2 (the substring-match bug seen in G0)
    assert s["target_ns"] == 11520 + 12160 + 15841
    assert s["exec_time"] == s["target_ns"]
    assert s["target_dispatches"] == 3
    # fix round 1 (R1): __amd_rocclr_* kernels are ordinary kernels
    copy = [d for n, d in rows if n == "__amd_rocclr_copyBuffer"]
    assert s["kernels"] == {BFS_KERNEL: 39521, BFS_KERNEL2: 13600,
                            "__amd_rocclr_copyBuffer": sum(copy)}
    assert s["kernel_dispatches"] == {BFS_KERNEL: 3, BFS_KERNEL2: 3, "__amd_rocclr_copyBuffer": 6}


def test_summarize_extern_c_name_btree():
    rows = read_kernel_trace([FIXTURES / "rocm7_btree_kernel_trace.csv"])
    s = summarize_kernel_trace(rows, _frontier_apps()["b+tree"]["score_regex"])
    assert s["target_ns"] == 62880
    assert s["target_dispatches"] == 1
    assert s["kernels"] == {"findRangeK": 62880, "findK": 66720}
    assert s["kernel_dispatches"] == {"findRangeK": 1, "findK": 1}


def test_summarize_xsbench_counts_warmup_dispatch():
    rows = read_kernel_trace([FIXTURES / "rocm7_xsbench_kernel_trace.csv"])
    s = summarize_kernel_trace(rows, _frontier_apps()["xsbench"]["score_regex"])
    assert s["target_dispatches"] == 2  # warmup + timed launch both count
    assert s["target_ns"] == 64170539 + 64011018


def test_default_score_regex_matches_both_name_forms():
    rx = driver_rocprof.default_score_regex("findRangeK")
    s = summarize_kernel_trace(
        [("findRangeK", 5), ("findRangeK(long, knode*)", 7), ("findRangeK2(int)", 100)], rx,
    )
    assert s["target_ns"] == 12


def test_score_regex_matches_every_frontier_baseline_name():
    names = {
        "bfs": BFS_KERNEL,
        "backprop": "bpnn_layerforward_CUDA(float*, float*, float*, float*, int, int)",
        "b+tree": "findRangeK",
        "heartwall": "kernel()",
        "hotspot": "calculate_temp(int, float*, float*, float*, int, int, int, int, float, "
        "float, float, float, float, float)",
        "pathfinder": "dynproc_kernel(int, int*, int*, int*, int, int, int, int)",
        "nw": "needle_cuda_shared_1(int*, int*, int, int, int, int)",
        "streamcluster": "kernel_compute_cost(int, int, long, Point*, int, int, float*, float*, "
        "int*, bool*)",
        "xsbench": "xs_lookup_kernel(Inputs, SimulationData)",
    }
    decoys = [BFS_KERNEL2, "findK", "needle_cuda_shared_2(int*, int*, int, int, int, int)",
              "bpnn_adjust_weights_cuda(float*, int, float*, int, float*, float*)"]
    import re

    for app, entry in _frontier_apps().items():
        rx = re.compile(entry["score_regex"])
        assert rx.search(names[app]), app
        assert entry["score_regex"].startswith("^"), app
        assert not any(rx.search(d) for d in decoys), app


# --------------------------------------------------------------------------- scoring helpers


def _sample(target_ns, kernels, wall_s=1.0):
    return {"exec_time": target_ns, "target_ns": target_ns, "target_dispatches": 1,
            "kernels": kernels, "wall_s": wall_s, "backend": "rocprofv3"}


def test_scored_time_adds_kernels_absent_from_baseline():
    base = _sample(100, {"k(int)": 100, "other(int)": 50})
    opt = _sample(40, {"k(int)": 40, "other(int)": 50, "k_new(int)": 30})
    assert new_kernel_ns(opt, base["kernels"]) == 30
    assert scored_time_ns(opt, base["kernels"]) == 70
    renamed = _sample(0, {"k_v2(int)": 60, "other(int)": 50})  # target renamed away
    assert scored_time_ns(renamed, base["kernels"]) == 60


def test_new_target_matching_kernel_counts_once():
    """A renamed/templated target variant absent from the baseline is already in target_ns."""
    rx = r"^k(\(|<|$)"
    base = _sample(100, {"k(int)": 100, "other(int)": 50})
    variant = "k<64>(int)"  # templated target: new name, matches the regex
    opt = _sample(30 + 25, {"k(int)": 30, variant: 25, "other(int)": 50, "helper(int)": 10})
    assert summarize_kernel_trace([("k(int)", 30), (variant, 25), ("other(int)", 50),
                                   ("helper(int)", 10)], rx)["target_ns"] == 55
    assert new_kernel_ns(opt, base["kernels"], rx) == 10  # only helper(int) is new work
    assert scored_time_ns(opt, base["kernels"], rx) == 55 + 10  # the variant counts once
    # the same result through the caller-side workaround (target names passed as known)
    known = set(base["kernels"]) | {variant}
    assert scored_time_ns(opt, known) == scored_time_ns(opt, base["kernels"], rx)
    # without the regex the variant would be counted a second time
    assert scored_time_ns(opt, base["kernels"]) == 55 + 25 + 10
    # a target renamed so that it no longer matches is still counted once, as new work
    renamed = _sample(0, {"k_v2(int)": 60, "other(int)": 50})
    assert scored_time_ns(renamed, base["kernels"], rx) == 60


def test_integrity_guards_other_and_wall():
    base = [_sample(100, {"k(int)": 100, "other(int)": 1_000_000}, wall_s=2.0)] * 2
    ok = [_sample(50, {"k(int)": 50, "other(int)": 1_100_000}, wall_s=4.0)]
    g = integrity_guards(base, ok, r"^k\(")
    assert g["g_other_ok"] and g["g_wall_ok"] and g["ok"]
    assert g["other_ns_limit"] == pytest.approx(1.10 * 1_000_000 + 50_000)
    assert g["wall_s_limit"] == pytest.approx(1.5 * 2.0 + 1.0)
    moved = [_sample(1, {"k(int)": 1, "other(int)": 1_200_000}, wall_s=2.0)]
    assert not integrity_guards(base, moved, r"^k\(")["g_other_ok"]
    host = [_sample(1, {"k(int)": 1, "other(int)": 1_000_000}, wall_s=4.5)]
    g = integrity_guards(base, host, r"^k\(")
    assert g["g_other_ok"] and not g["g_wall_ok"] and not g["ok"]


# --------------------------------------------------------------------------- rocprof_time_app


class _FakeRocprofRunner:
    """Stands in for SubprocessRunner: writes a kernel trace where rocprofv3 would."""

    def __init__(self, trace: Path, returncode: int = 0, plain_returncode: int | None = None):
        self.trace = trace
        self.returncode = returncode
        self.plain_returncode = plain_returncode
        self.calls: list[tuple[list[str], Path, int | None]] = []

    def run(self, command, cwd, *, quiet=None, stdout_cap_bytes=None, measure_cpu=False):
        self.calls.append((command, cwd, stdout_cap_bytes))
        if "-d" not in command:  # the unprofiled re-run after a profiler failure
            rc = self.returncode if self.plain_returncode is None else self.plain_returncode
            return subprocess.CompletedProcess(command, rc, b"out", b"err")
        outdir = Path(command[command.index("-d") + 1])
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "trace_kernel_trace.csv").write_text(self.trace.read_text())
        return subprocess.CompletedProcess(command, self.returncode, b"out", b"err")


def _bfs_app(**extra):
    app = dict(_frontier_apps()["bfs"])
    app.update(extra)
    return app


def test_rocprof_time_app_per_sample_dict(tmp_path):
    runner = _FakeRocprofRunner(FIXTURES / "rocm7_bfs_kernel_trace.csv")
    (tmp_path / "rodinia" / "bfs-hip").mkdir(parents=True)
    data = rocprof_time_app(_bfs_app(), runner, tmp_path, 3, rocm_path=Path("/nonexistent"))
    assert len(data) == 3
    for d in data:
        assert set(d) == {"exec_time", "target_ns", "target_dispatches", "kernels",
                          "kernel_dispatches", "wall_s", "cpu_s", "backend", "valid",
                          "validation_output"}
        assert d["exec_time"] == d["target_ns"] == 39521
        assert d["valid"] is None  # no validate callback given
        assert d["backend"] == "rocprofv3"
        assert isinstance(d["wall_s"], float)
    cmd, cwd, cap = runner.calls[0]
    assert cmd[:4] == ["rocprofv3", "--kernel-trace", "--output-format", "csv"]
    assert cmd[cmd.index("--") + 1:] == ["./bfs", "../data/bfs/graph8M.txt"]
    assert cwd == tmp_path / "rodinia" / "bfs-hip"
    assert cap is None
    # traces are deleted after parsing unless retained
    assert not list((tmp_path / "profiles").glob("rocprof_*"))


def test_rocprof_time_app_keeps_full_stdout_and_retains(tmp_path):
    runner = _FakeRocprofRunner(FIXTURES / "rocm7_bfs_kernel_trace.csv")
    rocprof_time_app(_bfs_app(stdout_cap_bytes=1024), runner, tmp_path, 1, retain_profiles=True)
    assert runner.calls[0][2] is None  # timed runs are validated (R4): no cap
    assert list((tmp_path / "profiles").glob("rocprof_bfs_sample_0/*kernel_trace.csv"))


def test_rocprof_time_app_validates_each_sample_and_removes_old_output(tmp_path):
    run_dir = tmp_path / "rodinia" / "bfs-hip"
    run_dir.mkdir(parents=True)
    runner = _FakeRocprofRunner(FIXTURES / "rocm7_bfs_kernel_trace.csv")
    seen = []

    def validate(result, rd):
        seen.append((rd, (rd / "result.txt").exists()))
        return (len(seen) != 2, None if len(seen) != 2 else "wrong")

    for _ in range(3):
        (run_dir / "result.txt").write_text("stale")  # must be deleted before each timed run
        break
    data = rocprof_time_app(_bfs_app(), runner, tmp_path, 3, validate=validate)
    assert [d["valid"] for d in data] == [True, False, True]
    assert data[1]["validation_output"] == "wrong"
    assert seen[0] == (run_dir, False)


def test_rocprof_time_app_baseline_without_target_returns_none(tmp_path):
    runner = _FakeRocprofRunner(FIXTURES / "rocm7_btree_kernel_trace.csv")
    assert rocprof_time_app(_bfs_app(), runner, tmp_path, 2) is None


def test_rocprof_time_app_swap_without_target_is_data(tmp_path):
    from gpa_bench_driver.driver_src.driver_models import FileSwap, SwapConfig

    swap = SwapConfig("bfs", [FileSwap(Path("kernel.cu"), Path("rodinia/bfs-hip/kernel.cu"), "")],
                      "0", "0", None)
    runner = _FakeRocprofRunner(FIXTURES / "rocm7_btree_kernel_trace.csv")
    data = rocprof_time_app(_bfs_app(), runner, tmp_path, 1, swap_config=swap)
    assert data[0]["target_dispatches"] == 0
    assert data[0]["kernels"] == {"findRangeK": 62880, "findK": 66720}


def test_rocprof_time_app_failure_raises(tmp_path):
    runner = _FakeRocprofRunner(FIXTURES / "rocm7_bfs_kernel_trace.csv", returncode=3)
    with pytest.raises(RocprofError, match="return code 3"):
        rocprof_time_app(_bfs_app(), runner, tmp_path, 1)


def test_profiler_crash_after_app_started_is_agent(tmp_path):
    # H3: rocprofv3 exited non-zero but its output dir exists (the app started) -> agent failure,
    # never laundered by a clean plain rerun.
    runner = _FakeRocprofRunner(FIXTURES / "rocm7_bfs_kernel_trace.csv", returncode=139,
                                plain_returncode=0)
    with pytest.raises(RocprofError, match="failed under the profiler"):
        rocprof_time_app(_bfs_app(), runner, tmp_path, 1)


# --------------------------------------------------------------------------- expected_checksum

XS = {"name": "xsbench", "expected_checksum": {"regex": r"^Verification checksum: (\d+)",
                                               "value": 711949}}


def test_checksum_passes_and_ignores_self_verdict():
    out = "Runtime: 1 s\nVerification checksum: 711949 (WARNING - INAVALID CHECKSUM!)\n"
    assert validate_app(XS, _completed(out), Path("/nonexistent")) == (True, None)


def test_checksum_wrong_value_fails_even_if_valid():
    ok, msg = validate_app(XS, _completed("Verification checksum: 952131 (Valid)\n"), Path("."))
    assert not ok and "expected 711949" in msg and "952131" in msg


def test_checksum_doctored_extra_valid_line_fails():
    out = "Verification checksum: 711949 (Valid)\nVerification checksum: 711949 (Valid)\n"
    ok, msg = validate_app(XS, _completed(out), Path("."))
    assert not ok and "found 2" in msg


def test_checksum_missing_line_fails_and_mid_line_text_ignored():
    ok, msg = validate_app(XS, _completed("note: Verification checksum: 711949\n"), Path("."))
    assert not ok and "found 0" in msg


def test_checksum_default_regex_and_spec_errors():
    app = {"name": "x", "expected_checksum": {"value": "945990"}}
    assert validate_app(app, _completed("Verification checksum: 945990 (Valid)\n"), Path("."))[0]
    with pytest.raises(ValueError, match="value"):
        validate_app({"name": "x", "expected_checksum": {"regex": "a"}}, _completed(""), Path("."))


def test_checksum_check_precedes_other_checks():
    app = dict(XS, pass_check_text="(Valid)")
    ok, _ = validate_app(app, _completed("Verification checksum: 952131 (Valid)\n"), Path("."))
    assert not ok


# --------------------------------------------------------------------------- config / overrides


def _hip_config(**kw):
    kw.setdefault("gpu_backend", "hip")
    kw.setdefault("rocm_path", Path("/opt/rocm-7.0.2"))
    kw.setdefault("offload_arch", "gfx90a")
    return DriverConfig(**kw)


def test_hip_config_defaults():
    cfg = _hip_config(app="xsbench")
    assert cfg.config == FRONTIER_YAML
    assert cfg.no_sanitize is True
    assert cfg.offload_arch == "gfx90a"


def test_app_overrides_merge_into_selected_app(monkeypatch):
    monkeypatch.setattr(driver_config.resource, "setrlimit", lambda *a: None)
    over = {"run_command": "./XSBench -m event -s small",
            "expected_checksum": {"regex": r"^Verification checksum: (\d+)", "value": 945990}}
    cfg = _hip_config(app="XSBench", app_overrides=over)
    app_config, _, _ = setup_app_config(cfg)
    apps = {a["name"]: a for a in app_config["apps"]}
    assert apps["xsbench"]["run_command"] == over["run_command"]
    assert apps["xsbench"]["expected_checksum"]["value"] == 945990
    assert apps["xsbench"]["kernel_file"] == "XSBench-hip/Simulation.cu"  # rest kept
    assert apps["bfs"] == _frontier_apps()["bfs"]  # other apps untouched
    over["expected_checksum"]["value"] = 1  # deep-copied
    assert apps["xsbench"]["expected_checksum"]["value"] == 945990


def test_app_overrides_need_single_app():
    with pytest.raises(OperationCombinationError):
        setup_app_config(_hip_config(app="all", app_overrides={"run_command": "x"}))


def test_gpu_device_hip_pins_rocr_only(monkeypatch):
    monkeypatch.setattr(driver_config.resource, "setrlimit", lambda *a: None)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "5")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5")
    _, _, env = setup_app_config(_hip_config(app="bfs", gpu_device=3))
    assert env["ROCR_VISIBLE_DEVICES"] == "3"
    assert "HIP_VISIBLE_DEVICES" not in env and "CUDA_VISIBLE_DEVICES" not in env
    assert env["PATH"].startswith("/opt/rocm-7.0.2/bin:")
    assert env["ROCM_PATH"] == "/opt/rocm-7.0.2"


def test_hip_env_disables_core_dumps(monkeypatch):
    calls = []
    monkeypatch.setattr(driver_config.resource, "setrlimit", lambda *a: calls.append(a))
    setup_app_config(_hip_config(app="bfs"))
    assert calls and calls[0][0] == driver_config.resource.RLIMIT_CORE and calls[0][1][0] == 0


def test_gpu_device_cuda_sets_cuda_visible_devices():
    _, _, env = setup_app_config(DriverConfig(app="bfs", gpu_backend="cuda", sm_version=80,
                                              cuda_home=Path("/fake/cuda"), gpu_device=2))
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert "ROCR_VISIBLE_DEVICES" not in env or env["ROCR_VISIBLE_DEVICES"] == os.environ.get(
        "ROCR_VISIBLE_DEVICES")


def test_hip_rejects_postprocess_nsys():
    with pytest.raises(OperationCombinationError):
        setup_app_config(_hip_config(app="bfs", postprocess_nsys=True))


@pytest.mark.parametrize(
    ("platform", "kfd", "rocminfo", "expected"),
    [
        ("frontier", False, None, "hip"),
        ("perlmutter", True, "/opt/rocm/bin/rocminfo", "cuda"),
        ("", True, None, "hip"),
        ("", False, "/opt/rocm/bin/rocminfo", "hip"),
        ("", False, None, "cuda"),
    ],
)
def test_detect_gpu_backend(monkeypatch, platform, kfd, rocminfo, expected):
    monkeypatch.setenv("APPEB_PLATFORM", platform)
    monkeypatch.setattr(driver_utils.Path, "exists",
                        lambda self: kfd if str(self) == "/dev/kfd" else os.path.exists(self))
    monkeypatch.setattr(driver_utils.shutil, "which",
                        lambda name: rocminfo if name == "rocminfo" else None)
    assert driver_utils.detect_gpu_backend() == expected


def test_cli_flags_reach_driver_config(monkeypatch):
    monkeypatch.setattr("sys.argv", ["gpa", "--app", "bfs", "--gpu-backend", "hip",
                                     "--offload-arch", "gfx942", "--gpu-device", "6"])
    cfg = DriverConfig.from_args(gbd.parse_args())
    assert (cfg.gpu_backend, cfg.offload_arch, cfg.gpu_device) == ("hip", "gfx942", 6)
    assert cfg.config == FRONTIER_YAML


# --------------------------------------------------------------------------- build / staging


class _RecordingRunner:
    def __init__(self):
        self.commands = []

    def run(self, command, cwd, *, quiet=None, stdout_cap_bytes=None):
        self.commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, b"", b"")


def test_build_app_hip_passes_offload_arch(tmp_path):
    r = _RecordingRunner()
    build_app({"name": "bfs", "path": "p", "run_command": "./bfs", "build_command": "make -B -j8"},
              80, r, tmp_path, no_clean=False, gpu_backend="hip", offload_arch="gfx90a")
    assert r.commands == [["make", "clean"], ["make", "-B", "-j8", "OFFLOAD_ARCH=gfx90a"]]


def test_build_app_cuda_unchanged(tmp_path):
    r = _RecordingRunner()
    build_app({"name": "b", "path": "p", "run_command": "./b"}, 80, r, tmp_path, no_clean=False)
    assert r.commands == [["make", "clean"], ["make", "-B", "-j", "8", "SM_VERSION=80"]]


def test_stage_hip_app_copies_only_app_and_links_data(tmp_path):
    root = tmp_path / "gpa"
    (root / "rodinia" / "bfs-hip").mkdir(parents=True)
    (root / "rodinia" / "bfs-hip" / "kernel.cu").write_text("k")
    (root / "rodinia" / "bfs-hip" / "old.o").write_text("o")
    (root / "rodinia" / "other").mkdir()
    (root / "rodinia" / "data" / "bfs").mkdir(parents=True)
    (root / "XSBench-hip").mkdir()
    (root / "XSBench-hip" / "Simulation.cu").write_text("s")
    tmp = tmp_path / "t"
    tmp.mkdir()
    assert gbd._stage_hip_app({"path": "rodinia/bfs-hip"}, root, tmp) == "rodinia/bfs-hip"
    assert (tmp / "rodinia" / "bfs-hip" / "kernel.cu").read_text() == "k"
    assert not (tmp / "rodinia" / "bfs-hip" / "old.o").exists()
    assert not (tmp / "rodinia" / "other").exists()
    assert (tmp / "rodinia" / "data").is_symlink()
    assert gbd._stage_hip_app({"path": "XSBench-hip"}, root, tmp) == "XSBench-hip"
    assert (tmp / "XSBench-hip" / "Simulation.cu").exists()


def test_app_dirs_lookup_keeps_cuda_paths():
    def lookup(path):
        return next(d for d in gbd.APP_DIRS if d in path)

    assert lookup("XSBench/cuda") == "XSBench"
    assert lookup("XSBench-hip") == "XSBench-hip"
    assert lookup("rodinia/bfs") == "rodinia"
    assert lookup("LULESH/cuda/src") == "LULESH"


# --------------------------------------------------------------------------- stdout cap


def test_subprocess_runner_stdout_cap(tmp_path):
    runner = SubprocessRunner(env=dict(os.environ), config=SubprocessRunnerConfig(timeout=60))
    cmd = ["python3", "-c", "import sys; sys.stdout.write('x' * 5000)"]
    assert len(runner.run(cmd, tmp_path).stdout) == 5000
    assert runner.run(cmd, tmp_path, stdout_cap_bytes=100).stdout == b"x" * 100


# --------------------------------------------------------------------------- frontier yaml


def test_frontier_yaml_apps_and_fields():
    apps = _frontier_apps()
    assert set(apps) == {"bfs", "backprop", "b+tree", "heartwall", "hotspot", "pathfinder",
                         "nw", "streamcluster", "xsbench"}
    with CUDA_YAML.open() as f:
        cuda = {a["name"]: a for a in yaml.safe_load(f)["apps"]}
    for name, app in apps.items():
        assert isinstance(app["rocprof_compute_dispatch"], int), name
        assert app["path"].endswith("-hip"), name
        for key in ("kernel_file", *app.get("extra_files", [])):
            path = app[key] if key == "kernel_file" else key
            assert (GPA_ROOT / path).is_file(), path
        # same swap-target basename as on Perlmutter
        assert Path(app["kernel_file"]).name == Path(cuda[name]["kernel_file"]).name, name
        assert app["kernel_name"] == cuda[name]["kernel_name"], name
        assert "pass_check_text" not in app and "ncu_args" not in app, name
    assert apps["xsbench"]["expected_checksum"]["value"] == 711949
    assert apps["xsbench"]["profile_run_command"] == "./XSBench -m event -s large"
    assert apps["pathfinder"]["stdout_cap_bytes"] == 1 << 20
    # GPA's ncu --launch-skip (0 where none)
    assert {n: a["rocprof_compute_dispatch"] for n, a in apps.items()} == {
        "bfs": 8, "backprop": 0, "b+tree": 0, "heartwall": 4, "hotspot": 0, "pathfinder": 2,
        "nw": 127, "streamcluster": 300, "xsbench": 0}


def test_frontier_refs_manifest_covers_every_reference():  # noqa: D103
    manifest = GPA_ROOT / "frontier_refs.md5"
    listed = {line.split()[1] for line in manifest.read_text().splitlines() if line.strip()}
    for app in _frontier_apps().values():
        if "reference_output" in app:
            assert app["reference_output"] in listed, app["name"]
    assert "rodinia/data/bfs/graph8M.txt" in listed


# --------------------------------------------------------------------------- cuda unchanged

PRISTINE = "117dc99"
_PROBE = r'''
import json, os, sys
from pathlib import Path
root = Path(sys.argv[1]); sys.path.insert(0, str(root)); os.chdir(root)
os.environ.update({"CUDA_HOME": "/fake/cuda", "PATH": "/usr/bin:/bin", "LD_LIBRARY_PATH": "/l",
                   "APPEB_PLATFORM": "perlmutter"})
import subprocess
try:
    import alive_progress  # noqa: F401
except ImportError:  # the pristine driver imports it unconditionally
    import contextlib, types
    _m = types.ModuleType("alive_progress")
    @contextlib.contextmanager
    def _bar(*a, **k):
        yield lambda *a, **k: None
    _m.alive_bar = _bar
    sys.modules["alive_progress"] = _m
from gpa_bench_driver.driver_src.driver_models import DriverConfig
from gpa_bench_driver.driver_src.driver_config import setup_app_config, determine_operations
from gpa_bench_driver.driver_src.driver_operations import build_app, run_app
from gpa_bench_driver.driver_src.driver_validation import validate_app
from gpa_bench_driver import gpa_bench_driver as g

class R:
    def __init__(self): self.c = []
    def run(self, command, cwd, **kw):
        self.c.append([list(command), str(cwd)]); return subprocess.CompletedProcess(command, 0, b"", b"")

out = {}
for kw in ({"app": "bfs", "nsys": True}, {"app": "xsbench", "ncu": True, "no_sanitize": True},
           {"app": "all"}, {"app": "backprop", "swaps_override": {Path("backprop_cuda_kernel.cu"): "// backprop_cuda_kernel.cu\nX"}, "nsys": True}):
    cfg = DriverConfig(sm_version=80, cuda_home=Path("/fake/cuda"), temp_dir=Path("/tmp"), **kw)
    old_keys = sorted(k for k in vars(cfg) if k not in ("gpu_backend", "offload_arch", "rocm_path", "app_overrides", "gpu_device", "reference_from_baseline", "kernel_gate", "interleave", "pairs", "vram_reset_sha256"))
    app_config, swaps, env = setup_app_config(cfg)
    key = repr(sorted((k, repr(v)) for k, v in kw.items()))
    out[key] = {
        "cfg": {k: str(getattr(cfg, k)) for k in old_keys if k != "config"},
        "config_name": cfg.config.name,
        "ops": [o.value for o in determine_operations(cfg)],
        "count": g.count_operations_per_pass(cfg),
        "env": {k: env.get(k) for k in ("CUDA_HOME", "PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")},
        "apps": app_config,
        "swaps": repr(sorted((k, repr(v)) for k, v in (swaps or {}).items())),
    }
    builds = {}
    for app in app_config["apps"]:
        r = R(); build_app(app, 80, r, Path("/t"), no_clean=False); run_app(app, r, Path("/t"))
        builds[app["name"]] = r.c
    out[key]["builds"] = builds
vals = []
for app in app_config["apps"]:
    for stdout in (b"", b"PASS! vectors are matching!\nFailed!\n(Valid)\nsum: 1.0\n"):
        try:
            res = validate_app(app, subprocess.CompletedProcess([], 0, stdout, b""), Path("/nonexistent"))
        except Exception as e:  # noqa: BLE001
            res = type(e).__name__
        vals.append([app["name"], str(res)[:200]])
out["validate"] = vals
out["app_dirs"] = [next(d for d in g.APP_DIRS if d in a["path"]) for a in app_config["apps"]]
print(json.dumps(out, sort_keys=True, default=str))
'''


def _git(*args):
    return subprocess.run(["git", "-C", str(GPA_ROOT), *args], capture_output=True, check=False)


@pytest.mark.skipif(_git("cat-file", "-e", f"{PRISTINE}^{{commit}}").returncode != 0,
                    reason=f"GPA-Benchmark history with {PRISTINE} not available")
def test_cuda_backend_behaviour_identical_to_pristine(tmp_path):
    """Config, env, operations, build/run commands and validation of the cuda backend equal 117dc99."""
    old = tmp_path / "old"
    old.mkdir()
    archive = subprocess.run(
        ["git", "-C", str(GPA_ROOT), "archive", PRISTINE, "gpa_bench_driver", "driver_apps.yaml"],
        capture_output=True, check=True,
    ).stdout
    subprocess.run(["tar", "-x", "-C", str(old)], input=archive, check=True)
    probe = tmp_path / "probe.py"
    probe.write_text(_PROBE)
    import sys

    results = []
    for root in (old, GPA_ROOT):
        proc = subprocess.run([sys.executable, str(probe), str(root)], capture_output=True,
                              text=True, check=False, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        assert proc.returncode == 0, proc.stderr[-3000:]
        results.append(proc.stdout.strip().splitlines()[-1])
    assert results[0] == results[1]


def test_from_args_namespace_without_new_flags_is_cuda_default(monkeypatch):
    monkeypatch.setenv("APPEB_PLATFORM", "perlmutter")
    ns = argparse.Namespace(
        app="bfs", sm_version=80, cuda_home=None, no_clean=False, build_only=False, nsys=True,
        ncu=False, config=None, swaps=None, detect_regions=False, postprocess_nsys=False,
        retain_nsys_profiles=False, num_samples=3, output_file=None, temp_dir=None,
        log_level="WARNING", no_progress=True, timeout=300, subprocess_output_char_limit=25000,
        suppress_command_stdout=False, no_sanitize=False, srun=False, small_problem=False,
    )
    cfg = DriverConfig.from_args(ns)
    assert cfg.gpu_backend == "cuda"
    assert cfg.config == CUDA_YAML
    assert cfg.no_sanitize is False
