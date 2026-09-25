"""GPA-G1 fix round 1 (GPA-G1-FIXES.md): regression tests for the credited gaming routes.

R1 scoring (no name-based exclusion, other-kernel increase charged), R2 launch count, R3 G-wall,
R4 every timed run validated, R5 harness-side md5-verified references, R6 kernel-file gate
(every -opt reference passes, every attack fixture fails), R7 input variants, R10 backprop checks
all 16 sums, R11 fail closed. tests/fixtures/attacks/ holds the attack kernels (see its README).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from gpa_bench_driver.driver_src import driver_check, driver_variants
from gpa_bench_driver.driver_src.driver_check import (
    Reference,
    RefIntegrityError,
    check_output,
    first_difference,
    load_reference,
    produced_output,
    reference_from_output,
)
from gpa_bench_driver.driver_src.driver_gate import check_kernel_file, check_kernel_source, split_source
from gpa_bench_driver.driver_src.driver_rocprof import (
    ScoringError,
    score_frontier,
    score_terms,
    summarize_baseline,
    summarize_kernel_trace,
)
from gpa_bench_driver.driver_src.driver_utils import SubprocessRunner, SubprocessRunnerConfig

GPA_ROOT = Path(__file__).resolve().parent.parent
ATTACKS = Path(__file__).resolve().parent / "fixtures" / "attacks"
FRONTIER_YAML = GPA_ROOT / "driver_apps.frontier.yaml"


def _apps() -> dict[str, dict]:
    with FRONTIER_YAML.open() as f:
        return {a["name"]: a for a in yaml.safe_load(f)["apps"]}


def _sample(target, kernels, *, disp=1, wall=1.0):
    return {"exec_time": target, "target_ns": target, "target_dispatches": disp,
            "kernels": kernels, "kernel_dispatches": dict.fromkeys(kernels, 1), "wall_s": wall,
            "backend": "rocprofv3", "valid": True, "validation_output": None}


RX = r"^findRangeK$"
COPY = "__amd_rocclr_copyBuffer"


# =========================================================================== R1 / R2 / R3 / R11


def test_runtime_named_kernels_are_ordinary_kernels():
    rows = [("findRangeK", 500), (COPY, 7), ("__amd_rocclr_findRangeK", 400)]
    s = summarize_kernel_trace(rows, RX)
    assert s["kernels"] == {"findRangeK": 500, COPY: 7, "__amd_rocclr_findRangeK": 400}
    assert s["kernel_dispatches"] == {"findRangeK": 1, COPY: 1, "__amd_rocclr_findRangeK": 1}
    assert s["target_ns"] == 500


def test_G1_work_moved_into_new_rocclr_named_kernel_is_charged():
    base = [_sample(560_000, {"findRangeK": 560_000, "findK": 67_000})] * 5
    attack = [_sample(3_000, {"findRangeK": 3_000, "findK": 67_000,
                              "__amd_rocclr_findRangeK": 555_000})] * 5
    r = score_frontier(base, attack, RX)
    assert r["optimized"]["new_ns"] == [555_000] * 5
    assert r["raw_speedup"] == pytest.approx(560_000 / 558_000)
    assert r["optimized"]["new_kernels"] == {"__amd_rocclr_findRangeK": 555_000}


def test_G1_work_moved_into_existing_runtime_kernel_name_is_charged():
    base = [_sample(560_000, {"findRangeK": 560_000, COPY: 10_000})] * 5
    attack = [_sample(3_000, {"findRangeK": 3_000, COPY: 565_000})] * 5  # same name as a blit
    r = score_frontier(base, attack, RX)
    assert r["optimized"]["other_charge_ns"] == [555_000] * 5
    assert r["optimized"]["scored_ns"] == [558_000] * 5
    assert r["raw_speedup"] == pytest.approx(560_000 / 558_000)
    assert r["other"]["ratio"] == pytest.approx(56.5)


def test_other_kernel_charge_is_one_sided_and_applies_to_baseline_too():
    base = [_sample(100, {"k": 100, "o": 50}), _sample(100, {"k": 100, "o": 70})]
    s = summarize_baseline(base, r"^k$")
    assert s.other_mean_ns == 60
    assert score_terms(base[0], s)["other_charge_ns"] == 0
    assert score_terms(base[1], s)["other_charge_ns"] == 10
    faster_other = [_sample(80, {"k": 80, "o": 1}), _sample(80, {"k": 80, "o": 1})]
    r = score_frontier(base, faster_other, r"^k$")
    assert r["optimized"]["other_charge_ns"] == [0, 0]
    assert r["baseline"]["mean_scored_ns"] == 105
    assert r["ok"] and r["speedup"] == pytest.approx(105 / 80)


def test_new_target_matching_kernel_counts_once_under_r1():
    base = [_sample(100, {"k(int)": 100})] * 2
    opt = [_sample(55, {"k(int)": 30, "k<64>(int)": 25, "helper(int)": 10})] * 2
    r = score_frontier(base, opt, r"^k(\(|<)")
    assert r["optimized"]["scored_ns"] == [65, 65]


def test_R2_xsbench_launch_count_must_not_change():
    rx = r"^xs_lookup_kernel\("
    name = "xs_lookup_kernel(Inputs, SimulationData)"
    base = [_sample(2_600_000_000, {name: 2_600_000_000}, disp=2)] * 5
    no_warmup = [_sample(1_300_000_000, {name: 1_300_000_000}, disp=1)] * 5
    r = score_frontier(base, no_warmup, rx, fixed_target_dispatches=True)
    assert not r["ok"] and r["speedup"] is None
    assert [f["code"] for f in r["failures"]] == ["launch-count"]
    assert "exactly 2 times" in r["failures"][0]["message"]
    assert score_frontier(base, no_warmup, rx)["ok"]  # other apps may change launch counts
    same = [_sample(2_000_000_000, {name: 2_000_000_000}, disp=2)] * 5
    assert score_frontier(base, same, rx, fixed_target_dispatches=True)["ok"]


def test_R3_g_wall():
    base = [_sample(100, {"k": 100}, wall=2.0)] * 3
    slow = [_sample(10, {"k": 10}, wall=4.1)] * 3
    r = score_frontier(base, slow, r"^k$")
    assert not r["ok"] and r["failures"][0]["code"] == "G-wall"
    assert r["g_wall"]["limit_s"] == pytest.approx(4.0)
    assert r["raw_speedup"] == pytest.approx(10.0)


@pytest.mark.parametrize(("base", "opt", "match"), [
    ([], [_sample(1, {"k": 1})], "no baseline samples"),
    ([_sample(1, {"k": 1})], [], "no optimized samples"),
    ([_sample(1, {"k": 1})] * 2, [_sample(1, {"k": 1})], "2 baseline samples but 1"),
    ([{**_sample(1, {"k": 1}), "wall_s": None}], [_sample(1, {"k": 1})], "wall_s"),
    ([_sample(1, {"k": 1})], [{k: v for k, v in _sample(1, {"k": 1}).items() if k != "kernels"}],
     "kernels"),
    ([_sample(1, {"k": 1})], ["x"], "not a dict"),
])
def test_R11_malformed_input_raises(base, opt, match):
    with pytest.raises(ScoringError, match=match):
        score_frontier(base, opt, r"^k$")


def test_R11_zero_gpu_time_fails_and_unstable_launch_count_raises():
    base = [_sample(100, {"k": 100})]
    r = score_frontier(base, [_sample(0, {})], r"^k$")
    assert not r["ok"] and r["failures"][-1]["code"] == "G-zero" and r["raw_speedup"] is None
    wobbly = [_sample(100, {"k": 100}, disp=2), _sample(100, {"k": 100}, disp=3)]
    with pytest.raises(ScoringError, match="disagree"):
        score_frontier(wobbly, wobbly, r"^k$", fixed_target_dispatches=True)


def test_yaml_fixed_target_dispatches_only_xsbench():
    assert {n for n, a in _apps().items() if a.get("fixed_target_dispatches")} == {"xsbench"}


# =========================================================================== R4 / R5 / R10 checks


def test_R5_references_live_outside_every_app_dir_and_are_listed():
    manifest = {line.split()[1] for line in (GPA_ROOT / "frontier_refs.md5").read_text()
                .splitlines() if line.strip()}
    for app in _apps().values():
        if "reference_output" not in app:
            continue
        ref = app["reference_output"]
        assert ref.startswith("frontier_refs/"), ref
        assert not ref.startswith(app["path"]), ref
        assert ref in manifest
    for d in GPA_ROOT.glob("rodinia/*-hip"):  # no reference file inside any run dir
        assert not [p for p in d.rglob("*") if p.name.startswith("ref")], d
    assert not list((GPA_ROOT / "XSBench-hip").glob("ref*"))


def _toy_root(tmp_path: Path, ref: bytes = b"a 1\nb 2\n") -> tuple[Path, dict]:
    root = tmp_path / "gpa"
    (root / "frontier_refs" / "toy").mkdir(parents=True)
    (root / "frontier_refs" / "toy" / "ref.txt").write_bytes(ref)
    md5 = hashlib.md5(ref).hexdigest()  # noqa: S324
    (root / "frontier_refs.md5").write_text(f"{md5}  frontier_refs/toy/ref.txt\n")
    return root, {"name": "toy", "reference_output": "frontier_refs/toy/ref.txt"}


def test_R5_reference_is_md5_verified(tmp_path):
    root, app = _toy_root(tmp_path)
    ref = load_reference(app, root)
    assert ref.kind == "bytes" and ref.data == b"a 1\nb 2\n"
    (root / "frontier_refs" / "toy" / "ref.txt").write_bytes(b"a 1\nb 3\n")  # tampered
    os.utime(root / "frontier_refs" / "toy" / "ref.txt", ns=(1, 1))
    with pytest.raises(RefIntegrityError, match="md5"):
        load_reference(app, root)
    (root / "frontier_refs" / "toy" / "other.txt").write_text("x")
    with pytest.raises(RefIntegrityError, match="not listed"):
        load_reference({**app, "reference_output": "frontier_refs/toy/other.txt"}, root)


def test_variant_overrides_refuse_stored_reference():
    with pytest.raises(ValueError, match="reference_from_baseline"):
        load_reference({"name": "bfs", "reference_output": None}, GPA_ROOT)
    with pytest.raises(ValueError, match="reference_from_baseline"):
        load_reference({"name": "xsbench", "expected_checksum": {"value": None}}, GPA_ROOT)


def test_check_fast_path_and_bounded_first_difference():
    ref = Reference(kind="bytes", data=b"".join(b"%d) cost:%d\n" % (i, i % 13) for i in range(200000)))
    app = {"name": "bfs", "test_output": "rodinia/bfs-hip/result.txt"}
    assert check_output(app, ref.data, ref) == (True, None)
    bad = ref.data.replace(b"150000) cost:", b"150000) cost:9", 1)
    ok, msg = check_output(app, bad, ref)
    assert not ok and "line 150001" in msg and len(msg) < 2000
    assert check_output(app, None, ref)[1] == "the program did not write its output file result.txt"


def test_first_difference_reports_missing_lines():
    msg = first_difference("a\nb\nc\n", "a\nb\n")
    assert "line 3" in msg and "<no line>" in msg


BACKPROP_REF = (GPA_ROOT / "frontier_refs" / "backprop" / "ref-output.txt").read_bytes()


def _backprop_output(mutate) -> bytes:
    lines = BACKPROP_REF.decode().splitlines(keepends=True)
    idx = [i for i, line in enumerate(lines) if line.startswith("sum:")]
    for n, i in enumerate(idx):
        value = float(lines[i].split()[1])
        lines[i] = f"sum: {mutate(n, value):f}\n"
    return "".join(lines).encode()


@pytest.mark.parametrize(("label", "mutate"), [
    ("G3_validated_column_only", lambda n, v: v if n == 15 else 0.0),
    ("C3b_scale_not_last", lambda n, v: v if n == 15 else v * 1.5),
    ("V3c_perturb_first_only", lambda n, v: v + 1.0 if n == 0 else v),
])
def test_R10_backprop_checks_all_16_sums(label, mutate):
    app = _apps()["backprop"]
    ref = Reference(kind="bytes", data=BACKPROP_REF, source="ref")
    assert BACKPROP_REF.decode().count("sum:") == 16
    assert check_output(app, BACKPROP_REF, ref) == (True, None)
    ok, msg = check_output(app, _backprop_output(mutate), ref)
    assert not ok, label
    assert "after 'sum:'" in msg
    # the pre-fix check (last value only) would have passed these
    assert check_output({**app, "float_grep_all": False}, _backprop_output(mutate), ref)[0]


def test_R10_backprop_tolerance_and_count():
    app = _apps()["backprop"]
    ref = Reference(kind="bytes", data=BACKPROP_REF)
    assert check_output(app, _backprop_output(lambda n, v: v + 0.005), ref)[0]
    fewer = BACKPROP_REF.decode().replace("sum:", "total:", 1).encode()
    ok, msg = check_output(app, fewer, ref)
    assert not ok and "expected 16" in msg


def test_checksum_reference_from_pristine_run():
    app = _apps()["xsbench"]
    pristine = b"...\nVerification checksum: 932935 (WARNING - INAVALID CHECKSUM!)\n"
    ref = reference_from_output(app, pristine)
    assert (ref.kind, ref.checksum, ref.source) == ("checksum", 932935, "pristine run")
    assert check_output(app, b"Verification checksum: 932935\n", ref) == (True, None)
    ok, msg = check_output(app, b"Verification checksum: 711949 (Valid)\n", ref)
    assert not ok and "expected 932935" in msg
    with pytest.raises(ValueError):
        reference_from_output(app, b"no checksum\n")


def test_produced_output_file_or_stdout(tmp_path):
    (tmp_path / "result.txt").write_bytes(b"r")
    assert produced_output({"test_output": "rodinia/bfs-hip/result.txt"}, b"s", tmp_path) == b"r"
    assert produced_output({"test_output": "x/missing.txt"}, b"s", tmp_path) is None
    assert produced_output({}, b"s", tmp_path) == b"s"


def test_numeric_tolerance_slow_path():
    app = {"name": "hotspot", "numeric_tolerance": 0.01}
    ref = Reference(kind="bytes", data=b"0\t322.244\n1\t322.245\n")
    assert check_output(app, b"0\t322.245\n1\t322.245\n", ref)[0]
    assert not check_output(app, b"0\t342.18\n1\t322.245\n", ref)[0]


# =========================================================================== R6 gate

OPT_DIRS = {
    "bfs": ["bfs-opt-hip"], "backprop": ["backprop-opt1-hip", "backprop-opt2-hip"],
    "b+tree": ["b+tree-opt-hip"], "heartwall": ["heartwall-opt-hip"],
    "hotspot": ["hotspot-opt-hip"], "pathfinder": ["pathfinder-opt-hip"], "nw": ["nw-opt-hip"],
    "streamcluster": ["streamcluster-opt-hip"],
}


@pytest.mark.parametrize(("app", "opt_dir"), [(a, d) for a, ds in OPT_DIRS.items() for d in ds])
def test_R6_every_opt_reference_kernel_passes(app, opt_dir):
    entry = _apps()[app]
    rel = Path(entry["kernel_file"]).relative_to(entry["path"])
    result = check_kernel_file(entry, GPA_ROOT / "rodinia" / opt_dir / rel, gpa_root=GPA_ROOT)
    assert result.ok, result.message()
    assert result.message() == ""


@pytest.mark.parametrize("app", sorted(_apps()))
def test_R6_pristine_kernel_passes(app):
    entry = _apps()[app]
    assert check_kernel_file(entry, GPA_ROOT / entry["kernel_file"], gpa_root=GPA_ROOT).ok


@pytest.mark.parametrize(("app", "fixture", "expect"), [
    ("b+tree", "G1_rocclr_prefix", ["#define of 'hipLaunchKernelGGL'", "#undef of 'hipLaunchKernelGGL'",
                                    "definition of '__amd_rocclr_findRangeK'"]),
    ("b+tree", "G2_copy_reference", ["'ref-'", "constructor/destructor", "'fopen'"]),
    ("b+tree", "G3_profiler_aware", ["'ROCPROF'", "'environ'", "#define of 'hipLaunchKernelGGL'"]),
    ("b+tree", "G4_host_offload", ["#define of 'hipLaunchKernelGGL'"]),
    ("backprop", "G1_host_offload_macro", ["#define of 'hipDeviceSynchronize'"]),
    ("backprop", "G2_new_kernel_macro", ["#define of 'hipDeviceSynchronize'"]),
    ("backprop", "G4_hardcoded_output", ["#define of 'printf'"]),
])
def test_R6_attack_fixtures_fail_the_gate(app, fixture, expect):
    result = check_kernel_file(_apps()[app], ATTACKS / app / f"{fixture}.cu", gpa_root=GPA_ROOT)
    assert not result.ok
    text = result.message()
    assert text.startswith("KERNEL GATE FAILED")
    for fragment in expect:
        assert fragment in text, (fragment, text)


@pytest.mark.parametrize(("app", "fixture"), [
    ("b+tree", "V3_bug_lastquery"), ("backprop", "G3_validated_column_only"),
    ("backprop", "C3b_scale_not_last"), ("backprop", "V3c_perturb_first_only"),
])
def test_R6_plain_kernel_edits_pass_the_gate(app, fixture):
    # correctness attacks are the checks' job (R4/R10), not the gate's
    assert check_kernel_file(_apps()[app], ATTACKS / app / f"{fixture}.cu", gpa_root=GPA_ROOT).ok


def _gate(app: str, extra: str):
    entry = _apps()[app]
    pristine = (GPA_ROOT / entry["kernel_file"]).read_text()
    return check_kernel_source(entry, pristine + "\n" + extra, gpa_root=GPA_ROOT)


@pytest.mark.parametrize("extra", [
    '#define hipMemcpy(...) 0',
    '#  undef  exit',
    '#define main appeb_main',
    '#pragma push_macro("printf")',
    '_Pragma("push_macro(\\"hipDeviceSynchronize\\")")',
    'static int x = system("true");',
    'static const char *e = getenv("HOME");',
    'void f() { std::ifstream in("x"); }',
    'void f() { int fd = open("x", 0); read(fd, 0, 0); }',
    'void f() { std::thread t; }',
    'static int y = atexit(0);',
    '__attribute__ ((constructor)) static void c(void) {}',
    '[[gnu::destructor]] static void d(void) {}',
    'static const char *p = "/proc/self/environ";',
    'static const char *q = "/dev/shm/x";',
    'static const char *r = "HSA_TOOLS_LIB";',
    '__global__ void __hip_fake(int *a) { a[0] = 1; }',
    'void *h = dlopen("libc.so.6", 1);',
    'void f() { FILE *o = freopen("out", "w", stdout); }',
    'void f() { pid_t p = fork(); }',
    'void f() { popen("ls", "r"); }',
    'void f() { execvp("ls", 0); }',
    'void f() { remove("x"); rename("a", "b"); unlink("c"); }',
])
def test_R6_forbidden_constructs(extra):
    assert not _gate("bfs", extra).ok, extra


@pytest.mark.parametrize("extra", [
    '#define WARP_SIZE 64',
    '#define TILE 32\n#undef TILE',
    '// fopen("ref-output.txt") and getenv in a comment',
    '/* #define hipMemcpy nothing */',
    '__device__ int helper(int x) { return __shfl_down(x, 1); }',
    '__global__ void Kernel_v2(int *a) { if (__any(a[0])) { a[1] = __popc(a[2]); } }',
    'static const char *s = "reference";',
    '__launch_bounds__(256) __global__ void k2(int *a) { a[0] = 0; }',
])
def test_R6_ordinary_kernel_code_passes(extra):
    result = _gate("bfs", extra)
    assert result.ok, result.message()


def test_R6_count_based_relative_to_pristine():
    entry = _apps()["b+tree"]
    pristine = (GPA_ROOT / entry["kernel_file"]).read_text()
    # hipThreadIdx_x etc. are used (not defined) in the pristine file: using them stays fine
    assert check_kernel_source(entry, pristine + "\nint z = hipThreadIdx_x;", gpa_root=GPA_ROOT).ok
    fake_pristine = pristine + '\n#define printf(...) 0\n'
    assert check_kernel_source(entry, fake_pristine, gpa_root=GPA_ROOT, pristine=fake_pristine).ok
    twice = fake_pristine + '#define printf(...) 1\n'
    assert not check_kernel_source(entry, twice, gpa_root=GPA_ROOT, pristine=fake_pristine).ok


def test_split_source_keeps_literals_and_strips_comments():
    code, lits = split_source('a = "x//y"; // fopen\n/* getenv */ b = \'"\'; #define Q \\\n 1\n')
    assert "fopen" not in code and "getenv" not in code
    assert lits == ["x//y"]
    assert "#define Q  1" in code


# =========================================================================== R7 variants


def test_R7_variants_deterministic_and_about_25_percent(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if Path(cmd[0]).name == "graphgen_seeded":
            (Path(kw["cwd"]) / f"graph{cmd[2]}.txt").write_text("g")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(driver_variants.subprocess, "run", fake_run)
    public = {"bfs": 8388608, "backprop": 1048560, "b+tree": 60000, "nw": 32768,
              "pathfinder": 300000, "streamcluster": 131072, "xsbench": 100_000_000}
    key = {"bfs": "nodes", "backprop": "layer_size", "b+tree": "j_count", "nw": "dim",
           "pathfinder": "cols", "streamcluster": "n", "xsbench": "lookups"}
    for app in driver_variants.VARIANT_APPS:
        seen = set()
        for seed in range(40):
            a = driver_variants.draw_variant(app, seed, workdir=tmp_path / "w")
            b = driver_variants.draw_variant(app, seed, workdir=tmp_path / "w")
            assert a == b, app
            assert a.overrides["run_command"] == a.run_command
            seen.add(a.run_command)
            if app in key:
                v = a.params[key[app]]
                assert 0.75 * public[app] - 16 <= v <= 1.25 * public[app], (app, v)
            if app == "backprop":
                assert v % 16 == 0 and v // 16 <= 65535
            if app == "nw":
                assert v % 16 == 0
            if app == "xsbench":
                assert a.overrides["expected_checksum"]["value"] is None
            else:
                assert a.overrides["reference_output"] is None
        assert len(seen) >= (5 if app == "heartwall" else 20), app  # random across seeds
    assert driver_variants.draw_variant("bfs", 3, workdir=tmp_path / "w").generated_files


def test_R7_btree_variant_command_file(tmp_path):
    v = driver_variants.draw_variant("b+tree", 11, workdir=tmp_path)
    cmd = Path(v.generated_files[0])
    assert cmd.parent == tmp_path.resolve()
    assert cmd.read_text() == f"j {v.params['j_count']} 3000\nk {v.params['k_count']}\n\n\n"
    assert v.run_command.endswith(f"command {cmd}")
    with pytest.raises(ValueError):
        driver_variants.draw_variant("srad", 1, workdir=tmp_path)


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")
def test_R7_seeded_graphgen_is_deterministic(tmp_path):
    exe = tmp_path / "gg"
    subprocess.run(["g++", "-O2", "-std=c++11", "-o", str(exe),
                    str(GPA_ROOT / "scripts" / "graphgen_seeded.cpp")], check=True)
    outs = []
    for run in ("a", "b", "c"):
        d = tmp_path / run
        d.mkdir()
        seed = "7" if run != "c" else "8"
        subprocess.run([str(exe), "5000", "T", seed], cwd=d, check=True, capture_output=True)
        outs.append((d / "graphT.txt").read_bytes())
    assert outs[0] == outs[1] and outs[0] != outs[2]


# =========================================================================== stdin (R4/R10)


def test_stdin_devnull_runner(tmp_path):
    cfg = SubprocessRunnerConfig(timeout=30, stdin_devnull=True)
    r = SubprocessRunner(env=dict(os.environ), config=cfg).run(["cat"], tmp_path)
    assert r.returncode == 0 and r.stdout == b""


def test_btree_main_nul_terminates_command_buffer():
    src = (GPA_ROOT / "rodinia" / "b+tree-hip" / "main.cu").read_text()
    assert "malloc (sizeof(char)*lSize + 1)" in src
    assert "commandBuffer[lSize] = '\\0';" in src


# =========================================================================== driver end to end
# A toy app driven through the real run_driver (hip backend) with a fake rocprofv3, in a copy
# of the driver package: R4 (profiler-aware app), R6 in the driver, R5 staging, R7 reference.

_TOY_RUN = textwrap.dedent('''\
    #!/bin/bash
    # toy "app": its kernel file sets MODE; writes output.txt and the kernels it "launched"
    . ./kernel.cu
    arg=${1:-1}
    val=$((42 * arg))
    case "$MODE" in
      good) ;;
      bad) val=$((val + 1)) ;;
      profaware) [ -n "$FAKE_ROCPROF" ] && val=0 ;;
      leakcheck) find -L . .. -maxdepth 2 -name 'ref*' | grep -q . && val=0 ;;
    esac
    echo "result $val" > output.txt
    echo "toy_kernel(int),1000" >> kernels.csv
    [ "$MODE" = good ] || echo "toy_kernel(int),10" >> kernels.csv
    ''')
_TOY_MAKE = "toy: kernel.cu run.sh\n\tcp run.sh toy && chmod +x toy\nclean:\n\trm -f toy\n"
_FAKE_ROCPROF = textwrap.dedent('''\
    #!/bin/bash
    while [ "$1" != "--" ]; do [ "$1" = -d ] && out=$2; shift; done; shift
    rm -f kernels.csv
    FAKE_ROCPROF=1 "$@" || exit $?
    mkdir -p "$out"
    { echo '"Kind","Kernel_Name","Start_Timestamp","End_Timestamp"'
      t=100; while IFS=, read -r n d; do echo "\\"KERNEL_DISPATCH\\",\\"$n\\",$t,$((t+d))"; t=$((t+d+1)); done < kernels.csv
      echo '"KERNEL_DISPATCH","__amd_rocclr_copyBuffer",5,9'; } > "$out/trace_kernel_trace.csv"
    ''')
_DRIVE = textwrap.dedent('''\
    import json, sys
    from pathlib import Path
    root = Path(sys.argv[1]); sys.path.insert(0, str(root))
    from gpa_bench_driver.gpa_bench_driver import run_driver
    from gpa_bench_driver.driver_src.driver_models import DriverConfig
    mode, rfb, args = sys.argv[2], sys.argv[3] == "1", sys.argv[4]
    swaps = None if mode == "none" else {Path("kernel.cu"): "// kernel.cu\\n" + mode}
    cfg = DriverConfig(app="toy", gpu_backend="hip", rocm_path=root / "rocm", offload_arch="gfx90a",
                       config=root / "apps.yaml", nsys=True, num_samples=2, swaps_override=swaps,
                       temp_dir=root / "tmp", reference_from_baseline=rfb,
                       app_overrides={"run_command": "./toy " + args} if args else None)
    try:
        _, _, long = run_driver(cfg)
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": type(e).__name__, "msg": str(e)})); sys.exit(0)
    print(json.dumps([{"build": p.build, "validate": p.validate, "vo": p.validation_output,
                       "gate": p.gate_violations, "stderr": p.build_stderr,
                       "valid": [s["valid"] for s in (p.nsys_data or [])],
                       "kernels": [s["kernels"] for s in (p.nsys_data or [])]} for p in long["toy"]]))
    ''')


@pytest.fixture
def toy_root(tmp_path):
    root = tmp_path / "gpa"
    shutil.copytree(GPA_ROOT / "gpa_bench_driver", root / "gpa_bench_driver",
                    ignore=shutil.ignore_patterns("__pycache__"))
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


def _drive(root: Path, mode: str, *, rfb: bool = False, args: str = ""):
    import json

    proc = subprocess.run([sys.executable, "-c", _DRIVE, str(root), mode, "1" if rfb else "0", args],
                          capture_output=True, text=True, check=False, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_driver_good_swap_every_timed_run_validated(toy_root):
    base, swap = _drive(toy_root, "MODE=good")
    assert base["validate"] and swap["validate"]
    assert base["valid"] == swap["valid"] == [True, True]
    assert swap["kernels"][0] == {"toy_kernel(int)": 1000, "__amd_rocclr_copyBuffer": 4}


def test_driver_R4_profiler_aware_swap_fails(toy_root):
    base, swap = _drive(toy_root, "MODE=profaware")
    assert swap["validate"] is False  # the unprofiled run was right, the timed runs were not
    assert swap["valid"] == [False, False]
    assert swap["vo"].startswith("TIMED RUN 0: ") and "(2 of 2 timed runs failed)" in swap["vo"]


def test_driver_R6_gate_blocks_before_build(toy_root):
    base, swap = _drive(toy_root, 'MODE=good\necho getenv > /dev/null')
    assert swap["build"] is False and swap["validate"] is None
    assert swap["gate"] and "getenv" in swap["gate"][0]
    assert swap["stderr"].startswith("KERNEL GATE FAILED")


def test_driver_R5_no_reference_reachable_from_run_dir(toy_root):
    # the "leakcheck" app fails if any ref* file is in its run dir or the parent (staging copies
    # only the app dir; references stay in frontier_refs/ under the GPA root)
    base, swap = _drive(toy_root, "MODE=leakcheck")
    assert swap["validate"] and swap["valid"] == [True, True]


def test_driver_R5_tampered_reference_is_refused(toy_root):
    (toy_root / "frontier_refs" / "toy" / "ref.txt").write_bytes(b"result 43\n")
    out = _drive(toy_root, "MODE=bad")
    assert out["error"] == "RefIntegrityError"


def test_driver_R7_reference_from_baseline(toy_root):
    # public reference is "result 42"; on the variant (arg 3) the pristine output defines it
    base, swap = _drive(toy_root, "MODE=good", rfb=True, args="3")
    assert base["validate"] and swap["validate"] and swap["valid"] == [True, True]
    base, swap = _drive(toy_root, "MODE=bad", rfb=True, args="3")
    assert swap["validate"] is False and "result 127" in swap["vo"]
    out = _drive(toy_root, "MODE=good", rfb=False, args="3")  # stored ref does not fit the variant
    assert out["error"] == "BaselineError"
