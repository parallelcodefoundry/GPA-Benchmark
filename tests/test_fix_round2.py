"""GPA-G1 fix round 2 (GPA-G1-FIXES2.md): F1 interleaved timing + mean charge, F2 template-aware
score_regex, F3 gate on the preprocessed kernel file, F4 G-cpu, F5 DriverInfraError.

tests/fixtures/round2/ holds the round-2 red-team kernels (evasions must FAIL the gate, legit
probes must PASS it); see its README.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from gpa_bench_driver.driver_src.driver_gate import check_kernel_file, check_kernel_source
from gpa_bench_driver.driver_src.driver_rocprof import (
    ScoringError,
    cpu_guard,
    score_frontier,
    summarize_kernel_trace,
)
from gpa_bench_driver.driver_src.driver_utils import (
    DriverInfraError,
    SubprocessRunner,
    SubprocessRunnerConfig,
    head_tail,
)

GPA_ROOT = Path(__file__).resolve().parent.parent
FIX = Path(__file__).resolve().parent / "fixtures"
ROUND2 = FIX / "round2"
HIPCC = Path(os.environ.get("ROCM_PATH", "/opt/rocm-7.0.2")) / "bin" / "hipcc"
needs_hipcc = pytest.mark.skipif(not HIPCC.exists() and shutil.which("hipcc") is None,
                                 reason="the preprocessed gate needs hipcc (ROCm)")


def _apps() -> dict[str, dict]:
    with (GPA_ROOT / "driver_apps.frontier.yaml").open() as f:
        return {a["name"]: a for a in yaml.safe_load(f)["apps"]}


def _s(target, kernels, *, disp=1, wall=1.0):
    return {"exec_time": target, "target_ns": target, "target_dispatches": disp,
            "kernels": kernels, "wall_s": wall}


# =========================================================================== F2


REAL_TEMPLATED = {  # demangled rocprofv3 names seen in the round-2 red team (rt2-*/out)
    "xsbench": "void xs_lookup_kernel<0>(Inputs, SimulationData)",
    "streamcluster": "void kernel_compute_cost<128>(int, int, long, Point*, int, int, float*, "
                     "float*, int*, bool*)",
}
PLAIN = {
    "bfs": "Kernel(Node*, int*, bool*, bool*, bool*, int*, int)",
    "backprop": "bpnn_layerforward_CUDA(float*, float*, float*, float*, int, int)",
    "b+tree": "findRangeK", "heartwall": "kernel()",
    "hotspot": "calculate_temp(int, float*, float*, float*, int, int, int, int, float, float, "
               "float, float, float, float)",
    "pathfinder": "dynproc_kernel(int, int*, int*, int*, int, int, int, int)",
    "nw": "needle_cuda_shared_1(int*, int*, int, int, int, int)",
    "streamcluster": "kernel_compute_cost(int, int, long, Point*, int, int, float*, float*, "
                     "int*, bool*)",
    "xsbench": "xs_lookup_kernel(Inputs, SimulationData)",
}
DECOYS = ["Kernel2(bool*, bool*, bool*, bool*, int)", "findK",
          "needle_cuda_shared_2(int*, int*, int, int, int, int)",
          "bpnn_adjust_weights_cuda(float*, int, float*, int, float*, float*)",
          "xs_lookup_kernel_v2(Inputs, SimulationData)", "__amd_rocclr_copyBuffer",
          "void my_kernel<1>(int)", "calculate_temp2(int)"]


@pytest.mark.parametrize("app", sorted(PLAIN))
def test_F2_score_regex_accepts_templates_prefix_and_namespace(app):
    rx = re.compile(_apps()[app]["score_regex"])
    name = PLAIN[app]
    base = name.split("(")[0]
    args = name[len(base):] or "(int)"
    assert rx.search(name)
    assert rx.search(f"void {base}<256, float>{args}")
    assert rx.search(f"void ns::{base}<(GridType)1>{args}")
    if app in REAL_TEMPLATED:
        assert rx.search(REAL_TEMPLATED[app])
    assert not any(rx.search(d) for d in DECOYS if not d.startswith(base + "(")), app


def test_F2_R2_counts_every_instantiation():
    rx = _apps()["xsbench"]["score_regex"]
    rows = [("void xs_lookup_kernel<0>(Inputs, SimulationData)", 10),
            ("void xs_lookup_kernel<1>(Inputs, SimulationData)", 20),
            ("__amd_rocclr_copyBuffer", 1)]
    s = summarize_kernel_trace(rows, rx)
    assert s["target_dispatches"] == 2 and s["target_ns"] == 30
    base = [_s(2600, {"xs_lookup_kernel(Inputs, SimulationData)": 2600}, disp=2)] * 3
    templ = [{**s, "wall_s": 1.0}] * 3
    r = score_frontier(base, templ, rx, fixed_target_dispatches=True)
    assert r["ok"], r["failures"]  # the round-2 L3 false positive (launch-count) is gone


# =========================================================================== F1


def test_F1_charge_on_means_no_op_not_biased():
    base = [_s(100, {"k": 100, "o": 40}), _s(100, {"k": 100, "o": 60})]
    noop = [_s(100, {"k": 100, "o": 60}), _s(100, {"k": 100, "o": 40})]
    r = score_frontier(base, noop, r"^k$")
    assert r["other"]["charge_ns"] == 0
    assert r["speedup"] == pytest.approx(1.0)
    assert r["optimized"]["scored_ns"] == [100, 100]


# =========================================================================== F4


def test_F4_cpu_guard_math_and_failure():
    g = cpu_guard([1.0, 1.02], [1.01, 1.03], 0.02)
    assert g["ok"] and g["slack_s"] == pytest.approx(max(5 * 0.02 / 2 ** 0.5, 0.10))
    g = cpu_guard([1.0, 1.0], [1.5, 1.5], 0.02)
    assert not g["ok"] and g["delta_s"] == pytest.approx(0.5)
    assert cpu_guard([1.0], [1.09], 0.001)["ok"]  # the 100 ms floor (H1)
    assert not cpu_guard([1.0], [1.2], 0.001)["ok"]
    base = [{**_s(100, {"k": 100}), "cpu_s": 1.0}, {**_s(100, {"k": 100}), "cpu_s": 1.01}]
    opt = [{**_s(25, {"k": 25}), "cpu_s": 1.4}, {**_s(25, {"k": 25}), "cpu_s": 1.41}]  # "4x" on the CPU
    r = score_frontier(base, opt, r"^k$", cpu_sigma_s=0.01)
    assert not r["ok"] and [f["code"] for f in r["failures"]] == ["G-cpu"]
    assert "CPU TIME" in r["failures"][0]["message"] and r["raw_speedup"] == pytest.approx(4)
    assert score_frontier(base, opt, r"^k$")["cpu"] == {"checked": False}  # no sigma -> not checked


@pytest.mark.parametrize(("b", "o", "sigma", "match"), [
    ([], [1.0], 0.1, "at least one"), ([1.0], [], 0.1, "at least one"),
    ([1.0, 1.0], [1.0], 0.1, "2 baseline samples but 1"), ([1.0], [1.0], None, "cpu_sigma_s"),
    ([{"cpu_s": None}], [1.0], 0.1, "no cpu_s"),
])
def test_F4_cpu_guard_fails_closed(b, o, sigma, match):
    with pytest.raises(ScoringError, match=match):
        cpu_guard(b, o, sigma)


def test_F4_every_app_has_a_calibrated_sigma():
    for name, app in _apps().items():
        assert isinstance(app.get("cpu_sigma_s"), (int, float)) and app["cpu_sigma_s"] > 0, name


def test_F4_runner_measures_child_cpu_time(tmp_path):
    runner = SubprocessRunner(env=dict(os.environ), config=SubprocessRunnerConfig(timeout=60))
    burn = [sys.executable, "-c", "import time\nt=time.process_time()\n"
            "while time.process_time()-t<0.3: pass"]
    r = runner.run(burn, tmp_path, measure_cpu=True)
    assert 0.25 <= r.cpu_user_s + r.cpu_sys_s < 2.0
    idle = runner.run([sys.executable, "-c", "import time; time.sleep(0.3)"], tmp_path,
                      measure_cpu=True)
    assert idle.cpu_user_s + idle.cpu_sys_s < 0.2


# =========================================================================== F5


def test_F5_head_tail():
    text = "A" * 2000 + "MIDDLE" + "Z" * 2000
    out = head_tail(text, 100, 100)
    assert out.startswith("A" * 100) and out.endswith("Z" * 100)
    assert "chars omitted" in out and "MIDDLE" not in out
    assert head_tail(b"short") == "short"


def test_F5_vanished_working_dir_is_infra(tmp_path):
    cfg = SubprocessRunnerConfig(timeout=30, raise_infra=True)
    runner = SubprocessRunner(env=dict(os.environ), config=cfg)
    with pytest.raises(DriverInfraError, match="vanished"):
        runner.run(["true"], tmp_path / "gone")
    plain = SubprocessRunner(env=dict(os.environ), config=SubprocessRunnerConfig(timeout=30))
    assert plain.run(["true"], tmp_path / "gone").returncode != 0  # cuda path unchanged


# =========================================================================== F3


EVASIONS = sorted((ROUND2 / "evasion").glob("*/*.cu"))
LEGIT = sorted((ROUND2 / "legit").glob("*/*.cu"))
ROUND1_ATTACKS = [FIX / "attacks" / a for a in (  # the credited round-1 gaming kernels
    "b+tree/G1_rocclr_prefix.cu", "b+tree/G2_copy_reference.cu", "b+tree/G3_profiler_aware.cu",
    "b+tree/G4_host_offload.cu", "backprop/G1_host_offload_macro.cu",
    "backprop/G2_new_kernel_macro.cu", "backprop/G4_hardcoded_output.cu")]


@needs_hipcc
@pytest.mark.parametrize("path", EVASIONS + ROUND1_ATTACKS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_F3_every_evasion_fails_the_gate(path):
    res = check_kernel_file(_apps()[path.parent.name], path, gpa_root=GPA_ROOT)
    assert not res.ok and res.preprocessed, res.message()


@needs_hipcc
@pytest.mark.parametrize("path", LEGIT, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_F3_every_legit_probe_passes_the_gate(path):
    res = check_kernel_file(_apps()[path.parent.name], path, gpa_root=GPA_ROOT)
    assert res.ok, res.message()


@needs_hipcc
@pytest.mark.parametrize(("app", "fixture", "fragment"), [
    ("backprop", "a_paste", "definition of '__amd_rocclr_addfn' after macro expansion"),
    ("b+tree", "n1_fopen64_tamper", "call of 'fopen64' after macro expansion"),
    ("b+tree", "n3_reserved_alias", "definition of '__amd_rocclr_rt_helper' after macro expansion"),
    ("b+tree", "att4_reserved_tokenpaste", "definition of '__amd_rocclr_helper' after macro"),
    ("backprop", "a_exp_macro", "'exp': it is declared or defined by the included headers"),
])
def test_F3_round2_evasions_are_caught_by_the_preprocessed_pass(app, fixture, fragment):
    res = check_kernel_file(_apps()[app], ROUND2 / "evasion" / app / f"{fixture}.cu",
                            gpa_root=GPA_ROOT)
    assert fragment in res.message(), res.message()
    raw = check_kernel_file(_apps()[app], ROUND2 / "evasion" / app / f"{fixture}.cu",
                            gpa_root=GPA_ROOT, preprocess=False)
    if fixture != "a_exp_macro":
        assert raw.ok  # these evaded the round-1 (raw-text) gate


def _gate(app: str, extra: str, **kw):
    entry = _apps()[app]
    pristine = (GPA_ROOT / entry["kernel_file"]).read_text()
    return check_kernel_source(entry, pristine + "\n" + extra, gpa_root=GPA_ROOT, **kw)


@needs_hipcc
@pytest.mark.parametrize("extra", [
    "#include <fcntl.h>\n#include <unistd.h>\nstatic int e1(){ return ::open(\"x\", 0); }",
    "#include <stdio.h>\nstatic void e2(){ FILE *f = fopen64(\"x\", \"r\"); (void)f; }",
    "#include <fcntl.h>\nstatic int e3(){ return creat64(\"x\", 0644); }",
    "#define CALL(f) f\n#include <stdlib.h>\nstatic char *e4(){ return CALL(getenv)(\"HOME\"); }",
    "#define P(a,b) a##b\n__device__ int P(__hip_,x)(int v){ return v; }",
    "#define NAME __amd_rocclr_evil\n__global__ void NAME(int *a){ a[0]=0; }",
    "#include <cstdio>\n#define S(x) #x\nstatic const char *e7 = S(/proc/self/environ);",
    "#define OPENER(a) ::a\n#include <stdio.h>\nstatic int e8(){ return OPENER(remove)(\"x\"); }",
])
def test_F3_expanded_evasions_fail(extra):
    assert not _gate("hotspot", extra).ok, extra


@needs_hipcc
@pytest.mark.parametrize("extra", [
    "#define HIP_CHECK(call) do { hipError_t e_ = (call); (void)e_; } while (0)",
    "#define CUDA_CHECK(x) (x)\n#define HIPCHECK(x) (x)\n#define BLOCK 16\n#define TILE 32",
    "template <int B> __device__ __forceinline__ float t_helper(float v) { return v * B; }",
    "__global__ void __launch_bounds__(256) k2(float *a) { extern __shared__ float sm[]; "
    "sm[threadIdx.x] = a[threadIdx.x]; a[0] = __builtin_amdgcn_readfirstlane((int)sm[0]); }",
    "__device__ int h(int x) {\n#pragma unroll 4\n for (int i = 0; i < 4; i++) x += i; return x; }",
])
def test_F3_legit_constructs_pass(extra):
    res = _gate("hotspot", extra)
    assert res.ok, res.message()


def test_F3_raw_pass_allows_upper_case_check_macros():
    assert _gate("bfs", "#define HIP_CHECK(x) (x)", preprocess=False).ok
    assert not _gate("bfs", "#define hipMemcpy(...) 0", preprocess=False).ok


@needs_hipcc
def test_F3_unpreprocessable_candidate_fails_closed():
    res = _gate("hotspot", '#include "no_such_header_appeb.h"')
    assert not res.ok and "could not be preprocessed" in res.message()


def test_F3_every_app_names_its_translation_unit():
    for name, app in _apps().items():
        spec = app["gate_preprocess"]
        assert (GPA_ROOT / app["path"] / spec["tu"]).is_file(), name
        assert "-x hip" in spec["flags"], name


# =========================================================================== F1/F4/F5 driver


_TOY_RUN = textwrap.dedent('''\
    #!/bin/bash
    . ./kernel.cu
    val=42
    [ "$MODE" = cpuhog ] && { i=0; while [ $i -lt 400000 ]; do i=$((i+1)); done; }
    echo "result $val" > output.txt
    echo "toy_kernel(int),1000" >> kernels.csv
    ''')
_TOY_MAKE = "toy: kernel.cu run.sh\n\tcp run.sh toy && chmod +x toy\nclean:\n\trm -f toy\n"
_FAKE_ROCPROF = textwrap.dedent('''\
    #!/bin/bash
    while [ "$1" != "--" ]; do [ "$1" = -d ] && out=$2; shift; done; shift
    [ -n "$FAKE_ROCPROF_CRASH" ] && exit 139
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
    mode = sys.argv[2]
    cfg = DriverConfig(app="toy", gpu_backend="hip", rocm_path=root / "rocm", offload_arch="gfx90a",
                       config=root / "apps.yaml", nsys=True, num_samples=3, pairs=4,
                       swaps_override={Path("kernel.cu"): "// kernel.cu\\nMODE=" + mode},
                       temp_dir=root / "tmp", kernel_gate=False)
    try:
        _, _, long = run_driver(cfg)
    except DriverInfraError as e:
        print(json.dumps({"infra": str(e)[:300]})); sys.exit(0)
    b, s = long["toy"]
    print(json.dumps({"validate": s.validate, "order_b": [x["order"] for x in s.baseline_nsys_data],
                      "order_o": [x["order"] for x in s.nsys_data],
                      "cpu_b": [x["cpu_s"] for x in s.baseline_nsys_data],
                      "cpu_o": [x["cpu_s"] for x in s.nsys_data],
                      "base_same": b.nsys_data == s.baseline_nsys_data}))
    ''')


@pytest.fixture
def toy2(tmp_path):
    import hashlib

    root = tmp_path / "gpa"
    shutil.copytree(GPA_ROOT / "gpa_bench_driver", root / "gpa_bench_driver",
                    ignore=shutil.ignore_patterns("__pycache__"))
    tools = root / "frontier_tools"
    tools.mkdir(exist_ok=True)
    reset = tools / "vram_reset"
    reset.write_text("#!/bin/bash\necho 'vram_reset allocs=1 GiB=0.00'\n")
    reset.chmod(0o755)
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


def _drive2(root: Path, mode: str, env: dict | None = None) -> dict:
    proc = subprocess.run([sys.executable, "-c", _DRIVE, str(root), mode], capture_output=True,
                          text=True, check=False, timeout=600, env={**os.environ, **(env or {})})
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_F1_driver_interleaves_and_discards_warmup(toy2):
    out = _drive2(toy2, "good")
    assert out["validate"] is True
    # J0 T1: ABBA order (B O O B B O O B), not strict alternation (M.md: alternation aliases
    # with period-2 VRAM-placement states). Mutation check: strict B,O,B,O would give [0,2,4,6].
    assert out["order_b"] == [0, 3, 4, 7] and out["order_o"] == [1, 2, 5, 6]
    # H1: CPU rides in every profiled sample (no separate pairs)
    assert len(out["cpu_b"]) == len(out["cpu_o"]) == 4 and out["base_same"]


def test_H1_driver_measures_cpu_in_profiled_runs(toy2):
    out = _drive2(toy2, "cpuhog")
    assert out["validate"] is True
    assert len(out["cpu_o"]) == 4 and all(c is not None for c in out["cpu_o"])
    delta = sum(out["cpu_o"]) / len(out["cpu_o"]) - sum(out["cpu_b"]) / len(out["cpu_b"])
    assert delta > 0.2, out
    assert not cpu_guard(out["cpu_b"], out["cpu_o"], 0.01)["ok"]


def test_F5_profiler_crash_with_working_binary_is_infra(toy2):
    out = _drive2(toy2, "good", env={"FAKE_ROCPROF_CRASH": "1"})
    assert "runs fine without the profiler" in out["infra"]
