"""GPA-G1 fix round 4 (GPA-G1-FIXES4.md): J1 gate attribution redesign (closes the macro-#line
bypass, fixes honest-header false positives, robust main-param count), J2 j=0 failure is infra,
J4 blocking-sync gate rejection, plus the H1 profiler-env proof and the R2 fixture wiring.

Each J1 rule test uses a minimal candidate (no header noise) and asserts the specific rule's
message (mutation-sensitive). Round-3 fixtures are wired to static gate assertions; recorded
compute evidence lives in .planning/frontier/gpa-g1/A.md.
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

from gpa_bench_driver.driver_src.driver_gate import check_kernel_file, check_kernel_source
from gpa_bench_driver.driver_src.driver_rocprof import score_frontier, summarize_kernel_trace

GPA_ROOT = Path(__file__).resolve().parent.parent
FIX = Path(__file__).resolve().parent / "fixtures"
HIPCC = Path(os.environ.get("ROCM_PATH", "/opt/rocm-7.0.2")) / "bin" / "hipcc"
needs_hipcc = pytest.mark.skipif(not HIPCC.exists() and shutil.which("hipcc") is None,
                                 reason="the preprocessed gate needs hipcc (ROCm)")
SYS_HDR = "/opt/rocm-7.0.2/include/hip/hip_runtime.h"


def _apps() -> dict[str, dict]:
    with (GPA_ROOT / "driver_apps.frontier.yaml").open() as f:
        return {a["name"]: a for a in yaml.safe_load(f)["apps"]}


def _gate(app: str, extra: str, **kw):
    entry = _apps()[app]
    pristine = (GPA_ROOT / entry["kernel_file"]).read_text()
    return check_kernel_source(entry, pristine + "\n" + extra, gpa_root=GPA_ROOT, **kw)


ASM_EVASION = 'static long e(){ long r; asm volatile("syscall" : "=a"(r) : "a"(39)); return r; }'


# =========================================================================== J1a raw rules

def test_J1_macro_line_directive_rejected():
    r = _gate("hotspot", '#define LN 1\n#line LN "%s"' % SYS_HDR, preprocess=False)
    assert not r.ok and "#line directive" in r.message()
    # mutation check: a candidate without the #line passes the raw pass
    assert _gate("hotspot", "// nothing", preprocess=False).ok


def test_J1_macro_line_bypass_does_not_hide_an_evasion():
    # the reviewer's bypass prepended to a real asm evasion must FAIL, on the #line rule
    bypass = '#define LN 1\n#line LN "%s"\n%s' % (SYS_HDR, ASM_EVASION)
    r = _gate("hotspot", bypass, preprocess=False)
    assert not r.ok and "#line directive" in r.message()


def test_J1_gnu_line_marker_rejected():
    r = _gate("hotspot", '# 1 "%s"' % SYS_HDR, preprocess=False)
    assert not r.ok and "GNU line marker" in r.message()


def test_J1_macro_computed_include_rejected():
    r = _gate("hotspot", "#define H <hip/hip_fp16.h>\n#include H", preprocess=False)
    assert not r.ok and "must be a literal" in r.message()


@pytest.mark.parametrize("inc", ['</tmp/x.h>', '"/etc/passwd"', '"../secret.h"', '</opt/rocm/x.h>'])
def test_J1_absolute_or_dotdot_include_rejected(inc):
    r = _gate("hotspot", f"#include {inc}", preprocess=False)
    assert not r.ok and "include path" in r.message()


# =========================================================================== J1b honest headers

HONEST_HEADERS = ["<hip/hip_fp16.h>", "<hip/hip_bf16.h>", "<cstring>", "<vector>", "<chrono>",
                  "<iostream>", "<numeric>", "<algorithm>", "<cstdint>", "<utility>",
                  "<hip/hip_cooperative_groups.h>"]


@needs_hipcc
@pytest.mark.parametrize("app", sorted({"bfs", "backprop", "b+tree", "heartwall", "hotspot",
                                        "pathfinder", "nw", "streamcluster", "xsbench"}))
def test_J1_honest_new_system_headers_pass(app):
    r = _gate(app, "\n".join(f"#include {h}" for h in HONEST_HEADERS))
    assert r.ok, r.message()


@needs_hipcc
def test_J1_macro_line_spoof_caught_by_preprocessed_pass_too():
    # even with the raw pass disabled, the physical file is the kernel file (not root-owned under
    # a system dir), so the spoofed tokens are analysed and the asm evasion is caught
    entry = _apps()["hotspot"]
    pristine = (GPA_ROOT / entry["kernel_file"]).read_text()
    body = '#define LN 1\n#line LN "%s"\n%s' % (SYS_HDR, ASM_EVASION)
    # drive only the preprocessed pass by giving a candidate the raw pass would also reject;
    # assert the asm is caught after expansion regardless
    r = check_kernel_source(entry, pristine + "\n" + body, gpa_root=GPA_ROOT)
    assert not r.ok
    assert any("#line" in v or "asm" in v.lower() for v in r.violations)


# =========================================================================== J1c main params

@needs_hipcc
@pytest.mark.parametrize("form", [
    "int main(int argc, char **argv, char **envp){ (void)envp; return 0; }",
    "auto main(int argc, char **argv, char **envp) -> int { (void)envp; return 0; }",
])
def test_J1_main_with_more_params_rejected_on_whole_program_file(form):
    # hotspot.cu is a whole-program kernel file whose pristine main has 2 params
    r = _gate("hotspot", form)
    assert not r.ok and "main()" in r.message()


@needs_hipcc
def test_J1_pristine_whole_program_main_passes():
    entry = _apps()["hotspot"]
    assert check_kernel_file(entry, GPA_ROOT / entry["kernel_file"], gpa_root=GPA_ROOT).ok


# =========================================================================== J4 blocking sync

@pytest.mark.parametrize("flag", ["hipDeviceScheduleBlockingSync", "hipEventBlockingSync"])
def test_J4_blocking_sync_rejected(flag):
    if flag == "hipEventBlockingSync":
        extra = f"static void b(){{ hipEvent_t e; hipEventCreateWithFlags(&e, {flag}); }}"
    else:
        extra = f"static void b(){{ hipSetDeviceFlags({flag}); }}"
    r = _gate("streamcluster", extra, preprocess=False)
    assert not r.ok and "blocking-sync" in r.message()


@needs_hipcc
@pytest.mark.parametrize("name", ["FP1_blocking_sync_only", "B3_blocking_sync_offload"])
def test_J4_round3_blocking_sync_fixtures_fail(name):
    r = check_kernel_file(_apps()["streamcluster"],
                          FIX / "round3" / "streamcluster" / f"{name}.cu", gpa_root=GPA_ROOT)
    assert not r.ok and "blocking-sync" in r.message()


@needs_hipcc
def test_J4_pristine_streamcluster_still_passes():
    # streamcluster uses hipEvents for timing but no blocking-sync flag -> must pass
    entry = _apps()["streamcluster"]
    assert check_kernel_file(entry, GPA_ROOT / entry["kernel_file"], gpa_root=GPA_ROOT).ok


# =========================================================================== R2 (static)

@needs_hipcc
def test_R2_drop_warmup_fixture_gate_passes_but_scoring_fails_launch_count():
    # the R2 attack drops the warmup launch; it is not a gate violation (it edits only the launch
    # loop), but the fixed_target_dispatches rule fails it in scoring.
    f = FIX / "attacks" / "xsbench" / "R2_drop_warmup.cu"
    assert check_kernel_file(_apps()["xsbench"], f, gpa_root=GPA_ROOT).ok
    rx = _apps()["xsbench"]["score_regex"]
    name = "xs_lookup_kernel(Inputs, SimulationData)"
    base = [{"target_ns": 2_600_000, "target_dispatches": 2, "kernels": {name: 2_600_000},
             "wall_s": 7.0, "cpu_s": 7.0}] * 3
    no_warmup = [{"target_ns": 1_300_000, "target_dispatches": 1, "kernels": {name: 1_300_000},
                  "wall_s": 5.5, "cpu_s": 5.5}] * 3
    r = score_frontier(base, no_warmup, rx, fixed_target_dispatches=True)
    assert not r["ok"] and any(f["code"] == "launch-count" for f in r["failures"])


# =========================================================================== J2 + H1 driver toy

# A toy app whose kernel-file MODE controls host CPU burn, optionally conditioned on the profiler
# environment, and whose fake rocprofv3 can fail the ORIGINAL (j=0) run once.
_TOY_RUN = textwrap.dedent('''\
    #!/bin/bash
    . ./kernel.cu
    burn(){ i=0; while [ $i -lt 500000 ]; do i=$((i+1)); done; }
    prof=0; [ -n "$ROCP_OUTPUT_DIR$ROCPROFILER_OUTPUT_PATH$FAKE_PROFILED" ] && prof=1
    case "$MODE" in
      good) ;;
      cpu_when_profiled)   [ $prof = 1 ] && burn ;;
      cpu_when_unprofiled) [ $prof = 0 ] && burn ;;
    esac
    echo "result 42" > output.txt
    echo "toy_kernel(int),1000" >> kernels.csv
    ''')
_TOY_MAKE = "toy: kernel.cu run.sh\n\tcp run.sh toy && chmod +x toy\nclean:\n\trm -f toy\n"
_FAKE_ROCPROF = textwrap.dedent('''\
    #!/bin/bash
    while [ "$1" != "--" ]; do [ "$1" = -d ] && out=$2; shift; done; shift
    # J2: fail the FIRST invocation in this app dir once (simulates a flaky ORIGINAL run)
    marker="$(dirname "$out")/.rocprof_calls"
    n=$(cat "$marker" 2>/dev/null || echo 0); echo $((n+1)) > "$marker"
    if [ -n "$FAIL_FIRST" ] && [ "$n" = 0 ]; then echo "tool init failed" >&2; exit 0; fi
    export FAKE_PROFILED=1
    rm -f kernels.csv
    "$@" || exit $?
    mkdir -p "$out"
    { echo '"Kind","Kernel_Name","Start_Timestamp","End_Timestamp"'
      t=100; while IFS=, read -r nm d; do echo "\\"KERNEL_DISPATCH\\",\\"$nm\\",$t,$((t+d))"; t=$((t+d+1)); done < kernels.csv
    } > "$out/trace_kernel_trace.csv"
    ''')
_DRIVE = textwrap.dedent('''\
    import json, sys
    from pathlib import Path
    root = Path(sys.argv[1]); mode = sys.argv[2]; sys.path.insert(0, str(root))
    from gpa_bench_driver.gpa_bench_driver import run_driver
    from gpa_bench_driver.driver_src.driver_models import DriverConfig
    from gpa_bench_driver.driver_src.driver_utils import DriverInfraError
    cfg = DriverConfig(app="toy", gpu_backend="hip", rocm_path=root / "rocm", offload_arch="gfx90a",
                       config=root / "apps.yaml", nsys=True, num_samples=3, pairs=4,
                       swaps_override={Path("kernel.cu"): "// kernel.cu\\nMODE=" + mode},
                       temp_dir=root / "tmp", kernel_gate=False, vram_reset_sha256="build-record")
    try:
        _, _, long = run_driver(cfg)
    except DriverInfraError as e:
        print(json.dumps({"infra": str(e)[:200]})); sys.exit(0)
    b, s = long["toy"]
    print(json.dumps({"validate": s.validate,
                      "cpu_b": [x["cpu_s"] for x in s.baseline_nsys_data],
                      "cpu_o": [x["cpu_s"] for x in s.nsys_data]}))
    ''')


@pytest.fixture
def toy(tmp_path):
    import hashlib

    root = tmp_path / "gpa"
    shutil.copytree(GPA_ROOT / "gpa_bench_driver", root / "gpa_bench_driver",
                    ignore=shutil.ignore_patterns("__pycache__"))
    tools = root / "frontier_tools"
    tools.mkdir(exist_ok=True)
    reset = tools / "vram_reset"
    reset.write_text("#!/bin/bash\necho 'vram_reset allocs=1 GiB=0.00'\n")
    reset.chmod(0o755)
    (tools / "vram_reset.sha256").write_text(  # K3 build record
        __import__("hashlib").sha256(reset.read_bytes()).hexdigest() + "  vram_reset\n")
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


def _drive(root: Path, mode: str, env: dict | None = None) -> dict:
    proc = subprocess.run([sys.executable, "-c", _DRIVE, str(root), mode], capture_output=True,
                          text=True, check=False, timeout=600, env={**os.environ, **(env or {})})
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_H1_cpu_is_measured_in_the_profiled_scored_run(toy):
    # the toy burns CPU only WHEN profiled: since G-cpu is measured in the profiled scored runs,
    # the extra CPU shows up (a profiler-conditional offload cannot hide)
    out = _drive(toy, "cpu_when_profiled")
    assert out["validate"] is True
    delta = sum(out["cpu_o"]) / len(out["cpu_o"]) - sum(out["cpu_b"]) / len(out["cpu_b"])
    assert delta > 0.05, out


def test_H1_cpu_burned_only_when_unprofiled_does_not_appear(toy):
    # the mirror: burning CPU only when UNPROFILED never happens during scoring (there are no
    # unprofiled scored runs), so the profiled series shows no extra CPU
    out = _drive(toy, "cpu_when_unprofiled")
    delta = sum(out["cpu_o"]) / len(out["cpu_o"]) - sum(out["cpu_b"]) / len(out["cpu_b"])
    assert abs(delta) < 0.05, out


def test_J2_original_j0_failure_is_infra_not_agent(toy):
    # the fake rocprofv3 fails the FIRST call in each app dir; the first timed run is the ORIGINAL
    # (j=0). J2: that is a DriverInfraError, not the agent's failure.
    out = _drive(toy, "good", env={"FAIL_FIRST": "1"})
    assert "infra" in out, out
    assert out["infra"].startswith("the original toy (j=0)"), out  # (the tmp path says "original")


# =========================================================================== J3 recorded evidence

EVIDENCE = FIX / "evidence"


@pytest.mark.parametrize(("name", "expect_fail"), [
    ("round4_hs_offload", "G-cpu"),        # profitable full host offload (4.0x) -> G-cpu
    ("round4_xs_offload", "G-cpu"),        # profitable partial offload (grid shrinks) -> G-cpu
    ("round4_r2_drop_warmup", "launch-count"),  # drop the warmup launch (2.0x) -> R2
])
def test_J3_recorded_compute_evidence(name, expect_fail):
    """The round-3/4 gaming fixtures, run on MI250X (frontier10244, fix round 4), are NOT
    credited. This pins the recorded verdicts; re-run with RUN_GPA_COMPUTE via val4.py."""
    d = json.loads((EVIDENCE / f"{name}.json").read_text())
    assert d["ok"] is False
    assert expect_fail in d["failures"], d
    if name == "round4_hs_offload":
        assert d["raw_speedup"] > 3.0  # a profitable full host offload, still not credited
    if name == "round4_r2_drop_warmup":
        assert d["raw_speedup"] > 1.5  # dropping the warmup would nearly double the score
    if expect_fail == "G-cpu":
        assert d["cpu_delta_s"] > d["cpu_slack_s"]  # the offload's CPU cost exceeds the slack


# =========================================================================== J10 (driver side)

def test_J10_long_line_windowed_around_first_difference_mid_row():
    """pathfinder prints one ~300 000-value row; the report must show the first differing
    COLUMN, not the first 300 characters (which are identical)."""
    from gpa_bench_driver.driver_src.driver_check import first_difference

    row = " ".join(str(i % 10) for i in range(300000))
    col = 250000                                     # far beyond the old 300-char clip
    assert row[col] != "9"
    bad = row[:col] + "9" + row[col + 1:]
    ref = "hdr\n" + row + "\n"
    out = first_difference(ref, "hdr\n" + bad + "\n")
    lines = out.splitlines()
    assert lines[0].startswith("first difference at line 2 ")
    exp_line = next(x for x in lines if "line 2 expected (first difference at column" in x)
    got_line = next(x for x in lines if "line 2 got      (first difference at column" in x)
    assert f"column {col + 1})" in exp_line and f"column {col + 1})" in got_line
    exp_win = exp_line.split("): ", 1)[1]
    got_win = got_line.split("): ", 1)[1]
    # a bounded window centred on the difference, with the differing value inside it
    assert exp_win.startswith("...") and exp_win.endswith("...")
    assert len(exp_win) <= 2 * 48 + 6 and len(got_win) <= 2 * 48 + 6
    assert exp_win != got_win and "9" in got_win
    # the old behaviour (first 300 chars, identical on both sides) is gone
    assert row[:300] not in out


def test_J10_short_lines_keep_the_plain_format_byte_identical():
    from gpa_bench_driver.driver_src.driver_check import first_difference

    out = first_difference("a 1\nb 2\nc 3\n", "a 1\nb 9\nc 3\n")
    assert out == ("first difference at line 2 (reference has 3 lines, output has 3 lines):\n"
                   "  line 2 expected: b 2\n  line 2 got:      b 9\n"
                   "  line 3 expected: c 3\n  line 3 got:      c 3")


def test_J10_identical_long_trailing_line_is_reported_as_identical():
    from gpa_bench_driver.driver_src.driver_check import first_difference

    long = "x" * 1000
    out = first_difference("a\n" + long + "\n", "b\n" + long + "\n")
    assert "line 2: expected and got are identical (1000 characters)" in out
