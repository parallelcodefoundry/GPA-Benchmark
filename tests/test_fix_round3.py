"""GPA-G1 fix round 3 (GPA-G1-FIXES3.md): H1 G-cpu in the profiled runs, H2 gate additions
(asm, #line, includes, main/env introspection), H3 error classification, H4 fail-closed,
H5 anonymous-namespace regex, H6 final_samples, H7 bfs graph pool, H8 stdout docstring.

Gate unit cases and the round-3 red-team fixtures (tests/fixtures/round3 + fixtures/offload) whose
compute results are recorded in .planning/frontier/gpa-g1/A.md.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import pytest
import yaml

from gpa_bench_driver.driver_src import driver_variants
from gpa_bench_driver.driver_src.driver_gate import check_kernel_file, check_kernel_source
from gpa_bench_driver.driver_src.driver_rocprof import (
    ScoringError,
    cpu_guard,
    pooled_cpu_ok,
    score_frontier,
)

GPA_ROOT = Path(__file__).resolve().parent.parent
FIX = Path(__file__).resolve().parent / "fixtures"
HIPCC = Path(os.environ.get("ROCM_PATH", "/opt/rocm-7.0.2")) / "bin" / "hipcc"
needs_hipcc = pytest.mark.skipif(not HIPCC.exists() and shutil.which("hipcc") is None,
                                 reason="the preprocessed gate needs hipcc (ROCm)")


def _apps() -> dict[str, dict]:
    with (GPA_ROOT / "driver_apps.frontier.yaml").open() as f:
        return {a["name"]: a for a in yaml.safe_load(f)["apps"]}


def _s(target, kernels, *, disp=1, wall=1.0, cpu=1.0):
    return {"exec_time": target, "target_ns": target, "target_dispatches": disp,
            "kernels": kernels, "wall_s": wall, "cpu_s": cpu}


# =========================================================================== H5 regex

REAL = {
    "xsbench": "(anonymous namespace)::xs_lookup_kernel(Inputs, SimulationData)",
    "streamcluster": "void (anonymous namespace)::kernel_compute_cost<128>(int, int, long, "
                     "Point*, int, int, float*, float*, int*, bool*)",
    "b+tree": "(anonymous namespace)::findRangeK",
    "backprop": "ns::bpnn_layerforward_CUDA(float*, float*, float*, float*, int, int)",
}


@pytest.mark.parametrize(("app", "name"), sorted(REAL.items()))
def test_H5_score_regex_matches_anonymous_and_nested_namespaces(app, name):
    assert re.search(_apps()[app]["score_regex"], name), (app, name)


def test_H5_r2_counts_anonymous_namespace_instantiations():
    from gpa_bench_driver.driver_src.driver_rocprof import summarize_kernel_trace

    rx = _apps()["xsbench"]["score_regex"]
    rows = [("(anonymous namespace)::xs_lookup_kernel(Inputs, SimulationData)", 5),
            ("void (anonymous namespace)::xs_lookup_kernel<0>(Inputs, SimulationData)", 7)]
    s = summarize_kernel_trace(rows, rx)
    assert s["target_dispatches"] == 2 and s["target_ns"] == 12


# =========================================================================== H1 / H4 G-cpu

def test_H1_gcpu_reads_cpu_from_profiled_samples():
    base = [_s(100, {"k": 100}, cpu=1.0), _s(100, {"k": 100}, cpu=1.02)]
    off = [_s(30, {"k": 30}, cpu=1.6), _s(30, {"k": 30}, cpu=1.62)]  # host offload in the SAME runs
    r = score_frontier(base, off, r"^k$", cpu_sigma_s=0.02)
    assert not r["ok"] and [f["code"] for f in r["failures"]] == ["G-cpu"]
    assert r["cpu"]["checked"] and r["cpu"]["delta_s"] == pytest.approx(0.6)


def test_H1_slack_floor_is_100ms_and_marginal_flag():
    g = cpu_guard([1.0], [1.09], 0.001)
    assert g["ok"] and g["slack_s"] == pytest.approx(0.10)
    g = cpu_guard([1.0], [1.15], 0.001)  # over by 0.05, < 2x slack -> marginal
    assert not g["ok"] and g["marginal"]
    g = cpu_guard([1.0], [1.5], 0.001)  # far over -> not marginal
    assert not g["ok"] and not g["marginal"]
    assert pooled_cpu_ok([1.0, 1.0], [1.02, 1.03], 0.02)["ok"]


def test_H4_gcpu_fails_closed_on_missing_cpu():
    base = [_s(100, {"k": 100}, cpu=1.0)]
    bad = [{**_s(50, {"k": 50}), "cpu_s": None}]
    with pytest.raises(ScoringError, match="no cpu_s"):
        score_frontier(base, bad, r"^k$", cpu_sigma_s=0.02)


def test_H4_no_sigma_means_not_checked():
    base = [_s(100, {"k": 100})]
    r = score_frontier(base, [_s(50, {"k": 50})], r"^k$")
    assert r["cpu"] == {"checked": False} and r["ok"]


def test_every_app_sigma_and_j0_pairs_present():
    for name, app in _apps().items():
        assert isinstance(app.get("cpu_sigma_s"), (int, float)) and app["cpu_sigma_s"] > 0, name
        # J0 (M.md T1): ABBA pairs replace final_samples; both even
        assert app.get("test_pairs") == 6 and app.get("final_pairs") == 10, name
        assert "final_samples" not in app, name


# =========================================================================== H2 gate

def _gate(app: str, extra: str, **kw):
    entry = _apps()[app]
    pristine = (GPA_ROOT / entry["kernel_file"]).read_text()
    return check_kernel_source(entry, pristine + "\n" + extra, gpa_root=GPA_ROOT, **kw)


@pytest.mark.parametrize("extra", [
    'static long e1(){ long r; asm volatile("syscall" : "=a"(r) : "a"(39) : "memory"); return r; }',
    'extern "C" char *e2(const char *) asm("getenv");',
    '__asm__(".globl x");',
    'static char *e4(){ return __environ[0]; }',
    'static char *e5(){ return _environ[0]; }',
    'static char *e6(){ return secure_getenv("HOME"); }',
    '#include <link.h>\nstatic int e7(){ return dl_iterate_phdr(0, 0); }',
    '#include <sys/auxv.h>\nstatic long e8(){ return getauxval(3); }',
    'static long e9(){ return prctl(1); }',
    '#line 1 "hip/hip_runtime.h"',
    '# 1 "hipcc_internal.h"',
    '#include "/etc/passwd"',
    '#include "../secret.h"',
])
def test_H2_raw_pass_rejects(extra):
    # raw pass alone (no hipcc) must reject each construct
    assert not _gate("bfs", extra, preprocess=False).ok, extra


def test_H2_rejects_main_with_more_params():
    # nw's pristine main has 2 params; a 3-param main (argv+envp) is rejected
    entry = _apps()["nw"]
    pristine = (GPA_ROOT / entry["kernel_file"]).read_text()
    cand = pristine + "\nint main(int argc, char **argv, char **envp){ (void)envp; return 0; }\n"
    assert not check_kernel_source(entry, cand, gpa_root=GPA_ROOT, preprocess=False).ok


@needs_hipcc
@pytest.mark.parametrize(("name", "fixture", "fragment"), [
    ("B5_asm_syscall_gate_evasion", "streamcluster", "asm"),
    ("B6_fnptr_alias_gate_evasion", "streamcluster", "asm"),
])
def test_H2_round3_gate_evasions_fail(name, fixture, fragment):
    r = check_kernel_file(_apps()[fixture], FIX / "round3" / fixture / f"{name}.cu", gpa_root=GPA_ROOT)
    assert not r.ok and fragment in r.message().lower()


@needs_hipcc
# FP1_blocking_sync_only is now a J4 rejection (blocking-sync); see test_fix_round4.
@pytest.mark.parametrize("name", ["FP2_restrict_regcache", "FP3_blocksize_hipcheck"])
def test_H2_round3_legit_probes_pass(name):
    r = check_kernel_file(_apps()["streamcluster"], FIX / "round3" / "streamcluster" / f"{name}.cu",
                          gpa_root=GPA_ROOT)
    assert r.ok, r.message()


@needs_hipcc
@pytest.mark.parametrize("extra", [
    "#include <hip/hip_cooperative_groups.h>",
    "#define HIP_CHECK(x) (x)",
    "__global__ void __launch_bounds__(256) k2(int *a){ a[0]=0; }",
    "namespace { __device__ int anon_helper(int x){ return x + 1; } }",
    "template <int B> __device__ int t(int x){ return x*B; }",
])
def test_H2_does_not_reject_legit_constructs(extra):
    r = _gate("hotspot", extra)
    assert r.ok, r.message()


@needs_hipcc
def test_H2_line_spoof_cannot_hide_kernel_tokens():
    # even if the raw pass were bypassed, a #line spoof followed by forbidden code is attributed
    # to the kernel file (fail closed) and caught by the preprocessed pass
    entry = _apps()["hotspot"]
    pristine = (GPA_ROOT / entry["kernel_file"]).read_text()
    cand = pristine + '\nstatic int spoof(){ return (int) fopen64("x","r"); }\n'
    assert not check_kernel_source(entry, cand, gpa_root=GPA_ROOT).ok


# =========================================================================== H7 bfs pool

def test_H7_bfs_pool_uses_pregenerated_graph(tmp_path):
    v = driver_variants.draw_variant("bfs", 2, workdir=tmp_path, gpa_root=GPA_ROOT, pool=True)
    assert v.params["pool"] is True and not v.generated_files  # nothing generated
    assert "frontier_refs/bfs_variants/graph8M_v" in v.run_command
    seed = v.params["graph_seed"]
    assert seed in driver_variants.BFS_POOL_SEEDS
    # deterministic by seed within the pool
    assert driver_variants.draw_variant("bfs", 2, workdir=tmp_path, gpa_root=GPA_ROOT,
                                        pool=True).run_command == v.run_command


def test_H7_bfs_manifest_lists_the_pool():
    manifest = (GPA_ROOT / "frontier_refs.md5").read_text()
    for seed in driver_variants.BFS_POOL_SEEDS:
        assert f"frontier_refs/bfs_variants/graph8M_v{seed}.txt" in manifest, seed


# =========================================================================== H8 docstring

def test_H8_stdout_cap_docstring_matches_r4():
    from gpa_bench_driver.driver_src import driver_utils

    doc = driver_utils.SubprocessRunner.run.__doc__
    assert "keep and validate the full stdout" in doc
    assert "not validated" not in doc.split("stdout_cap_bytes")[1].split("measure_cpu")[0]
