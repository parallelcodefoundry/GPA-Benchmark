"""GPA-G1 fix round 5 (GPA-G1-FIXES5.md), agent A's items.

K1  #line bypasses (digraph %:line, digraph GNU marker %: N, backslash-space-newline splice,
    comment-joined directive, flagged marker): layer (a) raw pass on phase-1..3-normalised text;
    layer (b) the compiler's own line markers / include echoes / dependency list and the whole
    program text of -fkeep-system-includes output. Every form is prepended to every round-2
    evasion fixture; each layer is checked alone (mutation per layer).
K3  VRAM reset binary in the trust zone (pre-agent sha256, sealed in-memory copy, build record).
K4  infra attribution (swap-side profiler-only failures retried once; the original's output
    failures are infra; a DriverInfraError keeps its class on either side).
Minors: (main) declarator, T0 record in the score, pathfinder's G-cpu floor.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from collections import Counter
from pathlib import Path

import pytest
import yaml

from gpa_bench_driver.driver_src import driver_gate
from gpa_bench_driver.driver_src.driver_gate import check_kernel_source
from gpa_bench_driver.driver_src.driver_utils import DriverInfraError

GPA_ROOT = Path(__file__).resolve().parent.parent
ROUND2 = GPA_ROOT / "tests" / "fixtures" / "round2"
SYS = "/opt/rocm-7.0.2/include/hip/hip_runtime.h"
needs_hipcc = pytest.mark.skipif(
    not (Path("/opt/rocm-7.0.2/bin/hipcc").exists() or shutil.which("hipcc")),
    reason="needs hipcc (the preprocessed pass)")


def _apps() -> dict[str, dict]:
    with (GPA_ROOT / "driver_apps.frontier.yaml").open() as f:
        return {a["name"]: a for a in yaml.safe_load(f)["apps"]}


def _gate_text(app: str, text: str, **kw):
    return check_kernel_source(_apps()[app], text, gpa_root=GPA_ROOT, **kw)


# ============================================================================= K1

BYPASS = {
    "digraph_line": f'%:line 1 "{SYS}"\n',
    "digraph_marker": '%: 1 "/usr/include/stdio.h"\n',
    "splice_space": f'#\\ \nline 1 "{SYS}"\n',
    "comment_joined": f'#/* a\n*/line 1 "{SYS}"\n',
    "digraph_flagged_push": '%: 1 "/usr/include/stdio.h" 1 3\n',
}
EVASIONS = sorted((ROUND2 / "evasion").glob("*/*.cu"))
# the fixtures only the preprocessed pass catches (the reviewer's bypass made them pass)
PP_ONLY = [ROUND2 / "evasion" / a / f"{f}.cu" for a, f in (
    ("b+tree", "att4_reserved_tokenpaste"), ("b+tree", "att5_trust_tamper"),
    ("b+tree", "n1_fopen64_tamper"), ("b+tree", "n2_profiler_environ"),
    ("b+tree", "n3_reserved_alias"), ("backprop", "a_paste"))]
_NEVER = re.compile(r"(?!x)x")


def _has_line_rule(res) -> bool:
    return any("#line" in v for v in res.violations)


def _raw_layer_off(monkeypatch):
    monkeypatch.setattr(driver_gate, "_LINE_DIR_RE", _NEVER)
    monkeypatch.setattr(driver_gate, "_GNU_MARKER_RE", _NEVER)


@pytest.mark.parametrize("form", sorted(BYPASS))
def test_K1_clang_accepts_every_bypass_form_as_a_line_directive(form, tmp_path):
    """Sanity: each form really is a #line/marker directive for clang (the bypass is real)."""
    if not Path("/opt/rocm-7.0.2/bin/hipcc").exists():
        pytest.skip("needs hipcc")
    src = tmp_path / "t.cu"
    src.write_text("int a;\n" + BYPASS[form] + "int b;\n")
    out = subprocess.run(["/opt/rocm-7.0.2/bin/hipcc", "-E", "--cuda-host-only", "-x", "hip",
                          "--offload-arch=gfx90a", str(src)], capture_output=True, text=True,
                         check=True, timeout=300).stdout
    tail = out[out.rindex('"' + str(src)):] if str(src) in out else out
    assert re.search(r'^# 1 "(/usr/include/stdio.h|' + re.escape(SYS) + ')"', tail, re.M), tail


@pytest.mark.parametrize("form", sorted(BYPASS))
@pytest.mark.parametrize("path", EVASIONS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_K1a_raw_layer_alone_rejects_every_form_on_every_evasion(form, path):
    res = _gate_text(path.parent.name, BYPASS[form] + path.read_text(), preprocess=False)
    assert not res.ok and _has_line_rule(res), res.message()


@needs_hipcc
@pytest.mark.parametrize("form", sorted(BYPASS))
@pytest.mark.parametrize("path", EVASIONS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_K1_every_form_prepended_to_every_evasion_fails_on_the_line_rule(form, path):
    res = _gate_text(path.parent.name, BYPASS[form] + path.read_text())
    assert not res.ok and res.preprocessed and _has_line_rule(res), res.message()


@needs_hipcc
@pytest.mark.parametrize("form", sorted(BYPASS))
@pytest.mark.parametrize("path", PP_ONLY, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_K1b_compiler_marker_layer_alone_rejects_every_form(form, path, monkeypatch):
    _raw_layer_off(monkeypatch)  # mutation: the raw layer misses the form
    res = _gate_text(path.parent.name, BYPASS[form] + path.read_text())
    assert not res.ok, res.message()
    assert any("#line-style file switch in the compiler's own line markers" in v
               for v in res.violations), res.message()


@needs_hipcc
@pytest.mark.parametrize("path", PP_ONLY, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_K1b_flagged_push_right_after_a_skipped_include_is_caught(path, monkeypatch):
    """An #include the compiler skips (already included) echoes like a real one; the spoofed
    push after it still lacks clang's renumber marker of the includer."""
    _raw_layer_off(monkeypatch)
    spoof = '#include <stdio.h>\n%: 1 "/usr/include/stdio.h" 1 3\n'
    res = _gate_text(path.parent.name, spoof + path.read_text())
    assert any("enter without an #include" in v for v in res.violations), res.message()


@needs_hipcc
@pytest.mark.parametrize("form", ["digraph_line", "digraph_marker", "splice_space",
                                  "comment_joined"])  # flag 3 hides text from this view; the
@pytest.mark.parametrize("path", PP_ONLY, ids=lambda p: f"{p.parent.name}/{p.name}")  # markers
def test_K1b_whole_program_text_alone_still_sees_the_evasion(form, path, monkeypatch):
    _raw_layer_off(monkeypatch)
    monkeypatch.setattr(driver_gate, "_marker_events", lambda *a, **k: Counter())
    res = _gate_text(path.parent.name, BYPASS[form] + path.read_text())
    assert not res.ok, res.message()
    assert any("found in the whole program text" in v for v in res.violations), res.message()


@needs_hipcc
def test_K1_mutation_without_both_layers_the_reviewer_bypass_passes(monkeypatch):
    """Reproduces the round-5 finding: raw layer bypassed + marker attribution trusted -> ok."""
    _raw_layer_off(monkeypatch)
    monkeypatch.setattr(driver_gate, "_marker_events", lambda *a, **k: Counter())
    monkeypatch.setattr(driver_gate, "_all_program_text", lambda text: "")
    path = ROUND2 / "evasion" / "b+tree" / "n2_profiler_environ.cu"
    res = _gate_text("b+tree", BYPASS["digraph_line"] + path.read_text())
    assert res.ok, res.message()


@needs_hipcc
def test_K1_system_header_pragma_is_a_line_rule_event():
    res = _gate_text("hotspot", (GPA_ROOT / _apps()["hotspot"]["kernel_file"]).read_text()
                     + "\n#pragma clang system_header\n")
    # hotspot.cu is the main file (pragma ignored there): must not be flagged
    assert res.ok, res.message()
    path = ROUND2 / "evasion" / "b+tree" / "n2_profiler_environ.cu"  # included kernel file
    res = _gate_text("b+tree", "#pragma clang system_header\n" + path.read_text())
    assert not res.ok and any("system-header marking of a non-system file" in v
                              for v in res.violations), res.message()


@pytest.mark.parametrize(("line", "fragment"), [
    ('#import "x.h"', "#import"), ("#include_next <stdio.h>", "#include_next"),
    ('#embed "/etc/hostname"', "#embed"), ('%:embed "/etc/hostname"', "#embed"),
    ('%:include "/etc/hostname"', "absolute or '..' include path"),
    ("%:define printf(...) 0", "#define of 'printf'"),
])
def test_K1_siblings_and_digraph_directives_raw(line, fragment):
    pristine = (GPA_ROOT / _apps()["hotspot"]["kernel_file"]).read_text()
    res = _gate_text("hotspot", pristine + "\n" + line + "\n", preprocess=False)
    assert not res.ok and fragment in res.message(), res.message()


@needs_hipcc
def test_K1b_digraph_absolute_include_and_embed_seen_by_the_compiler(tmp_path, monkeypatch):
    _raw_layer_off(monkeypatch)
    monkeypatch.setattr(driver_gate, "_OTHER_INCLUDE_RE", _NEVER)
    monkeypatch.setattr(driver_gate, "_INCLUDE_RE", _NEVER)
    secret = tmp_path / "secret.h"
    secret.write_text("int leaked;\n")
    pristine = (GPA_ROOT / _apps()["hotspot"]["kernel_file"]).read_text()
    res = _gate_text("hotspot", pristine + f'\n%:include "{secret}"\n')
    assert any("as the compiler read it" in v for v in res.violations), res.message()
    res = _gate_text("hotspot", pristine + '\n#pragma clang diagnostic ignored "-Wc23-extensions"\n'
                     f'static const char blob[] = {{\n%:embed "{secret}"\n}};\n')
    assert any("without entering it as an #include" in v for v in res.violations), res.message()


@needs_hipcc
def test_K1_honest_digraphs_and_splices_pass():
    pristine = (GPA_ROOT / _apps()["hotspot"]["kernel_file"]).read_text()
    honest = ("\nstatic const int k_tab<:3:> = <%1, 2, 3%>;\n"
              "#define K_SUM(a, b) \\\n    ((a) + (b))\n"
              "static int k_twice(int x) { return K_SUM(x, x); }\n")
    res = _gate_text("hotspot", pristine + honest)
    assert res.ok, res.message()


def test_K1_phase12_splices_backslash_whitespace_newline_and_keeps_strings():
    assert driver_gate._phase12("#\\ \t\nline 1") == "#line 1"
    assert driver_gate._phase12("a\\\r\nb") == "ab"
    assert driver_gate._phase12("??=line", trigraphs=False) == "??=line"
    assert driver_gate._phase12("??=line", trigraphs=True) == "#line"
    code, lits, raw = driver_gate._lex('x = "%:line"; <:: y; a<:1:>;')
    assert lits == ["%:line"] and "<::" in raw and "a[1]" in raw


def test_K1_trigraphs_follow_the_app_flags():
    assert not driver_gate._trigraphs_enabled({"gate_preprocess": {"flags": "-x hip -std=c++17"}})
    assert driver_gate._trigraphs_enabled({"gate_preprocess": {"flags": "-x hip -std=c++14"}})
    assert driver_gate._trigraphs_enabled({"gate_preprocess": {"flags": "-x hip -trigraphs"}})


# ----------------------------------------------------------------------------- (main) minor

MAIN_ENVP = "int (main)(int argc, char** argv, char** envp)"


def _hotspot_with_paren_main() -> str:
    pristine = (GPA_ROOT / _apps()["hotspot"]["kernel_file"]).read_text()
    new = re.sub(r"\bint\s+main\s*\(\s*int\s+argc\s*,\s*char\s*\*\*\s*argv\s*\)", MAIN_ENVP,
                 pristine, count=1)
    assert new != pristine
    return new


def test_main_parenthesized_declarator_raw():
    res = _gate_text("hotspot", _hotspot_with_paren_main(), preprocess=False)
    assert not res.ok and "main() with more parameters" in res.message(), res.message()


@needs_hipcc
def test_main_parenthesized_declarator_preprocessed(monkeypatch):
    monkeypatch.setattr(driver_gate, "_MAIN_DEF_RE", _NEVER)  # mutation: raw rule misses it
    res = _gate_text("hotspot", _hotspot_with_paren_main())
    assert not res.ok and "after preprocessing" in res.message(), res.message()


# ============================================================================= K3

def _tool_root(tmp_path: Path, body: str = "#!/bin/bash\necho reset-ok\n", record: bool = True):
    root = tmp_path / "gpa"
    (root / "frontier_tools").mkdir(parents=True)
    tool = root / "frontier_tools" / "vram_reset"
    tool.write_text(body)
    tool.chmod(0o755)
    if record:
        (root / "frontier_tools" / "vram_reset.sha256").write_text(
            hashlib.sha256(tool.read_bytes()).hexdigest() + "  vram_reset\n")
    return root, tool


def test_K3_snapshot_and_verified_run(tmp_path):
    from gpa_bench_driver.driver_src.driver_t0 import VramResetTool, snapshot

    root, tool = _tool_root(tmp_path)
    snap = snapshot(root)
    assert snap["sha256"] == snap["record_sha256"] == hashlib.sha256(tool.read_bytes()).hexdigest()
    with VramResetTool(snap["sha256"], root) as t:
        assert t.sealed
        assert t.run(dict(os.environ), tmp_path) >= 0


def test_K3_sealed_copy_ignores_a_later_change_on_disk(tmp_path):
    from gpa_bench_driver.driver_src.driver_t0 import VramResetTool

    root, tool = _tool_root(tmp_path)
    with VramResetTool(None, root) as t:
        tool.write_text("#!/bin/bash\nexit 7\n")  # replaced after the check
        t.run(dict(os.environ), tmp_path)  # still the verified bytes: exit 0


def test_K3_reviewer_tamper_binary_and_record_rewritten_is_caught_by_the_snapshot(tmp_path):
    from gpa_bench_driver.driver_src.driver_t0 import VramResetTool, snapshot

    root, tool = _tool_root(tmp_path)
    pre_agent = snapshot(root)["sha256"]
    tool.write_text("#!/bin/bash\n( sleep 0.2 ) &\nexit 0\n")  # the agent's script ...
    (root / "frontier_tools" / "vram_reset.sha256").write_text(  # ... with a matching record
        hashlib.sha256(tool.read_bytes()).hexdigest() + "  vram_reset\n")
    VramResetTool(None, root).close()  # the build record alone is fooled (why K3 exists)
    with pytest.raises(DriverInfraError, match="pre-agent sha256"):
        VramResetTool(pre_agent, root)


def test_K3_missing_binary_or_record(tmp_path):
    from gpa_bench_driver.driver_src.driver_t0 import VramResetTool, snapshot

    root, tool = _tool_root(tmp_path, record=False)
    with pytest.raises(DriverInfraError, match="build record"):
        VramResetTool(None, root)
    with pytest.raises(DriverInfraError, match="missing"):
        snapshot(root)
    tool.unlink()
    with pytest.raises(DriverInfraError, match="missing"):
        VramResetTool("0" * 64, root)


def _prepare_root(tmp_path: Path) -> Path:
    root = tmp_path / "prep"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(GPA_ROOT / "scripts" / "frontier_prepare.sh", root / "scripts")
    (root / "data.txt").write_text("x\n")
    (root / "frontier_refs.md5").write_text(
        hashlib.md5(b"x\n").hexdigest() + "  data.txt\n")  # noqa: S324
    return root


def _verify(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(root / "scripts" / "frontier_prepare.sh"), "verify"],
                          capture_output=True, text=True, check=False, timeout=120,
                          env={**os.environ, "ROCM_PATH": "/opt/rocm-7.0.2"})


def test_K3_prepare_verify_checks_the_binary_and_its_record(tmp_path):
    root = _prepare_root(tmp_path)
    out = _verify(root)
    assert out.returncode != 0 and "vram_reset is missing" in out.stderr
    (root / "frontier_tools").mkdir()
    tool = root / "frontier_tools" / "vram_reset"
    tool.write_text("#!/bin/bash\n")
    tool.chmod(0o755)
    (root / "frontier_tools" / "vram_reset.sha256").write_text(
        hashlib.sha256(tool.read_bytes()).hexdigest() + "  vram_reset\n")
    assert _verify(root).returncode == 0
    tool.write_text("#!/bin/bash\nexit 0\n")
    out = _verify(root)
    assert out.returncode != 0 and "does not match its build record" in out.stderr


# ============================================================================= K4 (+K3) toy driver

_TOY_RUN = textwrap.dedent('''\
    #!/bin/bash
    . ./kernel.cu
    n=$(cat .runs 2>/dev/null || echo 0); n=$((n+1)); echo $n > .runs
    if [ "$MODE" = flaky_base ] && [ $n -ge 3 ]; then echo "result 41" > output.txt
    elif [ "$MODE" = bad_base ]; then echo "result 40" > output.txt
    else echo "result 42" > output.txt; fi
    echo "toy_kernel(int),1000" >> kernels.csv
    ''')
_TOY_MAKE = "toy: kernel.cu run.sh\n\tcp run.sh toy && chmod +x toy\nclean:\n\trm -f toy\n"
_FAKE_ROCPROF = textwrap.dedent('''\
    #!/bin/bash
    while [ "$1" != "--" ]; do [ "$1" = -d ] && out=$2; shift; done; shift
    # K4: fail the first $FAIL_SWAP_N profiler starts in the SWAP copy (never starts the app)
    marker="$(dirname "$out")/.prof_fail"
    n=$(cat "$marker" 2>/dev/null || echo 0)
    case "$out" in */swap0/*)
      if [ "$n" -lt "${FAIL_SWAP_N:-0}" ]; then echo $((n+1)) > "$marker"; echo "tool init failed" >&2; exit 1; fi ;;
    esac
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
    root = Path(sys.argv[1]); opts = json.loads(sys.argv[2]); sys.path.insert(0, str(root))
    import gpa_bench_driver.gpa_bench_driver as G
    from gpa_bench_driver.gpa_bench_driver import run_driver
    from gpa_bench_driver.driver_src.driver_models import DriverConfig
    from gpa_bench_driver.driver_src.driver_utils import DriverInfraError
    if opts.get("swap_infra"):
        orig = G.profile_once
        def patched(app, runner, path, *a, **k):
            if "/swap0/" in str(path):
                raise DriverInfraError("worker left no result (toy)")
            return orig(app, runner, path, *a, **k)
        G.profile_once = patched
    cfg = DriverConfig(app="toy", gpu_backend="hip", rocm_path=root / "rocm", offload_arch="gfx90a",
                       config=root / "apps.yaml", nsys=True, pairs=2,
                       swaps_override={Path("kernel.cu"): "// kernel.cu\\nMODE=good"},
                       temp_dir=root / "tmp", kernel_gate=False,
                       vram_reset_sha256=opts.get("sha"))
    try:
        _, _, long = run_driver(cfg)
    except DriverInfraError as e:
        print(json.dumps({"infra": str(e)[:300], "cls": type(e).__name__})); sys.exit(0)
    b, s = long["toy"]
    print(json.dumps({"validate": s.validate, "run": s.run, "vo": s.validation_output,
                      "n": len(s.nsys_data or [])}))
    ''')


@pytest.fixture
def toy5(tmp_path):
    root = tmp_path / "gpa"
    shutil.copytree(GPA_ROOT / "gpa_bench_driver", root / "gpa_bench_driver",
                    ignore=shutil.ignore_patterns("__pycache__"))
    (root / "frontier_tools").mkdir()
    reset = root / "frontier_tools" / "vram_reset"
    reset.write_text("#!/bin/bash\necho 'vram_reset allocs=1 GiB=0.00'\n")
    reset.chmod(0o755)
    (root / "frontier_tools" / "vram_reset.sha256").write_text(
        hashlib.sha256(reset.read_bytes()).hexdigest() + "  vram_reset\n")
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


def _drive5(root: Path, env: dict | None = None, **opts) -> dict:
    proc = subprocess.run([sys.executable, "-c", _DRIVE, str(root), json.dumps(opts)],
                          capture_output=True, text=True, check=False, timeout=600,
                          env={**os.environ, **(env or {})})
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_K4_swap_profiler_only_failure_once_is_retried_as_infra(toy5):
    out = _drive5(toy5, env={"FAIL_SWAP_N": "1"})
    assert out.get("validate") is True and out["n"] == 2, out


def test_K4_swap_profiler_only_failure_twice_is_the_programs_failure(toy5):
    out = _drive5(toy5, env={"FAIL_SWAP_N": "2"})
    assert out.get("validate") is False and out.get("run") is False, out
    assert "fails under the profiler twice although it runs without it" in out["vo"], out


def test_K4_swap_side_driver_infra_error_stays_infra(toy5):
    out = _drive5(toy5, swap_infra=True)
    assert "infra" in out and "worker left no result" in out["infra"], out


def test_K4_original_timed_run_output_failure_is_infra(toy5):
    (toy5 / "rodinia" / "toy-hip" / "kernel.cu").write_text("MODE=flaky_base\n")
    out = _drive5(toy5)
    assert out.get("cls") == "BaselineInfraError", out
    assert "baseline timed run" in out["infra"], out


def test_K4_original_first_run_output_failure_is_infra(toy5):
    (toy5 / "rodinia" / "toy-hip" / "kernel.cu").write_text("MODE=bad_base\n")
    out = _drive5(toy5)
    assert out.get("cls") == "BaselineInfraError", out


def test_K3_driver_uses_the_pre_agent_sha256(toy5):
    good = hashlib.sha256((toy5 / "frontier_tools" / "vram_reset").read_bytes()).hexdigest()
    assert _drive5(toy5, sha=good).get("validate") is True
    out = _drive5(toy5, sha="0" * 64)
    assert "infra" in out and "pre-agent sha256" in out["infra"], out


# ============================================================================= minors

def test_t0_record_in_the_score():
    from gpa_bench_driver.driver_src.driver_rocprof import score_frontier

    def s(ns, reset):
        return {"target_ns": ns, "target_dispatches": 1, "kernels": {"k(int)": ns},
                "wall_s": 1.0, "cpu_s": 1.0, "vram_reset_s": reset}
    r = score_frontier([s(100, 0.8), s(100, 0.9)], [s(90, 0.7), s(90, None)], r"^k\(")
    assert r["t0"]["vram_reset_s"] == {"baseline": [0.8, 0.9], "optimized": [0.7, None]}
    assert r["t0"]["n_resets"] == 3 and r["t0"]["all_samples_reset"] is False
    assert r["t0"]["max_s"] == 0.9


def test_pathfinder_cpu_floor_is_025_and_reaches_the_score():
    from gpa_bench_driver.driver_src.driver_j0 import app_entry, score_kwargs
    from gpa_bench_driver.driver_src.driver_rocprof import cpu_guard

    kw = score_kwargs(app_entry("pathfinder", GPA_ROOT))
    assert kw["cpu_floor_s"] == 0.25
    assert score_kwargs(app_entry("hotspot", GPA_ROOT))["cpu_floor_s"] == 0.10
    g = cpu_guard([6.2] * 10, [6.2977] * 10, kw["cpu_sigma_s"], cpu_floor_s=kw["cpu_floor_s"])
    assert g["ok"] and g["slack_s"] == 0.25  # the reference opt's +0.098 s edge
