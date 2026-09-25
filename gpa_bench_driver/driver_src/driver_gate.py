"""Static kernel-file gate for the Frontier GPA apps (GPA-G1 fix round 1, R6).

An agent may only change the app's kernel file, but on several apps that file is ``#include``d
into host code, so macros, file access, constructors or reserved names in it reach the whole
program. The gate compares a candidate kernel file with the PRISTINE one and rejects NEW
occurrences (count-based: a construct already present in the pristine file stays allowed as
often as it occurs there) of:

a) ``#define`` / ``#undef`` / ``push_macro`` / ``pop_macro`` of an identifier that occurs in the
   app's other source files, or of a HIP/C/C++ API name (``hip*``, ``cuda*``, ``__hip*``, the
   printf family, allocation, exit/abort, ``main``, ``std``, common libc names);
b) file, process, environment, thread and dynamic-loading access (fopen, ifstream, open(,
   system(, getenv, environ, dlopen, std::thread, atexit, constructor/destructor attributes, ...)
   and string literals naming ``/proc``, ``/tmp``, ``/dev/shm``, ``ROCP``, ``ROCPROF``,
   ``HSA_TOOLS``, ``ref-``, ``ref_``;
c) definitions of functions or kernels whose names start with ``__`` (reserved names such as
   ``__amd_rocclr_*`` or ``__hip*``).

Comments are ignored (they are stripped before matching); string literals are checked by rule b
only. Used by the driver (every hip swap) and by APPEB's gpa_test and runner.

Fix round 2 (F3): after this raw-text pass (clear messages), the gate runs the app's own
preprocessor (``hipcc -E --cuda-host-only`` with the yaml ``gate_preprocess`` translation unit
and flags) on the pristine and the candidate kernel file and applies the rules to the EXPANDED
tokens that the line markers attribute to the kernel file: forbidden call names with any
qualification (``::open(``, ``fopen64``), definitions of ``__`` names built by token pasting or
aliases, and new macros named like an identifier of the other sources or a name declared or
defined in the included headers (``#define HIP_CHECK`` etc. stay allowed).

Documented residual (F9, not blocked): a namespace-scope static object whose constructor runs
host code before main. Everything such a constructor could use to cheat (file, environment,
process, thread access, macro hooks, reserved names) is rejected by the rules above, and the
program's inputs are unknown at static-initialization time.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from gpa_bench_driver.driver_src.driver_utils import DriverInfraError, detect_rocm_path, head_tail

SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hh", ".hpp", ".hip"}

# a) API names whose (re)definition as a macro is refused
_API_NAME_RE = re.compile(r"^(hip|cuda|__hip|__cuda|hsa_)")
_API_NAMES = frozenset({
    # printf family / stdio
    "printf", "fprintf", "sprintf", "snprintf", "vprintf", "vfprintf", "vsprintf", "vsnprintf",
    "dprintf", "puts", "fputs", "putchar", "fputc", "putc", "perror", "fflush", "fwrite", "fread",
    "fopen", "fclose", "freopen", "fscanf", "scanf", "sscanf", "fgets", "gets", "getchar",
    "stdout", "stderr", "stdin", "FILE",
    # allocation
    "malloc", "calloc", "realloc", "free", "aligned_alloc", "posix_memalign", "memalign",
    "new", "delete",
    # process control
    "exit", "_exit", "_Exit", "abort", "quick_exit", "atexit", "at_quick_exit", "main",
    "assert", "signal", "raise",
    # memory / string / time
    "memcpy", "memmove", "memset", "memcmp", "strcpy", "strncpy", "strcmp", "strncmp", "strlen",
    "strcat", "strstr", "time", "clock", "gettimeofday", "clock_gettime",
    # C++
    "std", "operator",
})

# b) forbidden tokens (regex on comment-free, string-free code) -> label
_FORBIDDEN_CODE = [
    (r"\bfopen\b", "fopen"),
    (r"\bfreopen\b", "freopen"),
    (r"\bfdopen\b", "fdopen"),
    (r"\b[io]?fstream\b", "file streams (fstream)"),
    (r"\bifstream\b", "ifstream"),
    (r"\bofstream\b", "ofstream"),
    (r"(?<![\w.>:])open\s*\(", "open("),
    (r"\bopenat\b", "openat"),
    (r"\bcreat\s*\(", "creat("),
    (r"(?<![\w.>:])read\s*\(", "read("),
    (r"(?<![\w.>:])write\s*\(", "write("),
    (r"\bpread(64)?\b", "pread"),
    (r"\bpwrite(64)?\b", "pwrite"),
    (r"\bmmap(64)?\b", "mmap"),
    (r"\bunlink(at)?\b", "unlink"),
    (r"(?<![\w.>:])remove\s*\(", "remove("),
    (r"(?<![\w.>:])rename(at)?\s*\(", "rename("),
    (r"(?<![\w.>:])system\s*\(", "system("),
    (r"\bpopen\b", "popen"),
    (r"\bexec(l|lp|le|v|vp|ve|vpe)\b", "exec*"),
    (r"(?<![\w.>:])v?fork\s*\(", "fork("),
    (r"\bposix_spawnp?\b", "posix_spawn"),
    (r"\bgetenv\b", "getenv"),
    (r"\bsecure_getenv\b", "secure_getenv"),
    (r"\bsetenv\b", "setenv"),
    (r"\bunsetenv\b", "unsetenv"),
    (r"\bputenv\b", "putenv"),
    (r"\benviron\b", "environ"),
    (r"\bdlopen\b", "dlopen"),
    (r"\bdlsym\b", "dlsym"),
    (r"\bstd\s*::\s*filesystem\b", "std::filesystem"),
    (r"\bstd\s*::\s*(thread|jthread|async)\b", "std::thread"),
    (r"\bpthread_create\b", "pthread_create"),
    (r"\batexit\b", "atexit"),
    (r"\bat_quick_exit\b", "at_quick_exit"),
    (r"__attribute__\s*\(\(\s*[^)]*\b(constructor|destructor)\b", "__attribute__((constructor/destructor))"),
    (r"\[\[\s*gnu\s*::\s*(constructor|destructor)", "[[gnu::constructor/destructor]]"),
    (r"\bsyscall\b", "syscall"),
    (r"\b__syscall\b", "__syscall"),
    (r"\bptrace\b", "ptrace"),
    (r"\b__secure_getenv\b", "__secure_getenv"),
    (r"\b__environ\b", "__environ"),
    (r"\b_environ\b", "_environ"),
    (r"\bdl_iterate_phdr\b", "dl_iterate_phdr"),
    (r"\bgetauxval\b", "getauxval"),
    (r"\bdladdr\b", "dladdr"),
    (r"\bdlinfo\b", "dlinfo"),
    (r"\bprctl\b", "prctl"),
    # inline assembly (incl. asm labels asm("name")) reaches libc symbols the gate cannot see
    (r"\b(?:asm|__asm__|__asm)\b", "inline asm"),
    # J4: blocking-sync scheduling only trades GPU wait for lower CPU time (a G-cpu lever) and
    # hangs under the Frontier profiler; it is never needed for a GPU kernel optimization.
    (r"\bhipDeviceScheduleBlockingSync\b", "blocking-sync scheduling"),
    (r"\bhipEventBlockingSync\b", "blocking-sync scheduling"),
    (r"\bcudaDeviceScheduleBlockingSync\b", "blocking-sync scheduling"),
    (r"\bcudaEventBlockingSync\b", "blocking-sync scheduling"),
]
_FORBIDDEN_CODE_RE = [(re.compile(rx), label) for rx, label in _FORBIDDEN_CODE]

# b) string literals naming these are refused
_FORBIDDEN_LITERAL_PARTS = ("/proc", "/tmp", "/dev/shm", "ROCP", "ROCPROF", "HSA_TOOLS",
                            "ref-", "ref_")

_IDENT_RE = re.compile(r"\b[A-Za-z_]\w*\b")
_DIRECTIVE_RE = re.compile(r"^[ \t]*#[ \t]*(define|undef)[ \t]+([A-Za-z_]\w*)", re.MULTILINE)
_PRAGMA_MACRO_RE = re.compile(r"\b(push_macro|pop_macro)\s*\(\s*\\?\"([A-Za-z_]\w*)\\?\"")
_RESERVED_DEF_RE = re.compile(r"\b(__[A-Za-z_]\w*)\s*\(")
_NOT_A_DEFINITION_PREFIX = re.compile(r"(\bif|\bwhile|\bfor|\bswitch|\breturn|[=,(!&|?:+\-*/<>])\s*$")
# H2 directives
# any #line (even macro-computed: "#line LN ...") and any GNU line marker "# 1 ..." (J1a)
_LINE_DIR_RE = re.compile(r"^[ \t]*#[ \t]*line\b", re.MULTILINE)
_GNU_MARKER_RE = re.compile(r"^[ \t]*#[ \t]*\d", re.MULTILINE)
# the whole #include operand (a literal <...>/"..." or anything else, e.g. a macro)
_INCLUDE_RE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*(\S.*?)[ \t]*$', re.MULTILINE)
# a C/C++ main definition: main ( <params> ) {  -> capture the parameter list
_MAIN_DEF_RE = re.compile(r"\bmain\s*\(([^)]*)\)\s*\{")


@dataclass
class GateResult:
    """Outcome of the kernel-file gate: ok, and one sentence per rejected construct."""

    ok: bool
    violations: list[str] = field(default_factory=list)
    kernel_file: str = ""
    preprocessed: bool = False
    notes: list[str] = field(default_factory=list)

    def message(self) -> str:
        """Agent-facing explanation ('' when the file passed)."""
        if self.ok:
            return ""
        head = (f"KERNEL GATE FAILED: {len(self.violations)} forbidden construct(s) added to "
                f"{self.kernel_file or 'the kernel file'}:")
        return "\n".join([head, *(f" - {v}" for v in self.violations)])


def split_source(text: str) -> tuple[str, list[str]]:
    """Strip comments; return (code with string/char literals blanked, the string literals).

    Newlines are kept so preprocessor directives stay on their own lines; a backslash-newline
    continuation is joined.
    """
    code, literals, _ = _lex(text)
    return code, literals


def _lex(text: str) -> tuple[str, list[str], str]:
    """(code with literals blanked, string literals, code with comments stripped only)."""
    text = text.replace("\\\r\n", "").replace("\\\n", "")
    code: list[str] = []
    raw: list[str] = []
    literals: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if c == "/" and nxt == "/":
            j = text.find("\n", i)
            i = n if j < 0 else j
            continue
        if c == "/" and nxt == "*":
            j = text.find("*/", i + 2)
            end = n if j < 0 else j + 2
            blank = " " + "\n" * text.count("\n", i, end)
            code.append(blank)
            raw.append(blank)
            i = end
            continue
        if c in "\"'":
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            raw.append(text[i:j + 1])
            if c == '"':
                literals.append(text[i + 1:j])
                code.append('""')
            else:
                code.append("' '")
            i = j + 1
            continue
        code.append(c)
        raw.append(c)
        i += 1
    return "".join(code), literals, "".join(raw)


def _balanced_close(code: str, open_idx: int) -> int:
    depth = 0
    for k in range(open_idx, len(code)):
        if code[k] == "(":
            depth += 1
        elif code[k] == ")":
            depth -= 1
            if depth == 0:
                return k
    return -1


def _reserved_definitions(code: str) -> list[str]:
    names = []
    for m in _RESERVED_DEF_RE.finditer(code):
        if _NOT_A_DEFINITION_PREFIX.search(code[max(0, m.start() - 40):m.start()]):
            continue  # a call inside an expression or condition
        close = _balanced_close(code, m.end() - 1)
        if close < 0:
            continue
        rest = code[close + 1:close + 200]
        if re.match(r"\s*(const\b\s*)?(noexcept\b\s*)?(->\s*[\w:<>*& ]+)?\{", rest):
            names.append(m.group(1))
    return names


def _macro_targets(code: str, raw: str) -> list[tuple[str, str]]:
    found = [(m.group(1), m.group(2)) for m in _DIRECTIVE_RE.finditer(code)]
    found += [(m.group(1), m.group(2)) for m in _PRAGMA_MACRO_RE.finditer(raw)]
    return found


def _is_api_name(name: str) -> bool:
    return bool(_API_NAME_RE.match(name)) or name in _API_NAMES


def _param_count(param_text: str) -> int:
    inner = param_text.strip()
    if not inner or inner == "void":
        return 0
    depth = 0
    n = 1
    for ch in inner:
        if ch in "(<[":
            depth += 1
        elif ch in ")>]":
            depth -= 1
        elif ch == "," and depth == 0:
            n += 1
    return n


def _features(text: str, other_idents: frozenset[str],
              pristine_includes: frozenset[str] | None = None,
              pristine_main_params: int | None = None) -> Counter:
    code, literals, raw = _lex(text)
    feats: Counter = Counter()
    for _ in _LINE_DIR_RE.finditer(raw):
        feats[("line_directive",)] += 1
    for _ in _GNU_MARKER_RE.finditer(raw):
        feats[("gnu_marker",)] += 1
    if pristine_includes is not None:
        # J1a: the operand must be a literal <...> or "..."; a macro (#include MACRO) is rejected.
        # Absolute paths and '..' are rejected for both bracket forms. A new plain <system>/app
        # header is allowed here (the preprocessed pass attributes its tokens); #ifdef-dead ones
        # are harmless.
        for m in _INCLUDE_RE.finditer(raw):
            operand = m.group(1)
            lit = re.match(r'^([<"])([^>"]*)[>"]\s*$', operand)
            if not lit:
                feats[("macro_include", operand[:40])] += 1
                continue
            name = lit.group(2)
            if name.startswith("/") or ".." in name.split("/"):
                feats[("bad_include", name)] += 1
    if pristine_main_params is not None:
        for m in _MAIN_DEF_RE.finditer(code):
            if _param_count(m.group(1)) > pristine_main_params:
                feats[("main_params",)] += 1
    for kind, name in _macro_targets(code, raw):
        if name in other_idents:
            feats[("macro", kind, name, "an identifier of the app's other source files")] += 1
        elif _is_api_name(name):
            feats[("macro", kind, name, "a HIP/C/C++ API name")] += 1
    for rx, label in _FORBIDDEN_CODE_RE:
        count = len(rx.findall(code))
        if count:
            feats[("token", label)] += count
    for lit in literals:
        for part in _FORBIDDEN_LITERAL_PARTS:
            if part in lit:
                feats[("literal", part)] += 1
    for name in _reserved_definitions(code):
        feats[("reserved", name)] += 1
    return feats


def _describe(feat: tuple, extra: int) -> str:  # noqa: PLR0911
    if feat[0] == "line_directive":
        return ("a #line directive: rewriting line markers to make kernel-file code look like "
                "another file is not allowed (even via a macro operand)")
    if feat[0] == "gnu_marker":
        return ("a '# N \"file\"' GNU line marker: rewriting line markers is not allowed")
    if feat[0] == "macro_include":
        return (f"#include {feat[1]}: the include operand must be a literal <...> or \"...\" "
                "header name (a macro-computed include is not allowed)")
    if feat[0] == "bad_include":
        return (f"#include of '{feat[1]}': an absolute or '..' include path is not allowed "
                "(only <system/HIP> headers and headers in the app directory)")
    if feat[0] == "main_params":
        return ("a main() with more parameters than the original: reading argv/envp to detect the "
                "profiler or the inputs is not allowed")
    return _describe_orig(feat, extra)


def _describe_orig(feat: tuple, extra: int) -> str:
    times = "" if extra == 1 else f" ({extra} more than the original)"
    if feat[0] == "token" and feat[1] == "inline asm":
        return (f"inline assembly{times}: raw asm (incl. asm labels) can reach syscalls or libc "
                "symbols the gate cannot see, and is not allowed in the kernel file")
    if feat[0] == "token" and feat[1] == "blocking-sync scheduling":
        return (f"blocking-sync scheduling{times}: blocking-sync device/event flags trade GPU "
                "wait for lower CPU time (which G-cpu governs) and hang under the Frontier "
                "profiler; they are not allowed in the kernel file")
    if feat[0] == "macro":
        _, kind, name, why = feat
        return (f"#{kind} of '{name}'{times}: redefining {why} is not allowed (the kernel file is "
                "compiled into the host program, so such macros change code outside the kernel)")
    if feat[0] == "token":
        return (f"'{feat[1]}'{times}: file, process, environment, thread or dynamic-loading access "
                "is not allowed in the kernel file")
    if feat[0] == "literal":
        return (f"string literal containing '{feat[1]}'{times}: references to profiler, "
                "reference-output or system paths are not allowed")
    return (f"definition of '{feat[1]}'{times}: names starting with '__' are reserved "
            "(runtime/compiler names) and may not be defined")


def other_source_identifiers(app: dict, gpa_root: Path) -> frozenset[str]:
    """Identifiers in the app's source files other than its kernel file."""
    app_dir = Path(gpa_root) / str(app["path"])
    kernel = (Path(gpa_root) / str(app["kernel_file"])).resolve()
    idents: set[str] = set()
    for path in sorted(app_dir.rglob("*")):
        if path.suffix not in SOURCE_SUFFIXES or not path.is_file():
            continue
        if path.resolve() == kernel:
            continue
        code, _ = split_source(path.read_text(encoding="utf-8", errors="replace"))
        idents.update(_IDENT_RE.findall(code))
    return frozenset(idents)


# ------------------------------------------------------------------ preprocessed pass (F3)

# names that must not be called/used in the kernel file after macro expansion, any qualification
_FORBIDDEN_CALLS = frozenset({
    "fopen", "fopen64", "freopen", "freopen64", "fdopen", "open", "open64", "openat",
    "openat64", "creat", "creat64", "read", "write", "pread", "pread64", "pwrite", "pwrite64",
    "readv", "writev", "mmap", "mmap64", "unlink", "unlinkat", "remove", "rename", "renameat",
    "renameat2", "system", "popen", "execl", "execlp", "execle", "execv", "execvp", "execve",
    "execvpe", "fork", "vfork", "clone", "posix_spawn", "posix_spawnp", "getenv",
    "secure_getenv", "setenv", "unsetenv", "putenv", "clearenv", "dlopen", "dlsym", "dlmopen",
    "pthread_create", "atexit", "at_quick_exit", "on_exit", "syscall", "__syscall", "ptrace",
    "kill", "raise", "signal", "sigaction", "secure_getenv", "__secure_getenv",
    "dl_iterate_phdr", "getauxval", "dladdr", "dlinfo", "prctl",
})
# H2: an asm label (extern "C" T f(...) asm("getenv")) reaches a forbidden symbol under any
# local name, so treat a call of a symbol whose asm-label is forbidden as that call; and forbid
# these names anywhere in the expanded kernel region.
_FORBIDDEN_NAMES = frozenset({"environ", "__environ", "_environ", "ifstream", "ofstream",
                              "fstream", "filebuf", "filesystem", "jthread"})
_ASM_RE = re.compile(r"\b(?:asm|__asm__|__asm)\b")
_CALL_RE = re.compile(r"(?<![\w.])(?<!->)([A-Za-z_]\w*)\s*\(")
_STD_THREAD_RE = re.compile(r"\bstd\s*::\s*(thread|async)\b")
_ATTR_CTOR_RE = re.compile(r"__attribute__\s*\(\(\s*[^)]*\b(constructor|destructor)\b")
_LINE_MARKER_RE = re.compile(r'^#\s*(\d+)\s+"((?:[^"\\]|\\.)*)"')
_DECL_AFTER_KW_RE = re.compile(r"\b(?:struct|class|union|enum|typedef|using)\s+(?:class\s+)?([A-Za-z_]\w*)")
_TYPEDEF_NAME_RE = re.compile(r"\btypedef\b[^;]*?\b([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*;")
_DEFINE_NAME_RE = re.compile(r"^#define\s+([A-Za-z_]\w*)", re.MULTILINE)

_PP_CACHE: dict[tuple, tuple] = {}


def _hipcc(rocm_path: Path | None) -> Path:
    hipcc = Path(rocm_path or detect_rocm_path()) / "bin" / "hipcc"
    if not hipcc.exists():
        found = shutil.which("hipcc")
        if not found:
            msg = f"the kernel gate needs hipcc (not found at {hipcc} or on PATH)"
            raise DriverInfraError(msg)
        hipcc = Path(found)
    return hipcc


def _pp_spec(app: dict) -> tuple[str, list[str]]:
    spec = app.get("gate_preprocess") or {}
    tu = spec.get("tu") or Path(str(app["kernel_file"])).relative_to(str(app["path"])).as_posix()
    flags = str(spec.get("flags", "-x hip")).split()
    return tu, flags


def _run_pp(app: dict, gpa_root: Path, kernel_text: str | None, hipcc: Path,
            extra: list[str], system_dirs: tuple[str, ...] = ()) -> subprocess.CompletedProcess:
    """Preprocess the app's TU in a scratch copy of the app dir with the kernel file replaced."""
    app_dir = Path(gpa_root) / str(app["path"])
    kernel_rel = Path(str(app["kernel_file"])).relative_to(str(app["path"]))
    tu, flags = _pp_spec(app)
    with tempfile.TemporaryDirectory(prefix="gpa_gate_") as tmp:
        work = Path(tmp) / "app"
        shutil.copytree(app_dir, work, symlinks=True,
                        ignore=shutil.ignore_patterns("*.o", ".rocprofv3", ".frontier_build.log"))
        if kernel_text is not None:
            (work / kernel_rel).write_text(kernel_text, encoding="utf-8")
        cmd = [str(hipcc), "-E", "--cuda-host-only", *flags, "--offload-arch=gfx90a", *extra, tu]
        proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True,  # noqa: S603
                              check=False, timeout=300)
        proc.kernel_abs = os.path.realpath(work / kernel_rel)  # type: ignore[attr-defined]
        proc.work = str(work)  # type: ignore[attr-defined]
        if proc.returncode == 0:
            # resolve marker paths while the scratch copy still exists
            kern, other, ids = _split_regions(proc.stdout, work, proc.kernel_abs, system_dirs)
            proc.regions = (kern, other)  # type: ignore[attr-defined]
            proc.marker_ids = ids  # type: ignore[attr-defined]
        return proc


_SYSDIR_CACHE: dict[str, tuple[str, ...]] = {}


def _system_include_dirs(hipcc: Path) -> tuple[str, ...]:
    """Toolchain system-include search dirs (J1b), from `hipcc -E -v` plus the usual roots."""
    key = str(hipcc)
    if key in _SYSDIR_CACHE:
        return _SYSDIR_CACHE[key]
    dirs: list[str] = []
    try:
        proc = subprocess.run([str(hipcc), "-E", "-v", "-x", "hip", "-"],  # noqa: S603
                              input="", capture_output=True, text=True, check=False, timeout=120)
        out = proc.stderr
        if "search starts here:" in out:
            body = out.split("search starts here:", 1)[1].split("End of search list.", 1)[0]
            for ln in body.splitlines():
                d = ln.strip()
                if d and os.path.isdir(d):
                    dirs.append(os.path.realpath(d))
    except (OSError, subprocess.SubprocessError):
        pass
    for d in ("/opt/rocm-7.0.2/include", "/opt/rocm/include", "/usr/include", "/usr/lib",
              "/usr/lib64", str(Path(hipcc).resolve().parent.parent)):
        rd = os.path.realpath(d)
        if os.path.isdir(rd) and rd not in dirs:
            dirs.append(rd)
    result = tuple(dirs)
    _SYSDIR_CACHE[key] = result
    return result


def _is_system_file(real: str, system_dirs: tuple[str, ...]) -> bool:
    """A marker file is SYSTEM iff it is under a toolchain system dir AND root-owned (J1b)."""
    if not any(real == d or real.startswith(d + os.sep) for d in system_dirs):
        return False
    try:
        return os.stat(real).st_uid == 0
    except OSError:
        return False


def _marker_id(name: str, work: Path, kernel_abs: str) -> str:
    if name.startswith("<"):
        return "<builtin>"
    path = name if os.path.isabs(name) else os.path.join(work, name)
    real = os.path.realpath(path)
    if real == kernel_abs:
        return "K"
    try:
        return "app:" + os.path.relpath(real, work)
    except ValueError:
        return "sys:" + real


def _split_regions(text: str, work: Path, kernel_abs: str,
                   system_dirs: tuple[str, ...] = ()) -> tuple[str, str, frozenset[str]]:
    """(kernel-file lines, all other lines) of preprocessor output, by line markers.

    J1b allow-list attribution. A marker file's tokens are "other" (not analysed) ONLY when it is
    the compiler's own builtin buffer or a SYSTEM header (root-owned, under a toolchain system
    dir). The kernel file, the app dir's own headers, and ANY unknown / unresolvable / spoofed
    path are the kernel file's own (fail closed) and are analysed. A macro-computed
    `#line "…/hip_runtime.h"` cannot hide tokens: the physical file is still the kernel file,
    which is not root-owned under a system dir.
    """
    kernel_lines: list[str] = []
    other_lines: list[str] = []
    in_kernel = True  # before the first marker, and (fail closed) for any unknown file
    seen: set[str] = set()
    for line in text.splitlines():
        m = _LINE_MARKER_RE.match(line)
        if m:
            name = m.group(2)
            seen.add(_marker_id(name, work, kernel_abs))
            if name.startswith("<"):
                in_kernel = False  # <built-in>, <command line>, <scratch space>
            else:
                path = name if os.path.isabs(name) else os.path.join(work, name)
                real = os.path.realpath(path)
                in_kernel = not _is_system_file(real, system_dirs)
            continue
        (kernel_lines if in_kernel else other_lines).append(line)
    return "\n".join(kernel_lines), "\n".join(other_lines), frozenset(seen)


def _declared_names(app: dict, gpa_root: Path, hipcc: Path) -> frozenset[str]:
    """Names declared or #defined outside the kernel file (headers + the other TU sources)."""
    key = ("declared", str(gpa_root), str(app["path"]), str(app["kernel_file"]), str(hipcc))
    if key in _PP_CACHE:
        return _PP_CACHE[key][0]
    sysdirs = _system_include_dirs(hipcc)
    empty = _run_pp(app, gpa_root, "", hipcc, [], sysdirs)
    macros = _run_pp(app, gpa_root, "", hipcc, ["-dM"], sysdirs)
    if empty.returncode != 0 or macros.returncode != 0:
        msg = ("the kernel gate could not preprocess the app's sources: "
               f"{head_tail(empty.stderr or macros.stderr)}")
        raise DriverInfraError(msg)
    _, other = empty.regions  # type: ignore[attr-defined]
    code, _, _ = _lex(other)
    names = {m.group(1) for m in _CALL_RE.finditer(code)}
    names |= {m.group(1) for m in _DECL_AFTER_KW_RE.finditer(code)}
    names |= {m.group(1) for m in _TYPEDEF_NAME_RE.finditer(code)}
    names |= set(_DEFINE_NAME_RE.findall(macros.stdout))
    result = frozenset(names)
    _PP_CACHE[key] = (result,)
    return result


_MAIN_NAME_RE = re.compile(r"(?<![\w:])main\s*\(")


def _max_main_params(code: str) -> int:
    best = -1
    for m in _MAIN_NAME_RE.finditer(code):
        close = _balanced_close(code, m.end() - 1)
        if close < 0:
            continue
        best = max(best, _param_count(code[m.end():close]))
    return best


def _expanded_features(kernel_region: str) -> Counter:
    code, literals, _ = _lex(kernel_region)
    feats: Counter = Counter()
    for m in _CALL_RE.finditer(code):
        if m.group(1) in _FORBIDDEN_CALLS:
            feats[("xcall", m.group(1))] += 1
    for name in _IDENT_RE.findall(code):
        if name in _FORBIDDEN_NAMES:
            feats[("xname", name)] += 1
    feats[("xname", "std::thread")] += len(_STD_THREAD_RE.findall(code))
    feats[("xname", "__attribute__((constructor/destructor))")] += len(_ATTR_CTOR_RE.findall(code))
    feats[("xasm",)] += len(_ASM_RE.findall(code))
    for lit in literals:
        for part in _FORBIDDEN_LITERAL_PARTS:
            if part in lit:
                feats[("xliteral", part)] += 1
    for name in _reserved_definitions(code):
        feats[("xreserved", name)] += 1
    return +feats


def _expanded_region(app: dict, gpa_root: Path, text: str, hipcc: Path,
                     system_dirs: tuple[str, ...] = ()):
    proc = _run_pp(app, gpa_root, text, hipcc, [], system_dirs)
    if proc.returncode != 0:
        return None
    return proc.regions[0]  # type: ignore[attr-defined]


def _describe_expanded(feat: tuple, extra: int) -> str:
    times = "" if extra == 1 else f" ({extra} more than the original)"
    if feat[0] == "xasm":
        return ("inline assembly after macro expansion: raw asm (incl. asm labels) can reach "
                "syscalls or libc symbols the gate cannot see, and is not allowed in the kernel file")
    kind, name = feat
    if kind == "xcall":
        return (f"call of '{name}'{times} after macro expansion: file, process, environment, "
                "thread or dynamic-loading access is not allowed in the kernel file")
    if kind == "xname":
        return (f"'{name}'{times} after macro expansion: file, process, environment or thread "
                "access is not allowed in the kernel file")
    if kind == "xliteral":
        return (f"string literal containing '{name}'{times} after macro expansion: references "
                "to profiler, reference-output or system paths are not allowed")
    return (f"definition of '{name}'{times} after macro expansion (token pasting or an alias "
            "macro): names starting with '__' are reserved and may not be defined")


def _preprocessed_violations(app: dict, candidate: str, pristine: str, gpa_root: Path,
                             rocm_path: Path | None, notes: list[str]) -> list[str]:
    hipcc = _hipcc(rocm_path)
    pkey = ("pristine", str(gpa_root), str(app["path"]), str(app["kernel_file"]), str(hipcc),
            hash(pristine))
    sysdirs = _system_include_dirs(hipcc)
    if pkey not in _PP_CACHE:
        region = _expanded_region(app, gpa_root, pristine, hipcc, sysdirs)
        if region is None:
            msg = f"the kernel gate could not preprocess the pristine {app['kernel_file']}"
            raise DriverInfraError(msg)
        _PP_CACHE[pkey] = (_expanded_features(region),)
    base = _PP_CACHE[pkey][0]
    proc = _run_pp(app, gpa_root, candidate, hipcc, [], sysdirs)
    if proc.returncode != 0:
        return [("the kernel file could not be preprocessed with the app's compiler flags, so it "
                 f"cannot be checked (fix the compile error first): {head_tail(proc.stderr, 600, 600)}")]
    cand_region = proc.regions[0]  # type: ignore[attr-defined]
    cand = _expanded_features(cand_region)
    violations = [_describe_expanded(f, n - base.get(f, 0))
                  for f, n in sorted(cand.items(), key=lambda kv: repr(kv[0]))
                  if n > base.get(f, 0)]
    # (a) new macros named like a declared/defined name of the headers or other sources
    declared = _declared_names(app, gpa_root, hipcc)
    raw_code, _, _ = _lex(candidate)
    pristine_code, _, _ = _lex(pristine)
    cand_macros = Counter(m.group(2) for m in _DIRECTIVE_RE.finditer(raw_code))
    base_macros = Counter(m.group(2) for m in _DIRECTIVE_RE.finditer(pristine_code))
    for name, n in sorted(cand_macros.items()):
        if n > base_macros.get(name, 0) and name in declared:
            violations.append(
                f"#define/#undef of '{name}': it is declared or defined by the included headers "
                "or the app's other sources; redefining it changes code outside the kernel")
    # J1c: main() parameter count on the preprocessed kernel-file tokens (any form)
    pkm = ("pristine_main", str(gpa_root), str(app["path"]), str(app["kernel_file"]), str(hipcc))
    if pkm not in _PP_CACHE:
        pr_region = _expanded_region(app, gpa_root, pristine, hipcc, sysdirs)
        base_code, _, _ = _lex(pr_region if pr_region is not None else "")
        _PP_CACHE[pkm] = (_max_main_params(base_code),)
    base_main = _PP_CACHE[pkm][0]
    cand_code, _, _ = _lex(cand_region)
    if _max_main_params(cand_code) > base_main:
        violations.append(
            "a main() with more parameters than the original (after preprocessing): reading "
            "argv/envp to detect the profiler or the inputs is not allowed")
    notes.append("preprocessed with " + " ".join([hipcc.name, *_pp_spec(app)[1], _pp_spec(app)[0]]))
    return violations


def _app_dir_headers(app: dict, gpa_root: Path) -> frozenset[str]:
    """Header/source names that exist in the app dir (allowed #include targets, by basename)."""
    app_dir = Path(gpa_root) / str(app["path"])
    names: set[str] = set()
    for path in app_dir.rglob("*"):
        if path.is_file() and path.suffix in SOURCE_SUFFIXES:
            names.add(path.name)
            names.add(path.relative_to(app_dir).as_posix())
    return frozenset(names)


def _pristine_main_params(pristine: str) -> int:
    code, _, _ = _lex(pristine)
    counts = [_param_count(m.group(1)) for m in _MAIN_DEF_RE.finditer(code)]
    return max(counts) if counts else 0  # a kernel file with no main() must not gain one with params


def check_kernel_source(app: dict, candidate: str, *, gpa_root: Path,
                        pristine: str | None = None, preprocess: bool = True,
                        rocm_path: Path | None = None) -> GateResult:
    """Gate a candidate kernel file (its full text) against the app's pristine kernel file.

    Args:
        app: the app's yaml entry (needs name, path, kernel_file)
        candidate: full text of the candidate kernel file
        gpa_root: GPA-Benchmark root holding the pristine sources
        pristine: pristine kernel text (default: read gpa_root / kernel_file)

    Returns:
        GateResult (ok False with one violation per construct the candidate adds)

    """
    kernel_rel = str(app["kernel_file"])
    if pristine is None:
        pristine = (Path(gpa_root) / kernel_rel).read_text(encoding="utf-8", errors="replace")
    others = other_source_identifiers(app, gpa_root)
    inc = _app_dir_headers(app, gpa_root)
    main_params = _pristine_main_params(pristine)
    base = _features(pristine, others, inc, main_params)
    cand = _features(candidate, others, inc, main_params)
    violations = [_describe(feat, count - base.get(feat, 0))
                  for feat, count in sorted(cand.items(), key=lambda kv: repr(kv[0]))
                  if count > base.get(feat, 0)]
    notes: list[str] = []
    if preprocess:
        for v in _preprocessed_violations(app, candidate, pristine, gpa_root, rocm_path, notes):
            if v not in violations:
                violations.append(v)
    return GateResult(ok=not violations, violations=violations,
                      kernel_file=Path(kernel_rel).name, preprocessed=preprocess, notes=notes)


def check_kernel_file(app: dict, candidate_path: Path, *, gpa_root: Path,
                      preprocess: bool = True) -> GateResult:
    """:func:`check_kernel_source` for a file on disk."""
    text = Path(candidate_path).read_text(encoding="utf-8", errors="replace")
    return check_kernel_source(app, text, gpa_root=gpa_root, preprocess=preprocess)
