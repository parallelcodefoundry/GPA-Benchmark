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
    (r"\bsyscall\s*\(", "syscall("),
    (r"\bptrace\b", "ptrace"),
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


def _features(text: str, other_idents: frozenset[str]) -> Counter:
    code, literals, raw = _lex(text)
    feats: Counter = Counter()
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


def _describe(feat: tuple, extra: int) -> str:
    times = "" if extra == 1 else f" ({extra} more than the original)"
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
    "pthread_create", "atexit", "at_quick_exit", "on_exit", "syscall", "ptrace", "kill",
    "raise", "signal", "sigaction",
})
_FORBIDDEN_NAMES = frozenset({"environ", "__environ", "ifstream", "ofstream", "fstream",
                              "filebuf", "filesystem", "jthread"})
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
            extra: list[str]) -> subprocess.CompletedProcess:
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
            proc.regions = _split_regions(proc.stdout, work, proc.kernel_abs)  # type: ignore[attr-defined]
        return proc


def _split_regions(text: str, work: Path, kernel_abs: str) -> tuple[str, str]:
    """(kernel-file lines, all other lines) of preprocessor output, by line markers."""
    kernel_lines: list[str] = []
    other_lines: list[str] = []
    in_kernel = False
    cache: dict[str, bool] = {}
    for line in text.splitlines():
        m = _LINE_MARKER_RE.match(line)
        if m:
            name = m.group(2)
            if name not in cache:
                path = name if os.path.isabs(name) else os.path.join(work, name)
                cache[name] = os.path.realpath(path) == kernel_abs
            in_kernel = cache[name]
            continue
        (kernel_lines if in_kernel else other_lines).append(line)
    return "\n".join(kernel_lines), "\n".join(other_lines)


def _declared_names(app: dict, gpa_root: Path, hipcc: Path) -> frozenset[str]:
    """Names declared or #defined outside the kernel file (headers + the other TU sources)."""
    key = ("declared", str(gpa_root), str(app["path"]), str(app["kernel_file"]), str(hipcc))
    if key in _PP_CACHE:
        return _PP_CACHE[key][0]
    empty = _run_pp(app, gpa_root, "", hipcc, [])
    macros = _run_pp(app, gpa_root, "", hipcc, ["-dM"])
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
    for lit in literals:
        for part in _FORBIDDEN_LITERAL_PARTS:
            if part in lit:
                feats[("xliteral", part)] += 1
    for name in _reserved_definitions(code):
        feats[("xreserved", name)] += 1
    return +feats


def _expanded_region(app: dict, gpa_root: Path, text: str, hipcc: Path) -> str | None:
    proc = _run_pp(app, gpa_root, text, hipcc, [])
    if proc.returncode != 0:
        return None
    return proc.regions[0]  # type: ignore[attr-defined]


def _describe_expanded(feat: tuple, extra: int) -> str:
    times = "" if extra == 1 else f" ({extra} more than the original)"
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
    if pkey not in _PP_CACHE:
        region = _expanded_region(app, gpa_root, pristine, hipcc)
        if region is None:
            msg = f"the kernel gate could not preprocess the pristine {app['kernel_file']}"
            raise DriverInfraError(msg)
        _PP_CACHE[pkey] = (_expanded_features(region),)
    base = _PP_CACHE[pkey][0]
    proc = _run_pp(app, gpa_root, candidate, hipcc, [])
    if proc.returncode != 0:
        return [("the kernel file could not be preprocessed with the app's compiler flags, so it "
                 f"cannot be checked (fix the compile error first): {head_tail(proc.stderr, 600, 600)}")]
    cand = _expanded_features(proc.regions[0])  # type: ignore[attr-defined]
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
    notes.append("preprocessed with " + " ".join([hipcc.name, *_pp_spec(app)[1], _pp_spec(app)[0]]))
    return violations


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
    base = _features(pristine, others)
    cand = _features(candidate, others)
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
