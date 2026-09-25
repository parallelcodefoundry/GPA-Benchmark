"""Output checks for the hip backend (Frontier): harness-side references and fast validation.

GPA-G1 fix round 1 (R4, R5, R7, R10):

* References live OUTSIDE every directory an app runs in: ``GPA-Benchmark/frontier_refs/<app>/``
  (yaml ``reference_output`` is relative to the GPA root). Before first use in a process each
  reference file is md5-checked against ``frontier_refs.md5``; a missing entry or a mismatch
  raises :class:`RefIntegrityError`.
* :func:`check_output` validates the bytes an app produced (its ``test_output`` file, else its
  stdout) against a :class:`Reference`, fast path first (byte equality); a diff is only built on
  failure and is bounded (the first differing lines), never a whole-file difflib.
* Check types (same yaml keys as driver_apps.yaml): ``expected_checksum`` (exactly one matching
  stdout line, number equal), ``numeric_tolerance`` (token-wise), ``float_grep`` (with
  ``float_grep_all: true`` EVERY occurrence is compared, same count required), exact otherwise.
* R7: :func:`reference_from_output` turns a PRISTINE run's output into the reference, so the
  agent's output on a random input variant is judged by the app's own check type/tolerance.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from gpa_bench_driver.driver_src.driver_validation import (
    DEFAULT_CHECKSUM_REGEX,
    validate_numeric_tolerance,
)

MANIFEST_NAME = "frontier_refs.md5"
_MAX_REPORT_LINES = 6
_MAX_LINE_CHARS = 300


class RefIntegrityError(RuntimeError):
    """A harness-side reference is unlisted in frontier_refs.md5 or does not match its md5."""


@dataclass(frozen=True)
class Reference:
    """What an app's output is checked against.

    kind "bytes": ``data`` holds the reference output; kind "checksum": ``checksum`` holds the
    expected integer. ``source`` says where it came from (a path, "yaml", "pristine run").
    """

    kind: str
    data: bytes | None = None
    checksum: int | None = None
    source: str = ""


# (resolved path, size, mtime_ns) -> verified bytes
_VERIFIED_CACHE: dict[tuple[str, int, int], bytes] = {}


def _manifest(gpa_root: Path) -> dict[str, str]:
    path = gpa_root / MANIFEST_NAME
    if not path.is_file():
        msg = f"{path} is missing: the Frontier references cannot be verified"
        raise RefIntegrityError(msg)
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:  # noqa: PLR2004
            entries[parts[1].lstrip("*")] = parts[0]
    return entries


def load_reference(app: dict, gpa_root: Path) -> Reference:
    """The app's stored reference: its md5-verified reference_output file or expected_checksum.

    Raises:
        RefIntegrityError: reference file unlisted in frontier_refs.md5, missing or modified
        ValueError: the app has no reference_output and no expected_checksum value

    """
    if "expected_checksum" in app:
        spec = app["expected_checksum"]
        if not isinstance(spec, dict) or spec.get("value") is None:
            msg = (f"expected_checksum of {app['name']} has no stored value for this input "
                   "(input variants need reference_from_baseline=True)")
            raise ValueError(msg)
        return Reference(kind="checksum", checksum=int(spec["value"]), source="yaml")
    if app.get("reference_output") is None:
        msg = (f"{app['name']} has no stored reference for this input (input variants need "
               "reference_from_baseline=True)")
        raise ValueError(msg)
    rel = str(app["reference_output"])
    path = (Path(gpa_root) / rel).resolve()
    if not path.is_file():
        msg = f"reference {rel} is missing (run scripts/frontier_prepare.sh)"
        raise RefIntegrityError(msg)
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _VERIFIED_CACHE:
        want = _manifest(Path(gpa_root)).get(rel)
        if want is None:
            msg = f"reference {rel} is not listed in {MANIFEST_NAME}"
            raise RefIntegrityError(msg)
        data = path.read_bytes()
        got = hashlib.md5(data).hexdigest()  # noqa: S324 - integrity pin, not security
        if got != want:
            msg = f"reference {rel} has md5 {got}, {MANIFEST_NAME} expects {want}"
            raise RefIntegrityError(msg)
        _VERIFIED_CACHE[key] = data
    return Reference(kind="bytes", data=_VERIFIED_CACHE[key], source=rel)


def produced_output(app: dict, stdout: bytes | None, run_dir: Path) -> bytes | None:
    """The bytes the app's check reads: its test_output file (in run_dir), else its stdout.

    Returns None when the app has a test_output file and the run did not write it.
    """
    if "test_output" in app:
        path = Path(run_dir) / Path(str(app["test_output"])).name
        return path.read_bytes() if path.is_file() else None
    return stdout if stdout is not None else b""


def _checksums(app: dict, text: str) -> list[re.Match]:
    spec = app.get("expected_checksum") or {}
    regex = spec.get("regex") if isinstance(spec, dict) else None
    return list(re.finditer(regex or DEFAULT_CHECKSUM_REGEX, text, re.MULTILINE))


def reference_from_output(app: dict, produced: bytes | None) -> Reference:
    """R7: the reference defined by a PRISTINE run's output (checksum apps: its checksum).

    Raises:
        ValueError: the pristine output is missing, or (checksum apps) has not exactly one
            checksum line

    """
    if produced is None:
        msg = f"the pristine {app['name']} run produced no output to use as the reference"
        raise ValueError(msg)
    if "expected_checksum" in app:
        found = _checksums(app, produced.decode("utf-8", errors="replace"))
        if len(found) != 1:
            msg = f"the pristine {app['name']} run printed {len(found)} checksum lines, not 1"
            raise ValueError(msg)
        return Reference(kind="checksum", checksum=int(found[0].group(1)), source="pristine run")
    return Reference(kind="bytes", data=bytes(produced), source="pristine run")


# Characters shown on each side of the first differing column of a long line. Matches
# APPEB's tools/gpa_harness/lib/gpa_failure_text.CONTEXT, so a line windowed here looks the same
# as one windowed there (that module leaves these pre-windowed lines unchanged).
_WINDOW_CONTEXT = 48


def _clip(line: str) -> str:
    return line if len(line) <= _MAX_LINE_CHARS else line[:_MAX_LINE_CHARS] + " [...]"


def _first_diff_col(a: str, b: str) -> int:
    """0-based index of the first differing character (min length if one is a prefix)."""
    n = min(len(a), len(b))
    return next((k for k in range(n) if a[k] != b[k]), n)


def _window(x: str, lo: int, hi: int) -> str:
    return ("..." if lo else "") + x[lo:hi] + ("..." if hi < len(x) else "")


def _line_pair(k: int, exp: str, got: str) -> list[str]:
    """Report lines for one expected/got pair.

    Short lines (and a missing line) keep the plain ``line N expected: ...`` form (unchanged).
    A long line is shown as a window around its first differing COLUMN (1-based), since the
    start of e.g. pathfinder's 300 000-value result row says nothing about where it differs.
    """
    missing = "<no line>" in (exp, got)
    if missing or max(len(exp), len(got)) <= _MAX_LINE_CHARS:
        return [f"  line {k + 1} expected: {_clip(exp)}", f"  line {k + 1} got:      {_clip(got)}"]
    if exp == got:
        return [f"  line {k + 1}: expected and got are identical ({len(exp)} characters)"]
    c = _first_diff_col(exp, got)
    lo, hi = max(0, c - _WINDOW_CONTEXT), c + _WINDOW_CONTEXT
    return [f"  line {k + 1} expected (first difference at column {c + 1}): {_window(exp, lo, hi)}",
            f"  line {k + 1} got      (first difference at column {c + 1}): {_window(got, lo, hi)}"]


def first_difference(ref: str, test: str) -> str:
    """Bounded report of where two texts first differ (line numbers are 1-based).

    Long lines are windowed around their first differing column (see :func:`_line_pair`).
    """
    ref_lines = ref.splitlines()
    test_lines = test.splitlines()
    n = min(len(ref_lines), len(test_lines))
    i = next((k for k in range(n) if ref_lines[k] != test_lines[k]), n)
    out = [f"first difference at line {i + 1} (reference has {len(ref_lines)} lines, "
           f"output has {len(test_lines)} lines):"]
    for k in range(i, min(i + _MAX_REPORT_LINES // 2, max(len(ref_lines), len(test_lines)))):
        exp = ref_lines[k] if k < len(ref_lines) else "<no line>"
        got = test_lines[k] if k < len(test_lines) else "<no line>"
        out += _line_pair(k, exp, got)
    return "\n".join(out)


_FLOAT = re.compile(r"(\d+\.\d+)")


def _grep_floats(text: str, key: str) -> list[float]:
    values = []
    for part in text.split(key)[1:]:
        m = _FLOAT.search(part)
        values.append(float(m.group(1)) if m else float("nan"))
    return values


def _check_float_grep(app: dict, produced: str, ref: str) -> tuple[bool, str | None]:
    key = str(app["float_grep"])
    tol = float(app["float_tolerance"])
    want = _grep_floats(ref, key)
    got = _grep_floats(produced, key)
    if not want:
        msg = f"reference has no float after {key!r}"
        raise ValueError(msg)
    if not app.get("float_grep_all"):
        want, got = want[-1:], got[-1:]
    if len(got) != len(want):
        return False, (f"expected {len(want)} value(s) after {key!r}, found {len(got)}")
    bad = [(i, w, g) for i, (w, g) in enumerate(zip(want, got, strict=True))
           if not abs(w - g) <= tol]
    if bad:
        shown = "; ".join(f"#{i + 1}: expected {w}, got {g}" for i, w, g in bad[:5])
        return False, (f"{len(bad)} of {len(want)} value(s) after {key!r} differ by more than "
                       f"{tol}: {shown}")
    return True, None


def check_output(app: dict, produced: bytes | None, ref: Reference) -> tuple[bool, str | None]:
    """Validate an app's produced output against a reference with the app's check type.

    Args:
        app: yaml entry (check keys: expected_checksum, numeric_tolerance, float_grep[_all])
        produced: bytes from :func:`produced_output` (None = expected output file missing)
        ref: from :func:`load_reference` or :func:`reference_from_output`

    Returns:
        (ok, message); message explains the failure (None on success)

    """
    if produced is None:
        name = Path(str(app.get("test_output", "output"))).name
        return False, f"the program did not write its output file {name}"
    if ref.kind == "checksum":
        found = _checksums(app, produced.decode("utf-8", errors="replace"))
        if len(found) != 1:
            return False, (f"expected exactly one 'Verification checksum' line in the output, "
                           f"found {len(found)}")
        got = int(found[0].group(1))
        if got != ref.checksum:
            return False, (f"checksum mismatch: expected {ref.checksum} for this input "
                           f"({ref.source}), got {got}")
        return True, None
    if ref.data is None:
        msg = "bytes reference without data"
        raise ValueError(msg)
    if produced == ref.data:  # fast path
        return True, None
    text = produced.decode("utf-8", errors="replace")
    ref_text = ref.data.decode("utf-8", errors="replace")
    if "numeric_tolerance" in app:
        ok, msg = validate_numeric_tolerance(text, ref_text, app)
        return ok, (None if ok else msg)
    if "float_grep" in app:
        return _check_float_grep(app, text, ref_text)
    return False, "output differs from the reference; " + first_difference(ref_text, text)
