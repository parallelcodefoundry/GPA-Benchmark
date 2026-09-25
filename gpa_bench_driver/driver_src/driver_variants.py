"""Random input variants for the post-agent generalization test (GPA-G1 fix round 1, R7).

``draw_variant(app, seed, workdir=...)`` draws one input variant of a Frontier GPA app: problem
parameters about +-25% around the public size, deterministic given (app, seed), cheap to make
(only bfs generates a data file: a seeded random graph; b+tree writes a 2-line command file).
Generated files go to ``workdir``, which must lie outside every directory an app runs in.

The variant is run with the driver's ``reference_from_baseline=True``: the PRISTINE baseline's
output on the variant is the reference, and the agent's output must pass the app's own check
(type and tolerance) against it; for xsbench the checksum comes from the pristine run.

    v = draw_variant("bfs", 1234, workdir=Path("/tmp/v"))
    DriverConfig(app="bfs", gpu_backend="hip", app_overrides=v.overrides,
                 reference_from_baseline=True, swaps_override={...}, nsys=True, ...)
"""

from __future__ import annotations

import random
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VARIANT_APPS = ("bfs", "backprop", "b+tree", "heartwall", "hotspot", "pathfinder", "nw",
                "streamcluster", "xsbench")

_GPA_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class Variant:
    """One drawn input variant (record ``seed``, ``params`` and ``run_command`` in results)."""

    app: str
    seed: int
    params: dict[str, Any]
    run_command: str
    overrides: dict[str, Any] = field(default_factory=dict)
    generated_files: tuple[str, ...] = ()


def _rng(app: str, seed: int) -> random.Random:
    return random.Random(f"gpa-frontier-variant:{app}:{int(seed)}")  # noqa: S311 - not crypto


def _around(rng: random.Random, public: int, *, lo: float = 0.75, hi: float = 1.25,
            multiple: int = 1, cap: int | None = None) -> int:
    low = max(multiple, int(public * lo) // multiple * multiple)
    high = int(public * hi) // multiple * multiple
    if cap is not None:
        high = min(high, cap // multiple * multiple)
    return rng.randrange(low, high + 1, multiple)


def _graphgen(workdir: Path, gpa_root: Path) -> Path:
    exe = workdir / "graphgen_seeded"
    if not exe.exists():
        src = gpa_root / "scripts" / "graphgen_seeded.cpp"
        subprocess.run(["g++", "-O2", "-std=c++11", "-o", str(exe), str(src)],  # noqa: S603,S607
                       check=True, capture_output=True)
    return exe


def draw_variant(app: str, seed: int, *, workdir: Path, gpa_root: Path | None = None) -> Variant:
    """Draw a random input variant of ``app`` (deterministic given app and seed).

    Args:
        app: canonical app name (one of VARIANT_APPS)
        seed: variant seed (record it)
        workdir: directory for generated inputs (created; must be outside the app's run dir)
        gpa_root: GPA-Benchmark root (default: this checkout)

    Returns:
        the Variant; pass ``variant.overrides`` as DriverConfig.app_overrides

    Raises:
        ValueError: unknown app

    """
    root = Path(gpa_root) if gpa_root is not None else _GPA_ROOT
    workdir = Path(workdir).resolve()
    rng = _rng(app, seed)
    files: list[str] = []
    overrides: dict[str, Any] = {}

    if app == "bfs":
        nodes = _around(rng, 8388608)
        graph_seed = rng.randrange(1, 2**31 - 1)
        params = {"nodes": nodes, "graph_seed": graph_seed}
        workdir.mkdir(parents=True, exist_ok=True)
        tag = f"v{int(seed)}_{nodes}"
        graph = workdir / f"graph{tag}.txt"
        if not graph.exists():
            exe = _graphgen(workdir, root)
            subprocess.run([str(exe), str(nodes), tag, str(graph_seed)],  # noqa: S603
                           cwd=workdir, check=True, capture_output=True)
        files.append(str(graph))
        run_command = f"./bfs {graph}"
    elif app == "backprop":
        # layer size must be a multiple of 16; blocks (size/16) stay <= 65535 (grid y limit)
        size = _around(rng, 1048560, hi=1.0, multiple=16, cap=16 * 65535)
        params = {"layer_size": size}
        run_command = f"./backprop {size}"
    elif app == "b+tree":
        count = _around(rng, 60000)
        kcount = _around(rng, 10000, cap=65535)
        params = {"j_count": count, "k_count": kcount}
        workdir.mkdir(parents=True, exist_ok=True)
        cmd = workdir / f"command_v{int(seed)}.txt"
        # same format as command_frontier.txt; the pristine parser reads j's count for both
        # count and rSize, so the second number is inert (kept for parity)
        cmd.write_text(f"j {count} 3000\nk {kcount}\n\n\n", encoding="utf-8")
        files.append(str(cmd))
        run_command = f"./b+tree.out file ../data/b+tree/mil.txt command {cmd}"
    elif app == "heartwall":
        frames = rng.randrange(8, 13)
        params = {"frames": frames}
        run_command = f"./heartwall ../data/heartwall/test.avi {frames}"
    elif app == "hotspot":
        pyramid = rng.randrange(4, 7)
        iterations = _around(rng, 100)
        params = {"pyramid_height": pyramid, "sim_time": iterations}
        run_command = (f"./hotspot 1024 {pyramid} {iterations} ../data/hotspot/temp_1024 "
                       "../data/hotspot/power_1024 output.out")
    elif app == "pathfinder":
        cols = _around(rng, 300000)
        rows = _around(rng, 300)
        pyramid = rng.randrange(15, 26)
        params = {"cols": cols, "rows": rows, "pyramid_height": pyramid}
        run_command = f"./pathfinder {cols} {rows} {pyramid}"
    elif app == "nw":
        dim = _around(rng, 32768, multiple=16)
        penalty = rng.randrange(8, 13)
        params = {"dim": dim, "penalty": penalty}
        run_command = f"./needle {dim} {penalty}"
    elif app == "streamcluster":
        n = _around(rng, 131072)
        dim = rng.choice((192, 224, 256, 288, 320))
        clustersize = _around(rng, 16000)
        params = {"n": n, "dim": dim, "clustersize": clustersize}
        run_command = f"./sc_gpu 10 20 {dim} {n} {n} {clustersize} none output.txt 1"
    elif app == "xsbench":
        lookups = _around(rng, 100_000_000)
        params = {"lookups": lookups}
        run_command = f"./XSBench -m event -s large -l {lookups}"
        # the stored checksum is only valid for the public count: the reference must come
        # from the pristine run (reference_from_baseline)
        overrides["expected_checksum"] = {"regex": r"^Verification checksum: (\d+)",
                                          "value": None}
    else:
        msg = f"no input variants for app {app!r} (known: {', '.join(VARIANT_APPS)})"
        raise ValueError(msg)

    overrides["run_command"] = run_command
    if app != "xsbench":
        overrides["reference_output"] = None  # the stored reference is for the public input
    return Variant(app=app, seed=int(seed), params=params, run_command=run_command,
                   overrides=overrides, generated_files=tuple(files))


def cleanup_variant(variant: Variant) -> None:
    """Delete a variant's generated files (the graphgen binary in workdir is kept)."""
    for f in variant.generated_files:
        Path(f).unlink(missing_ok=True)
