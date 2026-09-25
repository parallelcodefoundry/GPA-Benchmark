"""Random input variants for the post-agent generalization test (GPA-G1 fix round 1, R7).

``draw_variant(app, seed, workdir=...)`` draws one input variant of a Frontier GPA app. A
variant keeps the public input's SHAPE (problem dimensions, alignment / power-of-2 structure,
launch geometry, iteration counts) and changes the DATA (seeds, values, penalties, query sets,
graph edges), so the outputs differ from the public run and cannot be hardcoded or copied, while
an optimization that fixes a shape-specific pathology of the public input (e.g. nw's power-of-2
stride) keeps its speedup. Only pathfinder has no data knob outside its whole-program kernel
file; its variant changes the column count by at most 10% within the same alignment class.

Per-app knob (``VARIANT_KNOBS``; host-side seed arguments were added to NON-editable host files,
and the public run without them is byte-identical):

    bfs            graph seed (scripts/graphgen_seeded.cpp), same 8388608 nodes
    backprop       weight seed (facetrain.c 2nd argument), same layer size 1048560
    b+tree         query seed ("seed N" argument of main.cu), same command file (count, range)
    heartwall      first frame of the 10-frame window (main.cu 3rd argument)
    hotspot        temperature/power data (generated files), same 1024 grid, pyramid 5, 100 steps
    pathfinder     column count within +-10% (odd multiple of 32, like 300000), same rows/pyramid
    nw             input seed (needle.cu 3rd argument) and gap penalty, same dimension 32768
    streamcluster  point seed (streamcluster_cuda_cpu.cpp 10th argument), same sizes
    xsbench        nuclide/material data seed (XSBench -d), same 100000000 lookups

The variant is run with the driver's ``reference_from_baseline=True``: the PRISTINE baseline's
output on the variant is the reference, and the agent's output must pass the app's own check
(type and tolerance) against it; for xsbench the checksum comes from the pristine run.
Deterministic given (app, seed); generated files go to ``workdir`` (outside every run dir).

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

VARIANT_KNOBS = {
    "bfs": "graph seed (same 8388608 nodes)",
    "backprop": "weight seed (same layer size 1048560)",
    "b+tree": "query seed (same command file: j 60000 3000, k 10000)",
    "heartwall": "first frame of the 10-frame window",
    "hotspot": "temperature/power data (same 1024 grid, pyramid 5, 100 iterations)",
    "pathfinder": "columns within +-10% (odd multiple of 32; same 300 rows, pyramid 20)",
    "nw": "input seed + gap penalty (same dimension 32768)",
    "streamcluster": "point seed (same n, dim, chunk and cluster sizes)",
    "xsbench": "nuclide/material data seed -d (same 100000000 lookups)",
}

_GPA_ROOT = Path(__file__).resolve().parent.parent.parent
_PUBLIC_GRAPH_SEED = 20260924  # scripts/graphgen_seeded.cpp DEFAULT_SEED
_HEARTWALL_FRAMES = 104        # frames in rodinia/data/heartwall/test.avi


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


def _seed(rng: random.Random, *public: int) -> int:
    while True:
        value = rng.randrange(1, 2**31 - 1)
        if value not in public:
            return value


def _graphgen(workdir: Path, gpa_root: Path) -> Path:
    exe = workdir / "graphgen_seeded"
    if not exe.exists():
        src = gpa_root / "scripts" / "graphgen_seeded.cpp"
        subprocess.run(["g++", "-O2", "-std=c++11", "-o", str(exe), str(src)],  # noqa: S603,S607
                       check=True, capture_output=True)
    return exe


def _hotspot_inputs(rng: random.Random, gpa_root: Path, workdir: Path, tag: str) -> tuple[Path, Path]:
    """New 1024x1024 temperature and power files with the public files' value distribution.

    temperature = public temperature + uniform(-0.5, 0.5) K; power = a random permutation of the
    public power values (the hot spots move). Same format (one "%f" value per line).
    """
    data = gpa_root / "rodinia" / "data" / "hotspot"
    temp = [float(x) for x in (data / "temp_1024").read_text().split()]
    power = (data / "power_1024").read_text().split()
    temp_out = workdir / f"temp_1024_{tag}"
    power_out = workdir / f"power_1024_{tag}"
    if not temp_out.exists():
        temp_out.write_text("".join(f"{t + rng.uniform(-0.5, 0.5):f}\n" for t in temp))
    else:
        for _ in temp:  # keep the rng sequence identical whether or not the file is cached
            rng.uniform(-0.5, 0.5)
    rng.shuffle(power)
    if not power_out.exists():
        power_out.write_text("".join(f"{p}\n" for p in power))
    return temp_out, power_out


# H7: bfs generating an 8M-node graph costs ~15 s, too slow for gpa_test's per-call variant check.
# frontier_prepare.sh pre-generates a pool of POOL_SEEDS graphs under frontier_refs/bfs_variants/;
# gpa_test's check (pool=True) picks one, the runner's final variant always uses a fresh seed.
BFS_POOL_SEEDS = (911, 922, 933, 944)


def draw_variant(app: str, seed: int, *, workdir: Path, gpa_root: Path | None = None,
                 pool: bool = False) -> Variant:
    """Draw a random input variant of ``app`` (same shape as the public input, other data).

    Args:
        app: canonical app name (one of VARIANT_APPS)
        seed: variant seed (record it)
        workdir: directory for generated inputs (created; must be outside the app's run dir)
        gpa_root: GPA-Benchmark root (default: this checkout)
        pool: bfs only. Use a pre-generated pooled graph (H7) chosen by ``seed`` instead of
            generating one (for gpa_test's fast per-call check); ignored by other apps.

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
    tag = f"v{int(seed)}"

    if app == "bfs":
        if pool:
            pool_seed = BFS_POOL_SEEDS[seed % len(BFS_POOL_SEEDS)]
            graph = root / "frontier_refs" / "bfs_variants" / f"graph8M_v{pool_seed}.txt"
            params = {"nodes": 8388608, "graph_seed": pool_seed, "pool": True}
            run_command = f"./bfs {graph}"
        else:
            graph_seed = _seed(rng, _PUBLIC_GRAPH_SEED)
            params = {"nodes": 8388608, "graph_seed": graph_seed}
            workdir.mkdir(parents=True, exist_ok=True)
            graph = workdir / f"graph8M_{tag}.txt"
            if not graph.exists():
                exe = _graphgen(workdir, root)
                subprocess.run([str(exe), "8388608", f"8M_{tag}", str(graph_seed)],  # noqa: S603
                               cwd=workdir, check=True, capture_output=True)
            files.append(str(graph))
            run_command = f"./bfs {graph}"
    elif app == "backprop":
        weight_seed = _seed(rng, 7)
        params = {"layer_size": 1048560, "weight_seed": weight_seed}
        run_command = f"./backprop 1048560 {weight_seed}"
    elif app == "b+tree":
        query_seed = _seed(rng, 1)
        params = {"command_file": "command_frontier.txt", "query_seed": query_seed}
        run_command = (f"./b+tree.out file ../data/b+tree/mil.txt command ./command_frontier.txt "
                       f"seed {query_seed}")
    elif app == "heartwall":
        first = rng.randrange(1, _HEARTWALL_FRAMES - 10 + 1)
        params = {"frames": 10, "first_frame": first}
        run_command = f"./heartwall ../data/heartwall/test.avi 10 {first}"
    elif app == "hotspot":
        workdir.mkdir(parents=True, exist_ok=True)
        temp, power = _hotspot_inputs(rng, root, workdir, tag)
        files += [str(temp), str(power)]
        params = {"grid": 1024, "pyramid_height": 5, "sim_time": 100, "data": tag}
        run_command = f"./hotspot 1024 5 100 {temp} {power} output.out"
    elif app == "pathfinder":
        # 300000 = 32 * 9375: keep an odd multiple of 32 within +-10%, never the public count
        k = rng.randrange(8437, 10312, 2)  # odd k: 32 * k in [269984, 329984]
        cols = 32 * (k if k != 9375 else k + 2)
        params = {"cols": cols, "rows": 300, "pyramid_height": 20}
        run_command = f"./pathfinder {cols} 300 20"
    elif app == "nw":
        input_seed = _seed(rng, 7)
        penalty = rng.choice([p for p in range(1, 21) if p != 10])
        params = {"dim": 32768, "penalty": penalty, "input_seed": input_seed}
        run_command = f"./needle 32768 {penalty} {input_seed}"
    elif app == "streamcluster":
        point_seed = _seed(rng, 1)
        params = {"n": 131072, "dim": 256, "clustersize": 16000, "point_seed": point_seed}
        run_command = f"./sc_gpu 10 20 256 131072 131072 16000 none output.txt 1 {point_seed}"
    elif app == "xsbench":
        data_seed = _seed(rng, 42)
        params = {"lookups": 100_000_000, "data_seed": data_seed}
        run_command = f"./XSBench -m event -s large -l 100000000 -d {data_seed}"
        # the stored checksum is for the public data: the reference must come from the
        # pristine run (reference_from_baseline)
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
