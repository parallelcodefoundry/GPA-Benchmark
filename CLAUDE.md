# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPA-Benchmark is a driver framework for the GPU Performance Advisor (GPA). It automates the workflow of building, running, validating, profiling, and testing optimized code variants for GPU benchmark applications (Rodinia suite, LULESH, XSBench, ExaTENSOR, PeleC, Quicksilver, etc.).

## Commands

### Install / Setup
```bash
pip install -e .          # Install the driver package in editable mode
```

### Run the Driver
```bash
# Run all apps (build, sanitize, run, validate)
python -m gpa_bench_driver

# Run a specific app
python -m gpa_bench_driver --app lulesh
python -m gpa_bench_driver --app backprop

# Build only
python -m gpa_bench_driver --app xsbench --build-only

# Profile with Nsight Systems or Nsight Compute
python -m gpa_bench_driver --app lulesh --nsys
python -m gpa_bench_driver --app lulesh --ncu

# Test optimized code variants from a swaps directory
python -m gpa_bench_driver --app lulesh --swaps /path/to/swaps --ncu

# Use Slurm srun with timeout
python -m gpa_bench_driver --app lulesh --srun --timeout 45

# Skip sanitizer, specify SM version, save results to JSON
python -m gpa_bench_driver --app lulesh --no-sanitize --sm-version 90 --output-file results.json
```

Key flags:
- `--app` — app name or alias (see `driver_apps.yaml`); default `all`
- `--sm-version` — CUDA SM version (auto-detected from `nvidia-smi` if omitted)
- `--swaps` — directory of `.cu` swap files to test as optimization variants
- `--detect-regions` — only replace `>>> START EDITABLE REGION ... <<< END EDITABLE REGION` blocks instead of the entire file
- `--num-samples` — number of times to repeat run/profile for averaging
- `--srun` — prepend `srun` to all commands (for Slurm clusters)
- `--timeout` — per-command timeout in seconds
- `--no-sanitize` — skip GPU memory sanitizer
- `--output-file` — write results JSON to this path
- `--log-level` — DEBUG/INFO/WARNING/ERROR/CRITICAL

### Linting
```bash
ruff check .
ruff format .
```

### Utility Scripts
```bash
# Merge multiple JSON result files into one
python merge_driver_results.py result1.json result2.json ... -o merged.json

# Strip stdout/stderr from a results JSON to reduce file size
python condense_json.py input.json -o condensed.json
```

### Batch Driver Scripts
```bash
# Run full optimization study (DRGPU and NoDirectionality variants)
./drive-opt.sh

# Run ablation study (selector mechanism variants)
./drive-ablation-study-1.sh
```

## Architecture

### Workflow

The driver executes a pipeline of `Operation` enum steps:

`BUILD → SANITIZE → RUN → VALIDATE → NSYS_PROFILE → NCU_PROFILE → SWAP_BUILD → SWAP_RUN → SWAP_VALIDATE → SWAP_NCU_PROFILE`

Swap operations repeat the build/run/validate/profile cycle for each code variant found in the `--swaps` directory.

### Module Layout

```
gpa_bench_driver/
├── __main__.py                  # Entry point
├── gpa_bench_driver.py          # Orchestrator: CLI parsing, progress bar, main loop
└── driver_src/
    ├── driver_config.py         # YAML loading, op pipeline determination, env setup
    ├── driver_models.py         # Dataclasses (DriverConfig, AppResults, SwapConfig) and Operation enum
    ├── driver_operations.py     # build_app(), run_app(), sanitize_app()
    ├── driver_profiling.py      # nsys_profile_app(), ncu_profile_app(), SQLite postprocessing
    ├── driver_file_swapping.py  # swap_file_in_app(), swap_file_out_app(), editable region logic
    ├── driver_validation.py     # validate_app() with 4 strategies (text check, reference diff, float grep, window)
    ├── driver_reporting.py      # print_report_table(), save_results()
    └── driver_utils.py          # SubprocessRunner (timeout/truncation), CUDA/SM detection, path helpers
```

### Key Design Decisions

- **App config is data-driven**: All app metadata (build/run commands, kernel names, validation strategy, NCU args) lives in `driver_apps.yaml`. Adding a new benchmark only requires a new YAML entry.
- **Temporary working directories**: The driver copies app source into a temp dir before swapping files, ensuring originals are never permanently modified.
- **File swapping for optimization testing**: Swap variant files follow the naming convention `run_<N>_optimized_code_<M>.cu` in a directory named after the app. Editable-region markers (`>>> START EDITABLE REGION` / `<<< END EDITABLE REGION`) allow partial replacement within a file.
- **SubprocessRunner** (`driver_utils.py`) is the single execution primitive: handles Slurm `srun` integration, timeout enforcement (via multiprocessing), and middle-truncation of large stdout/stderr to stay within char limits.
- **Programmatic API**: `run_driver()` in `gpa_bench_driver.py` allows calling the driver from Python without the CLI.

### Benchmark Applications

All GPU benchmark apps are git submodules. Rodinia apps live under `rodinia/`. All Makefiles accept `SM_VERSION=<xx>` as a build parameter. The `get_data.sh` script downloads required Rodinia input data files.
