"""pytest setup for the GPA-Benchmark driver tests: make ``gpa_bench_driver`` importable."""

import sys
from pathlib import Path

GPA_ROOT = Path(__file__).resolve().parent.parent
if str(GPA_ROOT) not in sys.path:
    sys.path.insert(0, str(GPA_ROOT))
