#!/bin/bash

module load cuda/12.9.1 nsight-systems

cd /g/g20/davis306/llms4hpc/GPA-Benchmark
source .venv/bin/activate

app_names=("backprop" "b+tree" "huffman" "lavaMD" "lud" "nw" "pathfinder" "lulesh" "xsbench")
for i in "${app_names[@]}"; do
    python -m gpa_bench_driver --srun --timeout 45 --sm-version 90 --cuda-home /usr/tce/packages/cuda/cuda-12.9.1 -l DEBUG --swaps ../gpa-bench-drgpu --nsys --app "$i" -o drgpu-"$i".json
    python -m gpa_bench_driver --srun --timeout 45 --sm-version 90 --cuda-home /usr/tce/packages/cuda/cuda-12.9.1 -l DEBUG --swaps ../gpa-bench-nodr  --nsys --app "$i" -o nodr-"$i".json
done
