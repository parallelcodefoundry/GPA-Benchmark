#!/bin/bash

module load cuda/12.9.1 nsight-systems

cd /g/g20/davis306/llms4hpc/GPA-Benchmark
source .venv/bin/activate

study_names=("all_selectors_ablation" "file_profile_selector_ablation" "file_selector_ablation" "no_ablation")
app_names=("lulesh" "xsbench")
for j in "${study_names[@]}"; do
    for i in "${app_names[@]}"; do
        python -m gpa_bench_driver --no-sanitizer --srun --timeout 120 --sm-version 90 --cuda-home /usr/tce/packages/cuda/cuda-12.9.1 -l DEBUG --swaps ../study_1/"$j"  --nsys --app "$i" -o "$j"-"$i".json
    done
done
