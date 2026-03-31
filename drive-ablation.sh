#!/bin/bash

module load cuda/12.9.1 nsight-systems

cd /g/g20/davis306/llms4hpc/GPA-Benchmark
source .venv/bin/activate

app_names=("backprop" "b+tree" "huffman" "lavaMD" "lud" "nw" "pathfinder" "xsbench" "lulesh")
for i in "${app_names[@]}"; do
    python -m gpa_bench_driver --srun --timeout 45 --sm-version 90 --cuda-home /usr/tce/packages/cuda/cuda-12.9.1 -l DEBUG --swaps ../ablation-results/single_profile_metric_selector_ablation --nsys --app "$i" -o single-metric-"$i".json --no-sanitize
done

app_names=("xsbench" "lulesh")
for i in "${app_names[@]}"; do
    python -m gpa_bench_driver --srun --timeout 45 --sm-version 90 --cuda-home /usr/tce/packages/cuda/cuda-12.9.1 -l DEBUG --swaps ../ablation-results/multi_profile_metric_selector_ablation --nsys --app "$i" -o multi-metric-"$i".json --no-sanitize
done

app_names=("xsbench" "lulesh")
for i in "${app_names[@]}"; do
    python -m gpa_bench_driver --srun --timeout 45 --sm-version 90 --cuda-home /usr/tce/packages/cuda/cuda-12.9.1 -l DEBUG --swaps ../ablation-results/profile_selector_ablation --nsys --app "$i" -o profile-"$i".json --no-sanitize
done

app_names=("xsbench" "lulesh")
for i in "${app_names[@]}"; do
    python -m gpa_bench_driver --srun --timeout 45 --sm-version 90 --cuda-home /usr/tce/packages/cuda/cuda-12.9.1 -l DEBUG --swaps ../ablation-results/num_profiles_study --nsys --app "$i" -o num-profiles-"$i".json --no-sanitize
done
