#!/bin/bash
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -c 24
#SBATCH --mem-per-cpu=5g
#SBATCH -p gpu
#SBATCH --gpus-per-task=h100:1
#SBATCH -J drive-gpa
#SBATCH -o drive-gpa-%A_%a.out
#SBATCH --array=0-16

module load cuda/gcc/11.3.0/zen2/12.3.0
. ~/scratch/spack/share/spack/setup-env.sh
spack load nvhpc
cd ~/scratch/llms4hpc/GPA-Benchmark
. .venv/bin/activate

driver_srun() {
    srun -n 1 -c 24 --mem-per-cpu=5g --gpus-per-task 1 --exact python -m gpa_bench_driver \
    --app "$1" --nsys --swaps ~/scratch/llms4hpc/gpa-bench-nodr --output-file nodr-"$1".json \
    --no-progress
}

apps=(
    backprop
    bfs
    b+tree
    gaussian
    heartwall
    hotspot
    huffman
    lavaMD
    lud
    nw
    particlefilter
    pathfinder
    srad
    streamcluster
    exatensor
    lulesh
    xsbench
)

if [ -z "${SLURM_ARRAY_TASK_ID+x}" ]; then
    echo "SLURM_ARRAY_TASK_ID is not set."
    exit 1
fi

APP_NAME=${apps[$SLURM_ARRAY_TASK_ID]}

if [ -z "$APP_NAME" ]; then
    echo "Invalid SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

driver_srun "$APP_NAME"

wait

exit
