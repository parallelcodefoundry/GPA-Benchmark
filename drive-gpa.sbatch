#!/bin/bash
#SBATCH -n 3
#SBATCH -t 02:00:00
#SBATCH -c 24
#SBATCH -p gpu
#SBATCH --gpus-per-task=h100:1
#SBATCH -J drive-gpa
#SBATCH -o drive-gpa-%j.out

module load cuda/gcc/11.3.0/zen2/12.3.0
. ~/scratch/spack/share/spack/setup-env.sh
spack load nvhpc
cd ~/scratch/llms4hpc/GPA-Benchmark
. .venv/bin/activate

driver_srun() {
    srun -n 1 -c 24 --gpus-per-task 1 --exact ./driver.py --app "$1" --nsys --swaps ~/scratch/llms4hpc/gpa-bench-nodr --output-file nodr-"$1".json --no-progress &
}

#driver_srun backprop
#driver_srun bfs
driver_srun b+tree
#driver_srun gaussian
#driver_srun heartwall
#driver_srun hotspot
#driver_srun huffman
#driver_srun lavaMD
driver_srun lud
#driver_srun nw
#driver_srun particlefilter
#driver_srun pathfinder
#driver_srun srad
driver_srun streamcluster
#driver_srun exatensor
#driver_srun lulesh
#driver_srun xsbench

wait

exit
