#!/bin/bash
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -C gpu
#SBATCH --gpus 1
#SBATCH -A m2404
#SBATCH -o exatensor-profile-all-%j.out
#SBATCH -e exatensor-profile-all-%j.out

module load PrgEnv-nvidia
cd /global/homes/j/jhdavis/scratch/llms4hpc/GPA-Benchmark/ExaTENSOR
srun dcgmi profile --pause

subdirs=("exatensor" "exatensor-opt1" "exatensor-opt2" "exatensor-opt3" "exatensor-opt4")
for ver in "${subdirs[@]}"; do
    cd $ver
    ncu -f -o $ver --set full --import-source=yes ./main
    cd ..
done

srun dcgmi profile --resume
exit
