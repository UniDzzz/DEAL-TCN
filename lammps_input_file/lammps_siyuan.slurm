#!/bin/bash
#SBATCH --job-name=MoOS
#SBATCH --partition=64c512g
#SBATCH -N 1
#SBATCH --ntasks-per-node=64
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module load lammps/20230328-intel-2021.4.0-omp

mpirun lmp -i in.MoO3S
