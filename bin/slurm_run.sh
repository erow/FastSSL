#!/bin/bash
# Parameters
#SBATCH --cpus-per-task=10
#SBATCH --error=outputs/slurm/%j_%t_log.err
#SBATCH --gpus-per-node=8
#SBATCH --job-name=run
#SBATCH --mem=400GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --open-mode=append
#SBATCH --output=outputs/slurm/%j_%t_log.out

ARGS=${@}
torchrun --nproc_per_node=8 $ARGS
