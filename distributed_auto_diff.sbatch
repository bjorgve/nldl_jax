#!/bin/bash -e
#SBATCH --job-name=distributed_auto_diff
#SBATCH --account=project_465000096
#SBATCH --time=00:04:00
#SBATCH --partition=standard-g
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH -o distributed_auto_diff.out

ml purge
module load LUMI/23.09
module load jax

CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"
# export MPICH_GPU_SUPPORT_ENABLED=1


n_nodes=$SLURM_NNODES
n_gpus=$((n_nodes*$SLURM_GPUS_PER_NODE))
n_tasks=$n_gpus

srun \
  -N $n_nodes \
  -n $n_gpus \
  --cpu-bind=${CPU_BIND} \
  --gpus $n_gpus \
  singularity exec \
    --overlay=/project/project_465000096/bjorgve/ebconfigs/overlay.squashfs \
    $SIFJAX \
    bash -c '$WITH_CONDA; python distributed_auto_diff.py'
