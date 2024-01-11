#!/bin/bash -e
#SBATCH --job-name=single_node_auto_diff
#SBATCH --account=project_465000096
#SBATCH --time=00:04:00
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH -o single_node_auto_diff.out

ml purge
module load LUMI/23.09
module load jax

srun \
  -N 1 \
  -n 1 \
  --gpus 8 \
  singularity exec \
    --overlay=/project/project_465000096/bjorgve/ebconfigs/overlay.squashfs \
    $SIFJAX \
    bash -c '$WITH_CONDA; python single_node_auto_diff.py'
