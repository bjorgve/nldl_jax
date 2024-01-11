#!/bin/bash

n_nodes=$SLURM_NNODES
n_gpus=$((n_nodes*$SLURM_GPUS_PER_NODE))

CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"

srun \
  -N $n_nodes \
  -n $n_gpus \
  --gpus $n_gpus \
  --cpu-bind=${CPU_BIND} \
  singularity exec \
    --overlay=/project/project_465000096/bjorgve/ebconfigs/overlay.squashfs \
    $SIFJAX \
    bash -c '$WITH_CONDA; python distributed_auto_diff.py'
