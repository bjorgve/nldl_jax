#!/bin/bash

srun \
  -N 1 \
  -n 1 \
  --gpus 8 \
  singularity exec \
    --overlay=/project/project_465000096/bjorgve/ebconfigs/overlay.squashfs \
    $SIFJAX \
    bash -c '$WITH_CONDA; python single_node_auto_diff.py'
