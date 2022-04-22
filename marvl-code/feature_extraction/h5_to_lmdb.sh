#!/bin/bash

LANG="$1"
H5="/home/karoui/datasets/marvl/features/marvl-${LANG}_boxes36.h5"
LMDB="/home/karoui/datasets/marvl/features/marvl-${LANG}_boxes36.lmdb"

source ~/miniconda3/etc/profile.d/conda.sh

conda activate volta

python h5_to_lmdb.py --h5 $H5 --lmdb $LMDB

conda deactivate
