#!/bin/bash

source /cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/etc/profile.d/conda.sh
conda activate igwn-py38

job=$1
json=$2
outdir=$3

mkdir -p $outdir

python /home/thomas.callister/RedshiftDistributions/BBH-spin-q-correlations/injection-study/launchBilby.py \
    -job $job \
    -json $json \
    -outdir $outdir

conda deactivate
