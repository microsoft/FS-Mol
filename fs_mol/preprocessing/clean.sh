#!/bin/bash -x

# quick script to clean up assays, and do processing to graph form
mntPath="/mnt/genchemdata/preprocessed-data/metamol/metamol/"

# then do one pass with automated thresholds
python clean.py $mntPath --input-dir "raw/" --output-name "_autothreshold_new"

# DO THE ASSAY-TO-GRAPH PROCESSING
metadata="/home/megstanley/FS-Mol/fs_mol/preprocessing/"
input="/mnt/genchemdata/preprocessed-data/metamol/metamol/cleaned_autothreshold_new/"
output="/mnt/genchemdata/preprocessed-data/metamol/metamol/processed_autothreshold_new/"
# python featurize.py $input $output --load-metadata $metadata --min-size 32 --max-size 5000 --balance-limits 30.0 70.0