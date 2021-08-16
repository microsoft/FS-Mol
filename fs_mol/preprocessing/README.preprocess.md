# 1. Query ChEMBL for assays

An instance of ChEMBL needs to be accessible. The config is specified in a config.ini file with structure demonstrated in this directory in utils/config.ini .

    python query.py -h 

Options: 
--assay-list-file  Path to a .json with "assays" field, or csv with "chembl_id" field. The ChEMBL ids will be used to query.
--run-initial-query Runs an initial query of ChEMBL to find all assays with more than 32 datapoints. These are used as the list of assays
--save-dir The output directory in which all the resulting csvs (one for every assay) will be placed. 

Places raw csvs of all queried assays in to the output directory.

# 2. Cleaning raw csvs from ChEMBL

This step is for use after ChEMBL has been queried with query.py

### Run full pipeline end-to-end, using automated thresholding

input_path contains the directory raw/ with all csvs for assays

    python clean.py $input_path

output will be found in $input_path/cleaned/

### Run pipeline without thresholding step

    python clean.py $input_path --stop-step 1 --output-name "_regression_only"

### Pick up from regression step (consume files with only regression values -> convert to thresholded)

a) Run automated thresholding

    python clean.py $input_path --input-dir "cleaned_regression_only/" --output-name "_autothreshold" --start-step 2

b) Used fixed threshold of pKI 5, or 50% inhibition

    python clean.py $input_path --input-dir "cleaned_regression_only/" --output-name "_fixedthreshold" --start-step 2 --fixed-threshold

### Other options

a) Returns only examples with 'hard' labels -- they are outside of the buffer zone around the threshold.

    python clean.py $input_path --input-dir "cleaned_regression_only/" --output-name "_autothreshold" --start-step 2 --hard_only

# 3. Featurise cleaned csvs

    python featurise.py $INPUT_DIR $OUTPUT_DIR --load-metadata $metadata_dir

$INPUT_DIR and $OUTPUT_DIR specify the source directory of cleaned csvs, and the output directory for fully featurised data.

$metadata_dir contains a metadata.pkl.gz file which details how the molecules are to be featurised to graphs. 
This can be custom built for new datasets with new atom types, or to extend the feature set using the featuriser utils.

Other options permit specification of the maximum and minimum assay sizes that will complete featurization, and the permitted class imbalance limits.

Defaults were used in the extraction of the metalearning-dataset-of-molecules. 

# 4. Selection of the TRAIN/TEST/VALIDATION split for meta-learning

This proceeds as in the MetaSplit.ipynb notebook
