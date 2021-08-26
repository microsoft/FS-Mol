"""
Script to featurize ChEMBL assays.

Each assay should consist of a separate csv file with fields for the smiles, molecule id,
assay id and binary activity label.

Expected csv fields are given in CHEMBL_CSV_FORMAT

This also requires a metadata.pkl.gz with molecule featurizers
to allow graph featurization with pre-specified feature set.

train-test-splits are not performed within individual assays.

"""
import os
import sys
import logging
import pandas as pd
from glob import glob
from typing import List
from pathlib import Path

from dpu_utils.utils import run_and_debug, RichPath

from pyreporoot import project_root

sys.path.insert(0, str(project_root(Path(__file__), root_files="requirements.txt")))

from fs_mol.preprocessing.featurisers.featurise_utils import (
    get_featurizing_argparser,
    load_csv_assay_data,
    featurise_smiles_datapoints,
)
from fs_mol.preprocessing.utils.save_utils import (
    save_assay_data,
    save_metadata,
)
from fs_mol.utils.logging import set_up_logging

set_up_logging("featurisation.log")

logger = logging.getLogger(__name__)

# these features are in common across all ChEMBL assays from
# the original query
CHEMBL_CSV_FORMAT = {
    "SMILES": "smiles",
    "Property": "activity",
    "Assay_ID": "chembl_id",
    "RegressionProperty": "standard_value",
    "LogRegressionProperty": "log_standard_value",
    "Relation": "standard_relation",
    "AssayType": "assay_type",
}


def get_filenames(input_dir: str) -> List[str]:
    logger.info(f"Reading files from {input_dir}")
    return glob(os.path.join(input_dir, "CHEMBL*.csv"), recursive=True)


def filter_assays(summary: str, args) -> List[str]:

    """
    Perform a filter on assays using the summary csv
    that results from the cleaning steps.
    """

    sdf = pd.read_csv(summary)

    max_size = args.max_size
    if max_size is None:
        max_size = max(sdf["cleaned_size"])

    filter_balance = args.balance_limits
    if filter_balance is None:
        filter_balance = (0.0, 100.0)

    sdf = sdf.loc[
        (sdf["cleaned_size"] >= args.min_size)
        & (sdf["percentage_pos"] >= min(filter_balance))
        & (sdf["percentage_pos"] <= max(filter_balance))
        & (sdf["cleaned_size"] <= max_size)
    ]

    # please note this syntax breaks if pandas version < 1.2.4
    if args.sapiens_only:
        sdf = sdf.loc[sdf["assay_organism"].str.contains("sapiens", regex=False, na=False)]

    return sdf["chembl_id"].tolist()


def run(args):
    # get all the relevant raw datafiles to process
    filenames = get_filenames(args.INPUT_DIR)

    # load a summary file from the output of assay cleaning.
    summary_file = os.path.join(args.INPUT_DIR, "summary.csv")
    # if a summary file exists, use it to select files that are large enough and
    # pass the imbalance thresholds
    if os.path.exists(summary_file):
        assays_to_process = filter_assays(summary_file, args)
        files_to_process = []
        for f in filenames:
            assay = os.path.basename(f).split(".")[0]
            if assay in assays_to_process:
                files_to_process.append(f)
        logging.info(f"{len(files_to_process)} files passed filtering.")
    else:
        files_to_process = filenames
        logging.info(f"No filtering, processing all {len(files_to_process)} files.")

    # Metadata has to be loaded, as individual assay files are too small to
    # define the full set of atoms likely to be seen
    if args.load_metadata:

        print(f"Loading metadata from dir {args.load_metadata}")
        metapath = RichPath.create(args.load_metadata)
        path = metapath.join("metadata.pkl.gz")
        metadata = path.read_by_file_suffix()
        atom_feature_extractors = metadata["feature_extractors"]

    else:
        raise ValueError(
            "Metadata must be loaded for this processing, please supply "
            "directory containing metadata.pkl.gz."
        )

    # featurize and save data
    assays = set()
    failed_assays = set()
    featurised_data = None
    for filename in files_to_process:

        datapoints = load_csv_assay_data(
            filename,
            CHEMBL_CSV_FORMAT,
        )
        logger.info(f"{len(datapoints)} datapoints loaded.")
        assays.add(datapoints[0]["Assay_ID"])

        logger.info(f"Featurising data...")
        try:
            featurised_data = featurise_smiles_datapoints(
                train_data=datapoints,
                valid_data=datapoints[0],
                test_data=datapoints[0],
                atom_feature_extractors=atom_feature_extractors,
                num_processes=-1,
                include_descriptors=True,
                include_fingerprints=True,
            )
            logger.info(f"Completed featurization; saving data now.")

            save_assay_data(featurised_data, assay_id=filename, output_dir=args.OUTPUT_DIR)

        except IndexError:
            assay = datapoints[0]["Assay_ID"]
            failed_assays.add(assay)
            logger.info(f"Error in featurisation found for assay {assay}")
            continue

    # save out the metadata including now the properties/assays in this dataset
    properties_metadata = {"property_names": list(assays)}
    if featurised_data:
        save_metadata(
            featurised_data,
            output_dir=args.OUTPUT_DIR,
            extra_metadata=properties_metadata,
            failed=list(failed_assays),
        )


if __name__ == "__main__":
    parser = get_featurizing_argparser()
    args = parser.parse_args()
    # try:
    run_and_debug(lambda: run(args), args.debug)
    # except:
    #     raise ValueError("This data requires metadata to be passed.")
