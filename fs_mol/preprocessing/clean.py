"""
Cleaning ChEMBL assays.

There are three cleaning functions:

1. select_assays -- we currently accept only inhibition and XC50-type measurements

2. standardization -- SMILES are standardized, we convert XC50 measurements to -log10([C]/nM).
Duplicates are dropped if they are identical, or if they differ by a large amount.

3. (Optional) -- thresholding. We current implement two thresholding mechanisms:
fixed and autothreshold. Fixed applies pKX = 5/ 50% as a threshold, autothreshold finds
the median of the data and accepts it as a threshold provided it is in the range pKI 4-6
or 5-7 for enzymes. For specific tasks in a different phase of drug discovery we recommend
implementing a hand-tuned threshold.

The cleaning functions applied can be controlled with start_step and stop_step args,
where the functions are available via the CLEANING_STEPS dictionary.

"""
import os
import sys
import csv
import json
import logging
from dataclasses import dataclass
from glob import glob
from typing import List, Tuple
from pathlib import Path
from multiprocessing import cpu_count, Pool

import pandas as pd
import numpy as np

from pyreporoot import project_root

sys.path.insert(0, str(project_root(Path(__file__), root_files="requirements.txt")))

from fs_mol.preprocessing.utils.cleaning_utils import (
    clean_units,
    log_standard_values,
    standardize_smiles,
    clean_values,
    remove_far_duplicates,
    autothreshold,
    fixedthreshold,
)

logger = logging.getLogger(__name__)


def select_assays(x: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Initial cleaning of all datapoints in an assay/file.

    Removes any points that don't have units in the permitted set:
    standard_unit_set = {"nM", "%", "uM"},
    and converts the standard values to float.
    """

    # first step is to remove anything that doesn't have the approved units
    df = pd.DataFrame(x)
    df.drop(df[df.apply(clean_units, axis=1)].index, inplace=True)

    # make sure standard values are floats
    df["standard_value"] = df["standard_value"].astype(float)

    return df


def standardize(
    x: pd.DataFrame,
    num_workers: int = 6,
    max_mol_weight: float = 900.0,
    **kwargs,
) -> pd.DataFrame:

    """
    Second stage cleaning; SMILES standardization.

    1. desalt, canonicalise tautomers in SMILES
    2. remove > 900 Da molecular weight
    3. get log standard values (e.g. pKI)
    4. remove any repeats with conflicting measurements
    (conflicting is pKI differing by > 1.0)

    """

    # first clean the SMILES
    def parallelize_standardization(df):
        data_splits = np.array_split(df, num_workers)
        with Pool(num_workers) as p:
            df = pd.concat(p.map(standardize_smiles, data_splits))
        return df

    df = parallelize_standardization(x)

    # drop NaN standard values
    df.drop(df[df.apply(clean_values, axis=1)].index, inplace=True)

    # drop anything that the molecular weight is too high
    df = df.loc[df.molecular_weight <= max_mol_weight]

    # remove duplicates
    # first need to just keep one of the duplicates if smiles and value are *exactly* the same
    df = df.drop_duplicates(subset=["canonical_smiles", "standard_value"], keep="first")

    if len(df) > 0:

        # get log standard values -- need to convert uM first
        df.loc[(df["standard_units"] == "uM"), "standard_value"] *= 1000
        df.loc[(df["standard_units"] == "uM"), "standard_units"] = "nM"

        if df["standard_units"].iloc[0] != "%":
            df["log_standard_value"] = df.apply(log_standard_values, axis=1)
        else:
            df["log_standard_value"] = float("NaN")
        df.loc[(df["standard_units"] == "%"), "log_standard_value"] = float("NaN")

        # now drop duplicates if the smiles are the same and the values are outside of a threshold
        # close measurements are just noisy measurements of the same thing
        # NOTE: this currently scales badly so we only apply it to smaller dataframes,
        # larger assays are removed from the dataset in later stages.
        if len(df) < 5000:
            df = remove_far_duplicates(df)

        df["max_num_atoms"] = df.num_atoms.max()
        df["max_molecular_weight"] = df.molecular_weight.max()

    return df


def apply_thresholds(
    x: pd.DataFrame,
    hard_only: bool = False,
    automate_threshold: bool = True,
    **kwargs,
) -> pd.DataFrame:

    """
    Thresholding to obtain binary labels.

    """

    if automate_threshold:
        df, threshold = autothreshold(x)
    else:
        df, threshold = fixedthreshold(x)

    # convert activity classes to binary labels
    if hard_only:
        df = df[df.activity_string.isin(["active", "inactive"])]
    else:
        df = df[df.activity_string.isin(["active", "inactive", "weak active", "weak inactive"])]

    df.loc[df["activity_string"] == "active", "activity"] = 1.0
    df.loc[df["activity_string"] == "weak active", "activity"] = 1.0
    df.loc[df["activity_string"] == "inactive", "activity"] = 0.0
    df.loc[df["activity_string"] == "weak inactive", "activity"] = 0.0

    df["threshold"] = threshold

    return df


CLEANING_STEPS = {
    0: select_assays,
    1: standardize,
    2: apply_thresholds,
}

DEFAULT_CLEANING = {
    "hard_only": False,
    "automate_threshold": True,
    "num_workers": cpu_count(),
}


@dataclass(frozen=True)
class OutputSummary:
    chembl_id: str
    target_id: str
    assay_type: str
    assay_organism: str
    raw_size: int
    cleaned_size: int
    cleaning_failed: str
    cleaning_size_delta: int
    num_pos: int
    percentage_pos: float
    max_mol_weight: float
    threshold: float
    max_num_atoms: int
    confidence_score: int
    standard_units: str


def get_argparser():

    import argparse

    parser = argparse.ArgumentParser(
        description="First pass cleaning ChEMBL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    help_string = (
        "Directory containing the raw/ directory of csv files."
        " Cleaned data will be placed here under cleaned_(cleaning_pass)/"
    )
    parser.add_argument("BASE_PATH", type=str, help=help_string)

    parser.add_argument(
        "--input-dir",
        dest="input_dir",
        type=str,
        default="raw/",
        help="Directory under $BASE_PATH containing the input data to be cleaned.",
    )

    parser.add_argument(
        "--output-suffix",
        dest="output_suffix",
        type=str,
        default="",
        help="Suffix to directory to save in $BASE_PATH/cleaned$output_name.",
    )

    parser.add_argument(
        "--assays",
        dest="assays",
        nargs="+",
        type=str,
        default=None,
        help="Select assays to process by CHEMBL[ID] (e.g. CHEMBL999)",
    )

    parser.add_argument(
        "--hard-only",
        action="store_true",
        help="Only keep data with hard labels.",
    )

    parser.add_argument(
        "--fixed-threshold",
        action="store_true",
        help="Use the fixed threshold only.",
    )

    parser.add_argument(
        "--max-mol-weight",
        dest="max_mol_weight",
        type=float,
        default=900.0,
        help="Maximum allowed molecular weight in Da",
    )

    parser.add_argument(
        "--num-workers",
        dest="num_workers",
        type=int,
        default=None,
        help="Some tasks are parallelised: default workers is number cores available.",
    )

    parser.add_argument(
        "--confidence-lookup",
        action="store_true",
        help="If used, will look up each assay's confidence score from the utils/helper_files/confidence_lookup.json.",
    )

    return parser


def clean_assay(df: pd.DataFrame, assay: str) -> Tuple[pd.DataFrame, OutputSummary]:

    # remove index if it was saved with this file (back compatible)
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    original_size = len(df)

    failed = False
    try:
        print(f"Processing {assay}.")
        df = select_assays(df, **DEFAULT_CLEANING)
        df = standardize(df, **DEFAULT_CLEANING)
        df = apply_thresholds(df, **DEFAULT_CLEANING)
    except Exception as e:
        df = None
        logger.warning(f"Failed cleaning on {assay} : {e}")
        failed = True

    if df is None or len(df) == 0:
        logger.warning(f"Assay {assay} was empty post cleaning.")
        failed = True

    assay_dict = {}
    if failed:
        assay_dict = {
            "chembl_id": assay,
            "target_id": "NaN",
            "assay_type": "NaN",
            "assay_organism": "NaN",
            "raw_size": "NaN",
            "cleaned_size": 0,
            "cleaning_failed": str(True),
            "cleaning_size_delta": "NaN",
            "num_pos": "NaN",
            "percentage_pos": "NaN",
            "max_mol_weight": "NaN",
            "threshold": "NaN",
            "max_num_atoms": "NaN",
            "confidence_score": "NaN",
            "standard_units": "NaN",
        }

    else:
        target_id = df.iloc[0]["target_id"] if "target_id" in df.columns else None

        organism = None if df.iloc[0]["assay_organism"] == "nan" else df.iloc[0]["assay_organism"]
        assay_dict = {
            "chembl_id": assay,
            "target_id": target_id,
            "assay_type": df.iloc[0]["assay_type"],
            "assay_organism": organism,
            "raw_size": original_size,
            "cleaned_size": len(df),
            "cleaning_failed": str(failed),
            "cleaning_size_delta": original_size - len(df),
            "num_pos": df["activity"].sum(),
            "percentage_pos": df["activity"].sum() * 100 / len(df),
            "max_mol_weight": df.iloc[0]["max_molecular_weight"],
            "threshold": df.iloc[0]["threshold"],
            "max_num_atoms": df.iloc[0]["max_num_atoms"],
            "confidence_score": df.iloc[0]["confidence_score"],
            "standard_units": df.iloc[0]["standard_units"],
        }

    return df, OutputSummary(**assay_dict)


def process_all_assays(
    files_to_process: List[str],
    output_dir: str,
    basepath: str,
    use_confidence_lookup: bool = False,
) -> None:

    summary_file = os.path.join(output_dir, "summary.csv")

    logger.info(f"{len(files_to_process)} files remaining to process.")

    # confidence score lookup (if required for this particular pre-queried data)
    if use_confidence_lookup:
        with open(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "utils/helper_files/confidence_lookup.json",
            ),
            "r",
        ) as jsonfile:
            confidence_lookup = json.load(jsonfile)

    summaries = []
    for i, assay_file in enumerate(files_to_process):
        assay = os.path.basename(assay_file).split(".")[0]
        logger.info(f"Processing {i}: {assay}.")
        try:
            df = pd.read_csv(assay_file)
            if len(df) == 0:
                logger.warning(f"Loaded empty assay: {assay}")
            if "confidence_score" not in df.columns:
                df["confidence_score"] = confidence_lookup[assay]
            df, summary = clean_assay(df, assay)
            logger.info(f"Assay {assay} saving to output directory.")
            if df is not None and len(df) > 0:
                df.to_csv(
                    os.path.join(output_dir, os.path.basename(assay_file)),
                    index=False,
                )
            summaries.append(summary)
        except Exception as e:
            logger.warning(f"failed to clean assay: {e}")
            continue

    with open(summary_file, "a+", newline="") as csv_file:
        fieldnames = [
            "chembl_id",
            "target_id",
            "assay_type",
            "assay_organism",
            "raw_size",
            "cleaned_size",
            "cleaning_failed",
            "cleaning_size_delta",
            "num_pos",
            "percentage_pos",
            "max_mol_weight",
            "threshold",
            "max_num_atoms",
            "confidence_score",
            "standard_units",
        ]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

        for summary in summaries:
            csv_writer.writerow(
                {
                    "chembl_id": summary.chembl_id,
                    "target_id": summary.target_id,
                    "assay_type": summary.assay_type,
                    "assay_organism": summary.assay_organism,
                    "raw_size": summary.raw_size,
                    "cleaned_size": summary.cleaned_size,
                    "cleaning_failed": summary.cleaning_failed,
                    "cleaning_size_delta": summary.cleaning_size_delta,
                    "num_pos": summary.num_pos,
                    "percentage_pos": summary.percentage_pos,
                    "max_mol_weight": summary.max_mol_weight,
                    "threshold": summary.threshold,
                    "max_num_atoms": summary.max_num_atoms,
                    "confidence_score": summary.confidence_score,
                    "standard_units": summary.standard_units,
                }
            )


def get_files_to_process(input_dir: str, output_dir: str) -> List[str]:
    all_raw_files = glob(os.path.join(input_dir, "CHEMBL*.csv"), recursive=True)
    all_done_assays = set(
        [
            os.path.basename(x).split(".")[0]
            for x in glob(os.path.join(output_dir, "CHEMBL*.csv"), recursive=True)
        ]
    )
    files_to_process = []
    for f in all_raw_files:
        assay = os.path.basename(f).split(".")[0]
        if assay not in all_done_assays:
            files_to_process.append(f)
    return files_to_process


def clean_directory(args):

    basepath = args.BASE_PATH

    if args.hard_only:
        DEFAULT_CLEANING.update({"hard_only": True})
    if args.fixed_threshold:
        DEFAULT_CLEANING.update({"automate_threshold": False})
    if args.num_workers is not None:
        DEFAULT_CLEANING.update({"num_workers": args.num_workers})

    DEFAULT_CLEANING.update({"max_mol_weight": args.max_mol_weight})

    input_dir = os.path.join(basepath, args.input_dir)
    output_dir = os.path.join(basepath, f"cleaned{args.output_suffix}/")
    os.makedirs(output_dir, exist_ok=True)

    # do not repeat cleaning on already cleaned files.

    files_to_process = get_files_to_process(input_dir, output_dir)

    if args.assays is not None:
        filenames = [os.path.join(input_dir, f"{x}.csv") for x in args.assays]
        files_to_process = set(files_to_process).intersection(set(filenames))

    print(f"Processing {len(files_to_process)}.")
    process_all_assays(
        files_to_process, output_dir, basepath, use_confidence_lookup=args.confidence_lookup
    )


def run():

    parser = get_argparser()
    args = parser.parse_args()

    clean_directory(args)


if __name__ == "__main__":

    run()
