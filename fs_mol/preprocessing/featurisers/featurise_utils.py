import os
import sys
import csv
import argparse
import logging
import functools
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional, Callable

from dpu_utils.utils import RichPath

from rdkit import DataStructs
from rdkit.Chem import (
    Mol,
    RDConfig,
    Descriptors,
    MolFromSmiles,
    rdFingerprintGenerator,
)
from rdkit.Chem.QED import qed
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.Descriptors import ExactMolWt, BertzCT

from pyreporoot import project_root

sys.path.insert(0, str(project_root(Path(__file__), root_files="requirements.txt")))

from fs_mol.preprocessing.featurisers.featurised_data import FeaturisedData
from fs_mol.preprocessing.featurisers.featurisers import AtomFeatureExtractor
from fs_mol.preprocessing.utils.sequential_worker_pool import get_worker_pool

logger = logging.getLogger(__name__)


def get_featurizing_argparser():

    # define a parent parser that includes all options common to all data processing
    parser = argparse.ArgumentParser(
        description="Featurize a molecule dataset from a csv.", add_help=False
    )
    parser.add_argument(
        "INPUT_DIR",
        type=str,
        help="Directory which contains all the csv data components, including any labels.",
    )
    parser.add_argument(
        "OUTPUT_DIR",
        type=str,
        help="Directory which will hold the resulting featurized data.",
    )

    parser.add_argument(
        "--load-metadata",
        dest="load_metadata",
        type=str,
        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../utils/helper_files/"),
        help="Load metadata with featurisers from this directory.",
    )

    parser.add_argument(
        "--min-size",
        type=int,
        default=32,
        help="Smallest assay to permit in final set.",
    )

    parser.add_argument(
        "--max-size",
        dest="max_size",
        type=int,
        default=5000,
        help="Largest assay to permit in final set.",
    )

    parser.add_argument(
        "--balance-limits",
        dest="balance_limits",
        type=float,
        nargs="+",
        default=None,
        help=("Min and Max class imbalance permitted"),
    )

    parser.add_argument(
        "--sapiens-only",
        action="store_true",
        help="Only process those data points that refer to homo sapiens assay targets.",
    )

    # TODO: put context processsing in if required.

    parser.add_argument(
        "--seed", dest="random_seed", type=int, default=0, help="Random seed to use."
    )
    parser.add_argument("--debug", dest="debug", action="store_true", help="Enable debug routines.")
    return parser


def load_csv_assay_data(
    data_file: str,
    csv_format_dict: Dict[str, str],
) -> List[Dict[str, Any]]:

    print(f"Loading data from csv {data_file}.")

    with open(data_file, "rt") as datatable_csv_fh:
        csv_reader = csv.DictReader(datatable_csv_fh, delimiter=",", quotechar='"')
        data = list(csv_reader)

    datapoints = []
    for datapoint in data:
        datapoints.append({x: datapoint[y] for (x, y) in csv_format_dict.items()})

    return datapoints


def featurise_smiles_datapoints(
    *,
    train_data: List[Dict[str, Any]],
    valid_data: List[Dict[str, Any]],
    test_data: List[Dict[str, Any]],
    atom_feature_extractors: List[AtomFeatureExtractor],
    temporary_save_directory: RichPath = None,
    num_processes: int = 8,
    include_fingerprints: bool = False,
    include_descriptors: bool = False,
    include_molecule_stats: bool = False,
    quiet: bool = False,
    filter_failed: bool = False,
) -> FeaturisedData:
    """
    Args:
        train_data: a list of dictionaries representing the training set.
        valid_data: a list of dictionaries representing the validation set.
        test_data: a list of dictionaries representing the test set.
            Note: Each dict must contain a key "SMILES" whose value is a SMILES string
                representing the molecule.
        atom_feature_extractors: list of per-atom feature extractors; graph nodes will
            be labelled by concatenation of their outputs.
        temporary_save_directory: an (optional) directory to cache intermediate results to
            reduce unnecessary recalculation. If used, should be manually cleared if any changes
            have been made to the _smiles_to_mols function.
        num_processes: number of parallel worker processes to use for processing.

    Returns:
        A FeaturisedData tuple.
    """
    tmp_train_path, tmp_test_path, tmp_valid_path = None, None, None
    if temporary_save_directory is not None:
        temporary_save_directory.make_as_dir()
        tmp_train_path = temporary_save_directory.join("train_tmp_feat.pkl.gz")
        tmp_test_path = temporary_save_directory.join("test_tmp_feat.pkl.gz")
        tmp_valid_path = temporary_save_directory.join("valid_tmp_feat.pkl.gz")

    # Step 1: turn smiles into mols:
    logger.info("Turning smiles into mol")
    len_train_data = len(train_data)
    lazy_train_data = _lazy_smiles_to_mols(
        train_data,
        tmp_train_path,
        num_processes,
        include_fingerprints=include_fingerprints,
        include_descriptors=include_descriptors,
        include_molecule_stats=include_molecule_stats,
        filter_failed=filter_failed,
    )
    # Make a copy of the train_data iterator to use in the FeaturisedData class.
    featuriser_data_iter = _lazy_smiles_to_mols(
        train_data,
        tmp_train_path,
        num_processes,
        include_fingerprints=include_fingerprints,
        include_descriptors=include_descriptors,
        include_molecule_stats=include_molecule_stats,
        filter_failed=filter_failed,
    )

    len_valid_data = len(valid_data)
    valid_data_iter = _lazy_smiles_to_mols(
        valid_data,
        tmp_valid_path,
        num_processes,
        include_fingerprints=include_fingerprints,
        include_descriptors=include_descriptors,
        include_molecule_stats=include_molecule_stats,
        filter_failed=filter_failed,
    )
    len_test_data = len(test_data)
    test_data_iter = _lazy_smiles_to_mols(
        test_data,
        tmp_test_path,
        num_processes,
        include_fingerprints=include_fingerprints,
        include_descriptors=include_descriptors,
        include_molecule_stats=include_molecule_stats,
        filter_failed=filter_failed,
    )

    return FeaturisedData(
        train_data=lazy_train_data,
        len_train_data=len_train_data,
        valid_data=valid_data_iter,
        len_valid_data=len_valid_data,
        test_data=test_data_iter,
        len_test_data=len_test_data,
        atom_feature_extractors=atom_feature_extractors,
        featuriser_data=featuriser_data_iter,
        len_featurizer_data=len_train_data,
        quiet=quiet,
    )


def _lazy_smiles_to_mols(
    datapoints: Iterable[Dict[str, Any]],
    save_path: RichPath = None,
    num_processes: int = 8,
    include_fingerprints: bool = True,
    include_descriptors: bool = True,
    include_molecule_stats: bool = True,
    report_fail_as_none: bool = False,
    filter_failed: bool = False,
) -> Iterable[Dict[str, Any]]:
    # Early out if we have already done the work:
    if save_path is not None and save_path.exists():
        datapoints = save_path.read_by_file_suffix()
        logger.info(f"Loaded {len(datapoints)} molecules from {save_path}.")
        return datapoints

    # Turn smiles into mols, extract fingerprint data as well:
    with get_worker_pool(num_processes) as p:
        processed_smiles = p.imap(
            functools.partial(
                _smiles_to_rdkit_mol,
                include_fingerprints=include_fingerprints,
                include_descriptors=include_descriptors,
                include_molecule_stats=include_molecule_stats,
                report_fail_as_none=report_fail_as_none or filter_failed,
            ),
            datapoints,
            chunksize=16,
        )

        for processed_datapoint in processed_smiles:
            if filter_failed and processed_datapoint["mol"] is None:
                print("W: Failed to process {} - dropping".format(processed_datapoint["SMILES"]))
            else:
                yield processed_datapoint


def _smiles_to_rdkit_mol(
    datapoint,
    include_fingerprints: bool = True,
    include_descriptors: bool = True,
    include_molecule_stats: bool = False,
    report_fail_as_none: bool = False,
) -> Optional[Dict[str, Any]]:
    try:
        smiles_string = datapoint["SMILES"]
        rdkit_mol = MolFromSmiles(smiles_string)

        datapoint["mol"] = rdkit_mol

        # Compute fingerprints:
        if include_fingerprints:
            datapoint["fingerprints_vect"] = rdFingerprintGenerator.GetCountFPs(
                [rdkit_mol], fpType=rdFingerprintGenerator.MorganFP
            )[0]
            fp_numpy = np.zeros((0,), np.int8)  # Generate target pointer to fill
            DataStructs.ConvertToNumpyArray(datapoint["fingerprints_vect"], fp_numpy)
            datapoint["fingerprints"] = fp_numpy

        # Compute descriptors:
        if include_descriptors:
            datapoint["descriptors"] = []
            for descr in Descriptors._descList:
                _, descr_calc_fn = descr
                try:
                    datapoint["descriptors"].append(descr_calc_fn(rdkit_mol))
                except Exception:
                    datapoint["failed_to_convert_from_smiles"] = datapoint["SMILES"]

        # Compute molecule-based scores with RDKit:
        if include_molecule_stats:
            datapoint["properties"] = {
                "sa_score": compute_sa_score(datapoint["mol"]),
                "clogp": MolLogP(datapoint["mol"]),
                "mol_weight": ExactMolWt(datapoint["mol"]),
                "qed": qed(datapoint["mol"]),
                "bertz": BertzCT(datapoint["mol"]),
            }

        return datapoint
    except Exception:
        if report_fail_as_none:
            datapoint["mol"] = None
            return datapoint
        else:
            raise


# While the SAScore computation ships with RDKit, it is only in the contrib directory
# and cannot be directly imported. Hence, we need to do a bit of magic to load it,
# and we cache the loaded function in __compute_sascore:
__compute_sascore: Optional[Callable[[Mol], float]] = None


def compute_sa_score(mol: Mol, sascorer_path: Optional[str] = None) -> float:
    global __compute_sascore
    if __compute_sascore is None:
        # Guess path to sascorer in RDKit/Contrib if we are not given a path:
        if sascorer_path is None:
            sascorer_path = os.path.join(
                os.path.normpath(RDConfig.RDContribDir), "SA_Score", "sascorer.py"
            )

        # Now import "sascorer.py" by path as a module, and get the core function out:
        import importlib.util

        spec = importlib.util.spec_from_file_location("sascorer", sascorer_path)
        sascorer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sascorer)
        __compute_sascore = sascorer.calculateScore
    return __compute_sascore(mol)
