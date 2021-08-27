"""
Class to hold the result of a featurisation of a SMILES -> RDkit Mol -> features process.
"""
import sys
from pathlib import Path
from typing import Iterable, List, Any, Dict, Optional

from pyreporoot import project_root

sys.path.insert(0, str(project_root(Path(__file__), root_files="requirements.txt")))

from fs_mol.preprocessing.featurisers.featurisers import (
    AtomFeatureExtractor,
)
from fs_mol.preprocessing.featurisers.molgraph_utils import (
    molecule_to_graph,
    compute_smiles_dataset_metadata,
)


def featurise_mol_data(
    mol_data: Iterable[Dict[str, Any]],
    atom_feature_extractors: List[AtomFeatureExtractor],
) -> Iterable[Dict[str, Any]]:

    """
    Args:
        mol_data: a list of dictionaries representing molecules.
        atom_feature_extractors: list of per-atom feature extractors; graph nodes will
            be labelled by concatenation of their outputs.

    Returns:
        An iterator over molecules converted in to graphs consisting of adjacency lists
        and node features.
    """
    for datapoint in mol_data:
        try:
            datapoint = dict(datapoint)

            datapoint["graph"] = molecule_to_graph(
                datapoint["mol"],
                atom_feature_extractors,
            )

            yield datapoint
        except IndexError:
            print(
                f"Skipping datapoint {datapoint['SMILES']}, cannot featurise with current metadata."
            )
            continue


class FeaturisedData:
    """A tuple to hold the results of featurising a smiles based dataset.

    The class holds four properties about a dataset:
    * atom_feature_extractors: The feature extractors used on the atoms, which
        also store information such as vocabularies used.
    * train_data
    * valid_data
    * test_data
    """

    def __init__(
        self,
        *,
        train_data: Iterable[Dict[str, Any]],
        len_train_data: int,
        valid_data: Iterable[Dict[str, Any]],
        len_valid_data: int,
        test_data: Iterable[Dict[str, Any]],
        len_test_data: int,
        atom_feature_extractors: List[AtomFeatureExtractor],
        featuriser_data: Optional[Iterable[Dict[str, Any]]] = None,
        len_featurizer_data: Optional[int] = None,
        quiet: bool = False,
    ):
        # Store length properties
        self.len_train_data = len_train_data
        self.len_valid_data = len_valid_data
        self.len_test_data = len_test_data

        if featuriser_data is None:
            assert isinstance(
                train_data, list
            ), "If featuriser data is not supplied, then train data must be a list so that it can be iterated over twice."
            featuriser_data = train_data
            len_featurizer_data = len(train_data)

        # WARNING: If no metadata is passed, computes a new vocabulary for
        # every fresh dataset that is seen
        self._atom_feature_extractors = compute_smiles_dataset_metadata(
            mol_data=featuriser_data,
            data_len=len_featurizer_data,
            quiet=quiet,
            atom_feature_extractors=atom_feature_extractors,
        )

        # Do graph featurisation:
        self._train_data = featurise_mol_data(
            mol_data=train_data,
            atom_feature_extractors=self._atom_feature_extractors,
        )
        self._valid_data = featurise_mol_data(
            mol_data=valid_data,
            atom_feature_extractors=self._atom_feature_extractors,
        )
        self._test_data = featurise_mol_data(
            mol_data=test_data,
            atom_feature_extractors=self._atom_feature_extractors,
        )

    @property
    def train_data(self) -> Iterable[Dict[str, Any]]:
        return self._train_data

    @property
    def valid_data(self) -> Iterable[Dict[str, Any]]:
        return self._valid_data

    @property
    def test_data(self) -> Iterable[Dict[str, Any]]:
        return self._test_data

    @property
    def atom_feature_extractors(self) -> List[AtomFeatureExtractor]:
        return self._atom_feature_extractors
