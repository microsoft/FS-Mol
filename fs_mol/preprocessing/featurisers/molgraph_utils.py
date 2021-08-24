"""
Utils for converting an rdkit mol into a graph: a dict containing node features and
adjacency lists.
"""
import sys
import tqdm
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional, NamedTuple, Tuple

# rdkit imports
from rdkit import Chem
from rdkit.Chem import (
    Mol,
    rdmolops,
    MolToSmiles,
)

from pyreporoot import project_root

sys.path.insert(0, str(project_root(Path(__file__), root_files="requirements.txt")))

from fs_mol.preprocessing.featurisers.featurisers import (
    AtomFeatureExtractor,
    get_default_atom_featurisers,
)
from fs_mol.preprocessing.featurisers.rdkit_helpers import get_atom_symbol

logger = logging.getLogger(__name__)

# Note: we do not need to consider aromatic bonds because all molecules have been Kekulized:
# All of the aromatic bonds are converted into either single or double bonds, but the
# "IsAromatic" flag for the bond in unchanged.
BOND_DICT = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2}


class NodeFeatures(NamedTuple):
    """Tuple for holding the return value of `featurise_atoms`."""

    real_valued_features: List[np.ndarray]
    categorical_features: Optional[List[int]]
    num_categorical_classes: Optional[int]


def featurise_atoms(
    mol: Mol,
    atom_feature_extractors: List[AtomFeatureExtractor],
) -> NodeFeatures:
    """Computes features (real values, and possibly also categorical) for all atoms.

    Args:
        mol: the molecule to be processed.
        atom_feature_extractors: list of atom feature extractors.

    Returns:
        NamedTuple, containing node features, and optionally also node classes (i.e. additional node
        features expressed as categorical ids).
    """

    all_atom_class_ids = None
    num_atom_classes = None

    all_atom_features = []

    for atom in mol.GetAtoms():
        atom_features = [
            atom_featuriser.featurise(atom) for atom_featuriser in atom_feature_extractors
        ]

        atom_features = np.concatenate(atom_features).astype(np.float32)
        all_atom_features.append(atom_features)

    return NodeFeatures(
        real_valued_features=all_atom_features,
        categorical_features=all_atom_class_ids,
        num_categorical_classes=num_atom_classes,
    )


def compute_smiles_dataset_metadata(
    mol_data: Iterable[Dict[str, Any]],
    data_len: Optional[int] = None,
    quiet: bool = False,
    atom_feature_extractors: Optional[List[AtomFeatureExtractor]] = None,
) -> List[AtomFeatureExtractor]:
    """
    Given a dataset of molecules, compute metadata (such as atom featuriser vocabularies).

    Note: this should only be used if there is a large dataset with all possible required
    features represented -- otherwise atom_feature_extractors pre-initialized in a metadata
    file should be passed.
    """

    if atom_feature_extractors is None:
        uninitialised_featurisers = get_default_atom_featurisers()
        atom_feature_extractors = uninitialised_featurisers
    else:
        uninitialised_featurisers = [
            featuriser
            for featuriser in atom_feature_extractors
            if not featuriser.metadata_initialised
        ]

    if len(uninitialised_featurisers) == 0:
        logger.info("Using feature extractors.")
        return atom_feature_extractors

    logger.info("Initialising feature extractors.")

    for datapoint in tqdm(mol_data, total=data_len, disable=quiet):
        mol = datapoint["mol"]

        for atom in mol.GetAtoms():
            for featuriser in uninitialised_featurisers:
                featuriser.prepare_metadata(atom)

    for featuriser in uninitialised_featurisers:
        featuriser.mark_metadata_initialised()

    return atom_feature_extractors


def _need_kekulize(mol: Mol) -> bool:
    """Return whether the given molecule needs to be kekulized."""
    bonds = mol.GetBonds()
    bond_types = [str(bond.GetBondType()) for bond in bonds]
    return any(bond_type == "AROMATIC" for bond_type in bond_types)


def molecule_to_adjacency_lists(mol: Mol) -> List[List[Tuple[int, int]]]:
    """Converts an RDKit molecule to set of list of adjacency lists

    Args:
        mol: the rdkit.ROMol (or RWMol) to be converted.

    Returns:
        A list of lists of edges in the molecule.

    Raises:
        KeyError if there are any aromatic bonds in mol after Kekulization.
    """
    # Kekulize it if needed.
    if _need_kekulize(mol):
        rdmolops.Kekulize(mol)
        if mol is None:
            return None

    # Remove stereo information, such as inward and outward edges
    Chem.RemoveStereochemistry(mol)

    adjacency_lists: List[List[Tuple[int, int]]] = [[] for _ in range(len(BOND_DICT))]
    bonds = mol.GetBonds()
    for bond in bonds:
        bond_type_idx = BOND_DICT[str(bond.GetBondType())]
        adjacency_lists[bond_type_idx].append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    return adjacency_lists


def molecule_to_graph(
    mol: Mol,
    atom_feature_extractors: List[AtomFeatureExtractor],
):
    """Converts an RDKit molecule to an encoding of nodes and edges.

    Args:
        mol: the rdkit.ROMol (or RWMol) to be converted.
        atom_feature_extractors: list of per-atom feature extractors; graph nodes will
            be labelled by concatenation of their outputs.

    Returns:
        A dict: {node_labels, node_features, adjacency_list, node_masks)
        node_labels is a string representation of the atom type
        node_features is a vector representation of the atom type.
        adjacency_list is a list of lists of edges in the molecule.

    Raises:
        ValueError if the given molecule cannot be successfully Kekulized..

    """
    if mol is None:
        return None

    # Kekulize it if needed.
    if _need_kekulize(mol):
        rdmolops.Kekulize(mol)
        # Check that there are no aromatic bonds left, fail if there are:
        if _need_kekulize(mol):
            raise ValueError(
                f"Given molecule cannot be Kekulized successfully. "
                f"Molecule has smiles string:\n{MolToSmiles(mol)}"
            )
        if mol is None:
            return None

    # Remove stereo information, such as inward and outward edges
    Chem.RemoveStereochemistry(mol)

    # Calculate the edge information
    adjacency_lists = molecule_to_adjacency_lists(mol)

    graph: Dict[str, List[Any]] = {
        "adjacency_lists": adjacency_lists,
        "node_types": [],
        "node_features": [],
    }

    # Calculate the node information
    for atom in mol.GetAtoms():
        graph["node_types"].append(get_atom_symbol(atom))

    node_features = featurise_atoms(mol, atom_feature_extractors)

    graph["node_features"] = [
        atom_features.tolist() for atom_features in node_features.real_valued_features
    ]

    if node_features.num_categorical_classes is not None:
        graph["node_categorical_features"] = node_features.categorical_features
        graph["node_categorical_num_classes"] = node_features.num_categorical_classes

    return graph
