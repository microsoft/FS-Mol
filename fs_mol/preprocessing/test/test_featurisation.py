import os
import sys
import pytest
from typing import Dict

from dpu_utils.utils import RichPath
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root("fs_mol")))
from preprocessing.featurise import load_csv_assay_data
from preprocessing.featurisers.featurise_utils import featurise_smiles_datapoints


@pytest.fixture
def input_file():

    return os.path.join(os.path.dirname(__file__), "datasets/cleaned/CHEMBL1001235.csv")


@pytest.fixture
def feature_file():

    return os.path.join(os.path.dirname(__file__), "datasets/processed/CHEMBL1001235.jsonl.gz")


@pytest.fixture
def chembl_csv_format():
    return {
        "SMILES": "smiles",
        "Property": "activity",
        "Assay_ID": "chembl_id",
        "RegressionProperty": "standard_value",
        "LogRegressionProperty": "log_standard_value",
        "Relation": "standard_relation",
        "AssayType": "assay_type",
    }


@pytest.fixture
def metadata():

    metapath = RichPath.create(os.path.join(os.path.dirname(__file__), "../"))
    path = metapath.join("metadata.pkl.gz")
    metadata = path.read_by_file_suffix()
    return metadata["feature_extractors"]


def test_featurise(input_file: str, chembl_csv_format: Dict[str, str], feature_file: str, metadata):

    datapoints = load_csv_assay_data(
        input_file,
        chembl_csv_format,
    )

    # check the featurisers work
    featurised_data = featurise_smiles_datapoints(
        train_data=datapoints,
        valid_data=datapoints[0],
        test_data=datapoints[0],
        atom_feature_extractors=metadata,
        num_processes=-1,
        include_descriptors=True,
        include_fingerprints=True,
    )

    assert featurised_data.len_train_data == 21

    datapoints = [x for x in RichPath.create(feature_file).read_by_file_suffix()]

    assert len(datapoints) == 21
