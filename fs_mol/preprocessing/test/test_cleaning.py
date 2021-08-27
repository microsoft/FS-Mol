import os
import sys
from glob import glob
import pandas as pd

import pytest
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root("fs_mol")))

from preprocessing.clean import (
    select_assays,
    standardize,
    apply_thresholds,
    get_files_to_process,
    process_all_assays,
)


@pytest.fixture
def raw_dataset():

    return pd.read_csv(os.path.join(os.path.dirname(__file__), "datasets/CHEMBL1002396_raw.csv"))


@pytest.fixture
def processed_dataset():

    return pd.read_csv(
        os.path.join(os.path.dirname(__file__), "datasets/CHEMBL1002396_processed.csv")
    )


@pytest.fixture
def input_dir():

    return os.path.join(os.path.dirname(__file__), "datasets/raw/")


@pytest.fixture
def output_dir():
    return os.path.join(os.path.dirname(__file__), "datasets/cleaned")


@pytest.fixture
def final_dir():
    # dummy directory to allow repeat tests
    return os.path.join(os.path.dirname(__file__), "datasets/final_dir")


@pytest.fixture
def basepath():
    return os.path.join(os.path.dirname(__file__), "datasets")


@pytest.fixture
def summary_file():
    # dummy directory to allow repeat tests
    return os.path.join(os.path.dirname(__file__), "datasets/cleaned/summary.csv")


def test_len(raw_dataset: pd.DataFrame):
    assert len(raw_dataset) == 48


def test_cleaning_funcs(raw_dataset: pd.DataFrame, processed_dataset: pd.DataFrame):

    df = select_assays(raw_dataset)
    df = standardize(df)
    df = apply_thresholds(df)

    assert len(df) == len(processed_dataset)

    assert df.iloc[0]["threshold"] == processed_dataset.iloc[0]["threshold"]

    assert df.iloc[0]["smiles"] == processed_dataset.iloc[0]["smiles"]
    assert len(set(df.columns) - set(processed_dataset.columns)) == 0


def test_cleaning_pipeline(
    input_dir: str, output_dir: str, basepath: str, final_dir: str, summary_file: str
):

    os.makedirs(final_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    files_to_process = get_files_to_process(input_dir, final_dir)

    assert len(files_to_process) == 2

    process_all_assays(files_to_process, output_dir, basepath, 0, 2)

    fully_cleaned_files = glob(os.path.join(output_dir, "CHEMBL*.csv"), recursive=True)

    assert len(fully_cleaned_files) == 1

    summary = pd.read_csv(summary_file)

    assert len(summary) == 1

    assert len(summary.columns) == 15
