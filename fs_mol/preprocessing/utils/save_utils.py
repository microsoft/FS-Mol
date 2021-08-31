import os
import json
import gzip
import pickle
from tqdm import tqdm
from typing import Dict, Iterable, Any, List, Optional

from fs_mol.preprocessing.featurisers.featurised_data import FeaturisedData


def write_jsonl_gz_data(
    file_name: str, data: Iterable[Dict[str, Any]], len_data: int = None
) -> int:
    num_ele = 0
    with gzip.open(file_name, "wt") as data_fh:
        for ele in tqdm(data, total=len_data):
            save_element(ele, data_fh)
            num_ele += 1
    return num_ele


def save_element(element: Dict[str, Any], data_fh) -> None:
    ele = dict(element)
    ele.pop("mol", None)
    ele.pop("fingerprints_vect", None)
    if "fingerprints" in ele:
        ele["fingerprints"] = ele["fingerprints"].tolist()
    data_fh.write(json.dumps(ele) + "\n")


# the assays to be saved here are not given the train-test-valid prefixes
# that is assigned by an alternative file. Only the 'train' data is saved; which is all of the data.
def save_assay_data(featurised_data: FeaturisedData, assay_id: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for fold_name, data_fold in zip(
        ["train", "valid", "test"],
        [
            featurised_data.train_data,
            featurised_data.valid_data,
            featurised_data.test_data,
        ],
    ):
        if fold_name == "train":
            assay_name = os.path.basename(assay_id)[:-4]
            filename = os.path.join(output_dir, f"{assay_name}.jsonl.gz")
            num_written = write_jsonl_gz_data(
                filename, data_fold, len_data=featurised_data.len_train_data
            )
            print(f" Wrote {num_written} datapoints to {filename}.")

        else:
            continue


def save_metadata(
    featurised_data: FeaturisedData,
    output_dir: str,
    extra_metadata: Optional[Dict[str, Any]] = None,
    failed: Optional[List[str]] = None,
) -> None:

    metadata_file = os.path.join(output_dir, "metadata.pkl.gz")
    failed_file = os.path.join(output_dir, "failed_assays.txt")
    if extra_metadata is None:
        extra_metadata = {}
    with gzip.open(metadata_file, "wb") as data_fh:
        pickle.dump(
            {
                **extra_metadata,
                "feature_extractors": featurised_data.atom_feature_extractors,
            },
            data_fh,
        )

    print(f" Wrote metadata to {metadata_file}.")

    if failed:
        with open(failed_file, "w+") as ff:
            ff.write(failed)
        print(f"Wrote out failed processing assays to {failed_file}")
