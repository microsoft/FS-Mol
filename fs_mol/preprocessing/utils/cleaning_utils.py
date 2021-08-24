"""
A set of utils to clean assays taken from ChEMBL, used in
preprocessing/cleaning.py

"""

import math
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Callable

from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt

from fs_mol.preprocessing.utils.standardizer import Standardizer

standard_unit_set = {"nM", "%", "uM"}


class MolError(Exception):
    """Custom exception for failure to convert molecule."""

    def __init__(self, smiles: str, error: str):
        super().__init__()
        self._smiles = smiles
        self._error = error

    def __str__(self):
        return f"Cannot make rdkit mol from SMILES: {self._smiles} \n {self._error}"


def clean_units(x: pd.Series) -> bool:
    """Remove measurements that have units outside the permitted set"""
    return x["standard_units"] not in standard_unit_set


def clean_values(x: pd.Series) -> bool:
    return np.isnan(x["standard_value"])


def log_standard_values(x: pd.Series) -> float:
    """Convert standard value to -log10([C]/nM)"""
    if x["standard_value"] < 1e-13 or np.isnan(x["standard_value"]):
        return float("NaN")
    else:
        return -1 * math.log10(x["standard_value"] * 10 ** -9)


relation_set_lessthan = {"<", "<="}
relation_set_morethan = {">", ">="}
relation_equals = {"=", "~"}


def activity_threshold(x: float, threshold: float, buffer: float = 0.5) -> str:

    """
    Apply a threshold to activity measurements.

    For example, IC50/ EC50 measurements.
    """

    # for now just use the passed median value to threshold
    # this is calculated from the entire dataframe.
    val = x["log_standard_value"]
    relation = x["standard_relation"]

    # note: standard relations apply to standard value not log standard
    if val >= (threshold + buffer):
        return "active"
    elif val > threshold and val < (threshold + buffer) and relation in relation_set_lessthan:
        return "active"
    elif (
        val > threshold
        and val < (threshold + buffer)
        and relation in relation_set_morethan.union(relation_equals)
    ):
        return "weak active"
    elif (
        val > (threshold - buffer)
        and val <= threshold
        and relation in relation_set_lessthan.union(relation_equals)
    ):
        return "weak inactive"
    elif val > (threshold - buffer) and val <= threshold and relation in relation_set_morethan:
        return "inactive"
    elif val <= (threshold - buffer):
        return "inactive"


def inhibition_threshold(x: float, threshold: float, buffer: float = 5.0) -> str:

    """Apply a threshold to inhibition measurements."""

    # 5% default buffer around threshold for weak activity/inactivity
    # is not always used, but available to convert to four class problem.
    # for now just use the passed median value to threshold
    # this is calculated from the entire dataframe.
    val = x["standard_value"]
    relation = x["standard_relation"]

    # this is different to the activity as relations apply to standard value
    # not log standard values
    if val >= (threshold + buffer):
        return "active"
    elif val > threshold and val < (threshold + buffer) and relation in relation_set_morethan:
        return "active"
    elif (
        val > threshold
        and val < (threshold + buffer)
        and relation in relation_set_lessthan.union(relation_equals)
    ):
        return "weak active"
    elif (
        val > (threshold - buffer)
        and val <= threshold
        and relation in relation_set_morethan.union(relation_equals)
    ):
        return "weak inactive"
    elif val > (threshold - buffer) and val <= threshold and relation in relation_set_lessthan:
        return "inactive"
    elif val <= (threshold - buffer):
        return "inactive"


def autothreshold(x: pd.Series) -> Tuple[pd.DataFrame, float]:

    """
    Apply autothesholding procedure to data:

    1) Find the median for an assay
    2) Use the median as a threshold if it sits within the required range.
    For enzymes this is 5 <= median(pXC) <= 7, for everything else this is
    4 <= median(pXC) <= 6. Inhibition measurements use <= 50% as medians above
    this are found with assays containing almost entirely strong inhibitors.
    If the median is outside the required range, fix to pXC = 5.0 or inhibition = 50%
    3) Apply the threshold to the data series.

    For activity measurements, log standard value is used.
    """

    df = pd.DataFrame(x)

    # % inhibition measurements don't have log standard measurement
    if df.iloc[0]["standard_units"] == "%":

        # find median. Reject this as a classification threshold
        # if it is too far skewed towards low values
        # (this means there has been more a HTS and most measurements were
        # of obviously bad molecules. Permit > 50% as sign of some activity)
        median = df["standard_value"].median()
        threshold = median
        buffer = df["standard_value"].std() / 10
        if not 50.0 <= median:
            threshold = 50.0

        df["activity_string"] = df.apply(
            inhibition_threshold, args=(threshold,), buffer=buffer, axis=1
        )
    else:
        threshold_limits = (5, 7)
        # get median
        median = df["log_standard_value"].median()
        threshold = median
        buffer = df["log_standard_value"].std() / 10

        # use as a threshold provided it is in a sensible
        # range. This was chosen as pKI 4-6 in general, 5-7 for enzymes
        if "protein_class_desc" in df.columns:
            if any(("enzyme" in x) or ("ase" in x) for x in df.protein_class_desc.values):
                threshold_limits = (5, 7)
            else:
                threshold_limits = (4, 6)
        else:
            threshold_limits = (4, 6)

        if median < threshold_limits[0] or median > threshold_limits[1]:
            threshold = 5.0
        else:
            threshold = median

        df["activity_string"] = df.apply(
            activity_threshold, args=(threshold,), buffer=buffer, axis=1
        )

    return df, threshold


def fixedthreshold(x: pd.Series) -> Tuple[pd.DataFrame, float]:

    """
    Apply fixed threshold to the data.

    pXC = 5.0, or inhibition = 50%

    """

    df = pd.DataFrame(x)

    # % inhibition measurements don't have log standard measurement
    if df.iloc[0]["standard_units"] == "%":
        threshold = 50.0
        df["activity_string"] = df.apply(inhibition_threshold, args=(threshold,), axis=1)

    else:
        threshold = 5.0
        df["activity_string"] = df.apply(activity_threshold, args=(threshold,), axis=1)

    return df, threshold


def get_duplicated_rows(
    df: pd.DataFrame,
    comparison_fn: Callable,
    max_size: Optional[int] = None,
    block_by=None,
):

    """
    Find the duplicated rows.

    Permits a flexible comparison_fn that consumes two rows of a dataframe
    and returns a boolean if the two rows 'match' under the user's definition.

    """

    # recursively find matching groups of the dataframe with flexible comparison func
    # If block_by is provided, then we apply the algorithm to each block and
    # stitch the results back together (here would use if eg. obviously different measurements/ proteins)
    if block_by is not None:
        blocks = df.groupby(block_by).apply(
            lambda g: get_duplicated_rows(df=g, comparison_fn=comparison_fn, max_size=max_size)
        )

        keys = blocks.index.unique(block_by)
        for a, b in zip(keys[:-1], keys[1:]):
            blocks.loc[b, :] += blocks.loc[a].iloc[-1] + 1

        return blocks.reset_index(block_by, drop=True)

    records = df.to_records()
    partitions = []

    def get_record_index(r):
        return r[df.index.name or "index"]

    def find_partition(at=0, partition=None, indexes=None):

        r1 = records[at]

        if partition is None:
            partition = {get_record_index(r1)}
            indexes = [at]

        # Stop if enough duplicates have been found/ prevent inf
        if max_size is not None and len(partition) == max_size:
            return partition, indexes

        for i, r2 in enumerate(records):

            if get_record_index(r2) in partition or i == at:
                continue

            if comparison_fn(r1, r2):
                partition.add(get_record_index(r2))
                indexes.append(i)
                find_partition(at=i, partition=partition, indexes=indexes)

        return partition, indexes

    while len(records) > 0:
        partition, indexes = find_partition()
        partitions.append(partition)
        records = np.delete(records, indexes)

    return pd.Series(
        {idx: partition_id for partition_id, idxs in enumerate(partitions) for idx in idxs}
    )


def remove_far_duplicates(x: pd.DataFrame):

    """
    Find the duplicated rows.

    Permits a flexible comparison_fn that consumes two rows of a dataframe
    and returns a boolean if the two rows 'match' under the user's definition.

    """

    # judgement on whether a measurement is a bad duplicate:
    # if two measurements with the same SMILES differ by more than
    # one order of magnitude (diff pKI > 1.0 (pKI == "log_standard"))
    # then they are inconsistent and must be dropped

    df = pd.DataFrame(x)

    def far_duplicate_measurements(r1, r2) -> bool:

        if r1["canonical_smiles"] == r2["canonical_smiles"]:
            if r1["standard_units"] == "%":
                # only allow 5% variation here
                return np.abs(r1["standard_value"] - r2["standard_value"]) > 5.0
            else:
                return np.abs(r1["log_standard_value"] - r2["log_standard_value"]) > 1.0
        else:
            return False

    df["new_index"] = get_duplicated_rows(
        df, comparison_fn=far_duplicate_measurements, block_by="canonical_smiles"
    )
    # any duplicate indices here indicate that measurements for these molecules contradict
    df = df.drop_duplicates(subset=["new_index"], keep=False)

    df = df.drop(columns=["new_index"])
    return df


def standardize_smiles(x: pd.DataFrame, taut_canonicalization: bool = True) -> pd.DataFrame:

    """
    Standardization of a SMILES string.

    Uses the 'Standardizer' to perform sequence of cleaning operations on a SMILES string.
    """
    # reads in a SMILES, and returns it in canonicalized form.
    # can select whether or not to use tautomer canonicalization
    sm = Standardizer(canon_taut=taut_canonicalization)
    df = pd.DataFrame(x)

    def standardize_smile(x: str):
        try:
            mol = Chem.MolFromSmiles(x)
            mol_weight = MolWt(mol)  # get molecular weight to do downstream filtering
            num_atoms = mol.GetNumAtoms()
            standardized_mol, _ = sm.standardize_mol(mol)
            return Chem.MolToSmiles(standardized_mol), mol_weight, num_atoms
        except Exception:
            # return a fail as None (downstream filtering)
            return None

    standard = df["smiles"].apply(lambda row: standardize_smile(row))
    df["canonical_smiles"] = standard.apply(lambda row: row[0])
    df["molecular_weight"] = standard.apply(lambda row: row[1])
    df["num_atoms"] = standard.apply(lambda row: row[2])

    return df
