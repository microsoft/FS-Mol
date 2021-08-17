import argparse
import json
import os
from collections import Counter, OrderedDict
from typing import Any, Dict, Iterable, List, NamedTuple

from rdkit import Chem

from fs_mol.data.fsmol_dataset import DataFold, FSMolDataset
from fs_mol.data.fsmol_task import FSMolTask, MoleculeDatapoint
from fs_mol.utils.cli_utils import set_seed
from fs_mol.utils.test_utils import add_data_cli_args, set_up_dataset


class TaskData(NamedTuple):
    smiles: List[str]
    frac_positive: float
    numeric_labels: List[float]
    task_name: str


def parse_command_line():
    parser = argparse.ArgumentParser(
        description="Compute simple dataset statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_data_cli_args(parser)

    parser.add_argument("OUTPUT_PATH", type=str, help="JSON dictionary file to save stats.")
    return parser.parse_args()


def load_samples(samples: List[MoleculeDatapoint]) -> TaskData:
    """Load all samples into memory, but only keep the necessary bits around."""
    task_smiles = []
    task_labels = []

    for sample in samples:
        canonical_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(sample.smiles))

        task_smiles.append(canonical_smiles)
        task_labels.append(sample.numeric_label)

    return TaskData(
        smiles=task_smiles,
        frac_positive=float(sum([sample.bool_label for sample in samples])) / len(task_smiles),
        numeric_labels=task_labels,
        task_name=samples[0].task_name,
    )


def load_fold_data(tasks: Iterable[FSMolTask]) -> List[TaskData]:
    return [load_samples(task.samples) for task in tasks]


def get_counts_of_counts(counts: Counter) -> OrderedDict:
    counts_of_counts = Counter(counts.values()).items()
    return OrderedDict(sorted(counts_of_counts))


def compute_fold_stats(data: List[TaskData]) -> Dict[str, Any]:
    smiles_per_task = [d.smiles for d in data]
    unique_smiles_per_task = [list(set(smiles)) for smiles in smiles_per_task]

    num_smiles_per_task = [len(smiles) for smiles in smiles_per_task]
    num_unique_smiles_per_task = [len(smiles) for smiles in unique_smiles_per_task]

    # Count occurrences of each SMILES, both before and after deduplication within each task.
    smiles_counter = Counter(sum(smiles_per_task, []))
    smiles_counter_after_deduplication = Counter(sum(unique_smiles_per_task, []))

    return {
        "num_tasks": len(data),
        "num_samples": sum(num_smiles_per_task),
        "num_unique_smiles": len(smiles_counter),
        "num_samples_per_task": sorted(num_smiles_per_task),
        "num_unique_smiles_per_task": sorted(num_unique_smiles_per_task),
        "frac_positive_per_task": sorted([d.frac_positive for d in data]),
        "num_occ_per_smiles": get_counts_of_counts(smiles_counter),
        "num_occ_per_unique_smiles": get_counts_of_counts(smiles_counter_after_deduplication),
    }


def compute_fold_overlap(data_1: List[TaskData], data_2: List[TaskData]) -> int:
    def get_all_unique(data: List[TaskData]):
        return set(sum([d.smiles for d in data], []))

    unique_smiles_1 = get_all_unique(data_1)
    unique_smiles_2 = get_all_unique(data_2)

    return {
        "only_in_first": len(unique_smiles_1 - unique_smiles_2),
        "only_in_second": len(unique_smiles_2 - unique_smiles_1),
        "overlap": len(unique_smiles_1 & unique_smiles_2),
    }


def compute_dataset_stats(dataset: FSMolDataset) -> Dict[str, Any]:
    print("Computing dataset statistics.")
    fold_data = {}

    for fold in DataFold:
        fold_data[fold] = load_fold_data(dataset.get_task_reading_iterable(fold))
        print(f"Loaded {len(fold_data[fold])} tasks for fold {fold.name}.")

    # Compute stats taking all data together.
    stats = {"ALL": compute_fold_stats(sum(fold_data.values(), []))}

    # Compute stats within each fold.
    stats.update({fold.name: compute_fold_stats(data) for (fold, data) in fold_data.items()})

    # Compute stats between (unordered) pairs of distinct folds.
    stats.update(
        {
            f"{fold_1.name} {fold_2.name} overlap": compute_fold_overlap(data_1, data_2)
            for (fold_1, data_1) in fold_data.items()
            for (fold_2, data_2) in fold_data.items()
            if fold_1.value < fold_2.value
        }
    )

    # Also see how the union of training and validation overlaps with the test set.
    stats["TRAIN+VALIDATION TEST overlap"] = compute_fold_overlap(
        fold_data[DataFold.TRAIN] + fold_data[DataFold.VALIDATION], fold_data[DataFold.TEST]
    )

    return stats


def main():
    # We shouldn't be doing anything non-deterministic here, but lets fix the seed just in case.
    set_seed(0)

    args = parse_command_line()
    dataset = set_up_dataset(args, num_workers=0)
    stats = compute_dataset_stats(dataset)

    for key, value in stats.items():
        print(f"{key}: {value}")

    # Save stats to a json file.
    with open(os.path.join(args.OUTPUT_PATH, "stats.json"), "w+") as jsonfile:
        json.dump(stats, jsonfile)


if __name__ == "__main__":
    main()
