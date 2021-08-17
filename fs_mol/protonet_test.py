import argparse
import dataclasses
import logging
import os
import sys
from collections import defaultdict
from typing import DefaultDict, List, Optional

import torch
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from fs_mol.data.fsmol_dataset import DataFold, FSMolDataset
from fs_mol.data.protonet import get_protonet_task_sample_iterable
from fs_mol.models.protonet import PrototypicalNetwork
from fs_mol.utils.protonet_utils import PrototypicalNetworkTrainer, run_on_batches
from fs_mol.utils.molfilm_utils import resolve_starting_model_file
from fs_mol.utils.test_utils import (
    FSMolTaskSampleEvalResults,
    write_csv_summary,
    add_eval_cli_args,
    set_up_test_run,
)


logger = logging.getLogger(__name__)


def parse_command_line():
    parser = argparse.ArgumentParser(
        description="Test a Prototypical Network model on molecules.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "TRAINED_MODEL",
        type=str,
        help="File to load model from (determines model architecture).",
    )

    add_eval_cli_args(parser)

    parser.add_argument(
        "--batch-size",
        type=int,
        default=320,
        help="Maximum batch size to allow when running through inference on model.",
    )
    parser.add_argument(
        "--use-fresh-param-init",
        action="store_true",
        help="Do not use trained weights, but start from a fresh, random initialisation.",
    )
    args = parser.parse_args()
    return args


def test(
    model: PrototypicalNetwork,
    dataset: FSMolDataset,
    save_dir: str,
    context_sizes: List[int],
    target_size: Optional[int],
    num_samples: int,
    seed: int,
    batch_size: int,
):
    results: DefaultDict[str, List[FSMolTaskSampleEvalResults]] = defaultdict(list)
    for context_size in context_sizes:
        test_task_sample_iterator = get_protonet_task_sample_iterable(
            dataset=dataset,
            data_fold=DataFold.TEST,
            num_samples=num_samples,
            max_num_graphs=batch_size,
            support_size=context_size,
            query_size=target_size,
        )

        for run_idx, task_sample in enumerate(test_task_sample_iterator):
            _, result_metrics = run_on_batches(
                model,
                batches=task_sample.batches,
                batch_labels=task_sample.batch_labels,
                train=False,
            )
            logger.info(
                f"{task_sample.task_name}:"
                f" {task_sample.num_support_samples:3d} support samples,"
                f" {task_sample.num_query_samples:3d} query samples."
                f" Avg. prec. {result_metrics.avg_precision:.5f}.",
            )
            results[task_sample.task_name].append(
                FSMolTaskSampleEvalResults(
                    task_name=task_sample.task_name,
                    seed=seed + run_idx,
                    num_train=task_sample.num_support_samples,
                    num_test=task_sample.num_query_samples,
                    fraction_pos_train=task_sample.num_positive_support_samples
                    / task_sample.num_support_samples,
                    fraction_pos_test=task_sample.num_positive_query_samples
                    / task_sample.num_query_samples,
                    **dataclasses.asdict(result_metrics),
                )
            )

    for task_name, task_results in results.items():
        write_csv_summary(os.path.join(save_dir, f"{task_name}_eval_results.csv"), task_results)


def main():
    args = parse_command_line()
    out_dir, dataset = set_up_test_run("ProtoNet", args, torch=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_weights_file = resolve_starting_model_file(
        model_file=args.TRAINED_MODEL,
        model_cls=PrototypicalNetworkTrainer,
        out_dir=out_dir,
        use_fresh_param_init=args.use_fresh_param_init,
        device=device,
    )

    model = PrototypicalNetworkTrainer.build_from_model_file(
        model_weights_file,
        device=device,
    )

    test(
        model,
        dataset,
        save_dir=args.save_dir,
        context_sizes=args.train_sizes,
        target_size=None,
        num_samples=args.num_runs,
        seed=args.seed,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        import pdb

        _, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
