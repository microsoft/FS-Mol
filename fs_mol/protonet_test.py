import argparse
import logging
import sys
from typing import List

import torch
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))

from fs_mol.data import FSMolDataset, FSMolTaskSample
from fs_mol.data.protonet import get_protonet_batcher, task_sample_to_pn_task_sample
from fs_mol.models.abstract_torch_fsmol_model import resolve_starting_model_file
from fs_mol.models.protonet import PrototypicalNetwork
from fs_mol.utils.protonet_utils import PrototypicalNetworkTrainer, run_on_batches
from fs_mol.utils.metrics import BinaryEvalMetrics
from fs_mol.utils.test_utils import eval_model, add_eval_cli_args, set_up_test_run


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
    num_samples: int,
    seed: int,
    batch_size: int,
):
    batcher = get_protonet_batcher(max_num_graphs=batch_size)

    def test_model_fn(
        task_sample: FSMolTaskSample, temp_out_folder: str, seed: int
    ) -> BinaryEvalMetrics:
        pn_task_sample = task_sample_to_pn_task_sample(task_sample, batcher)

        _, result_metrics = run_on_batches(
            model,
            batches=pn_task_sample.batches,
            batch_labels=pn_task_sample.batch_labels,
            train=False,
        )
        logger.info(
            f"{pn_task_sample.task_name}:"
            f" {pn_task_sample.num_support_samples:3d} support samples,"
            f" {pn_task_sample.num_query_samples:3d} query samples."
            f" Avg. prec. {result_metrics.avg_precision:.5f}.",
        )

        return result_metrics

    return eval_model(
        test_model_fn=test_model_fn,
        dataset=dataset,
        train_set_sample_sizes=context_sizes,
        out_dir=save_dir,
        num_samples=num_samples,
        seed=seed,
    )


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
