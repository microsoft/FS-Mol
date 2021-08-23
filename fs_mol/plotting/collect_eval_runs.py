"""Collecting MAML test results"""
import os
import sys
import pandas as pd
from typing import Optional, Callable, List

from pyprojroot import here as project_root

sys.path.insert(0, str(project_root("fs_mol")))

from plotting.utils import (
    process_file,
    get_csv_paths,
    collate_experiment_results,
)


def collate_results(
    task_csvs: str,
    output_dir: str,
    model_name: str,
    support_set_sizes: List,
    grouping_column: str = "num_train_requested",
    plotting_metric: str = "average_precision_score",
    plot: bool = True,
    task_name_extraction_fn: Optional[Callable[[str], str]] = None,
) -> None:
    # summary df to summarise across all tasks
    summary_dfs = {val: pd.DataFrame() for val in support_set_sizes}

    for file in task_csvs:
        try:
            summary_dfs = process_file(
                summary_dfs=summary_dfs,
                file=file,
                output_dir=output_dir,
                model_name=model_name,
                select_values=support_set_sizes,
                task_name_extraction_fn=task_name_extraction_fn,
                grouping_column=grouping_column,
                plot=plot,
                plotting_metric=plotting_metric,
            )
        except Exception as e:
            print(f"Failed to grab and process csv for {file}, skipping: " + str(e))
            continue

    for val in support_set_sizes:
        v = "".join(str(val).split("."))
        summary_dfs[val].to_csv(
            os.path.join(output_dir, f"{model_name}_summary_{grouping_column}_{v}.csv"),
            header=True,
            index=False,
        )


def run(args):

    output_dir = os.path.join(args.INPUT_DIR, "summary")
    os.makedirs(output_dir, exist_ok=True)

    task_result_csvs = get_csv_paths(
        input_dir=args.INPUT_DIR,
        files_suffix=args.files_suffix,
        files_prefix=args.files_prefix,
    )

    # Collate the results for each support set size, including
    # all output metrics (eg. average precision, roc_auc etc.)
    collate_results(
        task_csvs=task_result_csvs,
        output_dir=output_dir,
        model_name=args.MODEL,
        support_set_sizes=args.support_set_sizes,
        plotting_metric=args.metric,
        plot=args.plot,
    )

    # join the results across different support set sizes
    # loads the summary_{grouping_column}_{num_points}.csv files
    # joins in to one output dataframe with results for just one metric (default avg precision)
    results = collate_experiment_results(
        results_dir=output_dir,
        model_name=args.MODEL,
        y_col=args.metric,
        task_number_prefix=args.task_number_prefix,
        support_set_size=args.support_set_sizes,
    )

    results.to_csv(os.path.join(output_dir, f"{args.MODEL}_summary.csv"), header=True, index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Collate testing results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "MODEL",
        type=str,
        help="Unique identifier of the model being tested.",
    )

    parser.add_argument(
        "INPUT_DIR",
        type=str,
        help=("Directory containing csv files from experiments to collate."),
    )

    parser.add_argument(
        "--support-set-sizes",
        default=[16, 32, 64, 128, 256],
        type=lambda s: [int(v) for v in s.split(",")],
        help="Values to group results for.",
    )

    parser.add_argument(
        "--files-suffix",
        default="",
        type=str,
        help="Suffix to identify files to include in collation from input directory",
    )

    parser.add_argument(
        "--files-prefix",
        default="",
        type=str,
        help="Prefix to identify files to include in collation from input directory",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot outputs for each run as function of number of support set points requested.",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="average_precision_score",
        help="Metric to plot",
    )

    parser.add_argument(
        "--task-number-prefix",
        default=None,
        type=str,
        help=(
            "The prefix used in labelling the tasks, if they consist of 'PREFIX+numerical_value "
            " Eg. Tasks/Assays are labelled 'CHEMBL999' etc within the CHEMBL results. "
            " If this is not set, the task name taken from the basic files will be used throughout."
        ),
    )

    args = parser.parse_args()

    run(args)
