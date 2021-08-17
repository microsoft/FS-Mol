"""Collecting MAML test results"""
import os
import sys
from typing import Optional, Callable, List

import pandas as pd
from inspect import getsourcefile


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

### Import Azure-specific things
from azureml.core import Run, Experiment

from azure_ml.utils import load_aml_config, get_workspace

from postprocessing.utils import (
    process_file,
    get_amlrun_csvs,
    get_csv_paths,
    collate_experiment_results,
)


def collate_aml_results(
    run_gen: Run,
    output_dir: str,
    model_name: str,
    select_values: List[int],
    grouping_column: str,
    plotting_metric: str = "roc_auc",
    plot: bool = True,
    task_name_extraction_fn: Optional[Callable[[str], str]] = None,
) -> None:
    summary_dfs = {val: pd.DataFrame() for val in select_values}

    for run in run_gen:

        run_id = run._run_id

        try:

            run_csvs = get_amlrun_csvs(run, output_dir=output_dir)

            if len(run_csvs) == 0:
                print(f"W: No results csvs in run {run._run_id}, skipping")
                continue

            for file in run_csvs:
                summary_dfs = process_file(
                    summary_dfs=summary_dfs,
                    file=file,
                    output_dir=output_dir,
                    model_name=model_name,
                    select_values=select_values,
                    task_name_extraction_fn=task_name_extraction_fn,
                    grouping_column=grouping_column,
                    plot=plot,
                    plotting_metric=plotting_metric,
                )

        except Exception as e:
            print(f"Failed to grab and process csv for {run_id}" + str(e))
            continue

    for val in select_values:
        v = "".join(str(val).split("."))
        summary_dfs[val].to_csv(
            os.path.join(output_dir, f"{model_name}_summary_{grouping_column}_{v}.csv"),
            header=True,
            index=False,
        )


def collate_results(
    task_csvs: str,
    output_dir: str,
    model_name: str,
    select_values: List,
    grouping_column: str = "requested_num_train",
    plotting_metric: str = "roc_auc",
    plot: bool = True,
    task_name_extraction_fn: Optional[Callable[[str], str]] = None,
) -> None:
    # summary df to summarise across all tasks
    summary_dfs = {val: pd.DataFrame() for val in select_values}

    for file in task_csvs:
        try:
            summary_dfs = process_file(
                summary_dfs=summary_dfs,
                file=file,
                output_dir=output_dir,
                model_name=model_name,
                select_values=select_values,
                task_name_extraction_fn=task_name_extraction_fn,
                grouping_column=grouping_column,
                plot=plot,
                plotting_metric=plotting_metric,
            )
        except Exception as e:
            print(f"Failed to grab and process csv for {file}, skipping: " + str(e))
            continue

    for val in select_values:
        v = "".join(str(val).split("."))
        summary_dfs[val].to_csv(
            os.path.join(output_dir, f"{model_name}_summary_{grouping_column}_{v}.csv"),
            header=True,
            index=False,
        )


def run(args):
    os.makedirs(args.OUTPUT_DIR, exist_ok=True)

    if args.aml_experiment_name is not None:
        current_dir = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))
        GENCHEM_FOLDER = "/".join(current_dir.split("/")[:4])
        genchem_aml_dir = os.path.join(GENCHEM_FOLDER, "azure_ml")
        aml_config_file = os.path.join(genchem_aml_dir, "aml_config_msr.json")
        aml_config = load_aml_config(aml_config_file)

        workspace = get_workspace(aml_config)
        EXPERIMENT_NAME = args.aml_experiment_name  # "MAML-GSK_001_testing"
        exp = Experiment(workspace=workspace, name=EXPERIMENT_NAME)

        # get the runs in this experiment
        run_gen = exp.get_runs(include_children=True)

        collate_aml_results(
            run_gen,
            output_dir=args.OUTPUT_DIR,
            model_name=args.MODEL,
            select_values=args.groupby_values,
            grouping_column=args.groupby,
            plotting_metric=args.metric,
            plot=args.plot,
        )

    else:
        if args.input_dir is None:
            raise ValueError("If no aml experiment is passed, require input directory.")

        task_result_csvs = get_csv_paths(
            input_dir=args.input_dir,
            files_suffix=args.files_suffix,
            files_prefix=args.files_prefix,
        )
        collate_results(
            task_csvs=task_result_csvs,
            output_dir=args.OUTPUT_DIR,
            model_name=args.MODEL,
            select_values=args.groupby_values,
            grouping_column=args.groupby,
            plotting_metric=args.metric,
            plot=args.plot,
        )

    # join the results across different select_values
    # loads the summary_{grouping_column}_{num_points}.csv files
    # joins in to one output dataframe
    results = collate_experiment_results(
        results_dir=args.OUTPUT_DIR,
        model_name=args.MODEL,
        x_col=args.groupby,
        y_col=args.metric,
        task_number_prefix=args.task_number_prefix,
        num_points=args.groupby_values,
    )

    results.to_csv(
        os.path.join(args.OUTPUT_DIR, f"{args.MODEL}_summary.csv"), header=True, index=False
    )


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
        "OUTPUT_DIR",
        type=str,
        help="Location to store summary csvs and plots",
    )

    parser.add_argument(
        "--groupby",
        default="num_train_requested",
        type=str,
        help="Name of property to group repeats by, eg. train_frac, requested_num_train.",
    )

    parser.add_argument(
        "--groupby-values",
        default=[16, 32, 64, 128, 256],
        type=lambda s: [int(v) for v in s.split(",")],
        help="Values to group results for.",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        type=str,
        help=(
            "Directory containing csv files from experiments to collate."
            " If aml-experiment-name is not set, this must be set."
        ),
    )

    parser.add_argument(
        "--aml-experiment-name",
        default=None,
        type=str,
        help="AML Experiment",
    )

    parser.add_argument(
        "--files-suffix",
        default="",
        type=str,
        help="Suffix to identify files to collate in input directory",
    )

    parser.add_argument(
        "--files-prefix",
        default="",
        type=str,
        help="Prefix to identify files to collate in input directory",
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot outputs for each run as function of $GROUPBY.",
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
