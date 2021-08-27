#!/usr/bin/env python3
import argparse
import contextlib
import json
import logging
import operator
import os
import sys
import time
from functools import reduce, partial
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional
from typing_extensions import Protocol

import tensorflow as tf
from azureml.core.run import Run
from dpu_utils.utils import run_and_debug, RichPath
from more_itertools import chunked
from tf2_gnn.cli_utils.model_utils import load_weights_verbosely
from tf2_gnn.cli_utils.training_utils import unwrap_tf_tracked_data
from tf2_gnn.layers import get_known_message_passing_classes

from pyreporoot import project_root

sys.path.insert(0, str(project_root(Path(__file__), root_files="requirements.txt")))

from fs_mol.data import (
    FSMolDataset,
    DataFold,
    FSMolTask,
    FSMolTaskSample,
    StratifiedTaskSampler,
)
from fs_mol.data.fsmol_task_sampler import SamplingException
from fs_mol.data.maml import FSMolStubGraphDataset, TFGraphBatchIterable
from fs_mol.models.metalearning_graph_binary_classification import (
    MetalearningGraphBinaryClassificationTask,
)
from fs_mol.utils.cli_utils import add_train_cli_args, set_up_train_run, str2bool
from fs_mol.utils.logging import FileLikeLogger, PROGRESS_LOG_LEVEL
from fs_mol.utils.maml_utils import save_model, eval_model_by_finetuning_on_tasks


logger = logging.getLogger(__name__)


MAML_MODEL_DEFAULT_HYPER_PARAMS = {
    "gnn_hidden_dim": 128,
    "gnn_num_layers": 8,
    "gnn_dense_every_num_layers": 32,
    "gnn_residual_every_num_layers": 2,
    "gnn_use_inter_layer_layernorm": True,
    "gnn_layer_input_dropout_rate": 0.1,
    "gnn_global_exchange_every_num_layers": 6,
    "optimizer": "sgd",  # single task optimizer now operates on tasks in inner loop
    "use_lr_schedule_single_task_optimizers": False,  # no lr schedule in inner loop
    "initial_emb_lr": 0.0005,
    "gnn_lr": 0.0005,
    "readout_lr": 0.001,
    "final_mlp_lr": 0.001,  # inner loop learning rates ~ x10 outer loop
}

VALIDATION_MODEL_DEFAULT_HYPER_PARAMS = {
    "optimizer": "adam",  # change to "sgd" for maml as this addresses the inner loop
    "use_lr_schedule_single_task_optimizers": True,  # default is to allow lr schedules. Set to False for MAML (inner loop)
    "initial_emb_lr": 0.00005,
    "gnn_lr": 0.00005,
    "readout_lr": 0.0001,
    "final_mlp_lr": 0.0001,
    "initial_final_lr_ratio": 0.1,  # let's try some warmup
    "learning_rate_warmup_steps": 10,
}

VALIDATION_SEEDS = (0, 1, 2)
VALIDATION_TRAIN_SIZES = (16, 128)


class MetatrainValidationCallback(Protocol):
    def __call__(self, model_weights: Dict[str, tf.Tensor]) -> float:
        ...


def metatrain_on_task_samples(
    model: MetalearningGraphBinaryClassificationTask,
    task_samples: List[FSMolTaskSample],
    max_num_nodes_in_batch: int,
    max_num_inner_train_steps: int = 1,
    metatrain_task_specific_parameters: bool = True,
    quiet: bool = False,
) -> Tuple[float, int]:
    # We make a copy of the current state of the model (at the beginning of an outer/meta
    # training step). Below, we will start finetuning for specific tasks from this state,
    # and then record the differences. At the end, we use the combined results to update
    # the stored parameters.
    cur_outer_parameter_values = {var.name: var.value() for var in model.trainable_variables}

    # loop over a batch of tasks and do training on each of them, accumulating gradients:
    grads_and_vars: List[Iterable[Tuple[tf.Tensor, tf.Variable]]] = []
    metatraining_test_loss = 0.0
    metatraining_num_test_graphs = 0
    for task_sample in task_samples:
        # Update parameters on a sample of the training data (the sampling happened
        # in the iterator)
        #   Alg 2, line 5: "Sample K datapoints D = {x^(j), y^(j)} from T_i"

        #   Alg 2, line 6: "Evaluate \nabla_\theta L_{T_i}(f_\theta) using D
        #                   and L_{T_i}" in Eq (2) or (3)"
        #   Alg 2, line 7: "Compute adapted parameters with gradient descent:
        #                   \theta'_i = \theta - \alpha \nabla_\theta L_{T_i}(f_\theta)"

        num_inner_train_steps = 0
        while num_inner_train_steps < max_num_inner_train_steps:
            train_data = TFGraphBatchIterable(
                samples=task_sample.train_samples,
                shuffle=True,
                max_num_nodes=max_num_nodes_in_batch,
            )
            for train_features, train_labels in train_data:
                # Only do steps as long as we are still under the limit; otherwise just drain the iterator:
                if num_inner_train_steps < max_num_inner_train_steps:
                    # num_graphs = train_features["num_graphs_in_batch"].numpy()
                    # log_fun(f"    Metatraining on {task_sample.name}, train data step {num_inner_train_steps}, {num_graphs} graphs")
                    model._run_step(train_features, train_labels, training=True)
                    num_inner_train_steps += 1

        if not quiet:
            logger.log(
                PROGRESS_LOG_LEVEL,
                f"   Trained on task {task_sample.name} for {num_inner_train_steps} steps.",
            )

        # Next, we do the meta-update using the test data for the task:
        test_data = TFGraphBatchIterable(
            samples=task_sample.test_samples, max_num_nodes=max_num_nodes_in_batch
        )
        test_loss, test_num_graphs = 0.0, 0
        with tf.GradientTape() as meta_update_tape:
            # Run test using a sample of the test data (the first batch):
            #   Alg 2, line 8: "Sample datapoints D_i' = {x^(j), y^(j)} from T_i
            #                   for the meta-update"
            for test_features, test_labels in test_data:
                # Alg 2, line 10 (partially): Compute \nabla_\theta L_{T_i}(f_{\theta'_i}(D'_i))
                # Note that after the training update, model currently implements
                # f_{\theta'_i}:
                test_output = model(test_features, training=False)
                test_metrics = model.compute_task_metrics(test_features, test_output, test_labels)
                # We now have L_{T_i}(f_{\theta'_i}(D'_i)):
                test_loss += test_metrics["loss"] * test_features["num_graphs_in_batch"]

                # Record some data for logging:
                test_num_graphs += test_features["num_graphs_in_batch"]

            # Compute mean loss per graph:
            test_loss = test_loss / test_num_graphs

        # We now need to compute \nabla_\theta and for that first need to
        # reset the parameters to the original value \theta:
        for var in model.trainable_variables:
            var.assign(cur_outer_parameter_values[var.name])

        # Now we compute \nabla_\theta L_{T_i}(f_{\theta'_i}(D'_i)):
        task_grads = meta_update_tape.gradient(test_loss, model.trainable_variables)
        grads_and_vars.append(zip(task_grads, model.trainable_variables))

        metatraining_test_loss += test_loss
        metatraining_num_test_graphs += test_num_graphs

    # Alg 2, line 10 (cont'ed): Update \theta <- \theta - \beta\nabla_\theta \sum_{...}
    #   We have already computed \nabla_\theta \sum_{...} (its components are stored in
    #   batch_task_grads). We now need to combine them appropriately across tasks (by
    #   variable name) and then just perform the update:
    combined_grads_and_vars: Dict[str, Tuple[tf.Tensor, tf.Variable]] = {}
    for task_grads_and_vars in grads_and_vars:
        for var_grad, var in task_grads_and_vars:
            # If required, ignore gradients for task-specific parameters and do not update them:
            if not metatrain_task_specific_parameters and var.name.startswith(
                model.__class__.__name__
            ):
                continue
            old_grad_and_var = combined_grads_and_vars.get(var.name)
            if old_grad_and_var is not None:
                var_grad += old_grad_and_var[0]
            combined_grads_and_vars[var.name] = (var_grad, var)
    # outer loop optimizers are separate and not reinitialised.
    model._apply_gradients(combined_grads_and_vars.values(), outer_loop=True)

    return metatraining_test_loss, metatraining_num_test_graphs


def metatrain_loop(
    model: MetalearningGraphBinaryClassificationTask,
    metatrain_valid_fn: MetatrainValidationCallback,
    dataset: FSMolDataset,
    max_num_nodes_in_batch: int,
    max_epochs: int,
    patience: int,
    save_dir: str,
    task_batch_size: int,
    train_size: int,
    test_size: int,
    min_test_size: int,
    max_num_inner_train_steps: int,
    metatrain_task_specific_parameters: bool = True,
    quiet: bool = False,
    aml_run: Optional[Run] = None,
) -> str:
    save_file = os.path.join(save_dir, f"best_validation.pkl")

    task_sampler = StratifiedTaskSampler(
        train_size_or_ratio=train_size,
        valid_size_or_ratio=0,
        test_size_or_ratio=(min_test_size, test_size),
    )

    def read_and_sample_from_task(paths: List[RichPath], id: int) -> Iterable[FSMolTaskSample]:
        for i, path in enumerate(paths):
            task = FSMolTask.load_from_file(path)
            try:
                yield task_sampler.sample(task, seed=id + i)
            except SamplingException as e:
                logger.debug(f"Sampling task failed:\n{str(e)}")

    # A metatesting epoch is when given a pre-trained model, we iterate over all validation tasks,
    # fine-tune the model to convergence and report back the results. The resulting metric
    # is the mean over the individual results.
    best_valid_epoch = 0
    best_valid_metric = metatrain_valid_fn(
        model_weights={var.name: var.value() for var in model.trainable_variables}
    )
    logger.info(f"Metatesting - initial mean valid metric:  {best_valid_metric:.4f}")
    save_model(save_file, model)
    train_time_start = time.time()

    for epoch in range(1, max_epochs + 1):
        train_task_samples = dataset.get_task_reading_iterable(
            data_fold=DataFold.TRAIN, task_reader_fn=read_and_sample_from_task
        )
        logger.info(f"== Epoch {epoch}")

        """
        Runs an epoch over the meta-training tasks. The model is trained according to standard
        MAML. Concretely, it implements the outer loop of Algorithm 2 of
        https://arxiv.org/pdf/1703.03400.pdf, i.e. lines 3-10.
        See comments in code for mapping of statements to the algorithm.
        """
        epoch_time_start = time.time()
        epoch_metatrain_test_loss, epoch_metatrain_num_graphs = 0.0, 0
        # Alg 2, line 3: "Sample batch of tasks T_i ~ p(T)"
        logger.log(PROGRESS_LOG_LEVEL, f" = Running metatraining on training tasks.")
        for step, task_batch in enumerate(chunked(train_task_samples, n=task_batch_size)):
            batch_metatrain_test_loss, batch_metatrain_num_test_graphs = metatrain_on_task_samples(
                model,
                task_batch,
                max_num_nodes_in_batch=max_num_nodes_in_batch,
                max_num_inner_train_steps=max_num_inner_train_steps,
                metatrain_task_specific_parameters=metatrain_task_specific_parameters,
                quiet=quiet,
            )
            epoch_metatrain_test_loss += batch_metatrain_test_loss
            epoch_metatrain_num_graphs += batch_metatrain_num_test_graphs
            task_batch_per_graph_loss = batch_metatrain_test_loss / batch_metatrain_num_test_graphs
            logger.info(
                f"  Step {step:4d}"
                f"  |  Graph avg. loss = {task_batch_per_graph_loss:.4f}"
                f"  |  Tasks included: {[task.name for task in task_batch]}",
            )

        total_time = time.time() - epoch_time_start
        metatrain_mean_test_loss = epoch_metatrain_test_loss / epoch_metatrain_num_graphs
        metatrain_speed = epoch_metatrain_num_graphs / total_time

        logger.info(
            f" Metatraining:  {metatrain_mean_test_loss:.4f} loss | {metatrain_speed:.2f} graphs/s"
        )
        if aml_run is not None:
            aml_run.log("metatrain_epoch_loss", metatrain_mean_test_loss)
            aml_run.log("metatrain_speed", metatrain_speed)
        logger.info(f" = Running metatesting on validation tasks.")
        valid_metric = metatrain_valid_fn(
            model_weights={var.name: var.value() for var in model.trainable_variables}
        )
        logger.info(f" Metatesting - mean valid metric:  {valid_metric:.4f}")

        # Save if good enough.
        if valid_metric > best_valid_metric:
            logger.info(
                f"  (Best epoch so far, target metric increased to {valid_metric:.5f} from {best_valid_metric:.5f}.)",
            )
            save_model(save_file, model)
            best_valid_metric = valid_metric
            best_valid_epoch = epoch
        elif epoch - best_valid_epoch >= patience:
            total_time = time.time() - train_time_start
            logger.info(
                f"Stopping training after {patience} epochs without "
                f"improvement on validation metric.",
            )

            logger.log(
                PROGRESS_LOG_LEVEL,
                f"Training took {total_time}s. Best validation metric: {best_valid_metric}",
            )
            break

    return save_file


def run_metatraining_from_args(args):
    # Get the housekeeping going and start logging:
    out_dir, fsmol_dataset, aml_run = set_up_train_run("MAML", args, tf=True)

    stub_graph_dataset = FSMolStubGraphDataset()

    # Create the MAML model:
    model_cls = MetalearningGraphBinaryClassificationTask
    model_params = model_cls.get_default_hyperparameters(args.gnn_type)
    model_params.update(MAML_MODEL_DEFAULT_HYPER_PARAMS)
    model_params.update(args.model_params_override or {})
    model_params.update({"metalearning_outer_loop_rate_scale": args.outer_loop_lr_scale})
    model = model_cls(model_params, dataset=stub_graph_dataset)
    data_description = stub_graph_dataset.get_batch_tf_data_description()
    model.build(data_description.batch_features_shapes)

    total_param_num = 0
    for var in model.trainable_variables:
        total_param_num += reduce(operator.mul, var.get_shape())
    logger.info(f"\tNum parameters {total_param_num}")

    # create a valid model copy with unique variable identifiers
    with tf.name_scope("valid"):
        # update validation model parameters so single task optimizers are not for inner loop
        model_params.update(VALIDATION_MODEL_DEFAULT_HYPER_PARAMS)
        model_params.update(args.validation_model_params_override or {})
        valid_model = model_cls(model_params, dataset=stub_graph_dataset)
        valid_model.build(data_description.batch_features_shapes)

    if args.pretrained_model is not None:
        print(f"I: Loading weights for initialisation from {args.pretrained_model}.")
        with contextlib.redirect_stdout(FileLikeLogger(logger, logging.INFO)):
            load_weights_verbosely(args.pretrained_model, model)

    logger.info(f"Model parameters: {json.dumps(unwrap_tf_tracked_data(model._params))}")

    metatrain_loop(
        model=model,
        metatrain_valid_fn=partial(
            eval_model_by_finetuning_on_tasks,
            model=valid_model,
            dataset=fsmol_dataset,
            max_num_nodes_in_batch=10000,
            metric_to_use=args.test_metric,
            train_set_sample_sizes=args.validation_train_set_sizes,
            test_set_size=args.validation_test_set_size,
            num_samples=args.validation_num_samples,
            aml_run=aml_run,
        ),
        dataset=fsmol_dataset,
        max_num_nodes_in_batch=10000,
        max_epochs=args.max_epochs,
        patience=args.patience,
        save_dir=out_dir,
        task_batch_size=args.task_batch_size,
        train_size=args.train_size,
        test_size=args.test_size,
        min_test_size=args.min_test_size,
        max_num_inner_train_steps=args.max_num_inner_train_steps,
        metatrain_task_specific_parameters=args.metatrain_task_specific_parameters,
        quiet=args.quiet,
        aml_run=aml_run,
    )


def get_metatraining_argparser():
    parser = argparse.ArgumentParser(
        description="Metatrain a GNN model.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    add_train_cli_args(parser)

    parser.add_argument(
        "--gnn-type",
        type=str,
        choices=sorted(get_known_message_passing_classes()),
        default="GNN_Edge_MLP",
        help="GNN model type to train.",
    )

    parser.add_argument(
        "--test-metric",
        type=str,
        default="avg_precision",
        help="Metric to report from metatesting on validation tasks.",
    )

    parser.add_argument(
        "--outer-loop-lr-scale",
        type=float,
        default=0.1,
        help="Scaling of outer loop learning rate relative to inner",
    )

    parser.add_argument(
        "--pretrained-model",
        type=str,
        help="Use weights from given pretrained model where possible.",
    )

    parser.add_argument(
        "--model-params-override",
        type=lambda s: json.loads(s),
        help="JSON dictionary overriding model hyperparameter values.",
    )

    parser.add_argument(
        "--validation-model-params-override",
        type=lambda s: json.loads(s),
        help="JSON dictionary overriding validation model hyperparameter values.",
    )

    parser.add_argument(
        "--max-epochs", type=int, default=10000, help="Maximal number of meta epochs to train for."
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Maximal number of meta epochs to continue training without improvement.",
    )

    parser.add_argument(
        "--task-batch-size",
        type=int,
        default=5,
        help="Number of tasks to use per step of the outer MAML loop.",
    )

    parser.add_argument(
        "--train-size",
        type=int,
        default=16,
        help="Number of samples to use per class as metatraining context.",
    )

    parser.add_argument(
        "--test-size",
        type=int,
        default=256,
        help="Number of samples to use as metatraining query (if possible, otherwise see --min-test-size).",
    )

    parser.add_argument(
        "--min-test-size",
        type=int,
        default=32,
        help="Minimal number of samples to use as metatraining query; if not possible, task will be skipped.",
    )

    parser.add_argument(
        "--max-num-inner-train-steps",
        type=int,
        default=1,
        help="Number of steps to take per task in inner loop of MAML on train data.",
    )

    parser.add_argument(
        "--metatrain-task-specific-parameters",
        type=str2bool,
        default=True,
        help="Toggles metatraining of task-specific (readout + final MLP) parameters.",
    )

    parser.add_argument(
        "--quiet", action="store_true", help="Generate less output during training."
    )

    parser.add_argument(
        "--validation-train-set-sizes",
        type=json.loads,
        default=[16, 128],
        help="JSON list selecting the number of datapoints sampled as training data during evaluation through finetuning on the validation tasks.",
    )

    parser.add_argument(
        "--validation-test-set-size",
        type=int,
        default=512,
        help="Maximum number of datapoints sampled as test data during evaluation through finetuning on the validation tasks.",
    )

    parser.add_argument(
        "--validation-num-samples",
        type=int,
        default=5,
        help="Number of samples considered for each train set size for each validation task during evaluation through finetuning.",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug routines")

    return parser


def run():
    parser = get_metatraining_argparser()
    args = parser.parse_args()

    # Make TF less noisy:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
    tf.get_logger().setLevel("ERROR")
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    run_and_debug(lambda: run_metatraining_from_args(args), enable_debugging=args.debug)


if __name__ == "__main__":
    run()
