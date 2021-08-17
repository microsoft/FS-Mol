from typing import Any, Dict, Tuple, Optional, Iterable, List, Union

import numpy as np
import sklearn.metrics as metrics
import tensorflow as tf
from tf2_gnn.data import GraphDataset
from tf2_gnn.models import GraphBinaryClassificationTask
from tf2_gnn.utils.polynomial_warmup_and_decay_schedule import PolynomialWarmupAndDecaySchedule

from fs_mol.utils.metrics import BinaryEvalMetrics, compute_binary_task_metrics


SMALL_NUMBER = 1e-7


class NoCachedStateError(Exception):
    def __init__(self):
        super().__init__()

    def __str__(self):
        return f"No cached state exists, model has not yet been built."


class MetalearningGraphBinaryClassificationTask(GraphBinaryClassificationTask):
    @classmethod
    def get_default_hyperparameters(cls, mp_style: Optional[str] = None) -> Dict[str, Any]:
        super_params = super().get_default_hyperparameters(mp_style)
        these_hypers: Dict[str, Any] = {
            "optimizer": "adam",  # change to "sgd" for maml as this addresses the inner loop
            "use_lr_schedule_single_task_optimizers": True,  # default is to allow lr schedules. Set to False for MAML (inner loop)
            "initial_emb_lr": 0.0005,
            "gnn_lr": 0.0005,
            "readout_lr": 0.001,
            "final_mlp_lr": 0.001,
            "use_loss_class_weights": True,
            "initial_final_lr_ratio": 0.1,  # if an lr warmup or decay is used, ratio of initial or final lr to regular lr.
            "apply_ANIL": False,  # No single task learning on model central components
            "outer_loop_optimizer": "adam",  # multitask optimizer
            "metalearning_outer_loop_rate_scale": 0.1,  # relative to single task learning rates
            "use_lr_schedules_maml_outer_loop": False,  # MAML outer loop optimizers default to no schedules.
        }
        super_params.update(these_hypers)
        return super_params

    def __init__(self, params: Dict[str, Any], dataset: GraphDataset, name: str = None):
        super().__init__(params, dataset=dataset, name=name)
        self._initial_emb_lr = params["initial_emb_lr"]
        self._gnn_lr = params["gnn_lr"]
        self._readout_lr = params["readout_lr"]
        self._final_mlp_lr = params["final_mlp_lr"]
        self._use_loss_class_weights = params["use_loss_class_weights"]

        self._initial_optimizer_states: Optional[Dict[str, List[tf.Tensor]]] = None

    def build(self, input_shapes: Dict[str, Any]):
        super().build(input_shapes)
        self._initialize_optimizers()
        self._initial_optimizer_states = self.optimizer_weights

    @property
    def optimizer_weights(self) -> Dict[str, List[tf.Tensor]]:
        # get current state of optimizers
        # neglect outer optimizers for now, for simplicity.
        state_dict = {}
        for opt in [
            "_initial_emb_optimizer",
            "_gnn_optimizer",
            "_readout_optimizer",
            "_final_mlp_optimizer",
        ]:
            optimizer = getattr(self, opt, None)
            if optimizer is not None:
                state_dict[opt] = optimizer.get_weights()
        return state_dict

    def reset_optimizer_state_to_initial(self) -> None:
        if self._initial_optimizer_states is None:
            raise NoCachedStateError()
        else:
            # reset optimizers to cached state if a state has been previously cached
            for opt, weights in self._initial_optimizer_states.items():
                optimizer = getattr(self, opt, None)
                optimizer.set_weights(weights)

    def _apply_gradients(
        self,
        gradient_variable_pairs: Iterable[Tuple[tf.Tensor, tf.Variable]],
        outer_loop: bool = False,
    ) -> None:
        # default is use regular optimizers. Outer loop gradients applied by hand, so flag is passed.

        # assign which gradient-variable pairs belong to which groups
        initial_embedding_weights, gnn_weights, readout_weights, final_mlp_weights = [], [], [], []

        for gv in gradient_variable_pairs:
            var_name = gv[1].name
            if var_name.startswith(self.__class__.__name__):
                if var_name.startswith(f"{self.__class__.__name__}/MLP_"):
                    final_mlp_weights.append(gv)
                else:
                    readout_weights.append(gv)
            elif var_name.startswith(
                f"{self._gnn._message_passing_class.__name__}_GNN/gnn_initial_node_projection"
            ):
                initial_embedding_weights.append(gv)
            else:
                gnn_weights.append(gv)

        # apply outer loop gradients if this is an outer loop gradient step
        if outer_loop:
            self._outer_initial_emb_optimizer.apply_gradients(initial_embedding_weights)
            self._outer_gnn_optimizer.apply_gradients(gnn_weights)
            self._outer_readout_optimizer.apply_gradients(readout_weights)
            self._outer_final_mlp_optimizer.apply_gradients(final_mlp_weights)
        # apply standard gradients otherwise: validation models and other use different 'standard'
        # optimizers.
        else:
            self._initial_emb_optimizer.apply_gradients(initial_embedding_weights)
            self._gnn_optimizer.apply_gradients(gnn_weights)
            self._readout_optimizer.apply_gradients(readout_weights)
            self._final_mlp_optimizer.apply_gradients(final_mlp_weights)

    def _make_learning_rate(
        self,
        learning_rate: Optional[
            Union[float, tf.keras.optimizers.schedules.LearningRateSchedule]
        ] = None,
        initial_final_lr_ratio: Optional[float] = None,
        allow_schedule: bool = True,
    ) -> Union[float, tf.keras.optimizers.schedules.LearningRateSchedule]:
        """
        Return learning rate.

        Initial and final learning rates are fractional quantities of learning rate.
        Args:
            learning_rate: Optional setting for learning rate; if unset, will
                use value from self._params["learning_rate"].
            initial_final_lr_ratio: Optional setting for multiplier for initial or final rate;
                if unset, will use value from self._params["initial_final_lr_ratio"].
        """
        if learning_rate is not None and isinstance(
            learning_rate, tf.keras.optimizers.schedules.LearningRateSchedule
        ):
            # if a schedule is passed, pass on directly
            return learning_rate

        else:

            if learning_rate is None:
                learning_rate = self._params["learning_rate"]

            num_warmup_steps = self._params.get("learning_rate_warmup_steps")
            num_decay_steps = self._params.get("learning_rate_decay_steps")
            # if warmup or decay is asked for, make a schedule (unless flag to stop schedule passed eg.
            # may be used in outer loop).
            # else, just return the passed learning rate.
            if (num_warmup_steps is not None or num_decay_steps is not None) and allow_schedule:
                # get a ratio of initial to final learning rates (can still be None)
                if initial_final_lr_ratio is None:
                    initial_final_lr_ratio = self._params["initial_final_lr_ratio"]
                # set up the initial and final learning rates
                if initial_final_lr_ratio is not None:
                    initial_learning_rate = learning_rate * initial_final_lr_ratio
                    final_learning_rate = learning_rate * initial_final_lr_ratio
                else:
                    initial_learning_rate = 0.00001
                    final_learning_rate = 0.00001
                # reset the learning rates for the warmup/decay that is *not* required.
                if num_warmup_steps is None:
                    num_warmup_steps = -1  # Make sure that we have no warmup phase
                    initial_learning_rate = learning_rate
                if num_decay_steps is None:
                    num_decay_steps = 1  # Value doesn't matter, but needs to be non-zero
                    final_learning_rate = learning_rate
                # assume a linear schedule
                learning_rate = PolynomialWarmupAndDecaySchedule(
                    learning_rate=learning_rate,
                    warmup_steps=num_warmup_steps,
                    decay_steps=num_decay_steps,
                    initial_learning_rate=initial_learning_rate,
                    final_learning_rate=final_learning_rate,
                    power=1.0,
                )

            return learning_rate

    def _make_optimizer(
        self,
        optimizer_name: str = "sgd",
        learning_rate: Optional[
            Union[float, tf.keras.optimizers.schedules.LearningRateSchedule]
        ] = None,
    ) -> tf.keras.optimizers.Optimizer:
        """
        Create fresh optimizer.
        Args:
            learning_rate: Optional setting for learning rate; if unset, will
                use value from self._params["learning_rate"].
            initialize_weights: Setting to initialize all variable slots when optimizer
            is first created, prior to first apply_gradients call.
        """
        if learning_rate is None:
            learning_rate = self._params["learning_rate"]

        if optimizer_name == "sgd":
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=learning_rate, momentum=self._params["momentum"]
            )
        elif optimizer_name == "rmsprop":
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=learning_rate,
                momentum=self._params["momentum"],
                rho=self._params["rmsprop_rho"],
            )
        elif optimizer_name == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        else:
            raise Exception('Unknown optimizer "%s".' % (self._params["optimizer"]))

        # create weights slots to enable meaningful state caching
        # this is a horrible hack to get around not using tf2 >= 2.3
        _ = optimizer.iterations
        optimizer._create_hypers()
        optimizer._create_slots(self.trainable_variables)

        return optimizer

    def compute_task_metrics(
        self,
        batch_features: Dict[str, tf.Tensor],
        task_output: Any,
        batch_labels: Dict[str, tf.Tensor],
    ) -> Dict[str, tf.Tensor]:

        ce_per_sample = tf.keras.losses.binary_crossentropy(
            y_true=tf.expand_dims(batch_labels["target_value"], -1),
            y_pred=tf.expand_dims(task_output, -1),
            from_logits=False,
        )

        if self._use_loss_class_weights:
            # We want to do class weighting but are too lazy to compute full dataset statistics (and
            # also know that in most cases, we only have a single batch anyway). Hence, compute per-batch
            # label stats:
            num_labels = tf.cast(tf.shape(batch_labels["target_value"])[0], tf.float32)
            sample_has_pos_label = tf.math.greater(batch_labels["target_value"], 0.0)
            num_pos_batch_labels = tf.reduce_sum(tf.cast(sample_has_pos_label, tf.float32))
            num_neg_batch_labels = num_labels - num_pos_batch_labels

            pos_label_weight = num_labels / (2.0 * num_pos_batch_labels + SMALL_NUMBER)
            neg_label_weight = num_labels / (2.0 * num_neg_batch_labels + SMALL_NUMBER)
            sample_weights = tf.where(sample_has_pos_label, pos_label_weight, neg_label_weight)
            ce = tf.reduce_mean(ce_per_sample * sample_weights)
        else:
            ce = tf.reduce_mean(ce_per_sample)

        num_graphs = tf.cast(batch_features["num_graphs_in_batch"], tf.float32)
        return {
            "loss": ce,
            "num_graphs": num_graphs,
            # We store the entirety of labels/results to enable more metric computations:
            "labels": batch_labels["target_value"],
            "predictions": task_output,
        }

    def compute_epoch_metrics(self, task_results: List[Any]) -> Tuple[float, str]:
        labels, predictions = [], []
        for task_result in task_results:
            predictions.append(task_result["predictions"])
            labels.append(task_result["labels"])
        predictions = tf.concat(predictions, axis=0).numpy()
        labels = tf.concat(labels, axis=0).numpy()

        average_precision = metrics.average_precision_score(y_true=labels, y_score=predictions)
        return -average_precision, f"Average Precision = {average_precision:.3f}"

    def _initialize_optimizers(
        self,
    ) -> None:
        # implementation of standard optimizers depends on use in validation or inner loop.
        # default behaviour ignores inner loop; this is set to True if building a MAML model.

        # initialise outer (unused if not MAML model)
        outer_optimizer_name = self._params["outer_loop_optimizer"].lower()
        outer_optimizer_config = {
            "initial_emb_optimizer": self._params["metalearning_outer_loop_rate_scale"]
            * self._initial_emb_lr,
            "gnn_optimizer": self._params["metalearning_outer_loop_rate_scale"] * self._gnn_lr,
            "readout_optimizer": self._params["metalearning_outer_loop_rate_scale"]
            * self._readout_lr,
            "final_mlp_optimizer": self._params["metalearning_outer_loop_rate_scale"]
            * self._final_mlp_lr,
        }
        for opt_name, opt_lr in outer_optimizer_config.items():
            if getattr(self, f"_outer_{opt_name}", None) is None:
                lr = self._make_learning_rate(
                    learning_rate=opt_lr,
                    allow_schedule=self._params["use_lr_schedules_maml_outer_loop"],
                )
                optimizer = self._make_optimizer(
                    optimizer_name=outer_optimizer_name,
                    learning_rate=lr,
                )
                setattr(self, f"_outer_{opt_name}", optimizer)

        # assemble single task optimizers
        optimizer_name = self._params["optimizer"].lower()

        # assemble correct learning rates
        if self._params["apply_ANIL"]:
            _initial_emb_lr = 0.0
            _gnn_lr = 0.0
        else:
            _initial_emb_lr = self._initial_emb_lr
            _gnn_lr = self._gnn_lr

        inner_optimizer_config = {
            "initial_emb_optimizer": _initial_emb_lr,
            "gnn_optimizer": _gnn_lr,
            "readout_optimizer": self._readout_lr,
            "final_mlp_optimizer": self._final_mlp_lr,
        }
        for opt_name, opt_lr in inner_optimizer_config.items():
            if getattr(self, f"_{opt_name}", None) is None:
                lr = self._make_learning_rate(
                    learning_rate=opt_lr,
                    allow_schedule=self._params["use_lr_schedule_single_task_optimizers"],
                )
                optimizer = self._make_optimizer(
                    optimizer_name=optimizer_name,
                    learning_rate=lr,
                )
                setattr(self, f"_{opt_name}", optimizer)

    def evaluate_model(self, dataset: tf.data.Dataset) -> BinaryEvalMetrics:
        predictions = self.predict(dataset).numpy()
        labels = []
        for _, batch_labels in dataset:
            labels.append(batch_labels["target_value"])
        return compute_binary_task_metrics(
            predictions=predictions, labels=np.concatenate(labels, axis=0)
        )
