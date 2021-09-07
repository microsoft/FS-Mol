# This is a collection of utility classes / torch.nn.Modules that can be used to
# obtain task-specific parameters (e.g., for featurewise linear modulation) in
# different places. For each different set of parameters you require, you should be
# calling the (one!) instance of TaskEmbeddingLayerProvider you have floating around,
# which in turn produces TaskEmbeddingLayer objects that can be used to map task ids
# to task embeddings of the right dimensions.
#
# Things are handled using (abstract) superclasses to allow different variants
# of implementing this behaviour. Currently, two variants are implemented:
#  (1) "Learnable": Each `LearnableTaskEmbeddingLayer` has its own embedding layer,
#      with its own separate task embedding.
#  (2) "Projected": There is one (core) learnable task embedding that is used
#      repeatedly, and each `ProjectedTaskEmbeddingLayer` produced appropriate
#      embeddings by projecting that central embedding.
#
# A third variant would be a variation of the "Projected" scheme, in which the task
# embeddings are not learnable, but produced by another subnetwork.

import math
from abc import ABC, abstractmethod
from typing import Optional, List

import torch
from torch import nn

"""
Task embedding layers
"""


class TaskEmbeddingLayer(nn.Module, ABC):
    """Abstract superclass of layers that map a list of task IDs to a vector representation."""

    def __init__(
        self,
        num_tasks: int,
        dim: int,
        target_mean: Optional[float] = None,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ):
        """
        Args:
            num_tasks: Number of tasks for use in task-dependent readouts.
            dim: Dimension of the task embedding.
            target_mean: Optionally determines the target mean of the value to return.
                This is important for FiLM uses, where \gamma should have a mean around 1.
            min_val: Minimal value (results will be clipped to that value)
            max_val: Maximal value (results will be clipped to that value)
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.dim = dim
        self.target_mean = target_mean
        assert (min_val is None) == (max_val is None)
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, task_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            task_ids: long tensor of rank 1.

        Returns:
            float tensor of shape [len(task_ids), dim].
        """
        task_embeddings = self.embed(task_ids)
        if self.min_val is not None and self.max_val is not None:
            task_embeddings = torch.clamp(task_embeddings, min=self.min_val, max=self.max_val)
        return task_embeddings

    @abstractmethod
    def embed(self, task_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            task_ids: long tensor of rank 1.

        Returns:
            float tensor of shape [len(task_ids), dim].
        """
        pass

    @abstractmethod
    def reinitialize_task_parameters(self, new_num_tasks: Optional[int] = None) -> None:
        """Reinitializes all task-specific information, but retains the connection
        between task embeddings and their different uses, if present.

        Args:
            new_num_tasks: Optional new number of tasks. If not present, number of tasks
                will not change.
        """
        pass


class LearnedTaskEmbeddingLayer(TaskEmbeddingLayer):
    """Class that can map task IDs to a vector representation obtained from a standard
    learnable embedding layer encapsulated by this class.
    """

    def __init__(self, *args, **kwargs):
        """See superclass."""
        super().__init__(*args, **kwargs)
        self.task_emb = self.__get_embedding()

    def __get_embedding(self) -> nn.Embedding:
        """Get a fresh embedding for the tasks.
        Weights can be initialised N(0,1), or with target mean.
        """
        init_weights = None
        if self.target_mean is not None:
            init_weights = torch.normal(
                mean=self.target_mean, std=0.03, size=(self.num_tasks, self.dim)
            )

        return nn.Embedding(
            num_embeddings=self.num_tasks,
            embedding_dim=self.dim,
            _weight=init_weights,
        )

    def embed(self, task_ids: torch.Tensor) -> torch.Tensor:
        """See superclass."""
        return self.task_emb(task_ids)

    def reinitialize_task_parameters(self, new_num_tasks: Optional[int] = None) -> None:
        """See superclass."""
        if new_num_tasks is not None:
            self.num_tasks = new_num_tasks
        self.task_emb = self.__get_embedding()


class ProjectedTaskEmbeddingLayer(TaskEmbeddingLayer):
    """Class that can map task IDs to a vector representation obtained by projecting
    a provided task embedding to the required dimension.
    """

    def __init__(self, input_task_embedding_dim: int, *args, **kwargs):
        """
        See superclass for most arguments.

        Args:
            input_task_embedding_dim: Size of the original embedding, which we are
                projecting to our target here.
        """
        super().__init__(*args, **kwargs)

        self.input_task_embedding_dim = input_task_embedding_dim
        self.embedding_projection = nn.Linear(
            in_features=self.input_task_embedding_dim, out_features=self.dim, bias=False
        )

        # This will be updated to a more sensible value in `update_task_embeddings`
        # once we use the layer:
        self.projected_task_embeddings = torch.zeros(size=(self.num_tasks, self.dim))

    def update_task_embeddings(self, task_embeddings: torch.Tensor):
        """Update the derived (mapped) task embeddings from the provided task embedding."""
        self.projected_task_embeddings = self.embedding_projection(task_embeddings)

    def reinitialize_task_parameters(self, new_num_tasks: Optional[int] = None) -> None:
        """See superclass."""
        if new_num_tasks is not None:
            self.num_tasks = new_num_tasks
        # Nothing else to do, as we do not have any task-specific task parameters

    def embed(self, task_ids: torch.Tensor) -> torch.Tensor:
        """See superclass."""
        task_embeddings = self.projected_task_embeddings[task_ids]  # [len(task_ids), self.dim]
        if self.target_mean:
            task_embeddings += self.target_mean
        return task_embeddings


# --------------------------------defining Linear layers for task embeddings----------------------------------------------


class TaskSpecificLinear(nn.Module):
    def __init__(
        self,
        num_tasks: int,
        in_size: int,
        out_size: int,
    ):
        super().__init__()

        """
        Args:
            num_tasks: Number of tasks to provision this for.
            in_size: Size of the inputs to the task-specific linear layers.
            out_size: Size of the outputs of the task-specific linear layers.
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.in_size = in_size
        self.out_size = out_size

    def extra_repr(self) -> str:
        return f"num_tasks={self.num_tasks}, in_size={self.in_size}, out_size={self.out_size}"

    def forward(self, inputs: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: float tensor of shape [I, self.in_size]
            task_ids: long tensor of shape [I]

        Returns:
            float tensor of shape [I, self.out_size].
        """
        aligned_matrices = self._get_matrices(task_ids)
        return torch.einsum("ni,nio->no", inputs, aligned_matrices)

    @abstractmethod
    def _get_matrices(self, task_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            task_ids: long tensor of shape [I]

        Returns:
            float tensor of shape [I, self.in_size, self.out_size].
        """
        pass

    @abstractmethod
    def reinitialize_task_parameters(self, new_num_tasks: Optional[int] = None) -> None:
        """Reinitializes all task-specific information, but retains the connection
        between task embeddings and their different uses, if present.

        Args:
            new_num_tasks: Optional new number of tasks. If not present, number of tasks
                will not change.
        """
        pass


class LearnedTaskSpecificLinear(TaskSpecificLinear):
    def __init__(self, *args, **kwargs):
        """See superclass."""
        super().__init__(*args, **kwargs)
        self._per_task_matrices = self.__get_matrices()

    def __get_matrices(self) -> nn.Parameter:
        if self.in_size == self.out_size:
            # Initialise things to be the identity matrix + noise:
            return torch.nn.Parameter(
                torch.eye(self.in_size, self.out_size).unsqueeze(0)
                + torch.normal(mean=0, std=0.03, size=(self.num_tasks, self.in_size, self.out_size))
            )
        else:
            # This is taken from the standard torch.nn.Linear initialisation ):
            weights = torch.zeros(size=(self.num_tasks, self.in_size, self.out_size))
            # nn.Linear uses kaiming with mode='fan_in', nonlinearity='leaky_relu'
            fan = self.in_size
            gain = math.sqrt(2.0 / (1 + math.sqrt(5) ** 2))
            std = gain / math.sqrt(fan)
            bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            with torch.no_grad():
                weights.uniform_(-bound, bound)
            return torch.nn.Parameter(weights)

    def _get_matrices(self, task_ids: torch.Tensor) -> torch.Tensor:
        """See superclass."""
        return self._per_task_matrices[task_ids]

    def reinitialize_task_parameters(self, new_num_tasks: Optional[int] = None) -> None:
        """See superclass."""
        if new_num_tasks is not None:
            self.num_tasks = new_num_tasks
        self._per_task_matrices = self.__get_matrices()


class ProjectedTaskSpecificLinear(TaskSpecificLinear):
    def __init__(self, input_task_embedding_dim: int, *args, **kwargs):
        """
        See superclass for most arguments.

        Args:
            input_task_embedding_dim: Size of the original embedding, which we are
                projecting to our target here.
        """
        super().__init__(*args, **kwargs)

        self.input_task_embedding_dim = input_task_embedding_dim
        # This will be updated to a more sensible value in `update_task_embeddings`
        # once we use the layer:
        self._per_task_matrices = torch.zeros(size=(self.num_tasks, self.in_size, self.out_size))

        raise NotImplementedError()

    def update_task_embeddings(self, task_embeddings: torch.Tensor):
        """Update the derived (mapped) task linears from the provided task embedding."""
        raise NotImplementedError()

    def reinitialize_task_parameters(self, new_num_tasks: Optional[int] = None) -> None:
        """See superclass."""
        if new_num_tasks is not None:
            self.num_tasks = new_num_tasks
        # Nothing else to do, as we do not have any task-specific task parameters

    def _get_matrices(self, task_ids: torch.Tensor) -> torch.Tensor:
        """See superclass."""
        return self._per_task_matrices[task_ids]


# --------------------------------Task Embedding Providers (to be called in overall model)-------------------------------


class TaskEmbeddingLayerProvider(nn.Module, ABC):
    """Abstract superclass of utility classes that provide `TaskEmbeddingLayer`s when
    required.
    """

    def __init__(self, num_tasks: int):
        """
        Args:
            num_tasks: Number of tasks to allow.
        """
        super().__init__()
        self.num_tasks = num_tasks

        # This will hold all of the modules that derive from the provided task embedding,
        # which we will use when we update that embedding.
        self.provided_task_specific_layers = nn.ModuleList()

    @abstractmethod
    def get_task_embedding_layer(
        self,
        embedding_dim: int,
        target_mean: Optional[float] = None,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> TaskEmbeddingLayer:
        """
        Args:
            embedding_dim: Dimension of the task embedding.
            target_mean: Optionally determines the target mean of the value to return.
                This is important for FiLM uses, where \gamma should have a mean around 1.
            min_val: Minimal value (results will be clipped to that value)
            max_val: Maximal value (results will be clipped to that value)

        Returns:
            A `TaskEmbeddingLayer` that can be used to map task IDs to task embeddings.
        """
        pass

    @abstractmethod
    def get_task_linear_layer(
        self,
        in_size: int,
        out_size: int,
    ) -> TaskSpecificLinear:
        """
        Args:
            in_size: Size of the inputs to the task-specific linear layers.
            out_size: Size of the outputs of the task-specific linear layers.

        Returns:
            A `TaskSpecificLinear` that can be used to do a task-specific linear mapping
            of a vector.
        """
        pass

    @abstractmethod
    def update_task_embeddings(self) -> None:
        """Hook to trigger updating of derived task embeddings, if required. Should be
        called every time after learnable parameters in the model are updated.
        """
        pass

    def reinitialize_task_parameters(self, new_num_tasks: Optional[int] = None) -> None:
        """Reinitializes all task-specific information, but retains the connection
        between task embeddings and their different uses, if present.

        Args:
            new_num_tasks: Optional new number of tasks. If not present, number of tasks
                will not change.
        """
        if new_num_tasks is not None:
            self.num_tasks = new_num_tasks
        for task_emb_layer in self.provided_task_specific_layers:
            task_emb_layer.reinitialize_task_parameters(self.num_tasks)


class LearnedTaskEmbeddingLayerProvider(TaskEmbeddingLayerProvider):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_task_embedding_layer(
        self,
        embedding_dim: int,
        target_mean: Optional[float] = None,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> TaskEmbeddingLayer:
        """See superclass."""
        l = LearnedTaskEmbeddingLayer(
            num_tasks=self.num_tasks,
            dim=embedding_dim,
            target_mean=target_mean,
            min_val=min_val,
            max_val=max_val,
        )
        self.provided_task_specific_layers.append(l)
        return l

    def get_task_linear_layer(
        self,
        in_size: int,
        out_size: int,
    ) -> TaskSpecificLinear:
        l = LearnedTaskSpecificLinear(
            num_tasks=self.num_tasks,
            in_size=in_size,
            out_size=out_size,
        )
        self.provided_task_specific_layers.append(l)
        return l

    def update_task_embeddings(self) -> None:
        """See superclass."""
        pass  # No udpate required


class ProjectedTaskEmbeddingLayerProvider(TaskEmbeddingLayerProvider):
    def __init__(self, task_embedding_dim: int, *args, **kwargs):
        """See superclass for most arguments.

        Args:
            task_embedding_dim: Dimension of the central task embedding that we are deriving
                uses from.
        """
        super().__init__(*args, **kwargs)
        self.task_embedding_dim = task_embedding_dim

        self.task_embedding = self.__get_task_embedding()

    def __get_task_embedding(self) -> nn.Parameter:
        return nn.Parameter(
            torch.normal(mean=0.0, std=1.0, size=(self.num_tasks, self.task_embedding_dim))
        )

    def get_task_embedding_layer(
        self,
        embedding_dim: int,
        target_mean: Optional[float] = None,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> TaskEmbeddingLayer:
        """See superclass."""
        l = ProjectedTaskEmbeddingLayer(
            input_task_embedding_dim=self.task_embedding_dim,
            num_tasks=self.num_tasks,
            dim=embedding_dim,
            target_mean=target_mean,
            min_val=min_val,
            max_val=max_val,
        )
        self.provided_task_specific_layers.append(l)
        return l

    def get_task_linear_layer(
        self,
        in_size: int,
        out_size: int,
    ) -> TaskSpecificLinear:
        l = ProjectedTaskSpecificLinear(
            input_task_embedding_dim=self.task_embedding_dim,
            num_tasks=self.num_tasks,
            in_size=in_size,
            out_size=out_size,
        )
        self.provided_task_specific_layers.append(l)
        return l

    def update_task_embeddings(self) -> None:
        """See superclass."""
        for l in self.provided_task_specific_layers:
            l.update_task_embeddings(self.task_embedding)

    def reinitialize_task_parameters(self, new_num_tasks: Optional[int] = None) -> None:
        """See superclass."""
        # Here, we only update the central task embedding, and then recompute all the derived ones:
        if new_num_tasks is not None:
            self.num_tasks = new_num_tasks
        self.task_embedding = self.__get_task_embedding()
        self.update_task_embeddings()


class TaskEmbeddingFiLMLayer(nn.Module):
    def __init__(
        self,
        task_embedding_provider: TaskEmbeddingLayerProvider,
        value_dim: int,
    ):
        super().__init__()

        # Initialise FiLM \beta values to 0 (as they are additive)
        # and \gamma values to 1 (as they are multiplicative):
        self.film_beta = task_embedding_provider.get_task_embedding_layer(
            embedding_dim=value_dim,
            target_mean=0.0,
        )
        self.film_gamma = task_embedding_provider.get_task_embedding_layer(
            embedding_dim=value_dim,
            target_mean=1.0,
            min_val=0.0,
            max_val=10.0,
        )

    def forward(self, values, task_ids):
        """
        Args:
            values: float32 tensor of shape [num_values, value_dim]
            task_ids: long tensor of shape [num_values]

        Returns:
            values \odot film_gamma[task_ids] + film_beta[task_ids]
        """
        return values * self.film_gamma(task_ids) + self.film_beta(task_ids)


class TaskEmbeddingFiLMMLP(nn.Module):
    def __init__(
        self,
        task_embedding_provider: TaskEmbeddingLayerProvider,
        input_dim: int,
        out_dim: int,
        hidden_layer_dims: List[int],
        activation=nn.ReLU(),
    ):
        super().__init__()

        self.activation = activation
        self.film_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        cur_hidden_dim = input_dim
        for hidden_layer_dim in hidden_layer_dims:
            self.film_layers.append(TaskEmbeddingFiLMLayer(task_embedding_provider, cur_hidden_dim))
            self.linear_layers.append(nn.Linear(cur_hidden_dim, hidden_layer_dim))
            cur_hidden_dim = hidden_layer_dim
        self.out_film_layer = TaskEmbeddingFiLMLayer(task_embedding_provider, cur_hidden_dim)
        self.out_layer = nn.Linear(cur_hidden_dim, out_dim)

    def forward(self, inputs, task_ids):
        cur_activations = inputs
        for film_layer, linear_layer in zip(self.film_layers, self.linear_layers):
            cur_activations = film_layer(cur_activations, task_ids)
            cur_activations = linear_layer(cur_activations)
            cur_activations = self.activation(cur_activations)
        cur_activations = self.out_film_layer(cur_activations, task_ids)
        cur_activations = self.out_layer(cur_activations)

        return cur_activations
