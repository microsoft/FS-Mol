import math
from dataclasses import dataclass
from typing import (
    Callable,
    Generic,
    List,
    Dict,
    Optional,
    Tuple,
    Any,
    TypeVar,
    Iterator,
    Iterable,
)

import numpy as np

from fs_mol.data.fsmol_dataset import NUM_EDGE_TYPES
from fs_mol.data.fsmol_task import MoleculeDatapoint


@dataclass(frozen=True)
class FSMolBatch:
    """General data structure for holding information about graph-featurized molecules in a
    batch. Note that batches of unequally sized graphs are formed from multiple graphs by
    combining into one large disconnected graph, where each new potential addition is tested
    to check that it will not exceed the allowed number of nodes or edges per batch.

    Args:
        num_graphs: total number of graphs in the batch.
        num_nodes: total number of nodes in the batch, V. Should be limited to a maximum.
        num_edges: total number of edges in batch; one batch contains multiple disconnected
            graphs where edges and nodes are renumbered accordingly.
        node_features: each node has a vector representation dependent on featurisation,
            e.g. atom type, charge, valency. [V, atom_features] float, where V is number of nodes
        adjacency_lists: Lists of all edges in the batch, for each edge type.
            list, len num_edge_types, elements [num edges, 2] int tensors
        edge_features: edges may also have vector representation carrying information specific
            to the edge. list, len num_edge_types, elements [num edges, ED] float tensors
        node_to_graph: Vector of indices of length V. Mapping from nodes to the graphs
            to which they belong.
    """

    num_graphs: int
    num_nodes: int
    num_edges: int
    node_features: np.ndarray  # [V, atom_features] float
    adjacency_lists: List[
        np.ndarray
    ]  # list, len num_edge_types, elements [num edges, 2] int tensors
    edge_features: List[
        np.ndarray
    ]  # list, len num_edge_types, elements [num edges, ED] float tensors
    node_to_graph: np.ndarray  # [V] long


BatchFeatureType = TypeVar("BatchFeatureType")
BatchLabelType = TypeVar("BatchLabelType")


def fsmol_batch_finalizer(batch_data: Dict[str, Any]) -> FSMolBatch:
    """
    Default implementation of a batch finalizer. Converts a batch that has reached maximum size
    into a final FSMolBatch object, which can then be consumed by e.g. a training loop.

    Args:
        batch_data: Dictionary containing batch data, initialised and populated
            from "MoleculeDatapoints" by the "FSMolBatcher.batch()" method.
    """
    adjacency_lists = []
    for adj_lists in batch_data["adjacency_lists"]:
        if len(adj_lists) > 0:
            adjacency_lists.append(np.concatenate(adj_lists, axis=0))
        else:
            adjacency_lists.append(np.zeros(shape=(0, 2), dtype=np.int64))

    edge_features = []
    for edge_type_idx, edge_feats in enumerate(batch_data["edge_features"]):
        if len(edge_feats) > 0:
            edge_features.append(np.concatenate(edge_feats, axis=0))
        else:
            edge_features.append(
                np.zeros(shape=(adjacency_lists[edge_type_idx].shape[0], 0), dtype=np.float32)
            )

    return FSMolBatch(
        num_graphs=batch_data["num_graphs"],
        num_nodes=batch_data["num_nodes"],
        num_edges=batch_data["num_edges"],
        node_features=np.concatenate(batch_data["node_features"], axis=0),
        adjacency_lists=adjacency_lists,
        edge_features=edge_features,
        node_to_graph=np.concatenate(batch_data["node_to_graph"], axis=0),
    )


class FSMolBatcher(Generic[BatchFeatureType, BatchLabelType]):
    """Create a batcher object. Acts on an iterable over MoleculeDatapoints to create
    suitably sized mini-batches. The batch method checks that adding another graph
    will not cause overflow of maximum number of total edges, nodes or graphs,
    and creates a new batch if that is the case.

    Sampling can only fail if a fixed size is requested for any of the folds that cannot be
    satisfied.

    Args:
        max_num_graphs (Optional): If set, the maximum number of graphs added to a batch.
        max_num_nodes (Optional): If set, maximum number of nodes added to a batch.
        max_num_edges (Optional): If set, maximum permitted edges in a batch.
        init_callback (Optional): Callable that can be passed to operate on the initial batch to,
            for example, add additional members to a batch data dictionary before building.
        per_datapoint_callback (Optional): Callable that permits a user to operate on a datapoint and batch
            prior to finalisation to perform additional operations. e.g. compute extra features.
        finalizer_callback (Optional): Callable to allow the final batch to be returned in a specific
            form, e.g. as instance of FSMolBatch.
    """

    def __init__(
        self,
        max_num_graphs: Optional[int] = None,
        max_num_nodes: Optional[int] = None,
        max_num_edges: Optional[int] = None,
        init_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        per_datapoint_callback: Optional[
            Callable[[Dict[str, Any], int, MoleculeDatapoint], None]
        ] = None,
        finalizer_callback: Optional[
            Callable[[Dict[str, Any]], Tuple[BatchFeatureType, BatchLabelType]]
        ] = None,
    ):
        if not (
            max_num_graphs is not None or max_num_nodes is not None or max_num_edges is not None
        ):
            raise ValueError(
                f"Need to specify at least one of max_num_graphs, max_num_nodes or max_num_edges!"
            )

        self._max_num_graphs = max_num_graphs or math.inf
        self._max_num_nodes = max_num_nodes or math.inf
        self._max_num_edges = max_num_edges or math.inf

        self._init_callback = init_callback
        self._per_datapoint_callback = per_datapoint_callback
        self._finalizer_callback = finalizer_callback

    def __init_batch(self) -> Dict[str, Any]:
        batch_data = {
            "num_graphs": 0,
            "num_nodes": 0,
            "num_edges": 0,
            "node_features": [],
            "adjacency_lists": [[] for _ in range(NUM_EDGE_TYPES)],
            "edge_features": [[] for _ in range(NUM_EDGE_TYPES)],
            "node_to_graph": [],
            "graph_task": [],
            "bool_labels": [],
            "numeric_labels": [],
        }

        if self._init_callback is not None:
            self._init_callback(batch_data)

        return batch_data

    def __finalize_batch(
        self, batch_data: Dict[str, Any]
    ) -> Tuple[BatchFeatureType, BatchLabelType]:
        if self._finalizer_callback is not None:
            return self._finalizer_callback(batch_data)

        batch_features = fsmol_batch_finalizer(batch_data)
        return batch_features, np.stack(batch_data["bool_labels"], axis=0)

    def batch(
        self, datapoints: Iterable[MoleculeDatapoint]
    ) -> Iterator[Tuple[BatchFeatureType, BatchLabelType]]:
        batch_data = self.__init_batch()

        for datapoint in datapoints:
            num_nodes = len(datapoint.graph.node_features)
            num_edges = sum(len(adj_list) for adj_list in datapoint.graph.adjacency_lists)

            # Decide if this batch is full:
            if (
                (batch_data["num_graphs"] + 1 > self._max_num_graphs)
                or (batch_data["num_nodes"] + num_nodes > self._max_num_nodes)
                or (batch_data["num_edges"] + num_edges > self._max_num_edges)
            ):
                yield self.__finalize_batch(batch_data)
                batch_data = self.__init_batch()

            sample_id_in_batch = batch_data["num_graphs"]

            # Collect the actual graph information:
            batch_data["node_features"].append(datapoint.graph.node_features)
            for edge_type, adj_list in enumerate(datapoint.graph.adjacency_lists):
                batch_data["adjacency_lists"][edge_type].append(adj_list + batch_data["num_nodes"])
            for edge_type, edge_feats in enumerate(datapoint.graph.edge_features):
                batch_data["edge_features"][edge_type].append(edge_feats)
            batch_data["node_to_graph"].append(
                np.full(shape=(num_nodes,), fill_value=sample_id_in_batch, dtype=np.int64)
            )

            # Label information:
            batch_data["bool_labels"].append(datapoint.bool_label)
            batch_data["numeric_labels"].append(datapoint.numeric_label)

            # Some housekeeping information:
            batch_data["graph_task"].append(datapoint.task_name)
            batch_data["num_graphs"] += 1
            batch_data["num_nodes"] += num_nodes
            batch_data["num_edges"] += num_edges

            if self._per_datapoint_callback is not None:
                self._per_datapoint_callback(batch_data, sample_id_in_batch, datapoint)

        if batch_data["num_graphs"] > 1:  # single-element batches are problematic for BatchNorm
            yield self.__finalize_batch(batch_data)


class FSMolBatchIterable(
    Iterable[Tuple[BatchFeatureType, BatchLabelType]], Generic[BatchFeatureType, BatchLabelType]
):
    def __init__(
        self,
        samples: List[MoleculeDatapoint],
        batcher: FSMolBatcher[BatchFeatureType, BatchLabelType],
        shuffle: bool = False,
        seed: int = 0,
    ):
        self._samples = samples
        self._batcher = batcher
        self._shuffle = shuffle
        self._rng = np.random.Generator(np.random.PCG64(seed=seed))

    def __iter__(self) -> Iterator[Tuple[BatchFeatureType, BatchLabelType]]:
        if self._shuffle:
            samples = list(self._samples)
            self._rng.shuffle(samples)
        else:
            samples = self._samples

        return self._batcher.batch(samples)
