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

from metamol.data.metamol_dataset import NUM_EDGE_TYPES
from metamol.data.metamol_task import MoleculeDatapoint


@dataclass(frozen=True)
class MetamolBatch:
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


def metamol_batch_finalizer(batch_data: Dict[str, Any]) -> MetamolBatch:
    adjacency_lists = []
    for adj_lists in batch_data["adjacency_lists"]:
        if len(adj_lists) > 0:
            adjacency_lists.append(np.concatenate(adj_lists, axis=0))
        else:
            adjacency_lists.append(np.zeros(shape=(0, 2), dtype=np.int32))

    edge_features = []
    for edge_type_idx, edge_feats in enumerate(batch_data["edge_features"]):
        if len(edge_feats) > 0:
            edge_features.append(np.concatenate(edge_feats, axis=0))
        else:
            edge_features.append(
                np.zeros(shape=(adjacency_lists[edge_type_idx].shape[0], 0), dtype=np.float32)
            )

    return MetamolBatch(
        num_graphs=batch_data["num_graphs"],
        num_nodes=batch_data["num_nodes"],
        num_edges=batch_data["num_edges"],
        node_features=np.concatenate(batch_data["node_features"], axis=0),
        adjacency_lists=adjacency_lists,
        edge_features=edge_features,
        node_to_graph=np.concatenate(batch_data["node_to_graph"], axis=0),
    )


class MetamolBatcher(Generic[BatchFeatureType, BatchLabelType]):
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

        batch_features = metamol_batch_finalizer(batch_data)
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
                np.full(shape=(num_nodes,), fill_value=sample_id_in_batch, dtype=np.int32)
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

        if batch_data["num_graphs"] > 0:
            yield self.__finalize_batch(batch_data)


class MetamolBatchIterable(
    Iterable[Tuple[BatchFeatureType, BatchLabelType]], Generic[BatchFeatureType, BatchLabelType]
):
    def __init__(
        self,
        samples: List[MoleculeDatapoint],
        batcher: MetamolBatcher[BatchFeatureType, BatchLabelType],
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
