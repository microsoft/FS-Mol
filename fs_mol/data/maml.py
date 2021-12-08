import logging
from typing import Dict, Any, Iterable, List, Tuple, Optional, Iterator

import numpy as np
import tensorflow as tf
from tf2_gnn.data.graph_dataset import GraphBatchTFDataDescription

from fs_mol.data.fsmol_dataset import NUM_EDGE_TYPES, NUM_NODE_FEATURES
from fs_mol.data.fsmol_batcher import FSMolBatcher, fsmol_batch_finalizer
from fs_mol.data.fsmol_task import MoleculeDatapoint

logger = logging.getLogger(__name__)


def maml_batch_finalizer(batch_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    fsmol_batch = fsmol_batch_finalizer(batch_data)

    batch_features = {
        "node_features": fsmol_batch.node_features,
        "node_to_graph_map": fsmol_batch.node_to_graph.astype(np.int32),
        "num_graphs_in_batch": fsmol_batch.num_graphs,
    }
    for edge_type_idx in range(NUM_EDGE_TYPES):
        batch_features[f"adjacency_list_{edge_type_idx}"] = fsmol_batch.adjacency_lists[
            edge_type_idx
        ].astype(np.int32)
        batch_features[f"edge_features_{edge_type_idx}"] = fsmol_batch.edge_features[edge_type_idx]

    batch_labels = {
        "target_value": np.stack(batch_data["bool_labels"], axis=0).astype(np.float32),
    }

    return batch_features, batch_labels


class FSMolStubGraphDataset:
    def __init__(self):
        self._params = {"edge_features_dims": {}}

    @property
    def params(self) -> Dict[str, Any]:
        return self._params

    @property
    def num_edge_types(self) -> int:
        return NUM_EDGE_TYPES

    @property
    def num_node_features(self) -> int:
        return NUM_NODE_FEATURES

    def get_batch_tf_data_description(self):
        batch_features_types = {
            "node_features": tf.float32,
            "node_to_graph_map": tf.int32,
            "num_graphs_in_batch": tf.int32,
        }
        batch_features_shapes = {
            "node_features": (None, self.num_node_features),
            "node_to_graph_map": (None,),
            "num_graphs_in_batch": (),
        }
        for edge_type_idx in range(self.num_edge_types):
            batch_features_types[f"adjacency_list_{edge_type_idx}"] = tf.int32
            batch_features_shapes[f"adjacency_list_{edge_type_idx}"] = (None, 2)
            batch_features_types[f"edge_features_{edge_type_idx}"] = tf.float32
            batch_features_shapes[f"edge_features_{edge_type_idx}"] = (
                None,
                self._params["edge_features_dims"].get(edge_type_idx, 0),
            )

        batch_labels_types = {"target_value": tf.float32}
        batch_labels_shapes = {"target_value": (None,)}

        return GraphBatchTFDataDescription(
            batch_features_types=batch_features_types,
            batch_features_shapes=batch_features_shapes,
            batch_labels_types=batch_labels_types,
            batch_labels_shapes=batch_labels_shapes,
        )


class TFGraphBatchIterable(Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]):
    def __init__(
        self,
        samples: List[MoleculeDatapoint],
        shuffle: bool = False,
        max_num_graphs: Optional[int] = None,
        max_num_nodes: Optional[int] = None,
        max_num_edges: Optional[int] = None,
    ):
        self._samples = samples
        self._shuffle = shuffle

        self._batcher = FSMolBatcher(
            max_num_graphs=max_num_graphs,
            max_num_nodes=max_num_nodes,
            max_num_edges=max_num_edges,
            finalizer_callback=maml_batch_finalizer,
        )

    def __iter__(self) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
        # If we need to shuffle, make a copy and do that:
        if self._shuffle:
            samples = list(self._samples)
            np.random.shuffle(samples)
        else:
            samples = self._samples

        return self._batcher.batch(self._samples)
