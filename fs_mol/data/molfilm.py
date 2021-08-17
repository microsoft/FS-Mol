import dataclasses
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Tuple, Optional, Dict, Any, List, Iterable, Iterator

import numpy as np
from dpu_utils.utils import RichPath
from pyprojroot import here as project_root

sys.path.insert(0, str(project_root()))


from metamol.data import (
    DataFold,
    MetamolDataset,
    MetamolTask,
    MetamolBatch,
    RandomTaskSampler,
    MetamolBatcher,
    MoleculeDatapoint,
    metamol_batch_finalizer,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetamolMolFiLMBatch(MetamolBatch):
    sample_to_task_id: np.ndarray


def molfilm_batcher_init_fn(batch_data: Dict[str, Any]):
    batch_data["sample_to_task_id"] = []


def molfilm_batcher_add_sample_fn(
    batch_data: Dict[str, Any],
    sample_id: int,
    sample: MoleculeDatapoint,
    task_name_to_id: Dict[str, int],
):
    batch_data["sample_to_task_id"].append(task_name_to_id[sample.task_name])


def molfilm_batcher_finalizer_fn(
    batch_data: Dict[str, Any]
) -> Tuple[MetamolMolFiLMBatch, np.ndarray]:
    plain_batch = metamol_batch_finalizer(batch_data)
    return (
        MetamolMolFiLMBatch(
            sample_to_task_id=np.stack(batch_data["sample_to_task_id"], axis=0),
            **dataclasses.asdict(plain_batch),
        ),
        np.stack(batch_data["bool_labels"], axis=0),
    )


def get_molfilm_batcher(
    task_name_to_id: Dict[str, int],
    max_num_graphs: Optional[int] = None,
    max_num_nodes: Optional[int] = None,
    max_num_edges: Optional[int] = None,
) -> MetamolBatcher[MetamolMolFiLMBatch, np.ndarray]:
    return MetamolBatcher(
        max_num_graphs=max_num_graphs,
        max_num_nodes=max_num_nodes,
        max_num_edges=max_num_edges,
        init_callback=molfilm_batcher_init_fn,
        per_datapoint_callback=partial(
            molfilm_batcher_add_sample_fn, task_name_to_id=task_name_to_id
        ),
        finalizer_callback=molfilm_batcher_finalizer_fn,
    )


def get_molfilm_inference_batcher(
    max_num_graphs: int,
) -> MetamolBatcher[MetamolMolFiLMBatch, np.ndarray]:
    # In this setting, we only consider a single task at a time, so they just all get the same ID:
    task_name_to_const_id: Dict[str, int] = defaultdict(lambda: 0)
    return get_molfilm_batcher(task_name_to_id=task_name_to_const_id, max_num_graphs=max_num_graphs)


class MolFiLMTaskSampleBatchIterable(Iterable[Tuple[MetamolMolFiLMBatch, np.ndarray]]):
    def __init__(
        self,
        dataset: MetamolDataset,
        data_fold: DataFold,
        task_name_to_id: Dict[str, int],
        max_num_graphs: Optional[int] = None,
        max_num_nodes: Optional[int] = None,
        max_num_edges: Optional[int] = None,
        num_chunked_tasks: int = 8,
        repeat: bool = False,
    ):
        self._dataset = dataset
        self._data_fold = data_fold
        self._num_chunked_tasks = num_chunked_tasks
        self._repeat = repeat

        self._task_sampler = RandomTaskSampler(
            train_size_or_ratio=1024, valid_size_or_ratio=0, test_size_or_ratio=0
        )
        self._batcher = get_molfilm_batcher(
            task_name_to_id, max_num_graphs, max_num_nodes, max_num_edges
        )

    def __iter__(self) -> Iterator[Tuple[MetamolMolFiLMBatch, np.ndarray]]:
        def paths_to_mixed_samples(
            paths: List[RichPath], idx: int
        ) -> Iterable[Tuple[MetamolMolFiLMBatch, np.ndarray]]:
            loaded_samples: List[MoleculeDatapoint] = []
            for i, path in enumerate(paths):
                task = MetamolTask.load_from_file(path)
                task_sample = self._task_sampler.sample(task, seed=idx + i)
                loaded_samples.extend(task_sample.train_samples)
            if self._data_fold == DataFold.TRAIN:
                np.random.shuffle(loaded_samples)

            for features, labels in self._batcher.batch(loaded_samples):
                yield features, labels

        return iter(
            self._dataset.get_task_reading_iterable(
                data_fold=self._data_fold,
                task_reader_fn=paths_to_mixed_samples,
                repeat=self._repeat,
                reader_chunk_size=self._num_chunked_tasks,
            )
        )
