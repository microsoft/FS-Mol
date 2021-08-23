import os
import logging
import time
from multiprocessing import Queue, Process, Event
from multiprocessing.synchronize import Event as EventType
from queue import Empty
from typing import List, Iterator, TypeVar, Callable, Iterable, Union, Type

import numpy as np
from dpu_utils.utils import RichPath


logger = logging.getLogger(__name__)


ReaderOutputType = TypeVar("ReaderOutputType")
QueueRichPathType = TypeVar("QueueRichPathType")
QueueReaderOutputType = TypeVar("QueueReaderOutputType")


def read_file_from_path_queue(
    input_paths: QueueRichPathType,
    output_queue: QueueReaderOutputType,
    reader_fn: Callable[[List[RichPath], int], Iterable[ReaderOutputType]],
    termination_signal: EventType,
    reader_chunk_size: int,
):
    try:
        chunk_idx = 0
        while not termination_signal.is_set():
            try:
                # The timeout here needs to be non-trivial as we might time out when we can't get GIL
                # on the main process because that one is busy reading a large dataset provided by
                # another worker.
                paths = []
                for _ in range(reader_chunk_size):
                    paths.append(input_paths.get(timeout=1))
                for reader_output in reader_fn(paths, chunk_idx):
                    output_queue.put(reader_output)
                chunk_idx += 1
            except Empty:
                # Empty can either mean ".get() failed due to lock contention" or queue is empty.
                # Only stop if we believe the queue is empty:
                if input_paths.empty():
                    # Give back whatever paths we have already claimed, so that other workers can
                    # pick them up:
                    for path in paths:
                        input_paths.put(path)
                    break
            except Exception as e:
                logger.warn(
                    f"Process {os.getpid()} threw {type(e)} exception when trying to read files {paths}."
                )
                import traceback

                logger.debug(traceback.format_exc())
                continue
    finally:
        # Put an empty into the output queue to signal that a worker finished:
        output_queue.put(Empty)


class BufferedFileReaderIterable(Iterable[ReaderOutputType]):
    def __init__(
        self,
        reader_fn: Callable[[List[RichPath], int], Iterable[ReaderOutputType]],
        data_paths: List[RichPath],
        shuffle_data: bool,
        repeat: bool,
        num_workers: int,
        buffer_size: int = 30,
        reader_chunk_size: int = 1,
    ):
        self._reader_fn = reader_fn
        self._data_paths = data_paths
        self._shuffle_data = shuffle_data
        self._repeat = repeat
        self._num_workers = num_workers
        self._buffer_size = buffer_size
        self._reader_chunk_size = reader_chunk_size

    def __iter__(self) -> Iterator[ReaderOutputType]:
        return BufferedFileReaderIterator(
            reader_fn=self._reader_fn,
            data_paths=self._data_paths,
            shuffle_data=self._shuffle_data,
            repeat=self._repeat,
            num_workers=self._num_workers,
            buffer_size=self._buffer_size,
            reader_chunk_size=self._reader_chunk_size,
        )


class BufferedFileReaderIterator(Iterator[ReaderOutputType]):
    """Read files in separate worker threads and expose them as an iterator.

    Our strategy is as follows:
     * We use one queue for paths that the workers are to read (self._file_queue)
       and one for outputs (self._output_queue) that we yield as iterator.
     * On start of a full iteration, we create the worker processes
     * Each worker reads and puts the read data into a shared output queue
     * Once the input queue is empty, the workers put an Empty into the output queue
       to signal that they are finished
     * Once we have collected as many Emptys as we have workers, we clean up the
       workers and either yield StopIteration or restart (if we are repeating)
    """

    def __init__(
        self,
        reader_fn: Callable[[List[RichPath], int], Iterable[ReaderOutputType]],
        data_paths: List[RichPath],
        shuffle_data: bool,
        repeat: bool,
        num_workers: int,
        buffer_size: int = 30,
        reader_chunk_size: int = 1,
    ):
        self._reader_fn = reader_fn
        self._data_paths = data_paths
        self._shuffle_data = shuffle_data
        self._repeat = repeat
        self._num_workers = num_workers
        self._buffer_size = buffer_size
        self._reader_chunk_size = reader_chunk_size

        # We'll set up the processes once we start iterating
        self._processes: List[Process] = []
        self._initialised_workers = False
        self._num_finished_workers = 0

        # Set up queues communicating files to the workers/collecting results:
        self._file_queue: "Queue[RichPath]" = Queue(len(self._data_paths))
        self._output_queue: "Queue[Union[Type[Empty], ReaderOutputType]]" = Queue(self._buffer_size)

        # Event to signal workers that __del__ function is called
        self._termination_signal = Event()

        # Start work on construction:
        self.initialise_workers()

    def initialise_workers(self):
        if self._initialised_workers:
            return

        # Set off the processes reading their files.
        self._processes = [
            Process(
                target=read_file_from_path_queue,
                args=(
                    self._file_queue,
                    self._output_queue,
                    self._reader_fn,
                    self._termination_signal,
                    self._reader_chunk_size,
                ),
            )
            for _ in range(self._num_workers)
        ]

        if self._shuffle_data:
            paths = list(self._data_paths)
            np.random.shuffle(paths)
        else:
            paths = self._data_paths

        for path in paths:
            self._file_queue.put(path)

        for worker in self._processes:
            worker.start()
        self._num_finished_workers = 0
        self._initialised_workers = True

    def __enter__(self) -> "BufferedFileReaderIterator":
        self.initialise_workers()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.cleanup_workers()
        return False  # Signal that exceptions should be re-raised, if needed

    def cleanup_workers(self):
        # Do nothing if we are already cleaned up:
        if not self._initialised_workers:
            return

        # Set the terminate flag:
        self._termination_signal.set()

        # Empty the input and output queues to unblock the workers:
        while not self._file_queue.empty():
            try:
                self._file_queue.get_nowait()  # queue.empty is not totally reliable, so get_nowait can raise errors.
            except Empty:
                pass
        while not self._output_queue.empty():
            try:
                self._output_queue.get_nowait()  # queue.empty is not totally reliable, so get_nowait can raise errors.
            except Empty:
                pass

        # Run over all workers, try to join them back:
        is_any_alive = False
        for worker in self._processes:
            worker.join(timeout=0.5)
            is_any_alive |= worker.is_alive()

        # The nuclear option to kill the stragglers:
        if is_any_alive:
            # Give them one more second to catch up:
            time.sleep(1)
            try:
                for worker in self._processes:
                    if worker.is_alive():
                        worker.terminate()
                        worker.join(timeout=0.1)
            except ValueError as e:
                raise e

        # Clean up the processes. Try one last time to join the child process. If this
        # fails, .close() will raise an Exception, but we have no other recourse at this point.
        try:
            for worker in self._processes:
                worker.join(timeout=60)
                worker.close()
        except Exception:
            # This is triggered when closing failed. At this point, we can't really do anything
            # anymore; the only thing we can hope is that the process eventually wakes up again,
            # gets its signal and dies. If that doesn't happen, we start to accumulate processes
            # and will eventually starve ourselves of resources.
            # The alternative is to die now, which would just kill the experiment directly. Hence,
            # we do the former and hope.
            # As a consequence, we do not re-use the termination signal (to let it remain set),
            # and instead create a fresh one below.
            pass
        self._processes.clear()
        self._termination_signal = Event()

        # Only reset to initial state if we want to repeat the dataset:
        if self._repeat:
            self._initialised_workers = False

    def __del__(self):
        self.cleanup_workers()

    def __next__(self) -> ReaderOutputType:
        if not self._initialised_workers:
            self.initialise_workers()

        while True:
            # If all workers are finished, either create fresh workers or bail out:
            if self._num_finished_workers >= self._num_workers:
                if self._repeat:
                    # Clean up the existing workers, start fresh ones:
                    # In principle, it would be better to re-use the existing ones, but meh,
                    # extra code...
                    self.cleanup_workers()
                    self.initialise_workers()
                else:
                    break

            next_element = self._output_queue.get()

            # Check if one of our worker reports that it's done:
            if next_element is Empty:
                self._num_finished_workers += 1
                # Just re-start the loop and see if the other workers are still there...
            else:
                # If everything is good, we have found our next element.
                return next_element

        # Reached if we break out of the while loop because we ran out of data:
        self.cleanup_workers()
        raise StopIteration


class SequentialFileReaderIterable(Iterable[ReaderOutputType]):
    def __init__(
        self,
        reader_fn: Callable[[List[RichPath], int], Iterable[ReaderOutputType]],
        data_paths: List[RichPath],
        shuffle_data: bool,
        repeat: bool,
        buffer_size: int = 30,
        reader_chunk_size: int = 1,
    ):
        self._reader_fn = reader_fn
        self._data_paths = data_paths
        self._shuffle_data = shuffle_data
        self._repeat = repeat
        self._buffer_size = buffer_size
        self._reader_chunk_size = reader_chunk_size

    def __iter__(self) -> Iterator[ReaderOutputType]:
        chunk_idx = 0
        input_paths = list(self._data_paths)
        np.random.shuffle(input_paths)
        while True:
            try:
                chunk_paths = []
                for _ in range(self._reader_chunk_size):
                    chunk_paths.append(input_paths.pop())
                for reader_output in self._reader_fn(chunk_paths, chunk_idx):
                    yield reader_output
                chunk_idx += 1
            except (StopIteration, IndexError):
                if self._repeat:
                    input_paths = list(self._data_paths)
                    np.random.shuffle(input_paths)
                else:
                    break
