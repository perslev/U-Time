import numpy as np
from utime.utils import ensure_list_or_tuple
from MultiPlanarUNet.logging.default_logger import ScreenLogger
from contextlib import contextmanager
from multiprocessing.pool import ThreadPool
from threading import Lock
from functools import partial
from queue import Queue, Empty
from time import sleep


def _add_loaded_to_dict_and_queue(dataset_access_dict,
                                  dataset_id_queue,
                                  study_id,
                                  lock):
    """

    Args:
        dataset_access_dict:
        dataset_id_queue:
        study_id:
        lock:

    Returns:

    """
    with lock:
        dataset_access_dict[study_id] = [0, 0]
        dataset_id_queue.put(study_id)


def _load_func(load_queue, access_dict, id_queue, lock, thread_num=None):
    """

    Args:
        load_queue:
        access_dict:
        id_queue:
        lock:
        thread_num:

    Returns:

    """
    while True:
        to_load, dataset_id = load_queue.get()
        to_load.load()
        _add_loaded_to_dict_and_queue(dataset_access_dict=access_dict[dataset_id],
                                      dataset_id_queue=id_queue[dataset_id],
                                      study_id=to_load.identifier,
                                      lock=lock)
        load_queue.task_done()


def _set_random_access_offset(access_count_dict, max_offset):
    """

    Args:
        access_count_dict:
        max_offset:

    Returns:

    """
    for dataset in access_count_dict:
        dataset = access_count_dict[dataset]
        for study in dataset:
            # Set random offset
            dataset[study][0] += np.random.randint(0, max_offset)


class LoadQueue:
    """
    Implements a SleepStudy loading queue
    Stores a reference to one or more SleepStudyDataset objects each storing
    one or more SleepStudy objects. Using the methods get_random_study and
    get_random_study_from methods, this method tracks the number of times a
    SleepStudy object has been accessed, and when exceeding a threshold,
    unloads it and loads a random SleepStudy from the same dataset.
    """
    def __init__(self,
                 datasets,
                 max_loaded_per_dataset=25,
                 num_access_before_reload=50,
                 preload_now=True,
                 n_load_threads=5,
                 logger=None):
        """
        Initialize a LoadQueue object from a list of SleepStudyDataset objects

        Args:
            datasets:                   (list) List of SleepStudyDataset obj.
            max_loaded_per_dataset:     (int)  Number of SleepStudy objects in
                                               each dataset that will be loaded
                                               at a given time.
            num_access_before_reload:   (int)  Number of times a SleepStudy obj
                                               can be accessed be
                                               get_random_study or
                                               get_random_study_from before
                                               a unload is invoked and a new
                                               data point is loaded.
        """
        self.datasets = ensure_list_or_tuple(datasets)
        self.dataset_id_to_dataset = {d.identifier: d for d in self.datasets}
        self.dataset_ids = list(self.dataset_id_to_dataset.keys())
        self.max_loaded_per_dataset = max_loaded_per_dataset
        self.num_access_before_reload = num_access_before_reload
        self.logger = logger or ScreenLogger()

        # Populated in self.preload()
        self.id_queues = {d.identifier: Queue() for d in self.datasets}
        self.access_count_dict = {d.identifier: {} for d in self.datasets}

        # Setup load thread pool
        self.load_queue_max_size = max_loaded_per_dataset * len(datasets)
        self.load_pool = ThreadPool(processes=n_load_threads)
        self.load_queue = Queue(maxsize=self.load_queue_max_size)
        self.lock = Lock()
        target = partial(_load_func, *(self.load_queue,
                                       self.access_count_dict,
                                       self.id_queues,
                                       self.lock))
        self.load_pool.map_async(target, range(n_load_threads))

        if preload_now:
            # Load specified number of obj and populate access count dict
            self.preload()

    def preload(self):
        """


        Returns:

        """
        # Set the number of loaded objects to 'max_loaded_per_dataset'
        self.logger(
            "Preloading {} SleepStudy objects from {} datasets:\n- {}".format(
                self.max_loaded_per_dataset,
                len(self.datasets),
                "\n- ".join(self.access_count_dict.keys())
            )
        )
        with self.lock:
            for dataset in self.datasets:
                if dataset.n_loaded != 0 or \
                        self.id_queues[dataset.identifier].qsize() != 0:
                    raise RuntimeError("Dataset {} seems to have already been "
                                       "loaded. Do not load any data before "
                                       "passing the SleepStudyDataset object "
                                       "to the queue class. Only call "
                                       "LoadQueue.preload once.".format(dataset))
                self._load_from(dataset,
                                size=self.max_loaded_per_dataset,
                                replace=False)
        self.logger("... awaiting preload")
        self.load_queue.join()
        self.logger("Preload complete.")

        # Increment counters to random off-set point
        max_offset = int(self.num_access_before_reload * 0.75)
        _set_random_access_offset(self.access_count_dict, max_offset)

    def get_random_study(self):
        """


        Returns:

        """
        dataset_id = np.random.choice(self.dataset_ids, 1)[0]
        return self.get_random_study_from(dataset_id)

    def load_queue_too_full(self, max_fraction=0.33):
        return self.load_queue.qsize() > self.load_queue_max_size*max_fraction

    @contextmanager
    def get_random_study_from(self, dataset_id):
        """

        Args:
            dataset:

        Returns:

        """
        if self.load_queue_too_full():
            self.logger.warn("Loading queue appears to be falling behind "
                             "(max_size={}, current={}). "
                             "Sleeping until loading queue is near empty "
                             "again.".format(self.load_queue_max_size,
                                             self.load_queue.qsize()))
            while self.load_queue.qsize() > 1:
                sleep(1)

        dataset = self.dataset_id_to_dataset[dataset_id]
        # Get random SleepStudy ID from the specified dataset
        sleep_study_id = self._get_and_reserve_id(dataset.identifier)
        sleep_study = dataset.id_to_study[sleep_study_id]
        try:
            yield sleep_study
        finally:
            self._release_id(dataset, sleep_study)

    def _is_ready_for_unload(self, dataset_id, study_id, consider_in_use=True):
        """

        Args:
            dataset_id:
            study_id:

        Returns:

        """
        n_accessed, in_use = self.access_count_dict[dataset_id][study_id]
        access_over = n_accessed >= self.num_access_before_reload
        no_longer_in_use = in_use <= 0
        if consider_in_use:
            return access_over and no_longer_in_use
        else:
            return access_over

    def _is_safe_to_load(self, dataset_id, study_id):
        """

        Args:
            dataset_id:
            study_id:

        Returns:

        """
        unloaded = study_id not in self.access_count_dict[dataset_id]
        if unloaded:
            return False
        else:
            return not self._is_ready_for_unload(dataset_id, study_id, False)

    def _unload(self, dataset_id, sleep_study):
        """

        Args:
            dataset_id:
            sleep_study:

        Returns:

        """
        # Remove entry from access dict
        del self.access_count_dict[dataset_id][sleep_study.identifier]
        # Unload the objects
        sleep_study.unload()

    def _load_from(self, dataset, size=1, replace=False):
        """

        Args:
            dataset:
            size:
            replace:

        Returns:

        """
        # Get a random (non-loaded) object to load
        # TODO: a bit slow, non_loaded_pairs iterates all stored pairs
        to_load = np.random.choice(dataset.non_loaded_pairs, size, replace)
        for ss in to_load:
            if self.load_queue.qsize() == self.load_queue_max_size:
                self.logger.warn("Loading queue seems about to block. This "
                                 "may indicate that the loader threads do not "
                                 "work fast enough to keep the loading queue "
                                 "empty. This may cause a data loading bottle-"
                                 "neck. Consider increasing the number of "
                                 "data loader threads on the LoadQueue.")
            self.load_queue.put((ss, dataset.identifier))

    def _release_id(self, dataset, sleep_study):
        """

        Args:
            dataset:
            sleep_study:

        Returns:

        """
        with self.lock:
            dataset_id = dataset.identifier
            study_id = sleep_study.identifier
            # Decrement in-use counter
            self.access_count_dict[dataset_id][study_id][1] -= 1
            if self._is_ready_for_unload(dataset_id, study_id):
                self._unload(dataset_id, sleep_study)
                self._load_from(dataset)

    def _get_and_reserve_id(self, dataset_id, timeout_s=30):
        """

        Args:
            dataset_id:
            timeout_s:

        Returns:

        """
        while True:
            try:
                if self.id_queues[dataset_id].qsize() < 1:
                    self.logger.warn("Study ID queue for dataset {} seems to"
                                     " block. This might indicate a data "
                                     "loading bottleneck.".format(dataset_id))
                study_id = self.id_queues[dataset_id].get(timeout=timeout_s)
            except Empty as e:
                raise Empty("Could not get SleepStudy ID from dataset {} with "
                            "timeout of {} seconds. Consider increasing the "
                            "number of load threads / max loaded per dataset /"
                            " access threshold".format(dataset_id,
                                                       timeout_s)) from e
            with self.lock:
                # Check that the study exists and not about the be unloaded
                if self._is_safe_to_load(dataset_id, study_id):
                    # Increment counters
                    self.access_count_dict[dataset_id][study_id][0] += 1
                    self.access_count_dict[dataset_id][study_id][1] += 1
                    self.id_queues[dataset_id].put(study_id)
                    return study_id
