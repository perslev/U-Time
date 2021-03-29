import numpy as np
from utime.dataset.queue.base_queue import BaseQueue
from utime.dataset.queue.study_loader import StudyLoader
from queue import Queue, Empty
from contextlib import contextmanager


class LimitationQueue(BaseQueue):
    """
    Implements a SleepStudy loading queue
    Stores a reference to a SleepStudyDataset objects each storing
    one or more SleepStudy objects. Using the methods get_random_study method,
    this method tracks the number of times a
    SleepStudy object has been accessed, and when exceeding a threshold,
    unloads it and loads a random SleepStudy from the same dataset.
    """
    def __init__(self,
                 dataset,
                 max_loaded=25,
                 num_access_before_reload=50,
                 preload_now=True,
                 await_preload=True,
                 study_loader=None,
                 n_load_jobs=7,
                 logger=None,
                 **kwargs):
        """
        Initialize a LoadQueue object from a of SleepStudyDataset object

        Args:
            dataset:                    (list) A SleepStudyDataset object
            max_loaded:                 (int)  Number of SleepStudy objects in
                                               a dataset that will be loaded
                                               at a given time.
            num_access_before_reload:   (int)  Number of times a SleepStudy obj
                                               can be accessed be
                                               get_random_study or
                                               a unload is invoked and a new
                                               data point is loaded.
            preload_now:   TODO
            study_loader:  TODO
            n_load_jobs:   TODO
            logger:        TODO
        """
        super(LimitationQueue, self).__init__(
            dataset=dataset,
            logger=logger
        )
        self.max_loaded = min(max_loaded or len(dataset), len(dataset))
        self.num_access_before_reload = num_access_before_reload

        # Queues of loaded and non-loaded objects
        self.loaded_queue = Queue(maxsize=self.max_loaded)
        self.non_loaded_queue = Queue(maxsize=len(dataset))

        # Fill non-loaded queue in random order
        inds = np.arange(len(dataset))
        np.random.shuffle(inds)
        for i in inds:
            self.non_loaded_queue.put(self.dataset.pairs[i])

        # Setup load thread pool
        self.study_loader = study_loader or StudyLoader(
            n_threads=n_load_jobs
        )
        # Register this dataset to become updated with new loaded studies
        # from the StudyLoader thread.
        self.study_loader.register_dataset(
            dataset_id=self.dataset.identifier,
            load_put_function=self._add_loaded_to_queue,
            error_put_function=self._load_error_callback,
        )

        # Increment counters to random off-set points for the first studies
        self.max_offset = int(self.num_access_before_reload * 0.75)
        self.n_offset = self.max_loaded

        if preload_now:
            # Load specified number of obj and populate access count dict
            self.preload(await_preload)

    def preload(self, await_preload=True):
        """


        Returns:

        """
        # Set the number of loaded objects to 'max_loaded_per_dataset'
        self.logger("Preloading {} SleepStudy objects from "
                    "{}".format(self.max_loaded, self.dataset.identifier))
        if self.dataset.n_loaded != 0 or self.loaded_queue.qsize() != 0:
            raise RuntimeError("Dataset {} seems to have already been "
                               "loaded. Do not load any data before "
                               "passing the SleepStudyDataset object "
                               "to the queue class. Only call "
                               "LoadQueue.preload once."
                               "".format(self.dataset.identifier))
        self._add_studies_to_load_queue(num=self.max_loaded)
        if await_preload:
            self.logger("... awaiting preload")
            self.study_loader.join()
            self.logger("Preload complete.")

    def load_queue_too_full(self, max_fraction=0.33):
        return self.study_loader.qsize() > \
               self.study_loader.maxsize*max_fraction

    def _warn_access_limit(self, min_fraction=0.10):
        qsize = self.loaded_queue.qsize()
        if qsize == 0:
            self.logger.warn("Study ID queue for dataset {} seems to"
                             " block. This might indicate a data loading "
                             "bottleneck.".format(self.dataset.identifier))
        elif qsize <= self.max_loaded*min_fraction:
            self.logger.warn("Dataset {}: Loaded queue getting too empty "
                             "(qsize={}, maxsize={})"
                             .format(self.dataset.identifier, qsize,
                                     self.max_loaded))

    @contextmanager
    def get_study_by_id(self, study_id):
        raise NotImplementedError

    @contextmanager
    def get_study_by_idx(self, study_idx):
        raise NotImplementedError

    @contextmanager
    def get_random_study(self):
        """
        TODO

        Returns:

        """
        with self.study_loader.thread_lock:
            self._warn_access_limit()
        # Get random SleepStudy ID from the specified dataset
        timeout_s = 30
        try:
            sleep_study, n_accesses = self.loaded_queue.get(timeout=timeout_s)
        except Empty as e:
            raise Empty("Could not get SleepStudy ID from dataset {} with "
                        "timeout of {} seconds. Consider increasing the "
                        "number of load threads / max loaded per dataset /"
                        " access threshold".format(self.dataset.identifier,
                                                   timeout_s)) from e
        try:
            yield sleep_study
        finally:
            self._release_study(sleep_study, n_accesses)

    def _add_studies_to_load_queue(self, num=1):
        """
        TODO

        Args:
            num:

        Returns:

        """
        for _ in range(num):
            ss = self.non_loaded_queue.get_nowait()  # Should never block!
            if ss.loaded:
                raise RuntimeWarning("Study ID {} in dataset {} seems to be "
                                     "already loaded, but it was fetched from "
                                     "the self.non_loaded_queue queue. This "
                                     "could be an implementation error!")
            self.study_loader.add_study_to_load_queue(ss, self.dataset.identifier)

    def _add_loaded_to_queue(self, sleep_study):
        """

        Args:
            study_id:

        Returns:

        """
        if self.n_offset >= 0:
            offset = np.random.randint(0, self.max_offset)
            self.n_offset -= 1
        else:
            offset = 0
        self.loaded_queue.put((sleep_study, offset))

    def _load_error_callback(self, sleep_study, *args, **kwargs):
        self.logger.warn("Load error on study {}".format(sleep_study))
        self._add_studies_to_load_queue(num=1)

    def _release_study(self, sleep_study, n_accesses):
        """
        TODO

        Args:
            sleep_study:
            n_accesses:

        Returns:

        """
        if n_accesses >= self.num_access_before_reload:
            # Unload, add to unloaded queue, start loading new study
            sleep_study.unload()
            self.non_loaded_queue.put(sleep_study)
            self._add_studies_to_load_queue(num=1)
        else:
            self.loaded_queue.put((sleep_study, n_accesses+1))
