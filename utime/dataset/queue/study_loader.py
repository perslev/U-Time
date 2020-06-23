from utime.errors import CouldNotLoadError
from mpunet.logging.default_logger import ScreenLogger
from threading import Lock, Thread
from queue import Queue, Empty
from time import sleep


def _load_func(load_queue, results_queue, load_errors_queue, lock, logger):
    """

    Args:
        load_queue:

    Returns:

    """
    while True:
        to_load, dataset_id = load_queue.get()
        try:
            to_load.load()
            results_queue.put((to_load, dataset_id))
        except CouldNotLoadError as e:
            with lock:
                logger.warn("[ERROR in StudyLoader] "
                            "Could not load study '{}'".format(to_load))
            load_errors_queue.put((to_load, dataset_id))
        finally:
            load_queue.task_done()


def _gather_loaded(output_queue, registered_datasets):
    while True:
        # Wait for studies in the output queue
        sleep_study, dataset_id = output_queue.get(block=True)
        load_put_function = registered_datasets[dataset_id][0]
        load_put_function(sleep_study)
        output_queue.task_done()


def _gather_errors(load_errors_queue, registered_datasets):
    while True:
        # Wait for studies in the output queue
        sleep_study, dataset_id = load_errors_queue.get(block=True)
        error_put_function = registered_datasets[dataset_id][1]
        error_put_function(sleep_study)
        load_errors_queue.task_done()


class StudyLoader:
    """
    Implements a multithreading SleepStudy loading queue
    """
    def __init__(self,
                 n_threads=5,
                 max_queue_size=50,
                 logger=None):
        """
        Initialize a StudyLoader object from a list of SleepStudyDataset objects

        Args:
            TODO
        """
        # Setup load thread pool
        self.logger = logger or ScreenLogger()
        self._load_queue = Queue(maxsize=max_queue_size)
        self._output_queue = Queue(maxsize=max_queue_size)
        self._load_errors_queue = Queue(maxsize=3)  # We probably want to raise
                                                    # an error if this queue
                                                    # gets to more than ~3!
        self.thread_lock = Lock()

        args = (self._load_queue, self._output_queue, self._load_errors_queue,
                self.thread_lock, self.logger)
        self.pool = []
        for _ in range(n_threads):
            p = Thread(target=_load_func, args=args, daemon=True)
            p.start()
            self.pool.append(p)

        # Prepare gathering thread
        self._registered_datasets = {}
        self.gather_loaded_thread = Thread(target=_gather_loaded,
                                           args=(self._output_queue,
                                                 self._registered_datasets),
                                           daemon=True)
        self.gather_errors_thread = Thread(target=_gather_errors,
                                           args=(self._load_errors_queue,
                                                 self._registered_datasets),
                                           daemon=True)
        self.gather_loaded_thread.start()
        self.gather_errors_thread.start()

    @property
    def qsize(self):
        """ Returns the qsize of the load queue """
        return self._load_queue.qsize

    @property
    def maxsize(self):
        """ Returns the maxsize of the load queue """
        return self._load_queue.maxsize

    def join(self):
        """ Join on all queues """
        self.logger("Awaiting preload from {} (train) datasets".format(
            len(self._registered_datasets)
        ))
        self._load_queue.join()
        self.logger("Load queue joined...")
        self._output_queue.join()
        self.logger("Output queue joined...")
        self._load_errors_queue.join()
        self.logger("Errors queue joined...")

    def add_study_to_load_queue(self, study, dataset_id):
        if dataset_id not in self._registered_datasets:
            raise RuntimeError("Dataset {} is not registered. "
                               "Call StudyLoader.register_dataset before adding"
                               " items from that dataset to the loading "
                               "queue".format(dataset_id))
        if self.qsize() == self.maxsize:
            self.logger.warn("Loading queue seems about to block! "
                             "(max_size={}, current={}). "
                             "Sleeping until loading queue is empty "
                             "again.".format(self.maxsize,
                                             self.qsize()))
            while self.qsize() > 1:
                sleep(1)
        self._load_queue.put((study, dataset_id))

    def register_dataset(self, dataset_id, load_put_function, error_put_function):
        with self.thread_lock:
            if dataset_id in self._registered_datasets:
                raise RuntimeWarning("A dataset of ID {} has already been "
                                     "registered.".format(dataset_id))
            self._registered_datasets[dataset_id] = (
                load_put_function, error_put_function
            )

    def de_register_dataset(self, dataset_id):
        with self.thread_lock:
            del self._registered_datasets[dataset_id]
