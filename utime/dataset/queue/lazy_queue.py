from utime.dataset.queue.base_queue import BaseQueue
from contextlib import contextmanager


class LazyQueue(BaseQueue):
    """
    Implements a queue-like object (same API interface as LoadQueue), but one
    that only loads data just-in-time when requested.
    This is useful for wrapping e.g. validation data in an object that behaves
    similar to the training queue object, but without consuming memory before
    needing to do validation.
    """
    def __init__(self, dataset, logger=None, **kwargs):
        """
        TODO
        Args:
            dataset:
            logger:
        """
        super(LazyQueue, self).__init__(
            dataset=dataset,
            logger=logger
        )

    @contextmanager
    def get_random_study(self):
        study = super().get_random_study()
        with study.loaded_in_context():
            yield study

    @contextmanager
    def get_study_by_idx(self, study_idx):
        study = super().get_study_by_idx(study_idx)
        with study.loaded_in_context():
            yield study

    @contextmanager
    def get_study_by_id(self, study_id):
        study = super().get_study_by_id(study_id)
        with study.loaded_in_context():
            yield study
