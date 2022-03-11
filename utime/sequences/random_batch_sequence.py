"""
A randomly sampling batch sequence object
Samples randomly across uniformly randomly selected SleepStudy objects
Does not sample uniformly across classes (see BalancedRandomBatchSequence).
"""

import logging
import numpy as np
from utime.sequences import BatchSequence

logger = logging.getLogger(__name__)


class RandomBatchSequence(BatchSequence):
    """
    BatchSequence sub-class that samples batches randomly across (uniformly)
    randomly selected SleepStudy objects with calls to self.__getitem__.

    See self.get_random_batch for detailed docstring.
    """
    def __init__(self,
                 dataset_queue,
                 batch_size,
                 data_per_period,
                 n_classes,
                 n_channels,
                 margin=0,
                 augmenters=None,
                 batch_scaler=None,
                 no_log=False,
                 identifier="",
                 **kwargs):
        """
        See BatchSequence docstring for argument descriptions
        """
        super().__init__(dataset_queue=dataset_queue,
                         batch_size=batch_size,
                         data_per_period=data_per_period,
                         n_classes=n_classes,
                         n_channels=n_channels,
                         margin=margin,
                         augmenters=augmenters,
                         batch_scaler=batch_scaler,
                         no_log=True,
                         scale_assertion=True,
                         identifier=identifier,
                         require_all_loaded=False,
                         **kwargs)
        if not no_log:
            self.log()

    def log(self):
        """ Log basic information on this object """
        logger.info(f"\n[*] RandomBatchSequence initialized{' ({})'.format(self.identifier) if self.identifier else ''}:\n"
                    f"    Data queue type: {type(self.dataset_queue)}\n"
                    f"    Batch shape:     {self.batch_shape}\n"
                    f"    N pairs:         {len(self.dataset_queue)}\n"
                    f"    Margin:          {self.margin}\n"
                    f"    Augmenters:      {self.augmenters}\n"
                    f"    Aug enabled:     {self.augmentation_enabled}\n"
                    f"    Batch scaling:   {bool(self.batch_scaler)}\n"
                    f"    All loaded:      {self.all_loaded}\n"
                    f"    N classes:       {self.n_classes}{' (AUTO-INFERRED)' if self._inferred else ''}")

    def __getitem__(self, idx):
        """
        Return a random batch of data
        See self.get_random_batch for docstring
        """
        # If multiprocessing, set unique seed for this particular process
        self.seed()
        return self.get_random_batch()

    def get_random_period(self):
        """
        Sample a random 'period/epoch/segment' of data from a (uniformly)
        random SleepStudy object in self.pairs.

        With self.margin > 0 multiple, connected periods is returned in a
        single call.

        Returns:
            X, ndarray of shape [margin*2+1, data_per_period, n_channels]
            y, ndarray of shape [margin*2+1, 1] class labels
        """
        with self.dataset_queue.get_random_study() as sleep_study:
            # Get random period idx
            n_periods = sleep_study.n_periods
            period_idx = int(np.random.randint(0, n_periods, 1))
            return self.get_period(sleep_study=sleep_study,
                                   period_idx=period_idx,
                                   allow_shift_at_border=True)

    def get_random_batch(self):
        """
        Returns a batch of data sampled randomly across SleepStudy pairs.

        Returns:
            X, float32 ndarray, batch of input data,
               shape [batch_size, data_per_prediction, n_channels] if margin=0
               else [batch_size, margin*2+1, data_per_prediction, n_channels]
            y, uint8 ndarray, batch of integer target values,
               shape [batch_size, 1] if margin=0
               else [batch_size, margin*2+1, 1]
        """
        X, y = self.get_empty_batch_arrays()
        for i in range(self.batch_size):
            xx, yy = self.get_random_period()
            X[i] = xx
            y[i] = yy
        return self.process_batch(X, y)
