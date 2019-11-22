"""
A randomly sampling batch sequence object
Samples randomly across uniformly randomly selected SleepStudy objects
Does not sample uniformly across classes (see BalancedRandomBatchSequence).
"""

import numpy as np
from utime.sequences import BatchSequence, requires_all_loaded
from utime.errors import NotLoadedError


class RandomBatchSequence(BatchSequence):
    """
    BatchSequence sub-class that samples batches randomly across (uniformly)
    randomly selected SleepStudy objects with calls to self.__getitem__.

    See self.get_random_batch for detailed docstring.
    """
    def __init__(self,
                 sleep_study_pairs,
                 batch_size,
                 data_per_period,
                 n_classes,
                 n_channels,
                 margin=0,
                 batch_scaler=None,
                 logger=None,
                 no_log=False,
                 identifier="",
                 **kwargs):
        """
        See BatchSequence docstring for argument descriptions
        """
        super().__init__(sleep_study_pairs=sleep_study_pairs,
                         batch_size=batch_size,
                         data_per_period=data_per_period,
                         n_classes=n_classes,
                         n_channels=n_channels,
                         margin=margin,
                         batch_scaler=batch_scaler,
                         logger=logger,
                         no_log=True,
                         scale_assertion=True,
                         identifier=identifier,
                         **kwargs)
        if not no_log:
            self.log()

    def log(self):
        """ Log basic information on this object """
        self.logger("[*] RandomBatchSequence initialized{}:\n"
                    "    Batch shape:     {}\n"
                    "    N pairs:         {}\n"
                    "    Margin:          {}\n"
                    "    Batch scaling:   {}\n"
                    "    All loaded:      {}\n"
                    "    N classes:       {}{}".format(" ()".format(self.identifier) if self.identifier else "",
                                                       self.batch_shape,
                                                       len(self.pairs),
                                                       self.margin,
                                                       bool(self.batch_scaler),
                                                       self.all_loaded,
                                                       self.n_classes,
                                                       " (AUTO-INFERRED)"
                                                       if self._inferred else ""))

    @requires_all_loaded
    def __getitem__(self, idx):
        """
        Return a random batch of data
        See self.get_random_batch for docstring
        """
        # If multiprocessing, set unique seed for this particular process
        self.seed()
        return self.get_random_batch()

    def get_random_period(self, assume_all_loaded=True):
        """
        Sample a random 'period/epoch/segment' of data from a (uniformly)
        random SleepStudy object in self.pairs.

        With self.margin > 0 multiple, connected periods is returned in a
        single call.

        Returns:
            X, a [data_per_prediction, n_channels] ndarray if margin == 0, else
               a list of len margin*2+1 of [data_per_prediction, n_channels]
               ndarrays if margin > 0
            y, integer label value if margin == 0 else a list of len margin*2+1
               of integer label values if margin >0
        """
        if assume_all_loaded:
            pairs = self.pairs
        else:
            pairs = [s for s in self.pairs if s.loaded]
        sleep_study = np.random.choice(pairs)
        if not sleep_study.loaded:
            raise NotLoadedError
        # Get random period idx
        n_periods = sleep_study.n_periods
        period_idx = int(np.random.randint(0, n_periods, 1))
        return self.get_period(study_id=sleep_study.identifier,
                               period_idx=period_idx,
                               allow_shift_at_border=True)

    def get_random_batch(self, assume_all_loaded=True):
        """
        Returns a batch of data sampled randomly across SleepStudy pairs.

        Args:
            assume_all_loaded: (bool) If True, select from all SleepStudy
                                      objects randomly. If False, select only
                                      from verified loaded objects. If True and
                                      an object is not loaded is randomly
                                      selected, raises NotLoadedError.

        Returns:
            X, float32 ndarray, batch of input data,
               shape [batch_size, data_per_prediction, n_channels] if margin=0
               else [batch_size, margin*2+1, data_per_prediction, n_channels]
            y, uint8 ndarray, batch of integer target values,
               shape [batch_size, 1] if margin=0
               else [batch_size, margin*2+1, 1]
        """
        X, y = [], []
        while len(X) != self.batch_size:
            X_, y_ = self.get_random_period(assume_all_loaded)
            X.append(X_), y.append(y_)
        return self.process_batch(X, y)
