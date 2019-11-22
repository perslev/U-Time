"""
A randomly sampling batch sequence object
Performs class-balanced sampling across uniformly randomly selected SleepStudy
objects.
"""

import numpy as np
from utime.sequences import BatchSequence, requires_all_loaded
from utime.errors import NotLoadedError


class BalancedRandomBatchSequence(BatchSequence):
    """
    BatchSequence sub-class that samples class-balanced random batches
    across (uniformly) randomly selected SleepStudy objects with calls to
    self.__getitem__.

    The 'sample_prob' property can be set to a list of values in [0...1] with
    sum == 1.0 that gives the probability that a period/segment/sleep epoch of
    ground truth label matching the sample_prob index will be sampled.

    See self.get_class_balanced_random_batch for detailed docstring.
    """
    def __init__(self,
                 sleep_study_pairs,
                 batch_size,
                 data_per_period,
                 n_classes,
                 n_channels,
                 sample_prob=None,
                 margin=0,
                 batch_scaler=None,
                 logger=None,
                 no_log=False,
                 identifier="",
                 **kwargs):
        """
        Args:
            sample_prob: (list, None) A list of length n_classes of sample
                                      probability values or None, in which
                                      case uniform class sampling will occur.

        See BatchSequence docstring for other argument descriptions
        """
        self._sample_prob = None
        super().__init__(sleep_study_pairs=sleep_study_pairs,
                         batch_size=batch_size,
                         data_per_period=data_per_period,
                         n_classes=n_classes,
                         n_channels=n_channels,
                         margin=margin,
                         batch_scaler=batch_scaler,
                         logger=logger,
                         no_log=True,
                         identifier=identifier,
                         **kwargs)
        self.sample_prob = sample_prob
        if not no_log:
            self.log()

    def log(self):
        """ Log basic information on this object """
        self.logger("[*] BalancedRandomBatchSequence initialized{}:\n"
                    "    Batch shape:     {}\n"
                    "    Sample prob.:    {}\n"
                    "    N pairs:         {}\n"
                    "    Margin:          {}\n"
                    "    Batch scaling:   {}\n"
                    "    All loaded:      {}\n"
                    "    N classes:       {}{}".format(" ()".format(self.identifier) if self.identifier else "",
                                                       self.batch_shape,
                                                       self.sample_prob,
                                                       len(self.pairs),
                                                       self.margin,
                                                       bool(self.batch_scaler),
                                                       self.all_loaded,
                                                       self.n_classes,
                                                       " (AUTO-INFERRED)"
                                                       if self._inferred else ""))

    @property
    def sample_prob(self):
        """ Returns the current class sampling probability vector """
        return self._sample_prob or [1.0/self.n_classes]*self.n_classes

    @sample_prob.setter
    def sample_prob(self, values):
        """
        Set a class-sampling probability vector.

        Args:
            values: A list of length self.n_classes of class-sampling
                    probability values. The list is normalized to sum to 1
        """
        if values is None:
            self._sample_prob = None
        else:
            if not isinstance(values, (list, tuple, np.ndarray)) or \
                    len(values) != self.n_classes:
                ValueError("'sample_prob' should be an array of"
                           " length n_classes. got {} "
                           "(type {})".format(values, type(values)))
            self._sample_prob = np.array(values)
            self._sample_prob /= np.sum(self._sample_prob)  # sum 1

    @requires_all_loaded
    def __getitem__(self, idx):
        """
        Return a random batch of data
        See self.get_class_balanced_random_batch for docstring
        """
        # If multiprocessing, set unique seed for this particular process
        self.seed()
        return self.get_class_balanced_random_batch()

    def get_class_balanced_random_period(self, assume_all_loaded=True):
        """
        Sample a class-balanced random 'period/epoch/segment' of data
        according to sample probabilities in self.sample_prob from
        a (uniformly) random SleepStudy object in self.pairs.

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
        # Get random class according to the sample probs.
        classes = np.arange(self.n_classes)
        cls = np.random.choice(classes, size=1, p=self.sample_prob)[0]
        found = False
        while not found:
            sleep_study = np.random.choice(pairs)
            if not sleep_study.loaded:
                raise NotLoadedError
            if cls not in sleep_study.class_to_period_dict or \
                    len(sleep_study.class_to_period_dict[cls]) == 0:
                # This SS does not have the given class
                continue
            try:
                # Get the period index of a randomly sampled class (according
                # to sample_prob distribution) within the SleepStudy pair
                idx = np.random.choice(sleep_study.class_to_period_dict[cls], 1)[0]
                if self.margin > 0:
                    # Shift the idx randomly within the window
                    idx += np.random.randint(-self.margin, self.margin+1)
                X_, y_ = self.get_period(study_id=sleep_study.identifier,
                                         period_idx=idx,
                                         allow_shift_at_border=True)
                return X_, y_
            except KeyError:
                continue

    def get_class_balanced_random_batch(self, assume_all_loaded=True):
        """
        Returns a batch of data sampled uniformly across SleepStudy pairs and
        randomly across target classes according to the distribution of
        self.sample_prob (for instance, [0.2, 0.2, 0.2, 0.2, 0.2] will sample
        uniformly across 5 classes from the uniformly chosen SleepStudy pair).

        Note: If the sampled SleepStudy object does not display the sampled
        target class, a new SleepStudy is sampled until success for the given
        label class.

        Note: For self.margin > 0 ('sequence' mode), sampling is conducted as
        normally, but the sampled position is shifted randomly by +- margin
        to either side to enforce sequence variation around rare classes.

        Note: For self.margin > 0 the output target classes may not be uniform
        in distribution as the sampling only ensures that at least one of each
        classes appear according to the sample distribution in a given sequence
        The remaining margin*2 targets in the sequence takes whatever values
        occur at these positions in the target sequence and are thus subject
        to class imbalance.

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
            X_, y_ = self.get_class_balanced_random_period(assume_all_loaded)
            X.append(X_), y.append(y_)
        return self.process_batch(X, y)
