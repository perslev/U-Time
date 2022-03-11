"""
A randomly sampling batch sequence object
Performs class-balanced sampling across uniformly randomly selected SleepStudy objects.
"""

import logging
import numpy as np
from utime.sequences import BatchSequence

logger = logging.getLogger(__name__)


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
                 dataset_queue,
                 batch_size,
                 data_per_period,
                 n_classes,
                 n_channels,
                 sample_prob=None,
                 margin=0,
                 augmenters=None,
                 batch_scaler=None,
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
        super().__init__(dataset_queue=dataset_queue,
                         batch_size=batch_size,
                         data_per_period=data_per_period,
                         n_classes=n_classes,
                         n_channels=n_channels,
                         margin=margin,
                         augmenters=augmenters,
                         batch_scaler=batch_scaler,
                         no_log=True,
                         identifier=identifier,
                         require_all_loaded=False,
                         **kwargs)
        self.sample_prob = sample_prob
        if not no_log:
            self.log()

    def log(self):
        """ Log basic information on this object """
        logger.info(f"\n[*] BalancedRandomBatchSequence initialized{f' ({self.identifier})' if self.identifier else ''}:\n"
                    f"    Data queue type: {type(self.dataset_queue)}\n"
                    f"    Batch shape:     {self.batch_shape}\n"
                    f"    Sample prob.:    {self.sample_prob}\n"
                    f"    N pairs:         {len(self.dataset_queue)}\n"
                    f"    Margin:          {self.margin}\n"
                    f"    Augmenters:      {self.augmenters}\n"
                    f"    Aug enabled:     {self.augmentation_enabled}\n"
                    f"    Batch scaling:   {bool(self.batch_scaler)}\n"
                    f"    All loaded:      {self.all_loaded}\n"
                    f"    N classes:       {self.n_classes}{' (AUTO-INFERRED)' if self._inferred else ''}")

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
                raise ValueError(f"'sample_prob' should be an array of"
                                 f" length n_classes={self.n_classes}. "
                                 f"Got {values} (type {type(values)})")
            self._sample_prob = np.array(values)
            self._sample_prob /= np.sum(self._sample_prob)  # sum 1

    def __getitem__(self, idx):
        """
        Return a random batch of data
        See self.get_class_balanced_random_batch for docstring
        """
        # If multiprocessing, set unique seed for this particular process
        self.seed()
        return self.get_class_balanced_random_batch()

    def get_class_balanced_random_period(self):
        """
        Sample a class-balanced random 'period/epoch/segment' of data
        according to sample probabilities in self.sample_prob from
        a (uniformly) random SleepStudy object in self.dataset_queue.

        With self.margin > 0 multiple, connected periods is returned in a
        single call.

        Returns:
            X, a [data_per_prediction, n_channels] ndarray if margin == 0, else
               a list of len margin*2+1 of [data_per_prediction, n_channels]
               ndarrays if margin > 0
            y, integer label value if margin == 0 else a list of len margin*2+1
               of integer label values if margin >0
        """
        # Get random class according to the sample probs.
        classes = np.arange(self.n_classes)
        cls = np.random.choice(classes, size=1, p=self.sample_prob)[0]
        tries, max_tries = 0, 1000
        while tries < max_tries:
            with self.dataset_queue.get_random_study() as sleep_study:
                try:
                    class_inds = sleep_study.get_class_indicies(cls)
                    if len(class_inds) == 0:
                        logger.warning(f"Found empty class inds array for study {sleep_study} and class {cls}")
                        raise KeyError
                except KeyError:
                    # This SS does not have the given class
                    tries += 1
                    continue
                else:
                    # Get the period index of a randomly sampled class
                    # (according to sample_prob distribution) within the
                    # SleepStudy pair
                    idx = np.random.choice(class_inds, 1)[0]
                    if self.margin > 0:
                        # Shift the idx randomly within the window
                        idx += np.random.randint(-self.margin, self.margin+1)
                    X_, y_ = self.get_period(sleep_study=sleep_study,
                                             period_idx=idx,
                                             allow_shift_at_border=True)
                    return X_, y_
        # Probably something is wrong, raise error.
        raise RuntimeError(f"Could not sample period for class {cls}, stopping after {max_tries} tries.")

    def get_class_balanced_random_batch(self):
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
        X, y = self.get_empty_batch_arrays()
        for i in range(self.batch_size):
            xx, yy = self.get_class_balanced_random_period()
            X[i] = xx
            y[i] = yy
        return self.process_batch(X, y)
