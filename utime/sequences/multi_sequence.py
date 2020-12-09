from utime.sequences.base_sequence import _BaseSequence
from mpunet.logging import ScreenLogger
import numpy as np


def _assert_comparable_sequencers(sequencers):
    """
    Takes a list of utime.sequencer Sequence objects and compares them for
    equality on a number of specified parameters.

    Raises ValueError if any sequence in the list deviates from the others
    with respect to any of the tested attributes.

    Args:
        sequencers: A list of Sequence objects
    """
    tests = (([], "margin"), ([], "batch_size"), ([], "n_classes"),
             ([], "data_per_period"), ([], "n_channels"))
    for s in sequencers:
        for list_, key in tests:
            list_.append(getattr(s, key))
    for list_, key in tests:
        if not all(np.asarray(list_) == list_[0]):
            raise ValueError("All sequences must have the same '{}' "
                             "property. Got {}.".format(key, list_))


class MultiSequence(_BaseSequence):
    """
    Light wrapper around a collection of BalancedRandomBatchSequence and/and
    RandomBatchSequence objects.

    Samples batches of size 1 uniformly from its set of sequence objects
    to form a cross-dataset batch of size equal to the original batch size.

    Inherits from _BaseSequence and implements a select set of methods that
    are distributed across the members of this MultiSequence.
    """
    def __init__(self, sequencers, batch_size, no_log=False, logger=None):
        # Make sure we can use the 0th sequencer as a reference that respects
        # all the sequences (same batch-size, margins etc.)
        _assert_comparable_sequencers(sequencers)
        super().__init__()
        self.logger = logger or ScreenLogger()
        self.sequences = sequencers
        self.batch_size = batch_size
        self.margin = sequencers[0].margin
        self.n_classes = sequencers[0].n_classes
        for s in self.sequences:
            s.batch_size = 1
        if not no_log:
            self.log()

    def log(self):
        self.logger("[*] MultiSequence initialized:\n"
                    "    --- Contains {} sequences\n"
                    "    --- Sequence IDs: {}"
                    "".format(len(self.sequences),
                              ", ".join(s.identifier for s in self.sequences)))

    def __len__(self):
        """ Returns the sum over stored sequencer lengths """
        return np.sum([len(s) for s in self.sequences])

    @property
    def batch_shape(self):
        """ Returns the batch shape as output from the MultiSequence """
        bs = self.sequences[0].batch_shape
        bs[0] = self.batch_size
        return bs

    @property
    def total_periods(self):
        """ Returns the sum of total periods over all sequences """
        return np.sum([s.total_periods for s in self.sequences])

    def get_class_counts(self):
        """ Returns the sum of class counts over all sequences """
        counts = np.zeros(shape=[self.sequences[0].n_classes], dtype=np.int)
        for seq in self.sequences:
            counts += seq.get_class_counts()
        return counts

    def get_class_frequencies(self):
        """ Returns the frequencies of classes over all sequences """
        counts = self.get_class_counts()
        return counts / np.sum(counts)

    def __getitem__(self, idx):
        """
        Returns a batch of size self.batch_size uniformly selected across the
        stored sequencer objects.
        """
        self.seed()
        seq_idxs = np.random.randint(0, len(self.sequences), self.batch_size)
        X, y = [], []
        for i, seq_idx in enumerate(seq_idxs):
            seq = self.sequences[seq_idx]
            try:
                # Currently only supported BalancedRandomBatchSequence
                # and RandomBatchSequence. Try balanced first.
                xx, yy = seq.get_class_balanced_random_period()
            except AttributeError:
                # Fall back to RandomBatchSequence
                xx, yy = seq.get_random_period()
            X.append(xx), y.append(yy)
        return self.sequences[0].process_batch(X, y)


class ValidationMultiSequence:
    """
    Light wrapper around a collection of BalancedRandomBatchSequence and/and
    RandomBatchSequence objects used for validation.

    The utime.callbacks Validation callback performs validation using all
    sequence objects stored under the 'sequences' and 'IDs' lists attributes
    of the ValidationMultiSequence.

    Inherits from _BaseSequence and implements a select set of methods that
    are distributed across the members of this MultiSequence..
    """
    def __init__(self, sequences, no_log=False, logger=None):
        _assert_comparable_sequencers(sequences)
        self.sequences = sequences
        self.IDs = [s.identifier.split("/")[0] for s in self.sequences]
        self.n_classes = self.sequences[0].n_classes
        self.logger = logger or ScreenLogger()
        if not no_log:
            self.log()

    def log(self):
        self.logger("[*] ValidationMultiSequence initialized:\n"
                    "    --- Contains {} sequences\n"
                    "    --- Sequence IDs: {}"
                    "".format(len(self.sequences),
                              ", ".join(self.IDs)))

    def __len__(self):
        """ Returns the sum over stored sequencer lengths """
        return np.sum([len(s) for s in self.sequences])

    def get_minimum_total_periods(self):
        """
        Returns the minimum number of total periods in any of the stored
        validation sequences.
        """
        vspe = []
        for vs in self.sequences:
            vspe.append(vs.total_periods)
        return np.min(vspe)

    @property
    def batch_size(self):
        """
        Returns the batch size as output by this ValidationMultiSequence
        """
        return self.sequences[0].batch_size

    @batch_size.setter
    def batch_size(self, value):
        """
        Updates the batch size on all stores sequnce objects

        Args:
            value: (int) New batch size to set
        """
        for s in self.sequences:
            s.batch_size = value
