import logging
import numpy as np
from utime.sequences.base_sequence import _BaseSequence
from psg_utils.errors import NotLoadedError

logger = logging.getLogger(__name__)


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
            raise ValueError(f"All sequences must have the same '{key}' property. Got {list_}.")


class MultiSequence(_BaseSequence):
    """
    Light wrapper around a collection of BalancedRandomBatchSequence and/and
    RandomBatchSequence objects.

    Samples batches of size 1 uniformly from its set of sequence objects
    to form a cross-dataset batch of size equal to the original batch size.

    Inherits from _BaseSequence and implements a select set of methods that
    are distributed across the members of this MultiSequence.
    """
    def __init__(self,
                 sequencers,
                 batch_size,
                 dataset_sample_alpha=0.5,
                 no_log=False):
        """
        TODO

        Args:
            sequencers:
            batch_size:
            dataset_sample_alpha:  TODO
            no_log:
        """
        # Make sure we can use the 0th sequencer as a reference that respects
        # all the sequences (same batch-size, margins etc.)
        _assert_comparable_sequencers(sequencers)
        super().__init__()
        self.sequences = sequencers
        self.sequences_idxs = np.arange(len(self.sequences))
        self.batch_size = batch_size
        self.margin = sequencers[0].margin
        self.n_classes = sequencers[0].n_classes

        # Compute probability of sampling a given dataset
        # We sample a given dataset either:
        #   1) Uniformly across datasets, independent of dataset size
        #   2) Unifomrly across records, sample a dataset according to size
        # The 'dataset_sample_alpha' parameter in [0...1] determines the
        # degree to which strategy 1 or 2 is followed:
        #   P = (1-alpha)*P_r + alpha*P_d
        # If alpha = 0, sample only according to
        n_samples = [len(s.dataset_queue) for s in sequencers]
        linear = n_samples / np.sum(n_samples)
        uniform = np.array([1/len(self.sequences)] * len(self.sequences))
        self.alpha = dataset_sample_alpha
        self.sample_prob = (1-self.alpha) * linear + self.alpha * uniform

        for s in self.sequences:
            s.batch_size = 1
        if not no_log:
            self.log()

    def log(self):
        logger.info(f"\n[*] MultiSequence initialized:\n"
                    f"    --- Contains {len(self.sequences)} sequences\n"
                    f"    --- Sequence IDs: {', '.join(s.identifier for s in self.sequences)}\n"
                    f"    --- Sequence sample probs (alpha={self.alpha}): {self.sample_prob}\n"
                    f"    --- Batch shape: {self.batch_shape}")

    def __len__(self):
        """ Returns the sum over stored sequencer lengths """
        try:
            return np.sum([len(s) for s in self.sequences])
        except NotLoadedError:
            # Queued data - return some reasonably large number, does not
            # matter as batches are normally randomly selected anyway.
            return 10000

    @property
    def batch_shape(self):
        """ Returns the batch shape as output from the MultiSequence """
        bs = self.sequences[0].batch_shape
        bs[0] = self.batch_size
        return bs

    @property
    def num_pairs(self):
        return np.sum([s.num_pairs for s in self.sequences])

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

        # OBS: Do not choose from self.sequences! The Sequence typed object is
        # array-like and numpy will attempt to iterate it to construct an
        # ndarray. For H5 datasets that may lead to all data stored in a
        # potentially very large dataset to be loaded every time this method is
        # called.
        sequences_idxs = np.random.choice(self.sequences_idxs,
                                          size=self.batch_size,
                                          replace=True,
                                          p=self.sample_prob)

        X, y = self.get_empty_batch_arrays()
        for i, sequence_idx in enumerate(sequences_idxs):
            sequence = self.sequences[sequence_idx]
            try:
                # Currently only supported BalancedRandomBatchSequence
                # and RandomBatchSequence. Try balanced first.
                xx, yy = sequence.get_class_balanced_random_period()
            except AttributeError:
                # Fall back to RandomBatchSequence
                xx, yy = sequence.get_random_period()
            X[i] = xx
            y[i] = yy
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
    def __init__(self, sequences, no_log=False):
        _assert_comparable_sequencers(sequences)
        self.sequences = sequences
        self.IDs = [s.identifier.split("/")[0] for s in self.sequences]
        self.n_classes = self.sequences[0].n_classes
        if not no_log:
            self.log()

    def log(self):
        logger.info(f"\n[*] ValidationMultiSequence initialized:\n"
                    f"    --- Contains {len(self.sequences)} sequences\n"
                    f"    --- Sequence IDs: {', '.join(self.IDs)}")

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
