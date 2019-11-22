from tensorflow.keras.utils import Sequence
from multiprocessing import current_process
from MultiPlanarUNet.logging import ScreenLogger
from utime.preprocessing.scaling import apply_scaling, assert_scaler
from utime.utils import assert_all_loaded
from utime.errors import NotLoadedError
from functools import wraps
import numpy as np


def requires_all_loaded(method):
    """
    Decorator for _BaseSequence derived class methods that ensures that all
    SleepStudies were loaded at init time. Otherwise, a NotLoadedError will
    be raised.

    OBS: Does not check the current status of loaded objects, but relies on the
         load check that occurred at init.
    """
    @wraps(method)
    def check_loaded_and_raise(self, *args, **kwargs):
        if not self.all_loaded:
            raise NotLoadedError("Method '{}' requires all stored SleepStudy "
                                 "objects to be "
                                 "loaded.".format(method.__name__))
        return method(self, *args, **kwargs)
    return check_loaded_and_raise


class _BaseSequence(Sequence):
    """
    ABC-like base class for the BaseSequence class
    Describes an interface of methods that should typically be implemented
    by one of its sub-classes.
    Most importantly implements the seed method which re-seeds the RNG of the
    current process if not done already.
    """
    def __init__(self):
        super().__init__()

        # A dictionary mapping process names to whether the process has been
        # seeded
        self.is_seeded = {}
        self._all_loaded = None
        self._periods_per_pair = None
        self._cum_periods_per_pair = None

    @property
    def all_loaded(self):
        return self._all_loaded

    @property
    def cum_periods_per_pair(self):
        """ Returns a list of cumulative sums over periods per pair """
        return self._cum_periods_per_pair

    @property
    def periods_per_pair(self):
        """ Returns a list of n_periods for each stored pair """
        return self._periods_per_pair

    def __getitem__(self, idx):
        raise NotImplemented

    def __iter__(self):
        raise NotImplemented

    def __len__(self):
        raise NotImplemented

    def batch_shape(self):
        raise NotImplemented

    def get_class_counts(self):
        raise NotImplemented

    def get_class_frequencies(self):
        raise NotImplemented

    def seed(self):
        """
        If multiprocessing, the processes will inherit the RNG state of the
        main process - here we reseed each process once so that the batches
        are randomly generated across multi-processes calls to the Sequence
        batch generator methods

        If multi-threading this method will just re-seed the 'MainProcess'
        process once
        """
        pname = current_process().name
        if pname not in self.is_seeded or not self.is_seeded[pname]:
            # Re-seed this process
            np.random.seed()
            self.is_seeded[pname] = True


class BaseSequence(_BaseSequence):
    """
    Basic Sequence class that implements methods needed across all Sequence
    sub-classes.
    """
    def __init__(self,
                 sleep_study_pairs,
                 n_classes,
                 n_channels,
                 batch_size,
                 batch_scaler,
                 logger=None,
                 require_all_loaded=True,
                 identifier=""):
        """
        Args:
            sleep_study_pairs: (list)   A list of SleepStudy objects
            n_classes:         (int)    Number of classes (sleep stages)
            n_channels:        (int)    The number of PSG channels to expect in
                                        data extracted from a SleepStudy object
            batch_size:        (int)    The size of the generated batch
            batch_scaler:      (string) The name of a sklearn.preprocessing
                                        Scaler object to apply to each sampled
                                        batch (optional)
            logger:            (Logger) A Logger object
            identifier:        (string) A string identifier name
        """
        super().__init__()
        self._all_loaded = assert_all_loaded(sleep_study_pairs,
                                             raise_=require_all_loaded)
        self.identifier = identifier
        self.pairs = sleep_study_pairs
        self.id_to_pair = {pair.identifier: pair for pair in self.pairs}
        self.n_classes = int(n_classes)
        self.n_channels = int(n_channels)
        self.logger = logger or ScreenLogger()
        self.batch_size = batch_size
        if self.all_loaded:
            self._periods_per_pair = np.array([ss.n_periods for ss in self.pairs])
            self._cum_periods_per_pair = np.cumsum(self.periods_per_pair)
        if batch_scaler not in (None, False):
            if not assert_scaler(batch_scaler):
                raise ValueError("Invalid batch scaler {}".format(batch_scaler))
            self.batch_scaler = batch_scaler
        else:
            self.batch_scaler = None

    @requires_all_loaded
    def get_class_counts(self):
        """
        Returns:
            An ndarray of class counts across all stored SleepStudy objects
            Shape [self.n_classes], dtype np.int
        """
        counts = np.zeros(shape=[self.n_classes], dtype=np.int)
        for im in self.pairs:
            count_dict = im.get_class_counts(as_dict=True)
            for cls, count in count_dict.items():
                counts[cls] += count
        return counts

    @requires_all_loaded
    def get_class_frequencies(self):
        """
        Returns:
            An ndarray of class frequencies comptued over all stored
            SleepStudy objects. Shape [self.n_classes], dtype np.int
        """
        counts = self.get_class_counts()
        return counts / np.sum(counts)

    @requires_all_loaded
    def _assert_scaled(self, warn_mean=5, warn_std=5, n_batches=5):
        """
        Samples n_batches random batches from the sub-class Sequencer object
        and computes the mean and STD of the values across the batches. If
        their absolute values are higher than 'warn_mean' and 'warn_std'
        respectively, a warning is printed.

        Note: Does not raise an Error or Warning

        Args:
            warn_mean: Maximum allowed abs(mean) before warning is invoked
            warn_std:  Maximum allowed std before warning is invoked
            n_batches: Number of batches to sample for mean/std computation
        """
        # Get a set of random batches
        batches = []
        for ind in np.random.randint(0, len(self), n_batches):
            X, _ = self[ind]  # Use __getitem__ of the given Sequence class
            batches.append(X)
        mean, std = np.abs(np.mean(batches)), np.std(batches)
        self.logger("Mean assertion ({} batches):  {:.3f}".format(n_batches,
                                                                  mean))
        self.logger("Scale assertion ({} batches): {:.3f}".format(n_batches,
                                                                  std))
        if mean > warn_mean or std > warn_std:
            self.logger.warn("OBS: Found large abs(mean) and std values over 5"
                             " sampled batches ({:.3f} and {:.3f})."
                             " Make sure scaling is active at either the "
                             "global level (attribute 'scaler' has been set on"
                             " individual SleepStudy objects, typically via the"
                             " SleepStudyDataset set_scaler method), or "
                             "batch-wise via the batch_scaler attribute of the"
                             " Sequence object.".format(mean, std))

    @property
    def batch_size(self):
        """ Returns the currently set batch size """
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        """
        Change the batch size of sampled batches

        Args:
            batch_size: (int) New batch size
        """
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("Batch size must be a positive integer.")
        self._batch_size = batch_size

    def scale(self, X):
        """
        Fit and apply scaler 'self.batch_scaler' to batch X
        Used only when self.batch_scaler is set, typically the entire PSG is
        scaled before training and this is not used.

        Args:
            X: (ndarray) A batch of data

        Returns:
            X (ndarray), scaled batch of data
        """
        # Loop over batch and scale each element
        for i, input_ in enumerate(X):
            org_shape = input_.shape
            input_ = input_.reshape(-1, org_shape[-1])
            scaled_input = apply_scaling(input_, self.batch_scaler)[0]
            X[i] = scaled_input.reshape(org_shape)

    def process_batch(self, X, y, copy=True):
        """
        Process a batch (X, y) of sampled data.

        The process_batch method should always be called in the end of any
        method that implements batch sampling.

        Processing includes:
          1) Casting of X to ndarray of dtype float32
          2) Ensures X has a channel dimension, even if self.n_channels == 1
          3) Ensures y has dtype uint8 and shape [-1, 1]
          4) Ensures both X and y has a 'batch dimension', even if batch_size
             is 1.
          5) If a 'batch_scaler' is set, scales the X data

        Args:
            X:     A list of ndarrays corresponding to a batch of X data
            y:     A list of ndarrays corresponding to a batch of y labels
            copy:  If True, force a copy of the X and y data. NOTE: data may be
                   copied in some cases even if copy=False, see np.asarray

        Returns:
            Batch of (X, y) data
            OBS: Currently does not return the w (weights) array
        """
        # Cast and reshape arrays
        arr_f = np.asarray if copy is False else np.array
        X = arr_f(X, dtype=np.float32).squeeze()
        if self.n_channels == 1:
            X = np.expand_dims(X, -1)
        y = np.expand_dims(arr_f(y, dtype=np.uint8).squeeze(), -1)

        expected_dim = len(self.batch_shape)
        if X.ndim == expected_dim-1:
            X, y = np.expand_dims(X, 0), np.expand_dims(y, 0)
        elif X.ndim != expected_dim:
            raise RuntimeError("Dimensionality of X is {} (shape {}), but "
                               "expected {}".format(X.ndim, X.shape,
                                                    expected_dim))

        if self.batch_scaler:
            # Scale the batch
            self.scale(X)
        # w = np.ones(len(X))

        return X, y  # , w  <- weights currently disabled, fix dice-loss first
