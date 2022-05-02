import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from functools import wraps
from multiprocessing import current_process
from utime import Defaults
from psg_utils.errors import NotLoadedError
from psg_utils.preprocessing.scaling import apply_scaling, assert_scaler
from psg_utils.dataset.utils import assert_all_loaded

logger = logging.getLogger(__name__)


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
            raise NotLoadedError(f"Method '{method.__name__}' requires all stored SleepStudy "
                                 "objects to be loaded.")
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

    def __call__(self):
        """
        Returns an iterator that iterates the dataset indefinitely, converting numpy arrays to tensors
        """
        while True:
            for i in range(len(self)):
                x, y = self.__getitem__(i)  # index does not matter
                yield tf.convert_to_tensor(x), tf.convert_to_tensor(y)

    def __getitem__(self, idx):
        raise NotImplemented

    def __iter__(self):
        raise NotImplemented

    def __len__(self):
        raise NotImplemented

    def get_pairs(self):
        raise NotImplemented

    @property
    def num_pairs(self):
        return len(self.get_pairs())

    def batch_shape(self):
        raise NotImplemented

    def get_class_counts(self):
        raise NotImplemented

    def get_class_frequencies(self):
        raise NotImplemented

    def get_batch_shapes(self, batch_size=None):
        x_shape = self.batch_shape
        y_shape = x_shape[:-2] + [1]
        if batch_size:
            # Overwrite
            x_shape[0] = batch_size
            y_shape[0] = batch_size
        return x_shape, y_shape

    def get_empty_batch_arrays(self):
        """
        TODO

        Returns:

        """
        x_shape, y_shape = self.get_batch_shapes()
        x = np.empty(shape=x_shape, dtype=Defaults.PSG_DTYPE)
        y = np.empty(shape=y_shape, dtype=Defaults.HYP_DTYPE)
        return x, y

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
            try:
                # Try fetch process number, add this to global seed to get different seeds in each process
                proc_seed = int(pname.split("-")[1])
            except IndexError:
                proc_seed = 0
            # Re-seed this process
            proc_seed = (Defaults.GLOBAL_SEED + proc_seed) if Defaults.GLOBAL_SEED is not None else None
            np.random.seed(proc_seed)
            self.is_seeded[pname] = True


class BaseSequence(_BaseSequence):
    """
    Basic Sequence class that implements methods needed across all Sequence
    sub-classes.
    """
    def __init__(self,
                 dataset_queue,
                 n_classes,
                 n_channels,
                 batch_size,
                 augmenters,
                 batch_scaler,
                 require_all_loaded=False,
                 identifier=""):
        """
        Args:
            dataset_queue:   (queue)    TODO
            n_classes:         (int)    Number of classes (sleep stages)
            n_channels:        (int)    The number of PSG channels to expect in
                                        data extracted from a SleepStudy object
            batch_size:        (int)    The size of the generated batch
            augmenters:        (list)   List of utime.augmentation.augmenters
            batch_scaler:      (string) The name of a sklearn.preprocessing
                                        Scaler object to apply to each sampled
                                        batch (optional)
            identifier:        (string) A string identifier name
        """
        super().__init__()
        self._all_loaded = assert_all_loaded(dataset_queue.dataset.pairs,
                                             raise_=require_all_loaded)
        self.identifier = identifier
        self.dataset_queue = dataset_queue
        self.n_classes = int(n_classes)
        self.n_channels = int(n_channels)
        self.augmenters = augmenters or []
        self.augmentation_enabled = bool(augmenters)
        self.batch_size = batch_size
        if self.all_loaded:
            try:
                self._periods_per_pair = np.array([ss.n_periods for ss in self.dataset_queue])
            except NotImplementedError:
                pass
            else:
                self._cum_periods_per_pair = np.cumsum(self.periods_per_pair)
        if batch_scaler not in (None, False):
            if not assert_scaler(batch_scaler):
                raise ValueError("Invalid batch scaler {}".format(batch_scaler))
            self.batch_scaler = batch_scaler
        else:
            self.batch_scaler = None

    def get_pairs(self):
        return self.dataset_queue.get_pairs()

    @requires_all_loaded
    def get_class_counts(self):
        """
        Returns:
            An ndarray of class counts across all stored SleepStudy objects
            Shape [self.n_classes], dtype np.int
        """
        counts = np.zeros(shape=[self.n_classes], dtype=np.int)
        for im in self.dataset_queue:
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

    def _assert_scaled(self, warn_mean=5, warn_std=5, n_studies=3,
                       periods_per_study=10):
        """
        Samples n_batches random batches from the sub-class Sequencer object
        and computes the mean and STD of the values across the batches. If
        their absolute values are higher than 'warn_mean' and 'warn_std'
        respectively, a warning is printed.

        Note: Does not raise an Error or Warning

        Args:
            warn_mean:         Maximum allowed abs(mean) before warning is invoked
            warn_std:          Maximum allowed std before warning is invoked
            n_studies:         Number of studies to (+ potentially load) sample from
            periods_per_study: Number of periods to sample from each study
        """
        # Get a set of random batches
        batches = []
        for _ in range(n_studies):
            xs = []
            with self.dataset_queue.get_random_study() as ss:
                seconds_per_study = periods_per_study * ss.period_length_sec
                start = np.random.randint(0, ss.last_period_start_second-seconds_per_study)
                start -= start % ss.period_length_sec
                xs.append(ss.extract_from_psg(start, start+seconds_per_study))
            batches.extend(xs)
        mean, std = np.abs(np.mean(batches)), np.std(batches)
        logger.info(f"Mean assertion ({periods_per_study} periods from each of {n_studies} studies): {mean:.3f}")
        logger.info(f"Scale assertion ({periods_per_study} periods from each of {n_studies} studies):  {std:.3f}")
        if mean > warn_mean or std > warn_std:
            logger.warning("OBS: Found large abs(mean) and std values over 5"
                           f" sampled batches ({mean:.3f} and {std:.3f})."
                           " Make sure scaling is active at either the "
                           "global level (attribute 'scaler' has been set on"
                           " individual SleepStudy objects, typically via the"
                           " SleepStudyDataset set_scaler method), or "
                           "batch-wise via the batch_scaler attribute of the"
                           " Sequence object.")

    @property
    def augmentation_enabled(self):
        """ Returns True if augmentation is currently enabled, see setter """
        return self._do_augmentation

    @augmentation_enabled.setter
    def augmentation_enabled(self, value):
        """
        Set augmentation on/off on this Sequence object.
        If augmentation_enabled = False no augmentation will be performed even
        if Augmenter objects are set in the self.augmenters list.
        If no Augmenter objects are set, augmentation_enabled has no effect.

        Args:
            value: (bool) Set augmentation enabled or not
        """
        if not isinstance(value, bool):
            raise TypeError("Argument to 'augmentation_enabled' must be a "
                            "boolean value. Got {} ({})".format(value,
                                                                type(value)))
        if value is True and not self.augmenters:
            raise ValueError("Cannot set 'augmentation_enabled' 'True' with "
                             "empty 'augmenters' list: {}".format(self.augmenters))
        self._do_augmentation = value

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

    @property
    def augmenters(self):
        """ Returns the current list of augmenters (may be empty) """
        return self._augmenters

    @augmenters.setter
    def augmenters(self, list_of_augs):
        """
        Initialize and set a list of utime.augmentation.augmenters Augmentor
        objects.

        Args:
            list_of_augs:

        Returns:

        """
        from utime.augmentation import augmenters
        if list_of_augs is None:
            init_aug = []
        else:
            c1 = not isinstance(list_of_augs, (tuple, list, np.ndarray))
            c2 = not all([isinstance(o, dict) for o in list_of_augs])
            if c1 or c2:
                raise TypeError("Property 'augmenters' must be a list or tuple "
                                "of dictionary elements, "
                                "got {}".format(list_of_augs))
            init_aug = []
            for d in list_of_augs:
                cls = augmenters.__dict__[d["cls_name"]]
                init_aug.append(cls(**d["kwargs"]))
                logger.info(f"Setting augmenter: {d['cls_name']}({d['kwargs']})")
        self._augmenters = init_aug

    def augment(self, X, y, w):
        """
        Apply Augmenters in self.augmenters to batch (X, y, w)
        OBS: Augmenters operate in-place

        Args:
            X: (ndarray) A batch of data
            y: (ndarray) A batch of corresponding labels
            w: (ndarray) A batch of weights associated to each sample in (X, y)

        Returns:
            None, performs in-place operations
        """
        if not self.augmentation_enabled:
            raise RuntimeError("Tried to do augmentation, but "
                               "augmentation_enabled is set to 'False'")
        for aug in self.augmenters:
            # OBS: in-place operations
            a = aug(X, y, w)
            if a is not None:
                raise TypeError("Output of augmenter {} was not None. Make "
                                "sure to implement all augmenters with "
                                "in-place operations on (X, y, w).")

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

    def process_batch(self, X, y):
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
          6) Performs augmentation on the batch if self.augmenters is set and
             self.augmentation_enabled is True

        Args:
            X:     A list of ndarrays corresponding to a batch of X data
            y:     A list of ndarrays corresponding to a batch of y labels

        Returns:
            Batch of (X, y) data
            OBS: Currently does not return the w (weights) array
        """
        # Cast and reshape arrays
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("Expected numpy array inputs.")
        X = np.squeeze(X).astype(Defaults.PSG_DTYPE)

        if self.n_channels == 1:
            X = np.expand_dims(X, -1)
        y = np.expand_dims(y.astype(Defaults.HYP_DTYPE).squeeze(), -1)

        expected_dim = len(self.batch_shape)
        if X.ndim == expected_dim-1:
            X, y = np.expand_dims(X, 0), np.expand_dims(y, 0)
        elif X.ndim != expected_dim:
            raise RuntimeError("Dimensionality of X is {} (shape {}), but "
                               "expected {}".format(X.ndim,
                                                    X.shape,
                                                    expected_dim))

        if self.batch_scaler:
            # Scale the batch
            self.scale(X)
        w = np.ones(len(X))
        if self.augmentation_enabled:
            # Perform augmentation
            self.augment(X, y, w=w)

        return X, y  # , w  <- weights currently disabled, fix dice-loss first
