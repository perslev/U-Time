import logging
import numpy as np
from .elastic_deformation import elastic_transform
from psg_utils.utils import exactly_one_specified

logger = logging.getLogger(__name__)


class Augmenter(object):
    """
    Base augmenter class

    Stores an augmentation function that in __call__ applies to each element
    of a batch (X, y) with per-element probability 'apply_prob'
    If a batch of format (X, y, w) is passed to __call__, 'aug_weight' is
    multiplied by the previous weight stored in w.

    OBS: Operates 'in-place'
    """
    def __init__(self, transform_func, apply_prob, aug_weight=0.5):
        """
        Args:
            transform_func: A callable function accepting parameters
                            (X, y, **kwargs) that modifies X in-place
            apply_prob:     A [0-1] float giving the probabilty that
                            transform_func is applied to an element of a batch
            aug_weight:     Multiplicative factor applied to elements in
                            (optional) list batch_w passed to transform_func.
                            batch_w is a list of sample weights for each
                            element in the batch.
        """
        assert callable(transform_func)
        self.transform_func = transform_func
        self.apply_prob = apply_prob
        self.aug_weight = aug_weight

    def __repr__(self):
        return "<{}>".format(self.__name__)

    @property
    def apply_prob(self):
        return self._apply_prob

    @apply_prob.setter
    def apply_prob(self, apply_prob):
        if apply_prob > 1 or apply_prob < 0:
            raise ValueError(
                "Apply probability is invalid with value %3.f" % apply_prob)
        self._apply_prob = apply_prob

    @property
    def aug_weight(self):
        return self._aug_weight

    @aug_weight.setter
    def aug_weight(self, aug_weight):
        if aug_weight < 0:
            raise ValueError("aug_weight must be >= 0 (got %3.f)" % aug_weight)
        self._aug_weight = aug_weight

    @staticmethod
    def separate_global_and_position_wise_kwargs(kwargs, batch_size):
        """
        Separates arguments passed with **kwargs to the self.transform_func
        into two sets:
          - Global arguments that should be passed to all self.trans_func calls
          - Position wise arguments that should be passed only for certain
            entities in batch_x and batch-y

        Specifically, list or tuple parameters of length == batch_size are
        considered position-wise arguments.

        Args:
            kwargs:   A dictionary of parameters, global and position-wise
            batch_size: Number of elements in the batch

        Returns:
            Dict of global arguments and dict of position-wise arguments
        """
        pos_keys, glob_keys = [], []
        for k, v in kwargs.items():
            is_pos = isinstance(v, (list, tuple)) and len(v) == batch_size
            pos_keys.append(k) if is_pos else glob_keys.append(k)
        glob = {k: kwargs[k] for k in glob_keys}
        pos = [{k: kwargs[k][i] for k in pos_keys} for i in range(batch_size)]
        return glob, pos

    def __call__(self, batch_x, batch_y, batch_w, **kwargs):
        """ Augment a batch of data """
        return self.augment(batch_x, batch_y, batch_w, **kwargs)

    def augment(self, batch_x, batch_y, batch_w=None, **kwargs):
        """
        Applies self.transform_func to elements of batch_x and batch_y with
        element-wise probability self.apply_prob

        Assumes len(batch_x) == len(batch_y) (== len(batch_w))

        List/tuples of len(batch_x) passed as kwargs will be passed with their
        position equivalents of batch_x and batch_y to the transform func. All
        other items in kwargs will be passed to all calls made to
        self.trans_func

        Args:
            batch_x:  A batch of data
            batch_y:  A batch of labels
            batch_w:  An optional batch of sample-weights
            **kwargs: Parameters passed to the transform functon
        """
        # Only augment some of the images (determined by apply_prob)
        augment_mask = np.random.rand(len(batch_x)) <= self.apply_prob

        # Get arguments that should be passed to all self.trans_func calls
        # (glob) and for only specific position-wise entities in batch_x and
        # batch_y
        glob, pos = self.separate_global_and_position_wise_kwargs(kwargs,
                                                                  len(batch_x))

        for i, augment in enumerate(augment_mask):
            if not augment:
                continue
            x_aug, y_aug = self.transform_func(batch_x[i], batch_y[i],
                                               **glob, **pos[i])
            batch_x[i], batch_y[i] = x_aug, y_aug
            if batch_w is not None:
                batch_w[i] *= self.aug_weight


class RegionalAugmenter(Augmenter):
    """
    The RegionalAugmenter is a base class for all augmentation classes that
    apply to only (a) sub-region(s) of the input signal.

    See base Augmenter class.
    """
    def __init__(self, transform_func, min_fraction, max_fraction, apply_prob,
                 log_sample, aug_weight):
        super().__init__(transform_func, apply_prob, aug_weight)

        self.log_sample = log_sample
        self.min_fraction = float(min_fraction)
        self.max_fraction = float(max_fraction)
        if self.min_fraction <= 0:
            raise ValueError("Minimum fraction must be > 0, got "
                             "{}".format(self.min_fraction))
        if self.max_fraction > 1:
            raise ValueError("Maximum fraction must be <= 1, "
                             "got {}".format(self.max_fraction))

    @staticmethod
    def reshape_x(x):
        """
        Reshapes a signal to shape [-1, n_channels]
        """
        org_shape = x.shape
        x = x.reshape(-1, x.shape[-1])
        return x, org_shape

    def get_aug_length(self, x_length):
        """
        Giving a signal of length x_length, sample a length of signal
        (number of samples) to augment according to properties
        self.log_sample, self.min_fraction, self.max_fraction
        """
        if self.log_sample:
            min_f = np.log10(self.min_fraction)
            max_f = np.log10(self.max_fraction)
            frac = np.power(10, np.random.uniform(min_f, max_f, 1)[0])
            length = int(frac * x_length)
        else:
            min_ = int(self.min_fraction * x_length)
            max_ = int(self.max_fraction * x_length)
            length = int(np.random.uniform(min_, max_, 1)[0])
        return length

    @staticmethod
    def get_start_point(x_length):
        """ Sample a random start position within a x_length long signal """
        start = np.random.randint(0, x_length, 1)[0]
        return start

    @staticmethod
    def _augment_with_transform(x, y, start, transform_func, x_length, aug_length):
        """
        Augment region with a function that applies (in-place) to the region
        """
        wrap = start+aug_length-x_length
        if wrap > 0:
            r1 = x[start:start + aug_length - wrap]
            r2 = x[0:wrap]
            r1[:] = transform_func(r1)
            r2[:] = transform_func(r2)
        else:
            r = x[start:start+aug_length]
            r[:] = transform_func(r)
        return x, y

    @staticmethod
    def _augment_with_insert(x, y, start, insert, x_length):
        """
        Augment region by replacing/inserting an array of data in the region
        """
        aug_length = len(insert)
        wrap = start+aug_length-x_length
        if wrap > 0:
            x[start:start + aug_length - wrap] = insert[:-wrap].copy()
            x[0:wrap] = insert[-wrap:].copy()
        else:
            x[start:start+aug_length] = insert.copy()
        return x, y

    def augment_region(self, x, y, start=None, transform_func=None, insert=None):
        """
        Augment a region/sub-sequence of a signal 'x'

        Args:
            x:                A sample from a batch, array, [..., n_channels]
            y:                A label value from a batch, int
            start:            Aug. region start position within x, int
            transform_func:   Function to apply to values within aug region
            insert:           Array to insert/replace at aug region

        Returns:
            x (augmented in-place), y
        """
        if not exactly_one_specified(transform_func, insert):
            raise ValueError("'augment_region' expected one of "
                             "'transform_func' and 'insert', got one or both.")
        x, org_shape = self.reshape_x(x)
        x_length = len(x)
        start = start or self.get_start_point(x_length)
        if transform_func is not None:
            aug_length = self.get_aug_length(x_length)
            x, y = self._augment_with_transform(x, y,
                                                start=start,
                                                transform_func=transform_func,
                                                x_length=x_length,
                                                aug_length=aug_length)
        else:
            x, y = self._augment_with_insert(x, y,
                                             start=start,
                                             insert=insert,
                                             x_length=x_length)
        return x.reshape(org_shape), y


class GlobalElasticDeformations(Augmenter):
    """
    1D Elastic augmenter
    """
    def __init__(self, alpha, sigma, apply_prob, aug_weight=0.5):
        """
        Args:
            alpha: A number of tuple/list of two numbers specifying a range
                   of alpha values to sample from in each augmentation call
                   The alpha value determines the strength of the deformation
            sigma: A number of tuple/list of two numbers specifying a range
                   of sigma values to sample from in each augmentation call
                   The sigma value determines the smoothness of the deformation
            apply_prob: Apply the transformation only with some probability
                        Otherwise, return the image untransformed
            aug_weight: If a list of weights of len(batch_x) elements is passed
                        the aug_weight will multiply with the passed weight at
                        index i of batch_x if i in batch_x is transformed.
        """
        self.__name__ = "GlobalElasticDeformations"
        # Initialize base
        super().__init__(elastic_transform, apply_prob, aug_weight)

        if isinstance(alpha, (list, tuple)):
            if len(alpha) != 2:
                raise ValueError("Invalid list of alphas specified '%s'. "
                                 "Should be 2 numbers." % alpha)
            if alpha[1] <= alpha[0]:
                raise ValueError("alpha upper is smaller than sigma lower (%s)" % alpha)
        if isinstance(sigma, (list, tuple)):
            if len(sigma) != 2:
                raise ValueError("Invalid list of sigmas specified '%s'. "
                                 "Should be 2 numbers." % sigma)
            if sigma[1] <= sigma[0]:
                raise ValueError("Sigma upper is smaller than sigma lower (%s)" % sigma)

        self._alpha = alpha
        self._sigma = sigma

    @property
    def alpha(self):
        """
        Return a randomly sampled alpha value in the range [alpha[0], alpha[1]]
        or return the integer/float alpha if alpha is not a list
        """
        if isinstance(self._alpha, (list, tuple)):
            return np.random.uniform(self._alpha[0], self._alpha[1], 1)[0]
        else:
            return self._alpha

    @property
    def sigma(self):
        """
        Return a randomly sampled sigma value in the range [sigma[0], sigma[1]]
        or return the integer/float sigma if sigma is not a list
        """
        if isinstance(self._sigma, (list, tuple)):
            return np.random.uniform(self._sigma[0], self._sigma[1], 1)[0]
        else:
            return self._sigma

    def __call__(self, batch_x, batch_y, batch_w, bg_values=0.0):
        return self.augment(batch_x, batch_y, batch_w,
                            bg_value=bg_values,
                            sigma=[self.sigma for _ in range(len(batch_x))],
                            alpha=[self.alpha for _ in range(len(batch_x))])

    def __str__(self):
        return "%s(alpha=%s, sigma=%s, apply_prob=%.3f)" % (
            self.__name__, self._alpha, self._sigma, self.apply_prob
        )


class GlobalAmplitude(Augmenter):
    """
    Scales the global amplitude of a signal.
    Simply multiplies the signal values by a constant
    """
    def __init__(self, min_scaling, max_scaling, apply_prob, aug_weight=0.5):
        self.__name__ = "GlobalAmplitude"
        # Initialize base
        super().__init__(self.scale, apply_prob, aug_weight)

        self.min_scaling = float(min_scaling)
        self.max_scaling = float(max_scaling)
        if self.max_scaling <= self.min_scaling:
            raise ValueError("Max scaling must be greater than min. scaling.")

    def scale(self, x, y):
        scale = np.random.uniform(self.min_scaling, self.max_scaling, 1)
        return x*scale, y


class GlobalShift(Augmenter):
    """
    Shifts the signal by adding a (positive or negative) constant
    """
    def __init__(self, min_shift, max_shift, apply_prob, aug_weight=0.5):
        self.__name__ = "GlobalShift"
        # Initialize base
        super().__init__(self.shift, apply_prob, aug_weight)

        self.min_shift = float(min_shift)
        self.max_shift = float(max_shift)
        if self.max_shift <= self.min_shift:
            raise ValueError("Max shift must be greater than min. shift.")

    def shift(self, x, y):
        shift = np.random.uniform(self.min_shift, self.max_shift, 1)
        return x+shift, y


class GlobalGaussianNoise(Augmenter):
    """
    Applies position-wise gaussian noise to all elements of a signal
    Note: Applies uniquely in each channel
    """
    def __init__(self, sigma, apply_prob, mean=0, aug_weight=0.5):
        self.__name__ = "GlobalGaussianNoise"
        # Initialize base
        super().__init__(self.apply_noise, apply_prob, aug_weight)

        self.mean = float(mean)
        self.sigma = float(sigma)

    def apply_noise(self, x, y):
        noise = np.random.normal(loc=self.mean, scale=self.sigma, size=x.shape)
        return x+noise, y


class ChannelDropout(Augmenter):
    """
    Drops whole channels at random with a certain probability, replacing
    all values in the channel with low sigma Gaussian noise.
    """
    def __init__(self, drop_fraction, apply_prob, aug_weight=0.5):
        self.__name__ = "ChannelDropout"
        super().__init__(self.drop_channels, apply_prob, aug_weight)
        self.drop_fraction = drop_fraction

    def drop_channels(self, x, y):
        n_channels = x.shape[-1]
        n_to_drop = max(int(n_channels * self.drop_fraction), 1)
        if n_to_drop >= n_channels:
            raise ValueError("Attempted to drop {} channels from 'x' with {}"
                             " channels (shape {})".format(n_to_drop,
                                                           n_channels,
                                                           x.shape))
        to_drop = np.random.choice(np.arange(n_channels), n_to_drop, False)
        for i in to_drop:
            x[..., i] = np.random.normal(loc=np.mean(x[..., i]),
                                         scale=0.01,
                                         size=x.shape[:-1])
        return x, y


class RegionalGaussianNoise(RegionalAugmenter):
    """
    Applies position-wise gaussian noise to a sub-region of a signal
    Note: Applies uniquely in each channel
    """
    def __init__(self, min_region_fraction, max_region_fraction,
                 apply_prob, mean=0, sigma=0.1, log_sample=True,
                 aug_weight=0.5):
        self.__name__ = "RegionalGaussianNoise"
        # Initialize base
        super().__init__(self.apply_noise, min_region_fraction,
                         max_region_fraction, apply_prob, log_sample,
                         aug_weight)
        self.mean = float(mean)
        self.sigma = float(sigma)

    def _noise_func(self, x):
        return x+np.random.normal(loc=self.mean,
                                  scale=self.sigma,
                                  size=x.shape)

    def apply_noise(self, x, y):
        x, y = self.augment_region(x, y, transform_func=self._noise_func)
        return x, y


class RegionalErase(RegionalAugmenter):
    """
    'Erases' a region of the input signal by replacing it with low sigma
    gaussian noise
    """
    def __init__(self, min_region_fraction, max_region_fraction,
                 apply_prob, log_sample=True, aug_weight=0.5):
        self.__name__ = "RegionalErase"
        super().__init__(self.random_erase, min_region_fraction,
                         max_region_fraction, apply_prob, log_sample,
                         aug_weight)

    def random_erase(self, x, y):
        x, org_shape = self.reshape_x(x)
        x_length = len(x)
        erase_start = self.get_start_point(x_length)
        erase_len = self.get_aug_length(x_length)
        noise = np.random.normal(loc=np.mean(x),
                                 scale=0.01,
                                 size=[erase_len, x.shape[-1]])
        x, y = self.augment_region(x, y, start=erase_start, insert=noise)
        return x.reshape(org_shape), y


class RegionalSignalMix(RegionalAugmenter):
    """
    Takes a signal from one region, insert the mean of this signal and a signal
    of another region into the second region's place.
    """
    def __init__(self, min_region_fraction, max_region_fraction,
                 apply_prob, log_sample=True, aug_weight=0.5):
        self.__name__ = "RegionalSignalMix"
        super().__init__(self.random_mix,
                         min_region_fraction, max_region_fraction,
                         apply_prob, log_sample, aug_weight)

    def random_mix(self, x, y):
        x, org_shape = self.reshape_x(x)
        x_length = len(x)
        insert_start = self.get_start_point(x_length)
        take_start = self.get_start_point(x_length)
        mix_len = self.get_aug_length(x_length)

        # Get signal to insert into 'insert_start' position
        take_inds = np.arange(take_start, take_start+mix_len)
        take_sig = x.take(take_inds, mode="wrap", axis=0).reshape(-1, x.shape[-1])
        insert_inds = np.arange(insert_start, insert_start+mix_len)
        insert_sig = x.take(insert_inds, mode="wrap", axis=0).reshape(-1, x.shape[-1])
        insert = (take_sig + insert_sig)/2

        x, y = self.augment_region(x, y, start=insert_start, insert=insert)
        return x.reshape(org_shape), y


class RegionalSignFlip(RegionalAugmenter):
    """
    Flips the sign of the signal within a sub-region.
    """
    def __init__(self, min_region_fraction, max_region_fraction,
                 apply_prob, log_sample=True, aug_weight=0.5):
        self.__name__ = "RegionalSignFlip"
        super().__init__(self.sign_flip,
                         min_region_fraction, max_region_fraction,
                         apply_prob, log_sample, aug_weight)

    def sign_flip(self, x, y):
        return self.augment_region(x, y, transform_func=lambda x: -x)
