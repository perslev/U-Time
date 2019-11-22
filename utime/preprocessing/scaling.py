"""
Functions for channel-wise scaling of PSG data

Implements the MultiChannelScaler, which fits and applies scalers from the
sklearn.preprocessing module individually to channels of a PSG ndarray
"""

import sklearn.preprocessing as preprocessing
import numpy as np


def assert_scaler(scaler):
    """
    Checks whether a scaler (string) exists in the sklearn.preprocessing
    module.

    Args:
        scaler: String class name representation of a potential
                sklearn.preprocessing Scaler object.

    Returns:
        True if the scaler exists in the module, otherwise False
    """
    if isinstance(scaler, str):
        scaler = [scaler]
    ok = []
    for sc in scaler:
        ok.append(sc in preprocessing.__dict__)
    return all(ok)


def get_scaler(scaler):
    """
    Returns a MultiChannelScaler initialized to perform scaling according to
    the 'scaler' argument

    Uses default parameters for all scalers except QuantileTransformer, which
    uses output data according to a normal distribution instead of the default
    uniform.

    Args:
        scaler: Either a string specifying a single scaler from the
                sklearn.preprocessing module to apply to all PSG channels
                (one such fit to each channel separately), or
                a list of strings specifying what scaler to fit and apply in
                each channel. E.g. ['StandardScaler', 'RobustScaler'] will
                apply the StandardScaler to channel 0 and RobustScaler to
                channel 1.

    Returns:
        A MultiChannelScaler object.
    """
    if isinstance(scaler, str):
        scaler = [scaler]
    scalers = []
    for sc in scaler:
        kwargs = {}  # Currently kwargs cannot be passed to the scalers
        if sc == "QuantileTransformer":
            kwargs["output_distribution"] = "normal"
        scalers.append((preprocessing.__dict__[sc], kwargs))
    return MultiChannelScaler(scalers=scalers)


def apply_scaling(X, scaler):
    """
    Initializes a MultiChannelScaler object and applies it immediately to the
    same data. Also returns the fit scaler

    Args:
        X:      A ndarray, PSG data, shape [N, C]
        scaler: A string or list of strings, see 'get_scaler'

    Returns:
        Transformed X data
        The fit MultiChannelScaler object
    """
    # Get scaler
    multi_scaler = get_scaler(scaler).fit(X)

    # Fit and apply transformation
    return multi_scaler.transform(X), multi_scaler


class MultiChannelScaler(object):
    """
    Wraps around Scaler objects from the sklearn.preprocessing module,
    initializing, fitting, storing and applying such scalers to/for individual
    channels of a [N, C] shaped ndarray (e.g. PSG data with C channels)
    """
    def __init__(self, scalers, with_centering=True):
        """
        Initializes the scaler object with a set of scaler strings. Does not
        fit or transform any data yet.

        Args:
            scalers:         A list of 2-tuples/lists each of format
                             (scaler class name, kwargs). The outer list should
                             be of length equal to the number of channels of the
                             PSG passed to self.fit, self.transform,
                             self.fit_transform.
            with_centering:  Apply centering to the data. If False, only
                             scaling of the data is applied.
        """
        err = "'scalers' should be a list of 2-tuples/lists (each of format " \
              "(scaler class, kwargs to scaler init), got {}".format(scalers)
        if not isinstance(scalers, (tuple, list, np.ndarray)):
            raise ValueError(err)
        if any([len(s) != 2 for s in scalers]):
            raise ValueError(err)
        # Store scaler class and passed parameters
        self.scaler_tuples = scalers
        self.with_centering = with_centering

        # Store list of initialized scalers fit to each channel
        self.scalers = []

        # Store number of channels
        self.n_channels = None

    def fit(self, X):
        """
        Fit all scalers specified in self.scalers to individual channels of X

        Args:
            X: ndarray of shape [N, C], where C == len(self.scaler_tuples)

        Returns:
            self
        """
        if X.ndim != 2:
            raise ValueError("Invalid shape for X (%s)" % X.shape)
        # Set number of channels
        self.n_channels = X.shape[-1]
        if len(self.scaler_tuples) == 1:
            scaler_tups = self.scaler_tuples * self.n_channels
        else:
            if len(self.scaler_tuples) != self.n_channels:
                raise ValueError("Number of passed scalers ({}) does not "
                                 "match the number of channels in X "
                                 "({})".format(len(self.scaler_tuples),
                                               self.n_channels))
            scaler_tups = self.scaler_tuples

        # Fit the scalers to each channel of X
        fit_scalers = []
        for i, (scaler_cls, scaler_kwargs) in enumerate(scaler_tups):
            try:
                scaler_cls = scaler_cls(**scaler_kwargs,
                                        with_centering=self.with_centering)
            except TypeError:
                scaler_cls = scaler_cls(**scaler_kwargs,
                                        with_mean=self.with_centering)
            xs = X[:, i]
            scaler_cls.fit(xs.reshape(-1, 1))
            fit_scalers.append(scaler_cls)
        self.scalers = fit_scalers
        return self

    def transform(self, X):
        """
        Transform each channel in X according to the (fitted) scalers in
        self.scalers

        Args:
            X: ndarray of shape [N, C], where C == len(self.scalers)

        Returns:
            X, transformed data, shape [N, C]
        """
        if X.shape[-1] != self.n_channels:
            raise ValueError("Invalid input of dimension %i, expected "
                             "last axis with %i channels" % (X.ndim,
                                                             self.n_channels))
        # Prepare volume like X to store results
        transformed = np.empty_like(X)
        for i in range(self.n_channels):
            scl = self.scalers[i]
            s = scl.transform(X[:, i].reshape(-1, 1))
            transformed[:, i] = s.reshape(X.shape[:-1])
        return transformed

    def fit_transform(self, X):
        """ Fits scalers to and immediately transform data in X """
        self.fit(X)
        return self.transform(X)
