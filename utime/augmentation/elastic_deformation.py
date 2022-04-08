import logging
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage.filters import gaussian_filter

logger = logging.getLogger(__name__)


def elastic_transform(signal, labels, alpha, sigma, bg_value=0.0):
    """
    Elastic deformation for 1D signals, modified from:
    [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
    Convolutional Neural Networks applied to Visual Document Analysis", in
    Proc. of the International Conference on Document Analysis and
    Recognition, 2003.

    Modified from:
    https://gist.github.com/chsasank/4d8f68caf01f041a6453e67fb30f8f5a

    Deforms both the signal and labels if len(labels) == len(signal)
    Signal linearly interpolated
    Labels nearest neighbour interpolated
    """
    assert signal.ndim in (1, 2, 3)
    org_sig_shape = signal.shape
    org_lab_shape = labels.shape
    if signal.ndim == 3:
        signal = signal.reshape(-1, signal.shape[-1])
        labels = labels.reshape(-1, 1)
    elif signal.ndim == 1:
        signal = np.expand_dims(signal, axis=-1)

    seg_length = signal.shape[0]
    channels = signal.shape[1]
    dtype = signal.dtype

    # Define coordinate system
    coords = (np.arange(seg_length),)

    # Initialize interpolators
    intrps = []
    for i in range(channels):
        intrps.append(RegularGridInterpolator(coords, signal[:, i],
                                              method="linear",
                                              bounds_error=False,
                                              fill_value=bg_value))

    # Get random elastic deformations
    dx = gaussian_filter((np.random.rand(seg_length) * 2 - 1), sigma,
                         mode="constant", cval=0.) * alpha

    # Define sample points
    indices = np.reshape(coords[0] + dx, (-1, 1))

    # Interpolate all signal channels
    signal = np.empty(shape=signal.shape, dtype=dtype)
    for i, intrp in enumerate(intrps):
        signal[:, i] = intrp(indices).astype(dtype)

    # Interpolate labels if passed, only if same shape as input
    if labels is not None and len(labels) == len(signal):
        lab_intrp = RegularGridInterpolator(coords, labels,
                                            method="nearest",
                                            bounds_error=False,
                                            fill_value=0)
        labels = lab_intrp(indices).astype(labels.dtype)

    return signal.reshape(org_sig_shape), labels.reshape(org_lab_shape)
