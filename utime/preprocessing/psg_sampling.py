"""
Set of functions for down- and re-sampling of PSG signals
"""


def fourier_resample(psg, new_sample_rate, old_sample_rate):
    from mne.filter import resample
    return resample(psg, new_sample_rate, old_sample_rate, axis=0)


def poly_resample(psg, new_sample_rate, old_sample_rate):
    from scipy.signal import resample_poly
    return resample_poly(psg, new_sample_rate, old_sample_rate, axis=0)


def set_psg_sample_rate(psg, new_sample_rate, old_sample_rate, method="poly"):
    """
    Resamples PSG of sample rate 'old_sample_rate' to new sample rate
    'new_sample_rate'. The length of the PSG will be a factor
    new_sample_rate/old_sample_rate the original.

    Args:
        psg:              ndarray of PSG data to resample, shape [N, C].
                          Resampling is performed over axis 0 (sample dim)
        new_sample_rate:  Sample rate of the new signal
        old_sample_rate:  Sample rate of the original signal
        method:           Resampling method, one of poly, fourier

    Returns:
        A resampled PSG ndarray
    """
    new_sample_rate = int(new_sample_rate)
    old_sample_rate = int(old_sample_rate)
    method = method.lower().split("_")[0]
    if method == "poly":
        return poly_resample(psg, new_sample_rate, old_sample_rate)
    elif method == "fourier":
        return fourier_resample(psg, new_sample_rate, old_sample_rate)
    else:
        raise ValueError("Invalid method {} selected.".format(method))
