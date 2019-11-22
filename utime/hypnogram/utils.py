"""
A collection of functions for converting dense hypnogram encodings to sparse
(start-duration-stage format) hypnogram encodings.

... and other small utility functions for handling hypnogram data.

-----
HYPNOGRAM FORMATS

utime operates with the following notation on hypnogram encodings:

'Sparse hypnogram':
    The hypnogram information is encoded in 3 identically sizes lists:
      - init time seconds (list of integers of initial period time points)
      - durations seconds (list of integers of seconds of period duration)
      - sleep stage (list of sleep stages, typically integer, for each period)
    ... from which a sleep stage at a particular point in time can be inferred.

'Dense hypnogram':
    A 'dense' hypnogram array is an array that stores 1 sleep stage for every
    'period_length_sec' seconds (typically 30s) of sleep.

'Signal dense hypnogram':
    A 'signal dense' hypnogram array is an array that stores the sleep stage
    for every sample in the input signal (so for 2 seconds of 100 Hz signal the
    signal dense array would have length 200).
"""

from utime.hypnogram import SparseHypnogram
import numpy as np


def create_class_int_to_period_idx_dict(hypnogram):
    """
    Takes a SparseHypnogram or DenseHypnogram object and returns a dictionary
    that maps from sleep stage integer labels (typically 0, ..., 5) to numpy
    arrays of indices that correspond to periods of the given sleep stage
    within the (dense) hypnogram.

    Args:
        hypnogram: A SparseHypnogram or DenseHypnogram object

    Returns:
        A dictionary
    """
    classes = hypnogram.classes
    if isinstance(hypnogram, SparseHypnogram):
        hypnogram = hypnogram.to_dense()
    stages = hypnogram["sleep_stage"].to_numpy()
    return {c: np.where(stages == c)[0] for c in classes}


def sparse_to_csv_file(inits, durs, stages, out_path, stage_map=None):
    """
    Takes hypnogram data in the Start-Duration-Stage format
    (see utime.hypnogram.formats) separated into the 3 sub-lists and outputs
    the data to a csv file at path 'out_path'.

    Args:
        inits:      List of integers of initial period time points
        durs:       List of integers of seconds of period duration
        stages:     List of integer sleep stages for each period
        out_path:   String path to an output csv file
        stage_map:  Optional dictionary mapping integers in 'stages' to other
                    labels values (such as string stage representations)
    """
    assert len(inits) == len(durs) == len(stages)
    with open(out_path, "w") as out_f:
        for i, d, s in zip(inits, durs, stages):
            if stage_map:
                s = stage_map[s]
            out_f.write("{},{},{}\n".format(i, d, s))


def dense_to_sparse(array, period_length_sec, allow_trim=False):
    """
    Takes a 'dense' hypnogram array (ndarray of shape [-1]) of sleep stage
    labels for every 'period_length_sec' second periods of sleep and returns a
    sparse, Start-Duration-Stage representation of 3 equally sizes lists of
    'inits', 'durations' and 'stages' (see utime.hypnogram.formats).

    Args:
        array:              1D ndarray of (dense) sleep stage labels
        period_length_sec:  Length in seconds of 1 sleep stage period.
        allow_trim:         Allow trimming of the hypnogram duration in the
                            (probably rarely occurring) situation that the last
                            sleep stage period is computed to have a duration
                            of less than 1 second, in which case it cannot be
                            represented by our integer-valued durations.
                            Same goes for when the final period has a length
                            that is not evenly divisible by the period length
                            of 'period_length_sec'.
                            If false, and trimming is needed due to either of
                            the two cases, an error is raised.

    Returns:
        inits, durs, stages format
        3 lists of identical length
    """
    array = array.squeeze()
    if array.ndim != 1:
        raise ValueError("Invalid dense array found of dim {} (expected 1)"
                         "".format(array.ndim))
    end_time = len(array) * period_length_sec

    # Get array of init dense inds
    start_inds = np.where(np.diff(array) != 0)[0] + 1
    start_inds = np.concatenate([[0], start_inds])

    # Get init times (second)
    inits = (start_inds * period_length_sec).astype(np.int)
    durs = np.concatenate([np.diff(inits), [end_time-inits[-1]]])
    stages = array[start_inds]

    if durs[-1] == 0:
        if not allow_trim:
            raise ValueError("Last duration is shorter than 1 second, "
                             "but allow_trim was set to False")
        # Remove trailing
        inits = inits[:-1]
        durs = durs[:-1]
        stages = stages[:-1]
    trail = durs[-1] % period_length_sec
    if trail:
        if not allow_trim:
            raise ValueError("Last duration of length {} seconds is not "
                             "divisible by the period length of {} seconds, "
                             "and allow_trim was set to False")
        durs[-1] -= trail
    return inits, durs, stages


def signal_dense_to_sparse(array, sample_rate, period_length_sec, allow_trim=False):
    """
    Takes a 'signal dense' hypnogram array (ndarray of shape [-1]) of sleep
    stages and returns a sparse, Start-Duration-Stage representation of 3
    equally sizes lists of 'inits', 'durations' and 'stages'
    (see utime.hypnogram.formats).

    A 'signal dense' hypnogram array is an array that stores the sleep stage
    for every sample in the input signal (so for 2 seconds of 100 Hz signal the
    signal dense array would have length 200).

    Converts the 'signal dense' array to a 'dense' array which is then
    converted to sparse.

    Args:
        array:              1D ndarray of (signal dense) sleep stage labels
        sample_rate:        The sample rate of the input signal
        period_length_sec:  Length in seconds of 1 sleep stage period.
        allow_trim:         See 'signal_dense_to_dense' and 'dense_to_sparse'

    Returns:
        inits, durs, stages format
        3 lists of identical length
    """
    d = signal_dense_to_dense(array, sample_rate, period_length_sec, allow_trim)
    return dense_to_sparse(d, period_length_sec, allow_trim)


def signal_dense_to_dense(array, sample_rate, period_length_sec, allow_trim=False):
    """
    Takes a 'signal dense' hypnogram array (ndarray of shape [-1]) of sleep
    stages and returns a dense array of 1 label for every 'period_length_sec'
    second periods of sleep.

    OBS: Assumes that all values within 'period_length_sec' seconds of signal
    are identical starting at the 0th value in the array.

    See 'signal_dense_to_sparse' for a description of signal dense arrays.

    Args:
        array:              1D ndarray of (signal dense) sleep stage labels
        sample_rate:        The sample rate of the input signal
        period_length_sec:  Length in seconds of 1 sleep stage period.
        allow_trim:         Allow the duration of array (in seconds) to be
                            non-evenly divisible by the 'period_length_sec'.
                            The remainder will be ignored.
                            Otherwise, an error is raised if the array cannot
                            be evenly split.

    Returns:
        ndarray of shape [-1] with 1 sleep stage label value for every
        'period_length_sec' seconds of input signal.
    """
    array = array.squeeze()
    if array.ndim != 1:
        raise ValueError("Invalid dense array found of dim {} (expected 1)"
                         "".format(array.ndim))
    if len(array) % sample_rate:
        raise ValueError("Signal dense array of shape {} is not divisible by "
                         "the sample rate of {}".format(array.shape,
                                                        sample_rate))
    end_time = int(len(array) / sample_rate)
    if end_time < period_length_sec:
        raise ValueError("Signal dense array too short (length {}) with period"
                         " length of {} seconds). Maybe the array is already "
                         "dense?".format(len(array), period_length_sec))
    trail = (end_time % period_length_sec) * sample_rate
    if trail:
        if not allow_trim:
            raise ValueError("Signal dense array of length {} ({} seconds) "
                             "is not evenly divisible by the period length "
                             "of {} seconds, and allow_trim was set to "
                             "False.")
        array = array[:-trail]

    # Make dense
    s = [period_length_sec * sample_rate, -1]
    return np.reshape(array, s, order="F")[0, :]
