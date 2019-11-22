"""
Functions for initializing a SparseHypnogram from hypnogram data output by
functions in utime.io.file_loaders.hyp_file_loaders.
"""

import numpy as np
from utime.hypnogram.stage_mapper import create_variable_ann_to_class_int_dict
from utime.hypnogram import SparseHypnogram
from utime.hypnogram.utils import signal_dense_to_sparse, dense_to_sparse
from utime import defaults


def load_start_duration_stage_format(tuples, period_length_sec,
                                     ann_to_class, **kwargs):
    """
    Initializes a SparseHypnogram from Start-Duration-Stage formatted data.

    Args:
        tuples:             3-tuple of equal length lists of starts, durations
                            and sleep stages (see utime.hypnogram)
        period_length_sec:  Sleep 'epoch'/period length in seconds.
        ann_to_class:       Dictionary mapping from labels in array to sleep
                            stage integer value representations. Can be None,
                            in which case annotations will be automatically
                            inferred.
        **kwargs            Catches sample_rate parameter that will often be
                            passed but not used.

    Returns:
        A SparseHypnogram object
    """
    start_sec, duration_sec, annotations = tuples
    if ann_to_class is None:
        ann_to_class = create_variable_ann_to_class_int_dict(annotations)

    # Translate annotations to class integers and init SparseHypnogram
    ann_class_ints = [ann_to_class[a] for a in annotations]
    sparse_hyp = SparseHypnogram(init_times_sec=start_sec,
                                 durations_sec=duration_sec,
                                 sleep_stages=ann_class_ints,
                                 period_length_sec=period_length_sec)
    return sparse_hyp, ann_to_class


def _load_dense_array(array, period_length_sec, ann_to_class, sample_rate):
    """
    See load_array docstring.
    """
    if array.dtype.type in (np.str_, np.string_):
        # If string were passed, convert to integer representation
        if ann_to_class is None:
            ann_to_class = create_variable_ann_to_class_int_dict(array)
        array = np.array([ann_to_class[a] for a in array])
    if ann_to_class is None:
        # We only need this to check that the labels match the default integer
        # labels for sleep stages.
        ann_to_class = defaults.stage_string_to_class_int
    unique_labels = np.unique(array)
    if np.any(~np.in1d(unique_labels, list(ann_to_class.values()))):
        raise RuntimeError("Labels {} do not match annotation dict of {}. "
                           "If the annotation dict was inferred automatically,"
                           " consider manually setting one.".format(unique_labels,
                                                                    ann_to_class))

    try:
        # Assume the array is 'signal dense' - a ValueError will be thrown if
        # this conversion fail, in which case we can assume the array is dense
        inits, durs, stages = signal_dense_to_sparse(array,
                                                     sample_rate,
                                                     period_length_sec,
                                                     allow_trim=True)
    except ValueError:
        # Dense already
        inits, durs, stages = dense_to_sparse(array,
                                              period_length_sec,
                                              allow_trim=True)
    # Create the SparseHypnogram from the sparse Start-Duration-Stage data
    sparse_hypno = SparseHypnogram(inits, durs, stages, period_length_sec)
    return sparse_hypno, ann_to_class


def load_array(array, period_length_sec, ann_to_class, sample_rate):
    """
    Loads flat ndarrays storing sleep stages and converts it to a
    SparseHypnogram. Supports both 'dense' and 'signal dense' arrays
    (see utime.hypnogram.utils for a description).

    Will attempt to convert from 'signal dense' --> sparse, which will fail if
    the array is already 'dense'. This will trigger converting dense --> sparse

    Args:
        array:              ndarray of shape [-1], integer type
        period_length_sec:  Sleep 'epoch'/period length in seconds.
        ann_to_class:       Dictionary mapping from labels in array to sleep
                            stage integer value representations. Can be None,
                            in which case annotations will be automatically
                            inferred.
        sample_rate:        The sample rate of the original data (needed for
                            signal dense conversion).

    Returns:
        A SparseHypnogram object
    """
    array = array.squeeze()
    if array.ndim == 1:
        return _load_dense_array(array, period_length_sec,
                                 ann_to_class, sample_rate)
    else:
        raise NotImplementedError("Received non-flat numpy array of shape {}"
                                  " for hypnogram data. Currently, a flat "
                                  "array of label values must be passed. Are "
                                  "the values stored in "
                                  "one-hot encoding?".format(array.shape))


_OBJ_TYPE_TO_HYP_LOADER = {
    "StartDurationStageFormat": load_start_duration_stage_format,
    "ndarray": load_array
}


def extract_hyp_data(hyp_obj, period_length_sec, annotation_dict, sample_rate):
    """
    Create a SparseHypnogram object from hypnogram data in formats as output
    by functions in utime.io.file_loaders.hyp_file_loaders.
    """
    loader = _OBJ_TYPE_TO_HYP_LOADER[type(hyp_obj).__name__]
    return loader(hyp_obj,
                  period_length_sec=period_length_sec or defaults.PERIOD_LENGTH_SEC,
                  ann_to_class=annotation_dict,
                  sample_rate=sample_rate)
