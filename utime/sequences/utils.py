import logging
import numpy as np
from .batch_sequence import BatchSequence
from .random_batch_sequence import RandomBatchSequence
from .balanced_random_batch_sequence import BalancedRandomBatchSequence

logger = logging.getLogger(__name__)


def batch_wrapper(generator, x_shape, y_shape,
                  x_dtype=np.float32, y_dtype=np.uint8):
    batch_size = x_shape[0]
    x_batch = np.empty(x_shape, dtype=x_dtype)
    y_batch = np.empty(y_shape, dtype=y_dtype)
    batch_length = 0
    for i, (xx, yy) in enumerate(generator):
        batch_ind = i % batch_size
        try:
            x_batch[batch_ind] = xx
            y_batch[batch_ind] = yy
        except ValueError:
            continue
        batch_length += 1
        if batch_ind == (batch_size-1):
            yield x_batch.copy(), y_batch.copy()
            batch_length = 0
    if batch_length:
        yield x_batch[:batch_length].copy(), y_batch[:batch_length].copy()


def get_sequence_class(random_batches, balanced_sampling):
    """
    Returns the appropriate BatchSequence sub-class given a set of parameters.

    Note: balanced_sampling cannot be True with random_batches=False

    Args:
        random_batches:     (bool) The BatchSequence should sample random
                                   batches across the SleepStudyDataset
        balanced_sampling:  (bool) The BatchSequence should sample randomly
                                   and uniformly across individual classes.

    Returns:
        A BatchSequence typed class (non-initialized)
    """
    if random_batches:
        if balanced_sampling:
            return BalancedRandomBatchSequence
        else:
            return RandomBatchSequence
    elif balanced_sampling:
        raise ValueError("Cannot use 'balanced_sampling' with "
                         "'random_batches' set to False.")
    else:
        return BatchSequence


def infer_dpe_and_chans(dataset_queue):
    with dataset_queue.get_random_study() as sleep_study:
        return sleep_study.data_per_period, sleep_study.n_sample_channels


def get_batch_sequence(dataset_queue,
                       batch_size=16,
                       random_batches=True,
                       balanced_sampling=True,
                       n_classes=None,
                       margin=0,
                       augmenters=None,
                       scaler=None,
                       batch_wise_scaling=False,
                       no_log=False,
                       **kwargs):
    """
    Return a utime.sequences BatchSequence object made from a dataset queue.
    A BatchSequence object is used to extract batches of data from all or
    individual SleepStudy objects represented by this SleepStudyDataset.

    All args pass to the BatchSequence object.
    Please refer to its documentation.

    Returns:
        A BatchSequence object
    """
    data_per_epoch, n_channels = infer_dpe_and_chans(dataset_queue)

    # Init and return the proper BatchSequence sub-class
    sequence_class = get_sequence_class(random_batches, balanced_sampling)
    return sequence_class(dataset_queue=dataset_queue,
                          batch_size=batch_size,
                          data_per_period=data_per_epoch,
                          n_classes=n_classes,
                          n_channels=n_channels,
                          margin=margin,
                          augmenters=augmenters,
                          batch_scaler=scaler if batch_wise_scaling else None,
                          identifier=dataset_queue.dataset.identifier,
                          no_log=no_log,
                          **kwargs)
