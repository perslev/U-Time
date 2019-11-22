import numpy as np


def get_steps(samples_per_epoch, sequence):
    """
    Computes the number of gradient update steps to use for training or
    validation.

    Takes an integer 'samples_per_epoch' specifying how many samples should be
    used in 1 epoch. Returns the (ceiled) number of batches of size
    'batch_size' needed for such epoch.

    If 'samples_per_epoch' is None, returns the length of the
    Sequence object.

    Args:
        samples_per_epoch: (int)      Number of samples to use in an epoch
        sequence:          (Sequence) The Sequence object from which samples
                                      will be generated

    Returns:
        (int) Number of steps to take in the epoch
    """
    if samples_per_epoch:
        return int(np.ceil(samples_per_epoch / sequence.batch_size))
    else:
        return len(sequence)
