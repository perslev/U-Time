from .utils import batch_wrapper
from .base_sequence import requires_all_loaded
from .batch_sequence import BatchSequence
from .random_batch_sequence import RandomBatchSequence
from .balanced_random_batch_sequence import BalancedRandomBatchSequence
from .multi_sequence import MultiSequence, ValidationMultiSequence


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
