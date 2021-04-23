"""
A set of functions for automatically applying simple preprocessing steps for
removing a certain class and/or ensure that a pair of PSG and HYP files match
each other in length

OBS: It is always assumed that the PSG and HYP files start at the same real
time. That is, they are aligned with respect to their first entries. Any
discrepancy between PSG and HYP lengths is assumed to follow from either of the
two extending longer in time at the end of the study. The data that extends
beyond the other file will normally be discarded (see strip functions below)
"""

import numpy as np
from utime import Defaults
from utime.hypnogram import SparseHypnogram
from utime.errors import NotLoadedError, StripError


_STRIP_ERR = StripError("Unexpected difference between PSG and HYP lengths.")


def _strip(hyp, mask, inits, durs, stages, pop_from_start):
    """
    Helper function for 'strip_class_leading' and 'strip_class_trailing'
    Removes elements from beginning of lists 'inits', 'durs', 'stages'
    according to 'mask' if pop_from_start=True, otherwise from the end of those
    lists.
    """
    for m in mask:
        if not m:
            break
        if pop_from_start:
            inits.pop(0), durs.pop(0), stages.pop(0)
        else:
            inits.pop(), durs.pop(), stages.pop()
    hyp.inits = np.array(inits, hyp.inits.dtype)
    hyp.durations = np.array(durs, hyp.durations.dtype)
    hyp.stages = np.array(stages, hyp.stages.dtype)


def strip_class_leading(psg, hyp, class_int, sample_rate, check_lengths=False, **kwargs):
    """
    Remove stage 'class_int' events from start and/or end of hypnogram
    Typically applied in 'strip_class_leading_and_trailing'
    See drop_class function for argument description.
    """
    remove_mask = np.asarray(hyp.stages) == class_int
    i, d, s = list(hyp.inits), list(hyp.durations), list(hyp.stages)
    _strip(hyp, remove_mask, i, d, s, pop_from_start=True)
    if check_lengths and not assert_equal_length(psg, hyp, sample_rate):
        raise _STRIP_ERR
    return psg, hyp


def strip_class_trailing(psg, hyp, class_int, sample_rate, check_lengths=False, **kwargs):
    """
    Remove stage 'class_int' events from the end of hypnogram
    Typically applied in 'strip_class_leading_and_trailing'
    See drop_class function for argument description.
    """
    remove_mask = np.asarray(hyp.stages) == class_int
    i, d, s = list(hyp.inits), list(hyp.durations), list(hyp.stages)
    _strip(hyp, reversed(remove_mask), i, d, s, pop_from_start=False)
    if check_lengths and not assert_equal_length(psg, hyp, sample_rate):
        raise _STRIP_ERR
    return psg, hyp


def strip_class_leading_and_trailing(psg, hyp, class_int, sample_rate, check_lengths=False, **kwargs):
    """
    Drops a class 'class_int' from the head and tail of a hypnogram file.
    Does not strip the PSG or HYP further. If this function is applied alone,
    the PSG and HYP lengths should precisely match after dropping the class
    See drop_class function for argument description.
    """
    strip_class_leading(psg, hyp, class_int, sample_rate)
    strip_class_trailing(psg, hyp, class_int, sample_rate)
    if check_lengths and not assert_equal_length(psg, hyp, sample_rate):
        raise _STRIP_ERR
    return psg, hyp


def strip_psg_to_match_hyp_len(psg, hyp, sample_rate, check_lengths=False, **kwargs):
    """
    Trims the tail of a PSG to match the length of a hypnogram.
    See drop_class function for argument description.
    """
    psg_len_sec = psg.shape[0] / sample_rate
    diff_sec = psg_len_sec - hyp.total_duration
    if diff_sec < 0:
        raise StripError("HYP length is larger than PSG length, "
                         "should not strip PSG. Consider the "
                         "'strip_hyp_match_psg_len' or 'strip_to_match' "
                         "functions")
    elif diff_sec == 0:
        return psg
    idx_to_strip = int(sample_rate * diff_sec)
    if check_lengths and not assert_equal_length(psg, hyp, sample_rate):
        raise _STRIP_ERR
    return psg[:-idx_to_strip]


def end_pad_psg(psg, hyp, sample_rate, pad_value=0.0, check_lengths=False, **kwargs):
    """
    TODO

    Args:
        psg:
        hyp:
        sample_rate:
        pad_value:
        check_lengths:

    Returns:

    """
    n_seconds = hyp.total_duration - psg.shape[0]/sample_rate
    if n_seconds < 0:
        raise StripError("Hypnogram should be longer than PSG for "
                         "'end_pad_psg' to make sense. Got a negative time "
                         "difference of {} seconds.".format(n_seconds))
    n_inserts = int(n_seconds * sample_rate)
    padded_psg = np.empty(shape=[psg.shape[0] + n_inserts, psg.shape[1]],
                          dtype=psg.dtype)
    padded_psg[:len(psg)] = psg
    padded_psg[len(psg):] = pad_value
    if check_lengths and not assert_equal_length(padded_psg, hyp, sample_rate):
        raise _STRIP_ERR
    return padded_psg


def strip_hyp_to_match_psg_len(psg, hyp, sample_rate, check_lengths=False, **kwargs):
    """
    Strips a (longer) hypnogram to match the length of a (shorter) PSG
    See the SparseHypnogram.set_new_end_time method
    See drop_class function for argument description.
    """
    psg_len_sec = psg.shape[0] / sample_rate
    diff_sec = hyp.end_time - psg_len_sec
    if diff_sec < 0:
        raise StripError("PSG length is larger than HYP length, "
                         "should not strip HYP. Consider the "
                         "'strip_psg_to_match_hyp_len' or 'strip_to_match' "
                         "functions")
    elif diff_sec == 0:
        return hyp
    if diff_sec % hyp.period_length_sec:
        raise StripError("Time difference between PSG and HYP ({} sec) not"
                         " evenly divisible by the period length "
                         "({} sec)".format(diff_sec, hyp.period_length_sec))
    hyp.set_new_end_time(hyp.end_time - diff_sec)
    if check_lengths and not assert_equal_length(psg, hyp, sample_rate):
        raise _STRIP_ERR
    return hyp


def strip_to_match(psg, hyp, sample_rate, class_int=None, check_lengths=False, **kwargs):
    """
    Strips to match the PSG and HYP lengths using the following ordered steps:
      1) Drops any potential "OUT_OF_BOUNDS" segments
      2) If a class_int is passed and if the hypnogram is longest, attempt
         to match by removing the class_int stages from the end of the
         hypnogram
      3) If the hypnogram is longest, reduce the length of the hypnogram
      4) If the PSG is longest, strip the PSG from the tail to match

    See drop_class function for argument description.
    """
    # Drop out of bounds segments
    psg, hyp = drop_class(psg, hyp,
                          sample_rate=sample_rate,
                          class_int=Defaults.OUT_OF_BOUNDS[1],
                          strip_only=True,
                          call_strip_to_match=False)
    psg_length_sec = psg.shape[0] / sample_rate
    if class_int and hyp.total_duration > psg_length_sec:
        # Remove trailing class integer
        strip_class_trailing(None, hyp, class_int, None)
    if hyp.total_duration > psg_length_sec:
        if not hyp.total_duration % 30 and psg.shape[0] % 30:
            # PSG is shorter than hyp, and hyp seems to be 'correct', pad PSG
            psg = end_pad_psg(psg, hyp, sample_rate, pad_value=0.0)
        else:
            # Trim PSG first to ensure length divisible by period_length_sec*sampl_rate
            psg, _ = trim_psg_trailing(psg, sample_rate, hyp.period_length_sec)
            hyp = strip_hyp_to_match_psg_len(psg, hyp, sample_rate)
            psg_length_sec = psg.shape[0] / sample_rate
    if psg_length_sec > hyp.total_duration:  # Note total_dur. is a property
        psg = strip_psg_to_match_hyp_len(psg, hyp, sample_rate)
    if check_lengths and not assert_equal_length(psg, hyp, sample_rate):
        raise _STRIP_ERR
    return psg, hyp


def strip_class(psg, hyp, class_int, sample_rate, check_lengths=False, **kwargs):
    """
    Remove class 'class_int' if leading or trailing, then strip to match
    See drop_class function for argument description.
    """
    raise DeprecationWarning("'strip_class' is no longer a supported strip "
                             "function. Use 'drop_class' with "
                             "strip_only=True instead.")


def convert_to_strip_mask(bool_mask):
    """
    Takes a bool list/array where True represents a class to drop; returns
    a bool array where True entries that have a False entry to the left or
    right in the list, and that are not in the first or last position in the
    list, are replaced by False.

    In other words, this function removes any True entries that are not part of
    a segment of potentially multiple Trues that connect to the first or last
    entry in the list.

    In:  [True, True, False, True,  True,  False, True, True]
    Out: [True, True, False, False, False, False, True, True]
                               *      *
    """
    def false_index(arr):
        for i, elem in enumerate(arr):
            if not elem:
                return i
    bool_mask = np.array(bool_mask, dtype=np.bool, copy=True)
    forward_false_idx = false_index(bool_mask)
    backward_false_idx = false_index(bool_mask[::-1])
    bool_mask[forward_false_idx:-backward_false_idx] = False
    return bool_mask


def drop_class(psg, hyp, class_int, sample_rate,
               strip_only=False, check_lengths=False,
               call_strip_to_match=True, **kwargs):
    """
    Drops a sleep stage / class with integer value 'class_int' entirely. That
    is, all 'class_int' stages in SparseHypnogram 'hyp' will be dropped and
    init times will be recomputed. The corresponding PSG signal will likewise
    be removed entirely.

    This function is used mostly to remove 'UNKNOWN', 'OTHERS', 'NOT SCORED'
    type sleep stage classes, often collectively assigned class integer 5.

    Note that due to the re-computing of the hypnogram init times, one should
    no longer look up sleep stages in the new, stripped hypnogram using real
    time stamps from the study (second '100' in the old and new hypnogram may
    no longer correspond to the same data).

    Also note that dropping a class this way will cause flanking PSG segments
    to transition sharply/non smoothly if the dropped class was not in the
    head or tail of the study. This is, however, in our experiance, not an
    issue as these stages - in our applications - mostly occur near the
    beginning or end of the study and rarely in general.

    Args:
        psg:           A ndarray, PSG data, of shape [N, C]
        hyp:           A SparseHypnogram
        class_int:     Integer value corresponding to the class that should be
                       dropped.
        sample_rate:   The sample rate (Hz) of the passed PSG
        strip_only:    If True, only drop segments for the class connected to
                       the start or end of the hypnogram. I.e. if the class
                       appears in the middle of the hypnogram flanked by other
                       classes, it will not be removed.
        check_lengths: Assert that the PSG and HYP have equal length after the
                       stripping function has been applied. This is usually
                       wanted, but the default parameter is set to False for
                       all strip functions, as they may be used inside other
                       strip functions. The high-level 'apply_strip_func'
                       function always sets check_lengths=True on the
                       'top-level' strip function.
        call_strip_to_match: Call call_strip_to_match() at the end of
                             this function (before length check)

    Returns:
        psg, hyp
    """
    # Get all stages of class 'class_int'
    mask = hyp.stages == class_int
    if strip_only:
        mask = convert_to_strip_mask(mask)

    # Get init and duration drop masks
    inits_to_drop = hyp.inits[mask]
    durs_to_drop = hyp.durations[mask]

    # Find all PSG indices that should be removed
    inds_to_remove = []
    for i, (start_sec, dur) in enumerate(zip(inits_to_drop, durs_to_drop)):
        end_sec = start_sec + dur
        # Convert to indices
        start_idx = int(start_sec * sample_rate)
        end_idx = int(end_sec * sample_rate)
        inds_to_remove.extend(range(start_idx, min(len(psg), end_idx)))

    # Drop PSG on inds
    psg = np.delete(psg, inds_to_remove, axis=0)

    # Drop the class from the hypnogram
    keep_mask = ~mask
    durations = hyp.durations[keep_mask]
    stages = hyp.stages[keep_mask]

    # Re-compute inits
    inits = np.array([0] + list(np.cumsum(durations)[:-1]))

    # Create new hypnogram (just to perform some value checks)
    hyp = SparseHypnogram(inits, durations, stages, hyp.period_length_sec)

    if call_strip_to_match:
        psg, hyp = strip_to_match(psg, hyp, sample_rate=sample_rate)
    if check_lengths and not assert_equal_length(psg, hyp, sample_rate):
        raise StripError("Unexpected difference between PSG length ({} "
                         "seconds) and HYP length ({} seconds). This error "
                         "occurred in 'drop_class' strip func on class {} "
                         "for SleepPair with sample rate:\n{}".format(
            psg.shape[0] / sample_rate, hyp.total_duration,
            class_int, sample_rate)
        )
    return psg, hyp


def trim_psg_trailing(psg, sample_rate, period_length_sec, hyp=None, **kwargs):
    """
    Trims the length of an input PSG array so that it is evenly divisible by a number
        i = sample_rate * period_length_sec

    Does not consider any of the following arguments (ignored):
        hyp, class_int, check_lengths

    Args:
        psg: array, shape [N, C] where N is the number of samples and C the number of channels
        sample_rate: The sample rate in 1/s of the input PSG
        period_length_sec: Length in seconds of 1 segment in the PSG
        hyp: Ignored

    Returns:
        PSG, hyp/None
    """
    i = sample_rate * period_length_sec
    if len(psg) % i != 0:
        psg = psg[:-int(len(psg) % i)]
    return psg, hyp


def assert_equal_length(psg, hyp, sample_rate):
    """ Return True if the PSG and HYP have equal lengths in seconds """
    return psg.shape[0] / sample_rate == hyp.total_duration


def apply_strip_func(sleep_study, sample_rate):
    """
    Applies the strip function set on a SleepStudy object to itself.

    Args:
        sleep_study: A SleepStudy object
        sample_rate: The sample rate of the currently set PSG.

    Returns:
        The PSG ndarray and SparseHypnogram objects to which the strip function
        has been applied.
    """
    if not sleep_study.loaded:
        raise NotLoadedError("Cannot apply strip func to {} "
                             "as it is not loaded.".format(sleep_study))
    func_str, kwargs = sleep_study.strip_func
    f = globals()[func_str]
    try:
        psg, hypnogram = f(psg=sleep_study.psg,
                           hyp=sleep_study.hypnogram,
                           sample_rate=sample_rate,
                           check_lengths=True,  # Always check in the end
                           period_length_sec=sleep_study.period_length_sec,
                           **kwargs)
    except StripError as e:
        sleep_study.raise_err(StripError,
                       "Could not perform strip using {} on class "
                       "{}. Please investigate "
                       "manually.".format(*sleep_study.strip_func), _from=e)
    return psg, hypnogram
