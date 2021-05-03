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

import numpy as np
from utime.hypnogram import SparseHypnogram
from utime.hypnogram.formats import StartDurationStageFormat
from utime.hypnogram.stage_mapper import create_variable_ann_to_class_int_dict


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
    start_inds = np.where([array[i+1] != array[i] for i in range(len(array)-1)])[0] + 1  # find transition indices
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
    if sample_rate is None or period_length_sec is None:
        raise ValueError("Must specify the 'sample_rate' and 'period_length_sec' parameters.")
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


def ndarray_to_ids_format(array, period_length_sec, sample_rate):
    """
    Loads flat ndarrays storing sleep stages and converts it to a
    init-duration-stage format. Supports both 'dense' and 'signal dense' arrays
    (see utime.hypnogram.utils for a description).

    Will attempt to convert from 'signal dense' --> sparse, which will fail if
    the array is already 'dense'. This will trigger converting dense --> sparse

    Args:
        array:              ndarray of shape [-1], integer type
        period_length_sec:  Sleep 'epoch'/period length in seconds.
        sample_rate:        The sample rate of the original data (needed for
                            signal dense conversion).

    Returns:
        A StartDurationStageFormat object
    """
    array = array.squeeze()
    if array.ndim == 1:
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
        return StartDurationStageFormat((inits, durs, stages))
    else:
        raise NotImplementedError("Received non-flat numpy array of shape {}"
                                  " for hypnogram data. Currently, a flat "
                                  "array of label values must be passed. Are "
                                  "the values stored in "
                                  "one-hot encoding?".format(array.shape))


def sparse_hypnogram_from_ids_format(ids_tuple, period_length_sec, ann_to_class):
    """
    Initializes a SparseHypnogram from Start-Duration-Stage formatted data.

    Args:
        ids_tuple:          3-tuple of equal length lists of starts, durations
                            and sleep stages (see utime.hypnogram)
        period_length_sec:  Sleep 'epoch'/period length in seconds.
        ann_to_class:       Dictionary mapping from labels in array to sleep
                            stage integer value representations. Can be None,
                            in which case annotations will be automatically
                            inferred.

    Returns:
        A SparseHypnogram object, annotation dict
    """
    start_sec, duration_sec, annotations = ids_tuple
    if ann_to_class is None:
        ann_to_class = create_variable_ann_to_class_int_dict(annotations)

    # Translate annotations to class integers and init SparseHypnogram
    ann_class_ints = [ann_to_class[a] for a in annotations]
    sparse_hyp = SparseHypnogram(init_times_sec=start_sec,
                                 durations_sec=duration_sec,
                                 sleep_stages=ann_class_ints,
                                 period_length_sec=period_length_sec)
    return sparse_hyp, ann_to_class


def hyp_has_gaps(init_times_sec, durations_sec):
    """
    Checks if a hypnogram as specified by a list of init times in seconds and a list of
    same length or length N-1 of durations in seconds has any gaps.

    :param init_times_sec: list of integer start times, length N
    :param durations_sec: list of integer duration times, length N or N-1
    :return: bool
    """
    if len(durations_sec) == len(init_times_sec):
        durations_sec = durations_sec[:-1]
    actual_diffs = np.diff(init_times_sec)
    return np.any(~np.isclose(actual_diffs, durations_sec))


def fill_hyp_gaps(init_times_sec, durations_sec, stages, fill_value):
    """
    Fill gaps in a hypnogram in inits, durs, stages form with a value 'fill_value'.

    E.g.:

    Inits: [0, 10, 50]
    Durs: [10, 10, 10]
    Stages: ['W', 'N1', 'N1']

    and fill_value 'UNKNOWN', returns:

    Inits: [0, 10, 20, 50]
    Durs: [10, 10, 30, 10]
    Stages: ['W', 'N1', 'UNKNOWN', 'N1']
    """
    if not hyp_has_gaps(init_times_sec, durations_sec):
        # Do nothing
        return init_times_sec, durations_sec, stages
    assert len(init_times_sec) == len(durations_sec) == len(stages), "Inits, durations and stages must have equal length"
    actual_diffs = np.diff(init_times_sec)
    gap_lengths = actual_diffs - durations_sec[:-1]
    gap_inds = np.where(gap_lengths)[0]
    init_times_sec, durations_sec, stages = map(list, (init_times_sec, durations_sec, stages))
    for ind in gap_inds:
        if ind == 0 or ind >= len(gap_lengths):
            raise NotImplementedError("The implementation has not yet been tested for its handling "
                                      "of this situation. Please raise an issue on GitHub.")
        # Insert gap section
        length = gap_lengths[ind]
        init_times_sec.insert(ind+1, init_times_sec[ind] + durations_sec[ind])
        durations_sec.insert(ind+1, length)
        stages.insert(ind+1, fill_value)
    return tuple(map(tuple, (init_times_sec, durations_sec, stages)))


def load_events_file(events_file_path):
    """
    Load an .ids events file.

    Args:
        events_file_path: Path to .ids events file
    
    Returns:
        A StartDurationStage format tuple
    """
    from utime.io.hypnogram.hyp_extractors import extract_from_start_dur_stage
    events = list(zip(*extract_from_start_dur_stage(events_file_path)))
    # Make sure events are sorted by init time
    events = sorted(events, key=lambda x: x[0])
    return events


def get_indices_from_events(events, period_length_sec):
    """

    :param events:
    :param period_length_sec:
    :return:
    """
    indices = []
    for onset, dur, event in events:
        event = event.upper()
        round_func = np.ceil if "START" in event else (np.floor if "STOP" in event else np.round)
        index = int(round_func(onset / period_length_sec))
        indices.append(index)
    return indices


def get_psg_start_stop_events(events):
    """
    :param events:
    :return:
        List of START/STOP events. Length 2 list of format:
            [[start_sec, 0?? (duration), "START PSG"], [start_sec, 0?? (duration), "STOP PSG"]]
    """
    # Filter to keep only start/stop PSG events
    start_stop_events = list(filter(lambda e: "psg" in e[-1].lower(), events))
    if len(start_stop_events) != 2:
        raise ValueError("Found != 2 start/stop PSG events: {}".format(start_stop_events))

    # Make sure events are sorted by init time
    start_stop_events = sorted(start_stop_events, key=lambda x: x[0])
    if not ("start" in start_stop_events[0][-1].lower() and "stop" in start_stop_events[-1][-1].lower()):
        raise ValueError("Invalid start/stop events: {}".format(start_stop_events))

    # Standardize
    return [
        (start_stop_events[0][0], start_stop_events[0][1], "PSG START"),
        (start_stop_events[1][0], start_stop_events[1][1], "PSG STOP"),
    ]


def get_light_events(events):
    """

    :param events:
    :return:
    """
    # Filter to keep only lights on and lights off events
    light_events = list(filter(lambda e: "light" in e[-1].lower(), events))

    # Make sure events are sorted by init time
    light_events = sorted(light_events, key=lambda x: x[0])

    # Standardize to (onset, duration, "LIGHTS ON/OFF (STOP/START)") tuples
    # Also subtract offset seconds (if array was shifted in time due to trimming, e.g. with start/stop PSG events)
    light_events = [(e[0], e[1], "LIGHTS ON (STOP)" if "on" in e[-1].lower() else "LIGHTS OFF (START)")
                    for e in light_events]

    return light_events


def check_start_stop_events(start_stop_events):
    """

    :param start_stop_events:
    :return:
    """
    # Make sure events are sorted by init time
    start_stop_events = sorted(start_stop_events, key=lambda x: x[0])
    # Check alternating pairs
    start_stops = ["START" if "START" in e[-1] else "STOP" for e in start_stop_events]
    for i in range(len(start_stops)-1):
        if start_stops[i] == start_stops[i+1]:
            raise ValueError(f"Not alternating START/STOP events, {start_stop_events}")
    # Check starts with START event
    if start_stops[0] != "START":
        raise ValueError(f"Fisrt START/STOP event must be a 'START' event, got {start_stop_events}")
    # Check ends with STOP event
    if start_stops[-1] != "STOP":
        raise ValueError(f"Last START/STOP event must be a 'STOP' event, got {start_stop_events}")
    return start_stop_events


def filter_events_by_start_stop_events(events, start_stop_events):
    """

    :param events:
    :return:
    """
    # Sort, check alternating, starting with "START" event
    start_stop_events = check_start_stop_events(start_stop_events)

    # Extract [[START, STOP]..] pairs
    start_stop_pairs = []
    for i in range(0, len(start_stop_events), 2):
        start_stop_pairs.append(start_stop_events[i:i+2])

    # TODO:
    #  Inefficient search, but does it matter?
    filtered_events = []
    for event in events:
        # Remove any event which is not in [START...STOP] (seconds) of any of the STRT/STOP pairs
        for (start_sec, _, _), (stop_sec, _, _) in start_stop_pairs:
            if not (event[0] < start_sec or event[0] + event[1] > stop_sec):
                # In period, add to filtered list and stop search
                filtered_events.append(event)
                break
    return filtered_events


def filter_hypnogram_by_start_stop_events(dense_hypnogram_array, start_stop_events,
                                          period_length_sec, offset_sec=0):
    """
    TODO
    """
    # Sort, check alternating, starting with "START" event, ends with "STOP" event
    start_stop_events = check_start_stop_events(start_stop_events)

    # Add offset sec to starts
    start_stop_events = [
        (max(0, e[0] - offset_sec), e[1], e[2]) for e in start_stop_events
    ]

    # Extract indicies from the start stop second events
    start_stop_indices = get_indices_from_events(start_stop_events, period_length_sec)

    # Split into sections
    hypnogram_sections = np.split(np.asarray(dense_hypnogram_array), start_stop_indices)

    # Keep only between START...STOP
    to_keep = [hypnogram_sections[i] for i in range(1, len(hypnogram_sections), 2)]
    return np.concatenate(to_keep)
