"""
Collection of data structures for storing hypnogram raw data (before passed to
one of the classes in utime.hypnogram.hypnograms) and informing (through its
type) about how this data is formatted.
"""


class StartDurationStageFormat(tuple):
    """
    3-tuple class that can store exactly 3 list-like elements of identical
    length.

    utime uses this format to pass 'sparse' hypnogram data around. That is,
    hypnogram information encoded in 3 identically sizes lists:

    - init time seconds (list of integers of initial period time points)
    - durations seconds (list of integers of seconds of period duration)
    - sleep stage (list of sleep stages, typically integer, for each period)

    ... from which a sleep stage at a particular point in time can be inferred.
    """
    def __init__(self, *args):
        super(StartDurationStageFormat, self).__init__()

        # Check length exactly 3 and all same length
        if len(self) != 3:
            raise RuntimeError("StartDurationStageFormat should contain exactly"
                               " 3 sub-tuples or lists (storing respectively: "
                               "init times in seconds, durations in second and"
                               " sleep stages.)")
        try:
            if not (len(self[0]) == len(self[1]) == len(self[2])):
                raise TypeError
        except TypeError:
            raise RuntimeError("StartDurationStageFormat should contain 3 "
                               "sub-tuples or lists of equal length.")
