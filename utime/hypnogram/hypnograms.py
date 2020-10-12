"""
Classes that represent hypnograms in either a 'sparse' or 'dense' format.
See SparseHypnogram and DenseHypnogram docstrings below.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from utime.utils import exactly_one_specified
from utime import Defaults


class SparseHypnogram(object):
    """
    Data structure for hypnogram internally represented sparsely by 3 lists:
    - init time seconds (list of integers of initial period time points)
    - durations seconds (list of integers of seconds of period duration)
    - sleep stage (list of sleep stages, integer, for each period)

    Implements methods to query a sleep stage at a given time point from the
    sparse representation.
    """
    def __init__(self, init_times_sec, durations_sec, sleep_stages,
                 period_length_sec):
        if not (len(init_times_sec) == len(durations_sec) == len(sleep_stages)):
            raise ValueError("Lists 'init_times_sec' and 'sleep_stages' must "
                             "be of equal length.")
        self.period_length_sec = period_length_sec
        init_times_sec, durations_sec, sleep_stages = map(list, (init_times_sec,
                                                                 durations_sec,
                                                                 sleep_stages))
        if init_times_sec[0] != 0:
            # Insert leading OUT_OF_BOUNDS class if hypnogram does not start at
            # second 0
            init_times_sec.insert(0, 0)
            durations_sec.insert(0, init_times_sec[1])
            sleep_stages.insert(0, Defaults.OUT_OF_BOUNDS[1])
        self.inits = np.array(init_times_sec, dtype=np.int32)
        self.durations = np.array(durations_sec, dtype=np.int32)
        self.stages = np.array(sleep_stages, dtype=np.uint8)
        self.class_int_to_idx = {
            c: np.where(self.stages == c)[0] for c in self.classes
        }

        # Check sorted init times
        if np.any(np.diff(init_times_sec) < 0):
            raise ValueError("Array of init times must be sorted.")
        # Check init times and durations match
        if not np.all(np.isclose(np.diff(init_times_sec),
                                 durations_sec[:-1])):
            raise ValueError("Init times and durations do not match.")

    def __str__(self):
        return "SparseHypnogram(start={}s, end={}s, " \
               "length={}s, stages={})".format(
            self.inits[0], self.end_time, self.total_duration, self.classes
        )

    def __repr__(self):
        return str(self)

    @property
    def n_classes(self):
        """
        Returns the current number of unique classes. Note this value could
        change after i.e. stripping functions have been applied the hypnogram
        """
        return len(self.classes)

    @property
    def classes(self):
        """ Returns the unique classes/stages of the hypnogram """
        return np.unique(self.stages)

    @property
    def end_time(self):
        """ Hypnogram end time (seconds) """
        return self.inits[-1] + self.durations[-1]

    @property
    def last_period_start_second(self):
        """ Returns the second at which the last period begins """
        return self.end_time - self.period_length_sec

    @property
    def total_duration(self):
        """
        Returns the total length of the hypnogram.
        Identical to self.end_time when inits[0] == 0 (currently, always the
        case)
        """
        return np.sum(self.durations)

    def set_new_end_time(self, new_end_second):
        """
        Trim the hypnogram from the tail by setting a new (shorter) end-time.

        Args:
            new_end_second: New seconds at which the vhypnogram is trimmed to
                            end
        """
        # Find index of the new end time
        if new_end_second > self.end_time:
            raise ValueError("New end second {} is out of bounds for "
                             "hypnogram of length {} seconds".format(
                new_end_second, self.end_time
            ))
        init_ind = np.where(new_end_second > self.inits)[0][-1]
        self.inits = self.inits[:init_ind+1]
        self.stages = self.stages[:init_ind+1]
        self.durations = self.durations[:init_ind+1]

        # Update last duration
        old_end = self.inits[-1] + self.durations[-1]
        self.durations[-1] -= old_end - new_end_second

    def get_stage_at_sec(self, second):
        """
        Returns the sleep stage at a given second in the hypnogram

        Args:
            second: (int) The integer second at which to query the sleep stage

        Returns:
            (int) The sleep stage integer class at the given time
        """
        second = int(second)
        # Check second is within bounds of study duration
        if second < 0:
            raise ValueError("Query second must be >= 0 (got {})".format(second))
        if second < self.inits[0]:
            raise ValueError("Query second out of bounds (got second {}, but "
                             "first init second is {})".format(second,
                                                               self.inits[0]))
        if second > self.last_period_start_second:
            raise ValueError("Query second out of bounds (got second {}, but"
                             " study ends at second {}, "
                             "last period starts a second {})".format(second,
                                                                      self.end_time,
                                                                      self.last_period_start_second))
        # Find index of second
        ind = np.searchsorted(self.inits, second)
        if ind == len(self.inits) or self.inits[ind] != second:
            ind -= 1
        assert ind >= 0
        return self.stages[ind]

    def get_class_counts(self):
        """
        Computes the class counts for the hypnogram.

        Returns:
            A dictionary mapping class labels to counts.
        """
        counts = defaultdict(int)
        for stage, dur in zip(self.stages, self.durations):
            counts[stage] += dur // self.period_length_sec
        return counts

    def get_random_class_period(self, class_int):
        """
        TODO

        Args:
            class_int:

        Returns:

        """
        idx = np.random.choice(self.class_int_to_idx[int(class_int)], 1)
        start = self.inits[idx]
        end = start + self.durations[idx]
        num_periods = (end-start)//self.period_length_sec
        start_sec = start + (np.random.randint(0, num_periods, 1) *
                             self.period_length_sec)
        return int(start_sec)

    def to_dense(self):
        """
        Returns a DenseHypnogram representation of the stored data
        """
        return DenseHypnogram(period_length_sec=self.period_length_sec,
                              sparse_hypnogram=self)


class DenseHypnogram(pd.DataFrame):
    """
    Data structure for hypnogram in dense representation (one sleep stage
    stored for each period of 'period_length_sec' seconds.
    """
    def __init__(self, period_length_sec, dense_array=None, start_time=0,
                 sparse_hypnogram=None):
        """
        Dense hypnograms are initialized either from 3 sparse lists or a
        SparseHypnogram.

        See doctstring of SparseHypnogram.
        """
        if period_length_sec is None:
            raise ValueError("Must specify 'period_length_sec' argument when "
                             "storing dense hypnograms.")
        if not exactly_one_specified(dense_array, sparse_hypnogram):
            raise ValueError("Must specify either or of 'dense_array' and "
                             "'sparse_hypnogram'")

        if sparse_hypnogram is not None:
            # Assert all durations are divisible by the period length
            if not all(map(lambda x: x % period_length_sec == 0,
                           sparse_hypnogram.durations)):
                raise ValueError(
                    "All values in 'durations_sec' must be divisible "
                    "by the period length of {} seconds."
                    "".format(period_length_sec)
                )

            # Convert to dense arrays with 1 value for each period
            dense_inits, dense_stages = self._to_dense(
                init_times=sparse_hypnogram.inits,
                durations=sparse_hypnogram.durations,
                sleep_stages=sparse_hypnogram.stages,
                period_length=period_length_sec
            )
        else:
            dense_inits = self._get_inits(start_time,
                                          len(dense_array),
                                          period_length_sec)
            dense_stages = dense_array

        # Init the DataFrame base object
        super(DenseHypnogram, self).__init__(data={
            "period_init_time_sec": dense_inits,
            "sleep_stage": dense_stages
        })

    @staticmethod
    def _get_inits(start_time, n_stages, period_length_sec):
        return np.arange(start=start_time,
                         stop=start_time + (period_length_sec * n_stages),
                         step=period_length_sec, dtype=np.int64)

    def _to_dense(self, init_times, durations, sleep_stages, period_length):
        # Get number of periods per period
        periods_pr_period = [d // period_length for d in durations]

        # Duplicate the sleep stages the given number of times and flatten
        stages = np.repeat(sleep_stages, periods_pr_period)

        # Get array of time points
        start_time = init_times[0]
        inits = self._get_inits(start_time, len(stages), period_length)

        return inits, stages
