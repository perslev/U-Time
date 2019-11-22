"""
The BatchSequence class implements the primary mode of access to batches of
sampled data from a list of SleepStudy objects.

See docstring below.
"""

import numpy as np
from utime.sequences.base_sequence import BaseSequence, requires_all_loaded
from utime.errors import MarginError


def _infer_n_classes(n_classes, pairs):
    """
    Helper function for inferring the n_classes parameter from a list of
    SleepStudy pairs
    """
    if n_classes is not None:
        return int(n_classes)
    else:
        pairs = [p for p in pairs if p.loaded]
        return int(np.max([p.n_classes for p in pairs]))


def _check_margin(n_periods, margin, at_idx=None):
    """
    Helper function for sampling batches with BatchSequence(margin > 0).

    Args:
        n_periods:
        margin:
        at_idx:

    Returns:

    """
    if not isinstance(margin, int):
        raise ValueError(
            "Margin must be an integer. Got '{}'".format(type(margin)))
    if margin < 0:
        raise ValueError(
            "Margin must be a non-negative integer, got {}".format(margin))
    # Check margin is not too large
    if margin > (n_periods-1) // 2:
        raise ValueError("Margin too large for dataset length.")
    if at_idx is not None:
        if margin > at_idx:
            raise MarginError("Margin of {} too large at period idx {} (extends"
                              " to negative indices)".format(margin, at_idx),
                              shift=margin - at_idx)
        if margin + at_idx >= n_periods:
            raise MarginError("Margin of {} too large at period idx {} (extends"
                              " to index >= to total number of periods"
                              " ({}))".format(margin, at_idx, n_periods),
                              shift=n_periods - (margin + at_idx + 1))


class BatchSequence(BaseSequence):
    """
    Implements the BatchSequence object.
    The BatchSequence currently provides the main mechanism for sampling batches
    of data across one or more SleepStudy objects.

    The BatchSequence acts like a tf.keras.Sequence object, returning batches
    of data when indexed.

    Supports both 'single-epoch' sampling and sampling with a margin
    (always used with UTime) giving sequences of multiple, connected segments
    of signal as output for each entry of a batch.
    """
    def __init__(self,
                 sleep_study_pairs,
                 batch_size,
                 data_per_period,
                 n_classes,
                 n_channels,
                 margin=0,
                 batch_scaler=None,
                 logger=None,
                 no_log=False,
                 scale_assertion=True,
                 require_all_loaded=True,
                 identifier="",
                 **kwargs):
        """
        Args:
            sleep_study_pairs: (list)   A list of SleepStudy objects
            batch_size:        (int)    The size of each sampled batch
            data_per_period    (int)    The dimensionality/number of samples
                                        in each 'period/epoch/segment' of data
            n_classes:         (int)    Number of classes (sleep stages)
            n_channels:        (int)    The number of PSG channels to expect in
                                        data extracted from a SleepStudy object
            margin             (int)    The margin (number of periods/segments)
                                        to include down- and up-stream from
                                        a selected center segment. E.g. a
                                        margin of 3 will lead to 7 connected
                                        segments being returned in a sample.
            batch_scaler:      (string) The name of a sklearn.preprocessing
                                        Scaler object to apply to each sampled
                                        batch (optional)
            logger:            (Logger) A Logger object
            no_log:            (bool)   Do not log information on this Sequence
            identifier:        (string) A string identifier name
        """
        self._inferred = n_classes is None
        n_classes = _infer_n_classes(n_classes, sleep_study_pairs)
        super().__init__(identifier=identifier,
                         sleep_study_pairs=sleep_study_pairs,
                         n_classes=n_classes,
                         n_channels=n_channels,
                         batch_size=batch_size,
                         batch_scaler=batch_scaler,
                         logger=logger,
                         require_all_loaded=require_all_loaded)

        self._cum_periods_per_pair_minus_margins = None  # Set in margin setter
        self.margin = margin
        self.data_per_period = data_per_period

        if scale_assertion:
            self._assert_scaled()
        if not no_log:
            self.log()

    def log(self):
        """ Log basic information on this object """
        self.logger("[*] BatchSequence initialized{}:\n"
                    "    Batch shape:     {}\n"
                    "    N pairs:         {}\n"
                    "    Margin:          {}\n"
                    "    Batch scaling:   {}\n"
                    "    All loaded:      {}\n"
                    "    N classes:       {}{}".format(" ()".format(self.identifier) if self.identifier else "",
                                                       self.batch_shape,
                                                       len(self.pairs),
                                                       self.margin,
                                                       bool(self.batch_scaler),
                                                       self.all_loaded,
                                                       self.n_classes,
                                                       " (AUTO-INFERRED)"
                                                       if self._inferred else ""))

    @property
    @requires_all_loaded
    def total_periods(self):
        """ Return the som of n_periods across all SleepStudy objects """
        return self.cum_periods_per_pair[-1]

    @property
    @requires_all_loaded
    def total_periods_minus_margins(self):
        """ Return the som of n_periods across all SleepStudy objects """
        return self.cum_periods_per_pair_minus_margins[-1]

    @property
    def batch_shape(self):
        """ Returns the shape of the X output array that will be sampled """
        if self.margin == 0:
            return [self.batch_size, self.data_per_period, self.n_channels]
        else:
            return [self.batch_size, self.margin * 2 + 1,
                    self.data_per_period, self.n_channels]

    @property
    def margin(self):
        """ Returns the current margin """
        return self._margin

    @margin.setter
    def margin(self, new_margin):
        """
        Set a new margin

        Args:
            new_margin: (int) new margin value
        """
        if new_margin < 0:
            raise ValueError("Margin must be >= 0, got {}".format(new_margin))
        self._margin = new_margin
        if self.all_loaded:
            # Compute cum. number of periods per pair minus newly set margin
            minus_margin_cum_periods = np.cumsum(self.periods_per_pair -
                                                 (new_margin*2))
            self._cum_periods_per_pair_minus_margins = minus_margin_cum_periods

    @property
    @requires_all_loaded
    def cum_periods_per_pair_minus_margins(self):
        """
        Returns the cum. sum of N periods over SleepStudies minus the margins
        """
        return self._cum_periods_per_pair_minus_margins

    @requires_all_loaded
    def __len__(self):
        """ Returns the total number of batches in this dataset """
        return int(np.ceil(self.total_periods_minus_margins/self.batch_size))

    @requires_all_loaded
    def __iter__(self):
        """ Yields the entire dataset in fixed ordered batches """
        for i in range(len(self)):
            yield self.get_batch(i)

    @requires_all_loaded
    def __getitem__(self, idx):
        """
        Return a batch of data by overall dataset batch index
        See self.get_batch for docstring
        """
        if idx < 0:
            # Allow negative indexing
            idx = len(self)+idx
        # If multiprocessing, set unique seed for this particular process
        self.seed()
        return self.get_batch(idx)

    def get_pair_by_id(self, study_id):
        """
        Return a SleepStudy object by its identifier string

        Args:
            study_id: String identifier of a specific SleepStudy

        Returns:
            A stored SleepStudy object
        """
        return self.id_to_pair[study_id]

    def get_single_study_full_seq(self, study_id):
        """
        Return all periods/epochs/segments of data (X, y) of a SleepStudy.
        Differs only from 'SleepStudy.get_all_periods' in that the batch is
        processed and thus may be scaled.

        Args:
            study_id: A string identifier matching a single SleepStudy object

        Returns:
            X: ndarray of PSG data, shape [-1, data_per_period, n_channels]
            y: ndarray of labels, shape [-1, 1]
        """
        ss = self.get_pair_by_id(study_id)
        return self.process_batch(*ss.get_all_periods())

    def single_study_seq_generator(self, study_id, margin=None,
                                   overlapping=True, batch_size=None):
        """
        Yields single (batch-size 1) sequence elements (margin > 0) from a
        SleepStudy object of identifier 'study_id'. A margin may be passed,
        otherwise the set self.margin property is used. One of the two must be
        set. A batch_size may be set, otherwise uses the self.batch_size
        property.

        Args:
            study_id:     A string identifier matching a single SleepStudy obj.
            margin:       Optional value to use for margin instead of self.margin
            overlapping:  Yield overlapping batches (sliding window). Otherwise
                          return non-overlapping, connected segments.
            batch_size:   Optional value to use for batch_size instead of
                          self.batch_size

        Yields:
            X: ndarray of PSG data,
               shape [1, 2*margin+1, data_per_period, n_channels]
            y: ndarray of labels, shape [1, 2*margin+1, 1]
        """
        margin = margin or self.margin
        if not margin:
            raise ValueError("Must set the self.margin property or pass a "
                             "margin to the 'single_study_seq_generator' "
                             "function. Consider using the "
                             "'single_study_batch_generator' function.")
        from utime.sequences import batch_wrapper
        seq_length = margin * 2 + 1
        batch_size = batch_size or self.batch_size
        ss = self.get_pair_by_id(study_id)
        sequence_generator = ss.to_batch_generator(batch_size=seq_length,
                                                   overlapping=overlapping)
        for X, y in batch_wrapper(sequence_generator, batch_size=batch_size):
            yield self.process_batch(X, y)

    def single_study_batch_generator(self, study_id, batch_size=None):
        """
        Yield batches of data from a single SleepStudy object. A batch_size may
        be set, otherwise uses the self.batch_size property.

        Cannot be used with the self.margin property set.

        Args:
            study_id:   A string identifier matching a single SleepStudy obj.
            batch_size: Optional value to use for batch_size instead of
                        self.batch_size

        Yields:
            X: ndarray of PSG data,
               shape [batch_size, data_per_period, n_channels]
            y: ndarray of labels, shape [batch_size, 1]
        """
        if self.margin:
            raise ValueError("Cannot use 'single_study_batch_generator' with "
                             "self.margin set. Consider using "
                             "'single_study_seq_generator' instead.")
        batch_size = batch_size or self.batch_size
        ss = self.get_pair_by_id(study_id)
        for batch in ss.to_batch_generator(batch_size=batch_size):
            yield self.process_batch(*batch)

    def get_period(self, study_id, period_idx, allow_shift_at_border=True,
                   return_shifted_idx=False, margin=False):
        """
        Return period with index 'period_idx' of SleepStudy 'sleep_study'.
        If self.margin > 0 a sequence will be returned with the period at
        period_idx in the center. The upper or lower tails of such sequence may
        extend beyond the boarders of the total sequence. If
        allow_shift_at_border is set to True, get_period will return the
        nearest sequence of the same length that fits within the full sequence
        (the center will be shifted by the number of periods that were out-of-
        bounds with the original index).

        Args:
            study_id:              A string identifier matching a single
                                   SleepStudy obj.
            period_idx:            The period/segment/epoch index within the
                                   SleepStudy to return
            allow_shift_at_border: Allow shifting of the index if margin > 0
                                   makes the sequence extend beyond the PSG
                                   boundaries
            return_shifted_idx:    Return the index to which the center was
                                   shifted (if not shifted, equal to input
                                   'period_idx'
            margin:                Optional value to use for margin instead of
                                   self.margin

        Returns:
            X, list of length margin*2+1 shape [data_per_period, n_channels]
               ndarrays
            y, list of margin*2+1 class labels
        """
        margin = margin or self.margin
        sleep_study = self.get_pair_by_id(study_id)
        n_periods = sleep_study.n_periods
        try:
            _check_margin(n_periods, margin, at_idx=period_idx)
        except MarginError as e:
            if allow_shift_at_border:
                period_idx += e.shift
            else:
                raise MarginError("Margin error with "
                                  "'allow_shift_at_border=False'") from e
        Xs, ys = [], []
        for idx in range(period_idx-margin, period_idx+margin+1):
            X, y = sleep_study.get_period_by_idx(idx)
            Xs.append(X), ys.append(y)
        if return_shifted_idx:
            return Xs, ys, period_idx
        else:
            return Xs, ys

    def _get_periods_in_range(self, sleep_study, start, end):
        """
        Helper method for the self.get_batch method.
        Returns a list of periods with indices in [start, ..., end-1] using
        self.get_period from SleepStudy  'sleep_study'

        See self.bet_batch docstring.
        """
        X, y = [], []
        for period_idx in range(start, end):
            X_, y_ = self.get_period(sleep_study.identifier, period_idx,
                                     allow_shift_at_border=False)
            X.append(X_), y.append(y_)
        return X, y

    @requires_all_loaded
    def get_batch(self, batch_idx):
        """
        Return a batch of data index by overall batch index across the dataset.
        The order of the stored SleepStudy pairs is assumed fixed.

        Note that the final batch may be smaller than self.batch_size due to
        boundary effects.

        Args:
            batch_idx: Overall batch index across the stored SleepStudy objects

        Returns:
            X: A batch of PSG data, ndarray of shape
               [batch_size, margin*2+1, data_per_period, n_channels] if margin
               else [batch_size, data_per_period, n_channels]
            y: A batch of label values, ndarray of shape
               [batch_size, margin*2+1, 1] if margin, else
               [batch_size, 1]
        """
        # batch_idx is the batch number. We first find the total period index
        # at which we should start sampling
        global_period_start = batch_idx * self.batch_size

        # Find the SleepStudy object in which the given period occurs
        pair_idx = np.searchsorted(self.cum_periods_per_pair_minus_margins,
                                   1+global_period_start)
        sleep_study = self.pairs[pair_idx]
        if pair_idx > 0:
            previous_periods = self.cum_periods_per_pair_minus_margins[pair_idx-1]
        else:
            previous_periods = 0

        # Get the number of periods to sample in the given pair
        local_period_start = global_period_start - previous_periods + self.margin
        local_period_end = local_period_start + self.batch_size

        # Get number of periods that span into the next pair
        max_ = sleep_study.n_periods - self.margin
        periods_in_next_ss = abs(min(max_ - local_period_end, 0))
        local_period_end -= periods_in_next_ss

        # Get all periods in the current SS
        X, y = self._get_periods_in_range(sleep_study,
                                          local_period_start,
                                          local_period_end)
        if periods_in_next_ss:
            # Add potential periods from next SleepStudy
            try:
                next_sleep_study = self.pairs[pair_idx + 1]
            except IndexError:
                pass  # at the end
            else:
                # No error - that is we were able to get the next_sleep_study
                if periods_in_next_ss > next_sleep_study.n_periods:
                    raise NotImplementedError("Batch spans three SleepPairs. "
                                              "Handling this situation is not "
                                              "yet implemented.")
                x_, y_ = self._get_periods_in_range(sleep_study, self.margin,
                                                    self.margin+periods_in_next_ss)
                X.extend(x_)
                y.extend(y_)
        return self.process_batch(X, y)
