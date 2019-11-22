"""
Implements the SleepStudy class which represents a sleep study (PSG)
"""

import os
import numpy as np
from contextlib import contextmanager
from utime import errors
from utime.io.high_level_file_loaders import load_psg, load_hypnogram
from utime.preprocessing import (apply_scaling, strip_funcs, apply_strip_func,
                                 assert_scaler, set_psg_sample_rate,
                                 quality_control_funcs, assert_equal_length,
                                 apply_quality_control_func)
from utime.hypnogram.utils import create_class_int_to_period_idx_dict
from utime.dataset.utils import find_psg_and_hyp
from utime import defaults
from MultiPlanarUNet.logging import ScreenLogger


def assert_header_fields(header):
    """ Check for minimally required fields of a header dict """
    check = (('sample_rate', False), ('channel_names', False), ('date', True))
    for c, replace_with_none in check:
        if c not in header:
            if replace_with_none:
                header[c] = None
            else:
                raise ValueError("Invalid header file loaded, did not find "
                                 "attribute {} in header {}".format(c, header))


class SleepStudy(object):
    """
    Represents a PSG sleep study and (optionally) a manually scored hypnogram
    """
    def __init__(self,
                 subject_dir,
                 psg_regex=None,
                 hyp_regex=None,
                 period_length_sec=None,
                 no_hypnogram=None,
                 annotation_dict=None,
                 load=False,
                 logger=None):
        """
        Initialize a SleepStudy object from PSG/HYP data

        PSG: A file that stores polysomnography (PSG) data
        HYP: A file that stores the sleep stages / annotations for the PSG

        Takes a path pointing to a directory in which two or more files are
        located. One of those files should be a PSG (data) file and unless
        no_hypnogram == True another should be a hypnogram/sleep stages/labels
        file. The PSG(/HYP) files are automatically automatically inferred
        using a set of simple rules when psg_regex or hyp_regex are None
        (refer to the 'utime.dataset.utils.find_psg_and_hyp' function).
        Otherwise, the psg_regex and/or hyp_regex is used to match against
        folder content. Each regex should have exactly one match within
        'subject_dir'.

        Args:
            subject_dir:      (str)    File path to a directory storing the
                                       subject data.
            psg_regex:        (str)    Optional regex used to select PSG file
            hyp_regex:        (str)    Optional regex used to select HYP file
            period_length_sec (int)    Sleep 'epoch' (segment/period) length in
                                       seconds
            no_hypnogram      (bool)   Initialize without ground truth data.
            annotation_dict   (dict)   A dictionary mapping from labels in the
                                       hyp_file_path file to integers
            load              (bool)   Load the PSG object at init time.
            logger            (Logger) A Logger object
        """
        self.logger = logger or ScreenLogger()

        self.subject_dir = os.path.abspath(subject_dir)
        try:
            psg, hyp = find_psg_and_hyp(subject_dir=self.subject_dir,
                                        psg_regex=psg_regex,
                                        hyp_regex=hyp_regex,
                                        no_hypnogram=no_hypnogram)
        except (ValueError, RuntimeError) as e:
            raise ValueError("Could not uniquely infer PSG/HYP files in subject"
                             " directory {}. Consider specifying/correcting "
                             "one or both of the psg_regex and hyp_regex "
                             "parameters to explicitly select the appropriate "
                             "file(s) within the "
                             "subject dir.".format(repr(subject_dir))) from e
        self.psg_file_path = psg
        self.hyp_file_path = hyp if not no_hypnogram else None

        self.no_hypnogram = no_hypnogram
        self.period_length_sec = period_length_sec or \
            defaults.get_default_period_length(self.logger)
        self.annotation_dict = annotation_dict

        # Hidden attributes controlled in property functions to limit setting
        # of these values to the load() function
        self._psg = None
        self._header = None
        self._identifier = None
        self._hypnogram = None
        self._last_period_start_second = None
        self._scaler = None
        self._scaler_obj = None
        self._select_channels = None
        self._alternative_select_channels = None
        self._channel_sampling_groups = None
        self._strip_func = None
        self._quality_control_func = None
        self._class_to_period_dict = None
        self._sample_rate = None
        self._org_sample_rate = None
        self._date = None

        if load:
            self.load()

    def __str__(self):
        if self.loaded:
            t = (self.identifier, len(self.select_channels), self.date,
                 self.sample_rate, self.hypnogram is not False)
            return "SleepStudy(loaded=True, identifier={:s}, N channels: {}, " \
                   "date: {}, sample_rate={:.1f}, hypnogram={})".format(*t)
        else:
            return repr(self)

    def __repr__(self):
        return "SleepStudy(loaded={}, identifier={})".format(self.loaded,
                                                             self.identifier)

    @property
    def psg(self):
        """ Returns the PSG object (an ndarray of shape [-1, n_channels]) """
        return self._psg

    @property
    def date(self):
        """ Returns the recording date, may be None """
        return self._date

    @property
    def hypnogram(self):
        """ Returns the hypnogram (see utime.hypnogram), may be None """
        return self._hypnogram

    @property
    def data_per_period(self):
        """
        Computes and returns the data (samples) per period of
        'period_length_sec' seconds of time (en 'epoch' in sleep research)
        """
        return self.period_length_sec * self.sample_rate

    @property
    def n_classes(self):
        """ Returns the number of classes represented in the hypnogram """
        return self.hypnogram.n_classes

    @property
    def n_channels(self):
        """ Returns the number of channels in the PSG array """
        return len(self.select_channels)

    @property
    def n_sample_channels(self):
        """
        Returns the number of channels that will be returned by
        self.extract_from_psg (this may be different from self.n_channels if
        self.channel_sampling_groups is set).
        """
        if self.channel_sampling_groups:
            return len(self.channel_sampling_groups())
        else:
            return self.n_channels

    @property
    def recording_length_sec(self):
        """ Returns the total length (in seconds) of the PSG recording """
        return self.psg.shape[0] / self.sample_rate

    @property
    def last_period_start_second(self):
        """ Returns the second that marks the beginning of the last period """
        return int(self.recording_length_sec - self.period_length_sec)

    @property
    def n_periods(self):
        """ Returns the total number of periods (segments/epochs) """
        return int(self.recording_length_sec / self.period_length_sec)

    @property
    def class_to_period_dict(self):
        """
        Returns the class_to_period_dict, which maps a class integer
        (such as 0) to an array of period indices that correspond to PSG
        periods/epochs/segments that in the ground truth is scored as the that
        class (such as 0).
        """
        return self._class_to_period_dict

    @property
    def identifier(self):
        """
        Returns an ID, which is simply the name of the directory storing
        the data
        """
        return os.path.split(self.subject_dir)[-1]

    @property
    def select_channels(self):
        """ See setter method. """
        return self._select_channels or []

    @select_channels.setter
    def select_channels(self, channels):
        """
        Sets select_channels; a property that when set marks a list of
        channel names to select from the PSG file on disk. All other channels
        are not loaded or removed after loading.

        OBS setting this propery when self.loaded is True forces a reload

        Args:
            channels: A list of channel names (strings) giving the names of
                      all channels to load when calling self.load().
        """
        if channels is not None:
            if not isinstance(channels, (list, tuple)):
                raise TypeError("'channels' must be a list or tuple, got "
                                "{}.".format(type(channels)))
            if not all([isinstance(c, str) for c in channels]):
                raise TypeError("Some values in 'select_channels' are not "
                                "strings, got {}. Expected a flat list of "
                                "strings.".format(channels))
        channels = channels or []
        self._select_channels = channels
        if self.loaded:
            self.reload(warning=True)

    @property
    def alternative_select_channels(self):
        """ See setter method """
        return self._alternative_select_channels or [[]]

    @alternative_select_channels.setter
    def alternative_select_channels(self, channels):
        """
        Set the alternative_select_channels; a property that when set defines
        a list of lists each similar to self.select_channels (see docstring).
        Each define an alternative set of channel names to be loaded in case of
        ChannelNotFound errors in self.load().

        OBS setting this propery when self.loaded is True forces a reload

        Args:
            channels: A list of lists of strings
        """
        e = "'channels' must be a list of lists, where the sub-lists are the " \
            "same lengths as the 'select_channels' list. Got {}."
        if not self.select_channels:
            raise ValueError("Must select primary channels before "
                             "alternative.")
        if channels is not None:
            if not isinstance(channels, (list, tuple)):
                raise TypeError(e.format(channels))
            if len(channels) == 1:
                from utime.utils import ensure_list_or_tuple
                channels = [ensure_list_or_tuple(channels[0])]
            for chan in channels:
                if not isinstance(chan, (list, tuple)):
                    raise TypeError(e.format(type(channels)))
                if len(chan) != len(self.select_channels):
                    raise ValueError(e.format(channels))
                if not all([isinstance(c, str) for c in chan]):
                    raise TypeError("Some values in one of the sub-list of "
                                    "alternative_select_channels are not "
                                    "strings, got {}. Expected a list of lists"
                                    " of strings.".format(channels))
        channels = channels or [[]]
        self._alternative_select_channels = channels
        if self.loaded:
            self.reload(warning=True)

    @property
    def channel_sampling_groups(self):
        """ Returns the channel_sampling_groups, see setter method """
        return self._channel_sampling_groups

    @channel_sampling_groups.setter
    def channel_sampling_groups(self, sampling_groups):
        """
        Sets the channel_sampling_groups; a property that when set groups
        channels together into 'sampling groups' that are considered a single
        entry and from which a single sub-channel will be randomly (uniformly)
        sampled at each call to self.extract_from_psg_with_channel_groups.

        For instance, a set of channels ['EEG1', 'EEG2', 'EMG'] with
        channel_sampling_groups [0, 0, 1] will cause
        self.extract_from_psg_with_channel_groups to return arrays of shape
        [-1, 2] where the first channel was randomly selected from {EEG1, EEG2}
        Please refer to self.extract_from_psg_with_channel_groups.

        Sets the attribute self._channel_sampling_groups to a function that
        when called randomly samples a set of channel indices according the the
        specified grouping.

        Args:
            sampling_groups: An array of length len(self.select_channels) of
                             integers
        """
        if self.select_channels is None:
            raise RuntimeError("Must set 'select_channels' before "
                               "'channel_sampling_groups'")
        if not isinstance(sampling_groups, (tuple, list)) or \
                len(sampling_groups) != len(self.select_channels):
            raise ValueError("'sampling_groups' argument must be a list of "
                             "length equal to 'select_channels', got {} "
                             "(type {})".format(sampling_groups,
                                                type(sampling_groups)))
        from collections import defaultdict
        groups = defaultdict(list)
        for chan, group in enumerate(sampling_groups):
            groups[group].append(chan)

        def sample_func():
            """
            Samples one channel from each group of channels in the
            sample dict, returns a list of channel indicies to sample
            """
            to_sample = []
            for chans in groups.values():
                to_sample.append(np.random.choice(chans))
            return to_sample
        self._channel_sampling_groups = sample_func

    @property
    def scaler(self):
        """ Returns the scaler type (string), see setter method """
        return self._scaler

    @scaler.setter
    def scaler(self, scaler):
        """
        Sets a scaler type.
        Is considered at load-time
        Setting with self.loaded == True forces a reload.

        Args:
            scaler: String, naming a sklearn.preprocessing scaler.
        """
        if not assert_scaler(scaler):
            raise ValueError("Invalid scaler, does not exist {}".format(scaler))
        self._scaler = scaler
        if self.loaded:
            self.reload(warning=True)

    @property
    def scaler_obj(self):
        """ Reference to the scaler object """
        return self._scaler_obj

    @property
    def org_sample_rate(self):
        """
        Returns the original sample rate
        """
        return self._org_sample_rate

    @property
    def sample_rate(self):
        """
        Returns the set/resampled sample rate, not the original
        See setter method
        """
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, sample_rate):
        """
        Set a new sample rate
        Is considered at load-time
        Setting with self.loaded == True forces a reload.
        """
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError("Sample rate must be a positive integer, "
                             "got {}".format(sample_rate))
        self._sample_rate = sample_rate
        if self.loaded:
            self.reload(warning=True)

    @property
    def strip_func(self):
        """
        See setter method
        strip_func - when set - is a 2-tuple (strip_func_name, kwargs)
        """
        return self._strip_func

    def set_strip_func(self, strip_func_str, **kwargs):
        """
        Sets a strip function. Strip functions are applied to the PSG/HYP pair
        at load time and may deal with minor differences between the length
        of the PSG and HYP (e.g. if the PSG is longer than the HYP file).
        See utime.preprocessing.strip_funcs

        Forces a reload if self.loaded is True

        Args:
            strip_func_str: A string naming a strip_func in:
                            utime.preprocessing.strip_funcs
            kwargs:         Other kw arguments that will be passed to the strip
                            function.
        """
        if strip_func_str not in strip_funcs.__dict__:
            self.raise_err(ValueError, "Invalid strip function "
                                       "{}".format(strip_func_str))
        self._strip_func = (strip_func_str, kwargs)
        if self.loaded:
            self.reload(warning=True)

    @property
    def quality_control_func(self):
        """ See setter method """
        return self._quality_control_func

    def set_quality_control_func(self, qc_func, **kwargs):
        """
        Sets a quality control function which is applied to all segments
        of a PSG (as determined by self.period_length_sec) and may alter the
        values of said segments.

        Applies at load-time, forces a reload if self.loaded == True.

        Args:
            qc_func:  A string naming a quality control function in:
                      utime.preprocessing.quality_control_funcs
            **kwargs: Parameters passed to the quality control func at load
        """
        if qc_func not in quality_control_funcs.__dict__:
            self.raise_err(ValueError, "Invalid quality control function "
                                       "{}".format(qc_func))
        self._quality_control_func = (qc_func, kwargs)
        if self.loaded:
            self.reload(warning=True)

    def to_batch_generator(self, batch_size, overlapping=False):
        """
        Yields batches of data from the SleepStudy PSG/HYP pair
        Note: With overlapping == False the last batch may be smaller than
        batch_size due to boundary effects.

        Args:
            batch_size:  An integer, number of periods/epochs/segments to return
                         in each batch.
            overlapping: Yield overlapping batches (sliding window). Otherwise
                         return non-overlapping, connected segments.

        Yields:
            X: ndarray of shape [batch_size, self.data_per_period, self.n_channels]
            y: ndarray of shape [batch_size, 1]
        """
        X, y = [], []
        for idx in range(self.n_periods):
            X_, y_ = self.get_period_by_idx(idx)
            X.append(X_), y.append(y_)
            if len(X) == batch_size:
                yield np.array(X), np.array(y)  # Must copy if overlapping=True
                if overlapping:
                    X.pop(0), y.pop(0)
                else:
                    X, y = [], []
        if len(X) != 0 and not overlapping:
            yield np.array(X), np.array(y)

    @property
    def loaded(self):
        """ Returns whether the SleepStudy data is currently loaded or not """
        return not any((self.psg is None,
                        self.hypnogram is None))

    def _load_with_any_in(self, channel_sets):
        """
        Normally not called directly, usually called from self._load.

        Circulates a list of lists of proposed channel names to load,
        attempting to load using any of the sets (in specified order), raising
        ChannelNorFound error if none of the sets could be loaded.

        Args:
            channel_sets: List of lists of strings, each naming a channel to
                          load.

        Returns:
            If one of the sets of channels could be loaded, returns the
            PSG array of shape [-1, n_channels], the header and the list of
            channels that were successfully loaded (an elem. of channel_sets).
        """
        for i, channel_set in enumerate(channel_sets):
            try:
                psg, header = load_psg(psg_file_path=self.psg_file_path,
                                       load_channels=channel_set or None)
                return psg, header, channel_set
            except errors.ChannelNotFoundError as e:
                if i < len(channel_sets) - 1:
                    # Try nex set of channels
                    continue
                else:
                    s, sa = self.select_channels, \
                            self.alternative_select_channels
                    err = errors.ChannelNotFoundError("Could not load "
                                                      "select_channels {} or "
                                                      "alternative_select_"
                                                      "channels "
                                                      "{}".format(s, sa))
                    raise err from e

    def _load(self):
        """
        Loads data from the PSG and HYP files
        -- If self.select_channels is set (aka non empty list), only the column
           names matching this list will be kept.
        -- PSG data is kept as a numpy array. Use self.select_channels to map
           between names and the numpy array
        -- If self.scaler is set, the PSG array will be scaled according to
           the specified sklearn.preprocessing scaler
        -- If self.hyp_strip_func is set, this function will be applied to the
           hypnogram object.
        """
        if len(self.alternative_select_channels[0]) != 0:
            try_channels = [self.select_channels] + self.alternative_select_channels
        else:
            try_channels = [self.select_channels]
        self._psg, header, loaded_chanls = self._load_with_any_in(try_channels)
        self._select_channels = loaded_chanls     # OBS must set private
        self._alternative_select_channels = None  # OBS must set private

        # Ensure all header information is available
        assert_header_fields(header)
        self._date = header["date"]
        self._org_sample_rate = header["sample_rate"]
        self._sample_rate = self._sample_rate or self._org_sample_rate

        if not self.select_channels:
            # OBS: Important to access private variable here as otherwise
            # the method may reload in a loop (depending on self.loaded state)
            self._select_channels = header["channel_names"]

        if self.hyp_file_path is not None and not self.no_hypnogram:
            self._hypnogram, \
            self.annotation_dict = load_hypnogram(self.hyp_file_path,
                                                  period_length_sec=self.period_length_sec,
                                                  annotation_dict=self.annotation_dict,
                                                  sample_rate=header["sample_rate"])
        else:
            self._hypnogram = False

        if self.strip_func:
            # Strip the data using the passed function on the passed class
            self._psg, self._hypnogram = apply_strip_func(self,
                                                          self.org_sample_rate)
        elif not assert_equal_length(self.psg,
                                     self.hypnogram, self.org_sample_rate):
            self.raise_err(RuntimeError, "PSG and hypnogram are not equally "
                                         "long in seconds. Consider setting a "
                                         "strip_function. "
                                         "See utime.preprocessing.strip_funcs.")

        if self.quality_control_func:
            # Run over epochs and assess if epoch-specific changes should be
            # made to limit the influence of very high noise level ecochs etc.
            self._psg = apply_quality_control_func(self, self.org_sample_rate)

        # Set different sample rate of PSG?
        if self.org_sample_rate != self.sample_rate:
            self._psg = set_psg_sample_rate(self._psg,
                                            new_sample_rate=self.sample_rate,
                                            old_sample_rate=self.org_sample_rate)
        if self.scaler:
            self._psg, self._scaler_obj = apply_scaling(self.psg, self.scaler)

        # Store dictionary mapping class integers to period idx of that class
        if self.hypnogram:
            self._class_to_period_dict = create_class_int_to_period_idx_dict(
                self.hypnogram
            )

    def load(self):
        """
        High-level function invoked to load the SleepStudy data
        """
        if not self.loaded:
            try:
                self._load()
            except Exception as e:
                raise errors.CouldNotLoadError("Unexpected load error for sleep "
                                               "study {}. Please refer to the "
                                               "above traceback.".format(self.identifier),
                                               study_id=self.identifier) from e
        return self

    def unload(self):
        """ Unloads the PSG, header and hypnogram data """
        self._psg = None
        self._header = None
        self._hypnogram = None

    def reload(self, warning=True):
        """ Unloads and loads """
        if warning and self.loaded:
            print("Reloading SleepStudy '{}'".format(self.identifier))
        self.unload()
        self.load()

    @contextmanager
    def loaded_in_context(self):
        """ Context manager from automatic loading and unloading """
        self.load()
        try:
            yield self
        finally:
            self.unload()

    def get_class_counts(self, as_dict=False):
        """
        Computes the class counts for the loaded hypnogram.

        Args:
            as_dict: (bool) return a dictionary mapping from class labels
                            (ints) to the count (int) for that class instead of
                            the typical array of class counts.

        Returns:
            An ndarray of length n_classes of counts if as_dict == False
            Otherwise a dictionary mapping class labels to counts.
        """
        classes = sorted(self._class_to_period_dict.keys())
        counts = np.array([len(self._class_to_period_dict[c]) for c in classes])
        if as_dict:
            return {cls: count for cls, count in zip(classes, counts)}
        else:
            return counts

    def raise_err(self, err_obj, err_msg, _from=None):
        """
        Helper method for raising an error specific to this SleepStudy object
        """
        e = err_obj("[{}] {}".format(repr(self), err_msg))
        if _from:
            raise e from _from
        else:
            raise e

    def get_full_hypnogram(self):
        """
        Returns the full (dense) hypnogram

        Returns:
            An ndarray of shape [self.n_periods, 1] of class labels
        """
        return self.hypnogram.to_dense()["sleep_stage"].to_numpy().reshape(-1, 1)

    def period_idx_to_sec(self, period_idx):
        """
        Helper method that maps a period_idx (int) to the first second in that
        period.
        """
        return period_idx * self.period_length_sec

    def get_all_periods(self):
        """
        Returns the full (dense) data of the SleepStudy

        Returns:
            X: An ndarray of shape [self.n_periods,
                                    self.data_per_period,
                                    self.n_channels]
            y: An ndarray of shape [self.n_periods, 1]
        """
        X = self.psg.reshape(-1, self.data_per_period, self.n_channels)
        y = self.get_full_hypnogram()
        if len(X) != len(y):
            err_msg = ("Length of PSG array does not match length dense "
                       "hypnogram array ({} != {}). If hypnogram "
                       "is longer, consider if a trailing or leading "
                       "sleep stage should be removed. (you may use "
                       "SleepStudyDataset.set_hyp_strip_func())".format(len(X),
                                                                        len(y)))
            self.raise_err(ValueError, err_msg)
        return X, y

    def get_period_at_sec(self, second):
        """
        Get a period of {X, y} data starting at 'second' seconds.

        Returns:
            X: An ndarray of shape [self.data_per_period, self.n_channels]
            y: An ndarray of shape [1]
        """
        X = self.get_psg_period_at_sec(second)
        y = self.hypnogram.get_stage_at_sec(second)
        return X, y

    def get_period_by_idx(self, period_idx):
        """
        Get a period of {X, y} data by index
        Period starting at second 0 is index 0.

        Returns:
            X: An ndarray of shape [self.data_per_period, self.n_channels]
            y: An ndarray of shape [1]
        """
        period_start_sec = self.period_idx_to_sec(period_idx)
        return self.get_period_at_sec(period_start_sec)

    def get_stage_by_idx(self, period_idx):
        """
        Get the hypnogram stage at period index 'period_idx'.

        Returns:
            y: An ndarray of shape [1]
        """
        period_start_sec = period_idx * self.period_length_sec
        return self.hypnogram.get_stage_at_sec(period_start_sec)

    def get_psg_period_at_sec(self, second):
        """
        Get PSG period starting at second 'second'.

        Returns:
            X: An ndarray of shape [self.data_per_period, self.n_channels]
        """
        if second % self.period_length_sec:
            raise ValueError("Invalid second of {}, not divisible by period "
                             "length of {} "
                             "seconds".format(second, self.period_length_sec))
        return self.extract_from_psg(start=second,
                                     end=second+self.period_length_sec)

    def extract_from_psg_with_channel_groups(self, start, end):
        """
        Extracts data from the PSG according to self.channel_sampling_groups
        Please refer to the self.channel_sampling_groups setter method.
        Please refer to self.extract_from_psg
        """
        if callable(self.channel_sampling_groups):
            channel_inds = tuple(self.channel_sampling_groups())
        else:
            raise RuntimeError("Unexpected error - 'channel_sampling_groups' "
                               "is set but is not a callable. Should be a "
                               "function that returns a list of channel "
                               "indices to sample.")
        return self.extract_from_psg(start, end)[:, channel_inds]

    def extract_from_psg(self, start, end):
        """
        Extract PSG data from second 'start' (inclusive) to second 'end'
        (exclusive)

        Args:
            start: int, start second to extract from
            end: int, end second to extract from

        Returns:
            A Pandas DataFrame view or numpy view
        """
        if start > self.last_period_start_second:
            raise ValueError("Cannot extract a full period starting from second"
                             " {}. Last full period of {} seconds starts at "
                             "second {}.".format(start, self.period_length_sec,
                                                 self.last_period_start_second))
        sr = self.sample_rate
        first_row = int(start * sr)
        last_row = int(end * sr)
        return self.psg[first_row:last_row]

    def plot_period(self, period_idx=None, period_sec=None, out_path=None):
        """
        Plot a period of data by index or second

        Args:
            period_idx: Period index to plot
            period_sec: The starting second of the period to plot
            out_path:   Path to save the figure to
        """
        if bool(period_idx) == bool(period_sec):
            raise ValueError("Must specify either period_idx or period_sec.")
        from utime.visualization.psg_plotting import plot_period
        period_sec = period_sec or self.period_idx_to_sec(period_idx)
        X, y = self.get_period_at_sec(period_sec)
        plot_period(X=X, y=defaults.class_int_to_stage_string[y],
                    channel_names=self.select_channels,
                    init_second=period_sec,
                    sample_rate=self.sample_rate,
                    out_path=out_path)

    def plot_periods(self, period_idxs=None, period_secs=None, out_path=None,
                     highlight_periods=True):
        """
        Plot multiple periods of data by indices or seconds

        Args:
            period_idxs:        Indices for all periods to plot
            period_secs:        The starting seconds of the periods to plot
            out_path:           Path to save the figure to
            highlight_periods:  Plot period-separating vertical lines
        """
        if bool(period_idxs) == bool(period_secs):
            raise ValueError("Must specify either period_idxs or period_secs.")
        from utime.visualization.psg_plotting import plot_periods
        period_secs = list(period_secs or map(self.period_idx_to_sec, period_idxs))
        if any(np.diff(period_secs) != self.period_length_sec):
            raise ValueError("Periods to plot must be consecutive.")
        X, y = zip(*map(self.get_period_at_sec, period_secs))
        plot_periods(X=X,
                     y=[defaults.class_int_to_stage_string[y_] for y_ in y],
                     channel_names=self.select_channels,
                     init_second=period_secs[0],
                     sample_rate=self.sample_rate,
                     out_path=out_path,
                     highlight_periods=highlight_periods)
