"""
Implements the SleepStudy class which represents a sleep study (PSG)
"""

import numpy as np
from utime import errors, Defaults
from utime.dataset.sleep_study.abc_sleep_study import AbstractBaseSleepStudy
from utime.io.channels import ChannelMontageTuple
from utime.io.high_level_file_loaders import get_org_include_exclude_channel_montages


class H5SleepStudy(AbstractBaseSleepStudy):
    """
    Represents a PSG sleep study and (optionally) a manually scored hypnogram
    """
    def __init__(self, h5_study_object, annotation_dict=None,
                 period_length_sec=None, no_hypnogram=False, logger=None):
        """
        TODO
        """
        self.h5_study_object = h5_study_object
        super(H5SleepStudy, self).__init__(
            annotation_dict=annotation_dict,
            period_length_sec=period_length_sec,
            no_hypnogram=no_hypnogram,
            logger=logger
        )
        if self.annotation_dict:
            self.annotation_dict = np.vectorize(annotation_dict.get)
        self._access_time_random_channel_selector = None
        self.load()  # Sets data visibility

    @property
    def identifier(self):
        """
        Returns an ID, which is simply the name of the directory storing
        the data
        """
        return self.h5_study_object.name.split("/")[-1]

    def __str__(self):
        if self.loaded:
            t = (self.loaded, self.open, self.identifier,
                 len(self.select_channels),self.sample_rate,
                 self.hypnogram is not False)
            return "H5SleepStudy(open={}, loaded={}, identifier={:s}, " \
                   "N channels: {}, sample_rate={:.1f}, " \
                   "hypnogram={})".format(*t)
        else:
            return repr(self)

    def __repr__(self):
        return 'H5SleepStudy(open={}, loaded={}, identifier={})' \
               ''.format(self.open, self.loaded, self.identifier)

    @property
    def loaded(self):
        """
        Returns whether the PSG and hypnogram properties are set or not.
        Only affects 'visibility' to the data, no data is actually loaded.
        """
        return not any((self.psg is None,
                        self.hypnogram is None))

    @property
    def open(self):
        """
        Returns whether the HDF5 file is currently open or not
        """
        return bool(self.h5_study_object)

    def _load_with_any_in(self, channel_sets, channels_in_file):
        """
        TODO

        Args:
            channel_sets:
            channels_in_file:

        Returns:

        """
        for i, channel_set in enumerate(channel_sets):
            try:
                # Work out which channels to include and exclude during loading
                org_channels, include_channels, _, _ = \
                    get_org_include_exclude_channel_montages(
                        load_channels=channel_set,
                        header={'channel_names': channels_in_file}
                    )
                return include_channels
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

    def load(self, reload=False):
        """
        Sets the PSG and hypnogram visibility according to self._try_channels.
        """
        # Get channels
        channels = ChannelMontageTuple(list(self.h5_study_object['PSG'].keys()))
        loaded_channels = self._load_with_any_in(self._try_channels,
                                                 channels_in_file=channels)
        self._psg = {
            chan: self.h5_study_object['PSG'][chan.original_name] for chan in loaded_channels
        }
        self._set_loaded_channels(loaded_channels)
        self._hypnogram = self.h5_study_object['hypnogram']

    def unload(self):
        """ Sets the PSG and hypnogram properties to None """
        self._psg = None
        self._hypnogram = None

    def reload(self, warning=True):
        """ Only sets the current channel visibility """
        self.unload()
        self.load()

    def get_psg_shape(self):
        """
        TODO

        Returns:

        """
        return list(self.psg[self.select_channels[0]].shape) + [len(self.psg)]

    def get_full_psg(self):
        """
        TODO

        Returns:

        """
        raise NotImplementedError
        psg = np.empty(shape=self.get_psg_shape(), dtype=Defaults.PSG_DTYPE)
        for i, c in enumerate(self.select_channels):
            psg[i] = self.psg[c]
        return psg

    def get_full_hypnogram(self):
        """
        TODO

        Returns:

        """
        return self.translate_labels(np.array(self.hypnogram))

    def get_all_periods(self):
        """
        TODO

        Returns:

        """
        return self.get_periods_by_idx(0, self.n_periods-1)

    def _second_to_idx(self, second):
        """
        TODO

        Args:
            second:

        Returns:

        """
        if second % self.period_length_sec:
            raise ValueError("Invalid second of {}, not divisible by period "
                             "length of {} "
                             "seconds".format(second, self.period_length_sec))
        rec_len = self.recording_length_sec
        if second >= rec_len:
            raise ValueError("Second {} outside range of sleep study {} "
                             "of length {} seconds".format(second,
                                                           self.identifier,
                                                           rec_len))
        return second // self.period_length_sec

    def get_class_indicies(self, class_int):
        """
        TODO

        Args:
            class_int:

        Returns:

        """
        return self.h5_study_object['class_to_index'][str(class_int)]

    def translate_labels(self, y):
        """
        TODO

        Returns:

        """
        if self.annotation_dict:
            return self.annotation_dict(y)
        else:
            return y

    def _get_sample_channels(self):
        """
        TODO

        Returns:

        """
        if not self.access_time_random_channel_selector:
            return self.select_channels
        else:
            return self.access_time_random_channel_selector.sample(
                available_channels=self.select_channels
            )

    def get_periods_by_idx(self, start_idx, end_idx):
        """
        Returns [N periods = end_idx - start_idx + 1] periods of data

        Args:
            start_idx: int, start index to extract
            end_idx: int, end index to extract

        Returns:
            X: ndarray of shape [N periods, data_per_period, num_channels]
            y: ndarray of shape [N, 1]
        """
        n_periods = end_idx-start_idx+1
        channels = self._get_sample_channels()
        x = np.empty(shape=[n_periods, self.data_per_period, len(channels)],
                     dtype=Defaults.PSG_DTYPE)
        for i, chan in enumerate(channels):
            x[..., i] = self.psg[chan][start_idx:end_idx+1]
        y = self.hypnogram[start_idx:end_idx+1]
        return x, self.translate_labels(y).reshape(-1, 1)

    def get_psg_period_at_sec(self, second):
        """
        TODO

        Args:
            second:

        Returns:

        """
        if self.access_time_random_channel_selector:
            raise RuntimeError("Calls to 'get_psg_period_at_sec' not permitted"
                               " when the 'access_time_random_channel_selector'"
                               " attribute is set - use 'get_periods_by_idx' "
                               "instead.")
        # Get idx
        idx = self._second_to_idx(second)
        channels = self.select_channels
        x = np.empty(shape=[self.data_per_period, len(channels)],
                     dtype=self.psg[channels[0]].dtype)
        for i, chan in enumerate(channels):
            x[:, i] = self.psg[chan][idx]
        return x

    def get_stage_at_sec(self, second):
        """
        TODO

        Args:
            second:

        Returns:

        """
        idx = self._second_to_idx(second)
        return self.translate_labels(self.hypnogram[idx])

    @property
    def sample_rate(self):
        """
        TODO

        Returns:

        """
        return self.h5_study_object.attrs.get('sample_rate')

    @property
    def date(self):
        """
        TODO

        Returns:

        """
        return self.h5_study_object.attrs.get('date')

    @property
    def n_classes(self):
        """
        TODO

        Returns:

        """
        return len(np.unique(self.get_full_hypnogram()))

    @property
    def recording_length_sec(self):
        """
        TODO

        Returns:

        """
        s1, s2, _ = self.get_psg_shape()
        return (s1*s2) / self.sample_rate

    @property
    def n_sample_channels(self):
        if self.access_time_random_channel_selector:
            return self.access_time_random_channel_selector.n_output_channels
        else:
            return self.n_channels

    @property
    def access_time_random_channel_selector(self):
        """
        TODO

        Returns:

        """
        return self._access_time_random_channel_selector

    @access_time_random_channel_selector.setter
    def access_time_random_channel_selector(self, channel_selector):
        """
        TODO

        Args:
            channel_selector:

        Returns:

        """
        from utime.io.channels import RandomChannelSelector
        if channel_selector is not None and not \
                isinstance(channel_selector, RandomChannelSelector):
            raise TypeError("Expected 'channel_selector' argument to be of "
                            "type {}, got {}".format(type(RandomChannelSelector),
                                                     type(channel_selector)))
        self._access_time_random_channel_selector = channel_selector

    def extract_from_psg(self, start, end, channel_inds=None):
        """
        Extract PSG data from second 'start' (inclusive) to second 'end'
        (exclusive)

        Args:
            start: int, start second to extract from
            end: int, end second to extract from
            channel_inds: list, list of channel indices to extract from

        Returns:
            X: ndarray of shape [N periods, data_per_period, C]
        """
        start_idx, end_idx = self._second_to_idx(start), self._second_to_idx(end)
        return self.get_periods_by_idx(start_idx, end_idx)[0]  # TODO, reads labels for no reason

    def to_batch_generator(self, batch_size, overlapping=False):
        """
        Yields batches of data from the SleepStudy PSG/HYP pair
        Note: With overlapping == False the last batch may be smaller than
        batch_size due to boundary effects.

        Args:
            batch_size:  An integer, number of periods/epochs/segments to
                         return in each batch.
            overlapping: Yield overlapping batches (sliding window). Otherwise
                         return non-overlapping, connected segments.

        Yields:
            X: ndarray of shape [batch_size, self.data_per_period,
                                 self.n_channels]
            y: ndarray of shape [batch_size, 1]
        """
        if overlapping:
            raise NotImplementedError("H5SleepStudy objects do not support "
                                      "to_batch_generator with "
                                      "overlapping=True yet.")
        end_point = self.n_periods-(self.n_periods % batch_size)
        for idx in range(0, end_point, batch_size):
            yield self.get_periods_by_idx(
                start_idx=idx,
                end_idx=idx+batch_size-1
            )
