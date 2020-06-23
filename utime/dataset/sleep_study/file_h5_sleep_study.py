"""
Implements the SleepStudyBase class which represents a sleep study (PSG)
"""

import numpy as np
from datetime import datetime
from utime import errors
from utime.io.high_level_file_loaders import load_hypnogram, open_h5_archive
from utime.dataset.sleep_study.subject_dir_sleep_study_base import SubjectDirSleepStudyBase


class FileH5SleepStudy(SubjectDirSleepStudyBase):
    """
    TODO
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
        TODO

        Initialize a SleepStudyBase object from PSG/HYP data

        PSG must be a .h5 file.
        Data is loaded from the h5 archive lazily on-request.

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
        super(FileH5SleepStudy, self).__init__(
            subject_dir=subject_dir,
            psg_regex=psg_regex,
            hyp_regex=hyp_regex,
            period_length_sec=period_length_sec,
            no_hypnogram=no_hypnogram,
            annotation_dict=annotation_dict,
            logger=logger
        )
        self._psg_obj = None
        if load:
            self.load()

    def __str__(self):
        if self.loaded:
            t = (self.identifier, len(self.select_channels), self.date,
                 self.sample_rate, self.hypnogram is not False)
            return "FileH5SleepStudy(loaded=True, identifier={:s}, " \
                   "N channels: {}, date: {}, sample_rate={:.1f}, " \
                   "hypnogram={})".format(*t)
        else:
            return repr(self)

    def __repr__(self):
        return "FileH5SleepStudy(loaded={}, " \
               "identifier={})".format(self.loaded, self.identifier)

    @property
    def psg_obj(self):
        """ TODO """
        return self._psg_obj

    @property
    def sample_rate(self):
        """ Returns the currently set sample rate """
        return self._psg_obj.attrs['sample_rate']

    @property
    def date(self):
        """ Returns the recording date, may be None """
        d = self._psg_obj.attrs.get("date")
        if not isinstance(d, str) and (isinstance(d, int) or
                                       np.issubdtype(d, np.integer)):
            d = datetime.fromtimestamp(d)
        return d

    @property
    def loaded(self):
        """ Returns whether the SleepStudyBase data is currently loaded or not """
        return not any((self.psg is None,
                        self.hypnogram is None))

    def _load_with_any_in(self, channel_sets):
        """
        TODO

        Args:
            channel_sets:

        Returns:

        """
        for i, channel_set in enumerate(channel_sets):
            try:
                h5_obj, psg, loaded_chnls = open_h5_archive(h5_file_path=self.psg_file_path,
                                                            load_channels=channel_set or None)
                return h5_obj, psg, loaded_chnls
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
        self._psg_obj, self._psg, loaded_chnls = self._load_with_any_in(self._try_channels)
        self._set_loaded_channels(loaded_chnls)

        if self.hyp_file_path is not None and not self.no_hypnogram:
            self._hypnogram, \
            self.annotation_dict = load_hypnogram(self.hyp_file_path,
                                                  period_length_sec=self.period_length_sec,
                                                  annotation_dict=self.annotation_dict,
                                                  sample_rate=self.sample_rate)
        else:
            self._hypnogram = False

    def load(self, reload=False):
        """
        High-level function invoked to load the SleepStudyBase data
        """
        if reload or not self.loaded:
            try:
                self._load()
            except Exception as e:
                raise errors.CouldNotLoadError("Unexpected load error for sleep "
                                               "study {}. Please refer to the "
                                               "above traceback.".format(self.identifier),
                                               study_id=self.identifier) from e
        return self

    def unload(self):
        """ Unloads the PSG and hypnogram """
        self.psg_obj.close()
        self._psg = None
        self._psg_obj = None
        self._hypnogram = None

    def reload(self, warning=True):
        """ Unloads and loads """
        if warning and self.loaded:
            print("Reloading SleepStudyBase '{}'".format(self.identifier))
        self.load(reload=True)

    def get_psg_shape(self):
        """
        TODO

        Returns:

        """
        n = self.psg[self.select_channels[0]].shape[0]
        return [n, len(self.psg)]

    def get_full_psg(self):
        """
        TODO

        Returns:

        """
        channels = [np.array(self.psg[c]) for c in self.select_channels]
        return np.stack(channels, axis=-1)

    def extract_from_psg(self, start, end, channel_inds=None):
        """
        Extract PSG data from second 'start' (inclusive) to second 'end'
        (exclusive)

        Args:
            start: int, start second to extract from
            end: int, end second to extract from
            channel_inds: list, list of channel indices to extract from

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

        channel_inds = channel_inds or range(len(self.select_channels))
        data = np.empty(shape=[last_row-first_row, len(channel_inds)],
                        dtype=self.psg[self.select_channels[channel_inds[0]]])
        for i, channel_ind in enumerate(channel_inds):
            chan_name = self.select_channels[channel_ind]
            data[:, i] = self.psg[chan_name][first_row:last_row]
        return data
