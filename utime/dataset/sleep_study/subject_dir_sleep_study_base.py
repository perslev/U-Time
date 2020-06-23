"""
Implements the SleepStudyBase class which represents a sleep study (PSG)
"""

import os

from utime.dataset.utils import find_psg_and_hyp
from utime.dataset.sleep_study.abc_sleep_study import AbstractBaseSleepStudy


class SubjectDirSleepStudyBase(AbstractBaseSleepStudy):
    def __init__(self,
                 subject_dir,
                 psg_regex=None,
                 hyp_regex=None,
                 period_length_sec=None,
                 no_hypnogram=None,
                 annotation_dict=None,
                 logger=None):
        """
        Initialize a SubjectDirSleepStudyBase object from PSG/HYP data

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
            logger            (Logger) A Logger object
        """
        super(SubjectDirSleepStudyBase, self).__init__(
            annotation_dict=annotation_dict,
            period_length_sec=period_length_sec,
            no_hypnogram=no_hypnogram,
            logger=logger
        )
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

    @property
    def identifier(self):
        """
        Returns an ID, which is simply the name of the directory storing
        the data
        """
        return os.path.split(self.subject_dir)[-1]

    @property
    def n_classes(self):
        """ Returns the number of classes represented in the hypnogram """
        return self.hypnogram.n_classes

    @property
    def recording_length_sec(self):
        """ Returns the total length (in seconds) of the PSG recording """
        return self.get_psg_shape()[0] / self.sample_rate

    def get_full_hypnogram(self):
        """
        Returns the full (dense) hypnogram

        Returns:
            An ndarray of shape [self.n_periods, 1] of class labels
        """
        return self.hypnogram.to_dense()["sleep_stage"].to_numpy().reshape(-1, 1)

    def get_periods_by_idx(self, start_idx, end_idx):
        """
        Get a range of period of {X, y} data by indices
        Period starting at second 0 is index 0.

        Args:
            start_idx (int): Index of first period to return
            end_idx   (int): Index of last period to return (inclusive)

        Returns:
            X: A list of ndarrays each of shape
               [self.data_per_period, self.n_channels]
            y: A list of ndarrays each of shape [1]
        """
        Xs, ys = [], []
        for idx in range(start_idx, end_idx+1):
            X, y = self.get_period_by_idx(idx)
            Xs.append(X), ys.append(y)
        return Xs, ys

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

    def get_stage_at_sec(self, second):
        """
        TODO

        Args:
            second:

        Returns:

        """
        return self.hypnogram.get_stage_at_sec(second)

    def get_all_periods(self):
        """
        Returns the full (dense) data of the SleepStudyBase

        Returns:
            X: An ndarray of shape [self.n_periods,
                                    self.data_per_period,
                                    self.n_channels]
            y: An ndarray of shape [self.n_periods, 1]
        """
        X = self.get_full_psg().reshape(-1, self.data_per_period, self.n_channels)
        if self.no_hypnogram:
            return X
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
