from utime.dataset import SleepStudy
from utime.dataset.sleep_study_dataset.subject_dir_sleep_study_dataset_base\
    import SubjectDirSleepStudyDatasetBase


class SleepStudyDataset(SubjectDirSleepStudyDatasetBase):
    """
    Represents a collection of SleepStudy objects
    """
    def __init__(self,
                 data_dir,
                 folder_regex=r'^(?!views).*$',
                 psg_regex=None,
                 hyp_regex=None,
                 no_labels=False,
                 period_length_sec=None,
                 annotation_dict=None,
                 identifier=None,
                 logger=None,
                 no_log=False):
        """
        Initialize a SleepStudyDataset from a directory storing one or more
        sub-directories each corresponding to a sleep/PSG study.
        Each sub-dir will be represented by a SleepStudy object.

        Args:
            data_dir:                (string) Path to the data directory
            folder_regex:            (string) Regex that matches folders to
                                              consider within the data_dir.
            psg_regex:               (string) Regex that matches files to
                                              consider 'PSG' (data) within each
                                              subject folder.
                                              Passed to each SleepStudy.
            hyp_regex:               (string) As psg_regex, but for hypnogram/
                                              sleep stages/label files.
                                              Passed to each SleepStudy.
            period_length_sec:       (int)    Ground truth segmentation
                                              period length in seconds.
            annotation_dict:         (dict)   Dictionary mapping labels as
                                              storred in the hyp files to
                                              label integer values.
            identifier:              (string) Dataset ID/name
            logger:                  (Logger) A Logger object
            no_log:                  (bool)   Do not log dataset details on init
        """
        super(SleepStudyDataset, self).__init__(
            data_dir=data_dir,
            sleep_study_class=SleepStudy,
            folder_regex=folder_regex,
            psg_regex=psg_regex,
            hyp_regex=hyp_regex,
            no_labels=no_labels,
            period_length_sec=period_length_sec,
            annotation_dict=annotation_dict,
            identifier=identifier,
            logger=logger,
            no_log=no_log
        )

    def __str__(self):
        return "SleepStudyDataset(identifier: {}, N pairs: {}, N loaded: {})" \
               "".format(self.identifier, len(self), self.n_loaded)

    def set_load_time_channel_sampling_groups(self, *channel_groups):
        """
        TODO

        Args:
            channel_groups:
        """
        if len(channel_groups) == 0 or channel_groups[0] is None:
            random_selector = None
        else:
            from utime.io.channels import RandomChannelSelector
            random_selector = RandomChannelSelector(*channel_groups)
        self.log("Setting load-time random channel selector: "
                 "{}".format(random_selector))
        for ss in self:
            ss.load_time_random_channel_selector = random_selector

    def set_scaler(self, scaler):
        """
        Sets the 'scaler' property on all stored SleepStudy objects.
        Please refer to the SleepStudy.scaler setter method
        """
        self.log("Setting '{}' scaler...".format(scaler))
        for ss in self:
            ss.scaler = scaler

    def set_sample_rate(self, sample_rate):
        """
        Sets the 'sample_rate' property on all stored SleepStudy objects.
        Please refer to the SleepStudy.sample_rate setter method
        """
        self.log("Setting sample rate of {} Hz".format(sample_rate))
        for ss in self:
            ss.sample_rate = sample_rate

    def set_strip_func(self, strip_func, **kwargs):
        """
        Sets the 'strip_func' property on all stored SleepStudy objects.
        Please refer to the SleepStudy.strip_func setter method
        """
        self.log("Setting '{}' strip function with parameters {}..."
                 "".format(strip_func, kwargs))
        for ss in self:
            ss.set_strip_func(strip_func, **kwargs)

    def set_quality_control_func(self, quality_control_func, **kwargs):
        """
        Sets the 'quality_control_func' property on all stored SleepStudy
        objects. Please refer to the SleepStudy.quality_control_func setter
        method
        """
        self.log("Setting '{}' quality control function with "
                 "parameters {}...".format(quality_control_func, kwargs))
        for ss in self:
            ss.set_quality_control_func(quality_control_func, **kwargs)
