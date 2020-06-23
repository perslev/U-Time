from utime.dataset.sleep_study import FileH5SleepStudy
from utime.dataset.sleep_study_dataset.subject_dir_sleep_study_dataset_base\
    import SubjectDirSleepStudyDatasetBase


class H5SleepStudyDataset(SubjectDirSleepStudyDatasetBase):
    """
    Represents a collection of H5SleepStudy objects
    """
    def __init__(self,
                 data_dir,
                 folder_regex=r'^(?!views).*$',
                 psg_regex=None,
                 hyp_regex=None,
                 period_length_sec=None,
                 annotation_dict=None,
                 identifier=None,
                 logger=None,
                 no_log=False):
        """
        Initialize a H5SleepStudyDataset from a directory storing one or more
        sub-directories each corresponding to a sleep/PSG study.
        Each sub-dir will be represented by a H5SleepStudy object.

        Args:
            data_dir:                (string) Path to the data directory
            folder_regex:            (string) Regex that matches folders to
                                              consider within the data_dir.
            psg_regex:               (string) Regex that matches files to
                                              consider 'PSG' (data) within each
                                              subject folder.
                                              Passed to each SleepStudyBase.
            hyp_regex:               (string) As psg_regex, but for hypnogram/
                                              sleep stages/label files.
                                              Passed to each SleepStudyBase.
            period_length_sec:       (int)    Ground truth segmentation
                                              period length in seconds.
            annotation_dict:         (dict)   Dictionary mapping labels as
                                              storred in the hyp files to
                                              label integer values.
            identifier:              (string) Dataset ID/name
            logger:                  (Logger) A Logger object
            no_log:                  (bool)   Do not log dataset details on init
        """
        super(H5SleepStudyDataset, self).__init__(
            data_dir=data_dir,
            sleep_study_class=FileH5SleepStudy,
            folder_regex=folder_regex,
            psg_regex=psg_regex,
            hyp_regex=hyp_regex,
            period_length_sec=period_length_sec,
            annotation_dict=annotation_dict,
            identifier=identifier,
            logger=logger,
            no_log=no_log
        )

    def __str__(self):
        return "H5SleepStudyDataset(identifier: {}, N pairs: {}, N loaded: {})" \
               "".format(self.identifier, len(self), self.n_loaded)
