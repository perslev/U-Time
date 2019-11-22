import os
import numpy as np

from utime.dataset import SleepStudy
from utime.errors import CouldNotLoadError
from utime.dataset.utils import find_subject_folders
from utime import defaults
from MultiPlanarUNet.logging import ScreenLogger


class SleepStudyDataset(object):
    """
    Represents a collection of SleepStudy objects
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
        if bool(psg_regex) != bool(hyp_regex):
            raise RuntimeError("Must specify both or none of the 'psg_regex' "
                               "and 'hyp_regex' arguments.")

        self.logger = logger or ScreenLogger()
        self.data_dir = os.path.abspath(data_dir)
        self.pairs = []
        self.period_length_sec = (period_length_sec or
                                  defaults.get_default_period_length(self.logger))

        # Get list of subject folders in the data_dir according to folder_regex
        subject_folders = find_subject_folders(self.data_dir, folder_regex)
        if len(subject_folders) == 0:
            raise RuntimeError("Found no subject folders in data directory "
                               "{} using folder regex {}.".format(self.data_dir,
                                                                  folder_regex))

        # Initialize SleepStudy objects
        for subject_dir in subject_folders:
            ss = SleepStudy(
                subject_dir=subject_dir,
                psg_regex=psg_regex,
                hyp_regex=hyp_regex,
                period_length_sec=self.period_length_sec,
                annotation_dict=annotation_dict,
                load=False,
                logger=self.logger
            )
            self.pairs.append(ss)
        if len(np.unique([p.identifier for p in self.pairs])) != len(self.pairs):
            raise RuntimeError("Two or more SleepStudy objects share the same"
                               " identifier, but all must be unique.")
        self._identifier = identifier or os.path.split(self.data_dir)[-1]
        if not no_log:
            self.log()

    def log(self, message=None):
        """ Log basic properties about this dataset """
        id_msg = "[Dataset: {}]".format(self.identifier)
        if message is None:
            message = str(self)
        self.logger("{} {}".format(id_msg, message))

    @property
    def identifier(self):
        """ Returns the dataset ID string """
        return self._identifier

    @property
    def n_loaded(self):
        """ Returns the number of stored pairs that have data loaded """
        return len(self.loaded_pairs)

    @property
    def loaded_pairs(self):
        """ Returns stored SleepStudy objects that have data loaded """
        return [s for s in self if s.loaded]

    @property
    def non_loaded_pairs(self):
        """ Returns stored SleepStudy objects that do not have data loaded """
        return [s for s in self if not s.loaded]

    def __len__(self):
        """ Returns the number of stored SleepStudy objects """
        return len(self.pairs)

    def __getitem__(self, item):
        """ Return an element from the list of stored SleepStudy objects """
        return self.pairs[item]

    def __iter__(self):
        """ Yield elements from the list of stored SleepStudy objects """
        for pair in self.pairs:
            yield pair

    def __str__(self):
        return "SleepStudyDataset(identifier: {}, N pairs: {}, N loaded: {})" \
               "".format(self.identifier, len(self), self.n_loaded)

    def load(self, N=None, random_order=True):
        """
        Load all or a subset of stored SleepStudy objects
        Data is loaded using a thread pool with one thread per SleepStudy.

        Args:
            N:              Number of SleepStudy objects to load. Defaults to
                            loading all.
            random_order:   Randomly select which of the stored objects to load
                            rather than starting from the beginning. Only has
                            an effect with N != None
        Returns:
            self, reference to the SleepStudyDataset object
        """
        from concurrent.futures import ThreadPoolExecutor
        if N is None:
            N = len(self)
            random_order = False
        not_loaded = self.non_loaded_pairs
        if random_order:
            to_load = np.random.choice(not_loaded, size=N, replace=False)
        else:
            to_load = not_loaded[:N]
        self.log("Loading {}/{} SleepStudy objects...".format(len(to_load),
                                                              len(self)))
        pool = ThreadPoolExecutor(max_workers=min(len(to_load), 7))
        res = pool.map(lambda x: x.load(), to_load)
        try:
            for i, ss in enumerate(res):
                print(" -- {}/{}".format(i+1, len(to_load)), end="\r", flush=True)
        except CouldNotLoadError as e:
            raise CouldNotLoadError("Could not load sleep study {}."
                                    " Please refer to the above "
                                    "traceback.".format(e.study_id)) from e
        finally:
            pool.shutdown()
        return self

    def get_by_id(self, sleep_study_id):
        """ Return a stored SleepStudy object by its ID string """
        matches = [s for s in self if s.identifier == sleep_study_id]
        if len(matches) == 0:
            raise ValueError("Did not find a match to id {}".format(sleep_study_id))
        elif len(matches) > 1:
            raise ValueError("Found multiple matches to identifier: {}: {}".format(
                sleep_study_id, matches
            ))
        else:
            return matches[0]

    def get_all_periods(self, stack=False):
        """
        Returns the output of SleepStudy.get_all_periods across all stored
        SleepStudy objects either as a list with one element for each
        (stack=False) or as single, stacked arrays (stack=True)
        Please refer to SleepStudy.get_all_periods

        Returns:
            X: ndarray shape [-1, data_per_period, n_channels] (stack=True)
            y: ndarray shape [-1, 1] (stack=True)
        """
        loaded = self.loaded_pairs
        self.log("Getting all periods from {} loaded pairs ({} total pairs)"
                 "".format(len(loaded), len(self)))
        X, y = zip(*[s.get_all_periods() for s in loaded])
        if stack:
            X, y = np.vstack(X), np.vstack(y)
        return X, y

    def get_class_counts(self, n_classes, log=True):
        """
        Computes the sum of class count across all loaded SleepStudy objects

        Args:
            n_classes: Number of expected classes.
            log:       Log the results of the counting

        Returns:
            classes, ndarray, shape [n_classes] of class label integers
            counts, ndarray, shape [n_classes], of class label counts
        """
        if log:
            self.log("Counting class occurrences across {} loaded samples..."
                     .format(self.n_loaded))
        counts = {i: 0 for i in range(n_classes)}
        _, y = self.get_all_periods(stack=True)
        cls_ints, cls_counts = np.unique(y, return_counts=True)
        for cls_int, cls_count in zip(cls_ints, cls_counts):
            counts[cls_int] += cls_count
        cls, counts = zip(*counts.items())
        if log:
            self.log("Classes: {}\n"
                     "Counts:  {}".format(cls, counts))
        return cls, counts

    def set_select_channels(self, channels):
        """
        Sets the 'select_channels' property on all stored SleepStudy objects.
        Please refer to the SleepStudy.select_channels setter method
        """
        self.log("Setting select channels: {}".format(channels))
        for ss in self:
            ss.select_channels = channels

    def set_alternative_select_channels(self, channels):
        """
        Sets the 'alternative_select_channels' property on all stored
        SleepStudy objects. Please refer to the
        SleepStudy.alternative_select_channels setter method
        """
        if channels is None:
            return
        self.log("Setting alternative select channels: {}".format(channels))
        for ss in self:
            ss.alternative_select_channels = channels

    def set_channel_sampling_groups(self, channel_sampling_groups):
        """
        Sets the 'channel_sampling_groups' property on all stored SleepStudy
        objects. Please refer to the SleepStudy.channel_sampling_groups
        setter method
        """
        if channel_sampling_groups is None:
            return
        self.log("Setting channel sampling "
                 "groups: {}".format(channel_sampling_groups))
        for ss in self:
            ss.channel_sampling_groups = channel_sampling_groups

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

    def get_batch_sequence(self,
                           batch_size,
                           random_batches=True,
                           balanced_sampling=True,
                           n_classes=None,
                           margin=0,
                           scaler=None,
                           batch_wise_scaling=False,
                           no_log=False,
                           **kwargs):
        """
        Return a utime.sequences BatchSequence object made from this dataset.
        A BatchSequence (sub derived) object is used to extract batches of data
        from all or individual SleepStudy objects represented by this
        SleepStudyDataset.

        All args pass to the BatchSequence object.
        Please refer to its documentation.

        Returns:
            A BatchSequence object
        """
        loaded = self.loaded_pairs
        if len(loaded) == 0:
            raise IndexError("At least 1 SleepStudy pair must be loaded"
                             " at batch sequence creation time.")
        # Assure all same dpe
        dpe = np.asarray([l.data_per_period for l in loaded])
        if not np.all(dpe == dpe[0]):
            raise ValueError("'get_batch_sequence' currently requires all "
                             "SleepStudy pairs to have an identical number of"
                             "data points pr. period. "
                             "Got: {}".format(dpe))
        # Assure all same channels
        cnls = np.asarray([l.n_sample_channels for l in loaded])
        if not np.all(cnls == cnls[0]):
            raise ValueError("'get_batch_sequence' currently requires all "
                             "SleepStudy pairs to have an identical number of"
                             "channels. Got: {}".format(cnls))

        # Init and return the proper BatchSequence sub-class
        from utime.sequences import get_sequence_class
        sequence_class = get_sequence_class(random_batches, balanced_sampling)
        return sequence_class(identifier=self.identifier,
                              sleep_study_pairs=self.pairs,
                              batch_size=batch_size,
                              data_per_period=dpe[0],
                              n_classes=n_classes,
                              n_channels=cnls[0],
                              margin=margin,
                              batch_scaler=scaler if batch_wise_scaling else None,
                              logger=self.logger,
                              no_log=no_log,
                              **kwargs)
