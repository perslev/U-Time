"""
A set of utility functions used across multiple scripts in utime.bin
"""

import logging
import os
from functools import wraps
from psg_utils.utils import ensure_list_or_tuple
from psg_utils.dataset import SleepStudyDataset
from psg_utils.preprocessing.utils import select_sample_strip_scale_quality
from utime import Defaults

logger = logging.getLogger(__name__)


def add_logging_file_handler(log_file_name, overwrite, logger_objects=None, log_dir=None, mode="w"):
    # Log to file if specified
    if log_file_name:
        Defaults.set_logging_file_handler(file_name=log_file_name,
                                          loggers=logger_objects,
                                          overwrite_existing=overwrite,
                                          log_dir=log_dir,
                                          mode=mode)
    else:
        relevant_loggers = logger_objects or \
                           Defaults.PACKAGE_LEVEL_LOGGERS or \
                           [logging.getLogger(Defaults.PACKAGE_NAME)]
        logger.info(f"Logs will not be saved to file for these loggers: {relevant_loggers} (--log_file_name is empty)")


def with_logging_level_wrapper(func, level, logger_names=None):
    loggers = [logging.getLogger(name) for name in logger_names] if logger_names \
        else Defaults.PACKAGE_LEVEL_LOGGERS or [logging.getLogger(Defaults.PACKAGE_NAME)]
    old_levels = {logger: logger.level for logger in loggers}
    @wraps(func)
    def with_logging_level(*args, **kwargs):
        for logger in loggers:
            logger.setLevel(level)
        outs = func(*args, **kwargs)
        for logger in loggers:
            logger.setLevel(old_levels[logger])
        return outs
    return with_logging_level


def assert_project_folder(project_folder, evaluation=False):
    """
    Raises RuntimeError if a folder 'project_folder' does not seem to be a
    valid U-Time folder in the training phase (evaluation=False) or evaluation
    phase (evaluation=True).

    Args:
        project_folder: A path to a folder to check for U-Time compat.
        evaluation:     Should the folder adhere to train- or eval time checks.

    Returns:
        empty_models_dir: Bool, whether the project_folder/models dir is empty or not.
    """
    import os
    import glob
    project_folder = os.path.abspath(project_folder)
    if not os.path.exists(Defaults.get_hparams_path(project_folder)) \
            and not os.path.exists(Defaults.get_pre_processed_hparams_path(project_folder)):
        # Folder must contain a 'hparams.yaml' file in all cases.
        raise RuntimeError("Folder {} is not a valid project folder."
                           " Must contain a hyperparameter "
                           "file.".format(project_folder))
    model_path = os.path.join(project_folder, "model")
    if evaluation:
        # Folder must contain a 'model' subfolder storing saved model files
        if not os.path.exists(model_path):
            raise RuntimeError("Folder {} is not a valid project "
                               "folder for model evaluation. Must contain a 'model' "
                               "subfolder.".format(project_folder))
        # There must be a least 1 model file (.h5) in the folder
        models = glob.glob(os.path.join(model_path, "*.h5"))
        if not models:
            raise RuntimeError("Did not find any model parameter files in "
                               "model subfolder {}. Model files should have"
                               " extension '.h5' to "
                               "be recognized.".format(project_folder))
    files_in_model_dir = os.path.exists(model_path) and bool(os.listdir(model_path))
    return not files_in_model_dir


def get_all_dataset_hparams(hparams, project_dir=None, dataset_ids=None):
    """
    Takes a YAMLHParams object and returns a dictionary of one or more entries
    of dataset ID to YAMLHParams objects pairs; one for each dataset described
    in 'hparams'.

    If 'hparams' has the 'datasets' attribute each mentioned dataset under this
    field will be loaded and returned. Otherwise, it is assumed that a single
    dataset is described directly in 'hparams', in which case 'hparams' as-is
    will be the only returned value (with no ID).

    Args:
        hparams: (YAMLHParams)    A hyperparameter object storing reference to
                                  one or more datasets in the 'datasets' field, or
                                  directly in 'hparams.
        project_dir: [None, str]  Optional path to a project directory storing
                                  hyperparameters relevant to the 'hparams' object.
                                  If not specified, will use the Default.PROJECT_DIR
                                  value which is set at runtime for all utime scripts.
        dataset_ids (None, list)  Only returns hparams for datasets with IDs in 'dataset_ids'.
                                  If None, return hparams for all datasets.

    Returns:
        A dictonary if dataset ID to YAMLHParams object pairs
        One entry for each dataset
    """
    from utime.hyperparameters import YAMLHParams
    dataset_hparams = {}
    if hparams.get("datasets"):
        # Multiple datasets specified in hparams configuration files
        ids_and_paths = hparams["datasets"].items()
        project_dir = project_dir or Defaults.PROJECT_DIRECTORY
        if not project_dir:
            raise ValueError("Must specify either the 'project_dir' argument or the "
                             "Defaults.PROJECT_DIRECTORY property must have been set before calling "
                             "this function (e.g., by invoking the utime entry script).")
        for id_, path in ids_and_paths:
            if dataset_ids and id_ not in dataset_ids:
                logger.warning(f"Ignoring dataset '{id_}' in hparams (not in 'dataset_ids' list).")
                continue
            yaml_path = os.path.join(Defaults.get_hparams_dir(project_dir), path)
            dataset_hparams[id_] = YAMLHParams(yaml_path,
                                               no_version_control=True)
    else:
        # Return as-is with no ID
        dataset_hparams[""] = hparams
    return dataset_hparams


def get_dataset_splits_from_hparams(hparams, splits_to_load, id=""):
    """
    Return all initialized and prepared (according to the prep. function of
    'select_sample_strip_scale_quality') SleepStudyDataset objects as described
    in a YAMLHparams object.

    Args:
        hparams:        A YAMLHparams object describing one or more datasets to
                        load
        splits_to_load: A string, list or tuple of strings giving the name of
                        all (sub-)datasets to load according to their hparams
                        descriptions. That is, 'load' could be ('TRAIN', 'VAL')
                        to load the training and validation data.
        id:             An optional id to prepend to the identifier of the
                        dataset. For instance, with id 'ABC' and sub-dataset
                        identifier 'TRAIN' the resulting dataset will have
                        identifier 'ABC/TRAIN'.

    Returns:
        A list of initialized and prepared datasets according to hparams.
    """
    ann_dict = hparams.get("sleep_stage_annotations")
    datasets = []
    for data_key in ensure_list_or_tuple(splits_to_load):
        if data_key not in hparams:
            raise ValueError("Dataset with key '{}' does not exists in the "
                             "hyperparameters file".format(data_key))
        new_id = f"{id}{'/' if id else ''}{hparams[data_key]['identifier']}"
        hparams[data_key]["identifier"] = new_id

        # Load either a standard SleepStudyDataset or from the SingleH5Dataset
        dataset = SleepStudyDataset(**hparams[data_key],
                                    annotation_dict=ann_dict)
        datasets.append(dataset)

    # Apply transformations, scaler etc.
    select_sample_strip_scale_quality(*datasets, hparams=hparams)
    return datasets


def get_dataset_splits_from_hparams_file(hparams_path, splits_to_load, id=""):
    """
    Loads one or more datasets according to hyperparameters described in yaml
    file at path 'hparams_path'. Specifically, this functions creates a temp.
    YAMLHparams object from the yaml file data and applies redirects to the
    'get_dataset_splits_from_hparams' function.

    Please refer to the docstring of 'get_dataset_splits_from_hparams' for
    details.
    """
    from utime.hyperparameters import YAMLHParams
    hparams = YAMLHParams(hparams_path, no_version_control=True)
    return get_dataset_splits_from_hparams(hparams, splits_to_load, id)


def get_splits_from_all_datasets(hparams, splits_to_load, return_data_hparams=False, dataset_ids=None):
    """
    Wrapper around the 'get_dataset_splits_from_hparams_file' and
    'get_dataset_splits_from_hparams' files loading all sub-datasets according
    to 'splits_to_load from each dataset specified in the file.
    The dataset is processed according to hparams in the prep. function
    'select_sample_strip_scale_quality'.

    I.e. if hparams refer to 2 different datasets, e.g. 'Sleep-EDF-153' and
    'DCSM' and you want to load the training and validation data from each
    of those you would pass load=('TRAIN', 'VAL') and the train/val pairs
    of each dataset would be yielded one by one.

    Please refer to 'get_dataset_splits_from_hparams' for details.

    Args:
        hparams:                  A YAMLHparams object storing references to one or more
                                  datasets
        splits_to_load:           A string, list or tuple of strings giving the name
                                  of all sub-datasets to load according to their hparams
                                  descriptions.
        return_data_hparams:      TODO
        dataset_ids (None, list)  Only returns hparams for datasets with IDs in 'dataset_ids'.
                                  If None, return hparams for all datasets.

    Returns:
        Yields one or more splits of data from datasets as described by
        'hparams'
    """
    data_hparams = get_all_dataset_hparams(hparams, dataset_ids=dataset_ids)
    for dataset_id, hparams in data_hparams.items():
        ds = get_dataset_splits_from_hparams(
                hparams=hparams,
                splits_to_load=splits_to_load,
                id=dataset_id)
        if return_data_hparams:
            yield ds, hparams
        else:
            yield ds


def get_dataset_from_regex_pattern(regex_pattern, hparams):
    """
    Initializes a SleepStudy dataset and applies prep. function
    'select_sample_strip_scale_quality' from all subject folders that match
    a regex statement.

    Args:
        regex_pattern: A string regex pattern used to match to all subject dirs
                       to include in the dataset
        hparams:       A YAMLHparams object to read settings from that should
                       apply to the initialized dataset.

    Returns:
        A SleepStudy object with settings set as per 'hparams'
    """
    ann_dict = hparams.get("sleep_stage_annotations")
    if 'prediction_params' in hparams:
        period_length_sec = hparams['prediction_params'].get('period_length_sec', None)
        pre_proc_params = hparams['prediction_params']
    else:
        period_length_sec = (hparams.get("train_data") or
                             hparams.get("test_data")).get('period_length_sec', None)
        pre_proc_params = hparams
    data_dir, pattern = os.path.split(os.path.abspath(regex_pattern))
    ssd = SleepStudyDataset(folder_regex=pattern,
                            data_dir=data_dir,
                            period_length_sec=period_length_sec,
                            annotation_dict=ann_dict)
    # Apply transformations, scaler etc.
    from utime.utils.scriptutils import select_sample_strip_scale_quality
    select_sample_strip_scale_quality(ssd, hparams=pre_proc_params)
    return ssd
