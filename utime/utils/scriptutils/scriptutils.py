"""
A set of utility functions used across multiple scripts in utime.bin
"""

import os
from utime.utils.utils import ensure_list_or_tuple
from MultiPlanarUNet.logging.default_logger import ScreenLogger


def assert_project_folder(project_folder, evaluation=False):
    """
    Raises RuntimeError if a folder 'project_folder' does not seem to be a
    valid U-Time folder in the training phase (evaluation=False) or evaluation
    phase (evaluation=True).

    Args:
        project_folder: A path to a folder to check for U-Time compat.
        evaluation:     Should the folder adhere to train- or eval time checks.
    """
    import os
    import glob
    project_folder = os.path.abspath(project_folder)
    if not os.path.exists(os.path.join(project_folder, "hparams.yaml")):
        # Folder must contain a 'hparams.yaml' file in all cases.
        raise RuntimeError("Folder {} is not a valid DeepSleep project folder."
                           " Must contain a 'hparams.yaml' "
                           "file.".format(project_folder))
    if evaluation:
        # Folder must contain a 'model' subfolder storing saved model files
        model_path = os.path.join(project_folder, "model")
        if not os.path.exists(model_path):
            raise RuntimeError("Folder {} is not a valid DeepSleep project "
                               "folder. Must contain a 'model' "
                               "subfolder.".format(project_folder))
        # There must be a least 1 model file (.h5) in the folder
        models = glob.glob(os.path.join(model_path, "*.h5"))
        if not models:
            raise RuntimeError("Did not find any model parameter files in "
                               "model subfolder {}. Model files should have"
                               " extension '.h5' to "
                               "be recognized.".format(project_folder))


def get_all_dataset_hparams(hparams):
    """
    Takes a YAMLHParams object and returns a dictionary of one or more entries
    of dataset ID to YAMLHParams objects pairs; one for each dataset described
    in 'hparams'.

    If 'hparams' has the 'datasets' attribute each mentioned dataset under this
    field will be loaded and returned. Otherwise, it is assumed that a single
    dataset is described directly in 'hparams', in which case 'hparams' as-is
    will be the only returned value (with no ID).

    Args:
        hparams: (YAMLHParams) A hyperparameter object storing reference to
                               one or more datasets in the 'datasets' field, or
                               directly in 'hparams.

    Returns:
        A dictonary if dataset ID to YAMLHParams object pairs
        One entry for each dataset
    """
    from utime.hyperparameters import YAMLHParams
    dataset_hparams = {}
    if hparams.get("datasets"):
        # Multiple datasets specified in hparams configuration files
        ids_and_paths = hparams["datasets"].items()
        for id_, path in ids_and_paths:
            yaml_path = os.path.join(hparams.project_path, path)
            dataset_hparams[id_] = YAMLHParams(yaml_path,
                                               no_log=True,
                                               no_version_control=True)
    else:
        # Return as-is with no ID
        dataset_hparams[""] = hparams
    return dataset_hparams


def get_dataset_splits_from_hparams(hparams, splits_to_load,
                                    logger=None, id=""):
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
        logger:         A Logger object
        id:             An optional id to prepend to the identifier of the
                        dataset. For instance, with id 'ABC' and sub-dataset
                        identifier 'TRAIN' the resulting dataset will have
                        identifier 'ABC/TRAIN'.

    Returns:
        A list of initialized and prepared datasets according to hparams.
    """
    from utime.dataset import SleepStudyDataset
    ann_dict = hparams.get("sleep_stage_annotations")
    datasets = []
    for data_key in ensure_list_or_tuple(splits_to_load):
        if data_key not in hparams:
            raise ValueError("Dataset with key '{}' does not exists in the "
                             "hyperparameters file".format(data_key))
        new_id = f"{id}{'/' if id else ''}{hparams[data_key]['identifier']}"
        hparams[data_key]["identifier"] = new_id
        dataset = SleepStudyDataset(**hparams[data_key],
                                    logger=logger,
                                    annotation_dict=ann_dict)
        datasets.append(dataset)

    # Apply transformations, scaler etc.
    from utime.utils.scriptutils import select_sample_strip_scale_quality
    select_sample_strip_scale_quality(*datasets, hparams=hparams, logger=logger)
    return datasets


def get_dataset_splits_from_hparams_file(hparams_path, splits_to_load,
                                         logger=None, id=""):
    """
    Loads one or more datasets according to hyperparameters described in yaml
    file at path 'hparams_path'. Specifically, this functions creates a temp.
    YAMLHparams object from the yaml file data and applies redirects to the
    'get_dataset_splits_from_hparams' function.

    Please refer to the docstring of 'get_dataset_splits_from_hparams' for
    details.
    """
    from utime.hyperparameters import YAMLHParams
    hparams = YAMLHParams(hparams_path, no_log=True, no_version_control=True)
    return get_dataset_splits_from_hparams(hparams, splits_to_load, logger, id)


def get_splits_from_all_datasets(hparams, splits_to_load, logger=None):
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
        hparams:        A YAMLHparams object storing references to one or more
                        datasets
        splits_to_load: A string, list or tuple of strings giving the name
                        of all sub-datasets to load according to their hparams
                        descriptions.
        logger:         A Logger object

    Returns:
        Yields one or more splits of data from datasets as described by
        'hparams'
    """
    data_hparams = get_all_dataset_hparams(hparams)
    for dataset_id, hparams in data_hparams.items():
        yield get_dataset_splits_from_hparams(
            hparams=hparams,
            splits_to_load=splits_to_load,
            logger=logger,
            id=dataset_id
        )


def get_dataset_from_regex_pattern(regex_pattern, hparams, logger=None):
    """
    Initializes a SleepStudy dataset and applies prep. function
    'select_sample_strip_scale_quality' from all subject folders that match
    the a regex statement.

    Args:
        regex_pattern: A string regex pattern used to match to all subject dirs
                       to include in the dataset
        hparams:       A YAMLHparams object to read settings from that should
                       apply to the initialized dataset.
        logger:        A Logger object

    Returns:
        A SleepStudy object with settings set as per 'hparams'
    """
    from utime.dataset.sleep_study_dataset import SleepStudyDataset
    ann_dict = hparams.get("sleep_stage_annotations")
    params = hparams["train_data"]
    data_dir, pattern = os.path.split(os.path.abspath(regex_pattern))
    params["data_dir"] = data_dir
    ssd = SleepStudyDataset(folder_regex=pattern,
                            **params,
                            logger=logger,
                            annotation_dict=ann_dict)
    # Apply transformations, scaler etc.
    from utime.utils.scriptutils import select_sample_strip_scale_quality
    select_sample_strip_scale_quality(ssd, hparams=hparams, logger=logger)
    return ssd


def select_sample_strip_scale_quality(*datasets, hparams, logger=None):
    """
    Helper function which calls the following 7 methods on a SleepStudyDataset
    with parameters inferred from a YAMLHparams object:
    - SleepStudyDataset.set_select_channels()
    - SleepStudyDataset.set_alternative_select_channels()
    - SleepStudyDataset.set_channel_sampling_groups()
    - SleepStudyDataset.set_sample_rate()
    - SleepStudyDataset.set_strip_func()
    - SleepStudyDataset.set_quality_control_func()
    - SleepStudyDataset.set_scaler()

    Args:
        *datasets: Any number of SleepStudyDataset objects
        hparams:   A YAMLHparams object parameterised the 3 methods called.
    """
    # Select channels if specified
    select = hparams.get("select_channels", [])
    list(map(lambda ds: ds.set_select_channels(select), datasets))

    # Set alternative select channels if specified
    alt_select = hparams.get("alternative_select_channels", [])
    list(map(lambda ds: ds.set_alternative_select_channels(alt_select), datasets))

    # Set channel sampling groups if specified
    groups = hparams.get("channel_sampling_groups", None)
    list(map(lambda ds: ds.set_channel_sampling_groups(groups), datasets))

    # Set sample rate
    sample_rate = hparams.get("set_sample_rate", None)
    list(map(lambda ds: ds.set_sample_rate(sample_rate), datasets))

    # Apply strip function if specified
    strip_settings = hparams.get("strip_func")
    if strip_settings:
        list(map(lambda ds: ds.set_strip_func(**strip_settings), datasets))

    # Apply quality control function if specified
    quality_settings = hparams.get("quality_control_func")
    if quality_settings:
        list(map(lambda ds: ds.set_quality_control_func(**quality_settings), datasets))

    # Set scaler
    if not hparams.get("batch_wise_scaling"):
        scl = hparams.get_from_anywhere("scaler") or "RobustScaler"
        list(map(lambda ds: ds.set_scaler(scl), datasets))
    else:
        from MultiPlanarUNet.logging import ScreenLogger
        logger = logger or ScreenLogger()
        logger.warn("Note: 'batch_wise_scaling' is set to True in 'fit' hparams."
                    " No scaling will be applied globally to members of the"
                    " datasets: {}. Scaling will instead be applied batch-wise."
                    " Check the log-output of the sequencer object to ensure "
                    "that a 'batch_wise_scaler' object has been set".format(
            list(map(str, datasets))
        ))


def make_multi_gpu_model(model, num_GPUs, logger=None):
    """
    Takes a compiled tf.keras Model object 'model' and applies
    from tensorflow.keras.utils import multi_gpu_model
    ... to mirror it across multiple visible GPUs. Input batches to 'model'
    are split evenly across the GPUs.

    Args:
        model:    (tf.keras Model) A compiled tf.keras Model object.
        num_GPUs: (int)            Number of GPUs to distribute the model over
        logger:   (Logger)         Optional Logger object

    Returns:
        The split, multi-GPU model.
        The original model
        Note: The two will be the same for num_GPUs=1
    """
    org_model = model
    if num_GPUs > 1:
        from tensorflow.keras.utils import multi_gpu_model
        model = multi_gpu_model(org_model, gpus=num_GPUs,
                                cpu_merge=False, cpu_relocation=False)
        logger = logger or ScreenLogger()
        logger("Creating multi-GPU model: N=%i" % num_GPUs)
    return model, org_model
