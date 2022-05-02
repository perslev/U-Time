"""
A set of functions for needed for running training in various settings
"""

import logging
import os
import shutil
import numpy as np
import pandas as pd
from utime.utils.scriptutils import get_all_dataset_hparams, get_splits_from_all_datasets
from utime.sequences import MultiSequence, ValidationMultiSequence, get_batch_sequence
from psg_utils.preprocessing.utils import select_sample_strip_scale_quality
from psg_utils.dataset.sleep_study_dataset import SingleH5Dataset
from psg_utils.errors import NotLoadedError

logger = logging.getLogger(__name__)


def remove_previous_session(project_folder):
    """
    Deletes various utime project folders and files from
    [project_folder].

    Args:
        project_folder: A path to a utime project folder
    """
    # Remove old files and directories of logs, images etc if existing
    paths = [os.path.join(project_folder, p) for p in ("logs",
                                                       "model",
                                                       "tensorboard")]
    for p in filter(os.path.exists, paths):
        if os.path.isdir(p):
            shutil.rmtree(p)
        else:
            os.remove(p)


def init_default_project_structure(project_folder, required_folders=('logs', 'model')):
    for folder in required_folders:
        folder = os.path.join(project_folder, folder)
        os.mkdir(folder)


def get_train_and_val_datasets(hparams, no_val, train_on_val, dataset_ids=None):
    """
    Return all pairs of (train, validation) SleepStudyDatasets as described in
    the YAMLHParams object 'hparams'. A list is returned, as more than 1
    dataset may be described in the parameter file.

    Also returns an updated version of 'no_val', see below. Specifically, if
    'train_on_val' is True, then no_val will be set to true no matter its
    initial value.

    Args:
        hparams:      (YAMLHParams) A hyperparameter object to load dataset
                                    configurations from.
        no_val:       (bool)        Do not load validation data
        train_on_val: (bool)        Load validation data, but merge it into
                                    the training data. Then return only the
                                    'trainin' (train+val) dataset.
        dataset_ids (None, list)    Only load datasets with IDs in 'dataset_ids'.
                                    If None, load all datasets.

    Returns:
        A list of training SleepStudyDataset objects
        A list of validation SleepStudyDataset objects, or [] if not val.
    """
    if no_val:
        load = ("train_data",)
        if train_on_val:
            raise ValueError("Should not specify --no_val with --train_on_val")
    else:
        load = ("train_data", "val_data")
    datasets = [*get_splits_from_all_datasets(hparams, load, dataset_ids=dataset_ids)]
    if train_on_val:
        if any([len(ds) != 2 for ds in datasets]):
            raise ValueError("Did not find a validation set for one or more "
                             "pairs in {}".format(datasets))
        logger.info("[OBS] Merging training and validation sets")
        datasets = [merge_train_and_val(*ds) for ds in datasets]
        no_val = True
    if not no_val:
        train_datasets, val_datasets = zip(*datasets)
    else:
        train_datasets = [d[0] for d in datasets]
        val_datasets = []
    return train_datasets, val_datasets


def get_h5_train_and_val_datasets(hparams, no_val, train_on_val, dataset_ids=None):
    """
    TODO

    Args:
        hparams:      (YAMLHParams) A hyperparameter object to load dataset
                                    configurations from.
        no_val:       (bool)        Do not load validation data
        train_on_val: (bool)        Load validation data, but merge it into
                                    the training data. Then return only the
                                    'trainin' (train+val) dataset.
        dataset_ids (None, list)    Only load datasets with IDs in 'dataset_ids'.
                                    If None, load all datasets.

    Returns:
        A list of training SleepStudyDataset objects
        A list of validation SleepStudyDataset objects, or [] if not val.
    """
    def _get_dataset(h5_dataset, regex, hparams):
        """
        Helper for returning a dataset from a H5Dataset object according to
        regex and a hyperparameter set for a single dataset.
        """
        h5_path = hparams['data_dir']
        if os.path.abspath(h5_path) != os.path.abspath(h5_dataset.h5_path):
            raise ValueError("Currently all data must be stored in a single "
                             ".h5 file. Found two or more different files.")
        dataset = h5_dataset.get_datasets(
            load_match_regex=regex,
            period_length_sec=hparams.get('period_length_sec'),
            annotation_dict=hparams.get('sleep_stage_annotations')
        )
        assert len(dataset) == 1
        return dataset[0]

    if train_on_val:
        raise NotImplementedError("Training on validation data is not yet "
                                  "implemented for preprocessed H5 datasets.")
    data_hparams = get_all_dataset_hparams(hparams, dataset_ids=dataset_ids)
    h5_dataset = None
    train_datasets, val_datasets = [], []
    for dataset_id, hparams in data_hparams.items():
        if h5_dataset is None:
            h5_dataset = SingleH5Dataset(hparams['train_data']['data_dir'])
        train = _get_dataset(
            h5_dataset=h5_dataset,
            regex=f'/{dataset_id}/TRAIN',
            hparams=hparams['train_data']
        )
        train_datasets.append(train)
        ds = [train]
        if not no_val:
            val = _get_dataset(
                h5_dataset=h5_dataset,
                regex=f'/{dataset_id}/VAL',
                hparams=hparams['val_data']
            )
            ds.append(val)
            val_datasets.append(val)
        select_sample_strip_scale_quality(*ds, hparams=hparams)
    return train_datasets, val_datasets


def get_generators(train_datasets_queues, hparams, val_dataset_queues=None):
    """
    Takes a list of training and optionally validation utime.dataset.queue
    type objects and returns a training and validation sequence object
    (see utime.sequences). If val_dataset_queues is None, returns only a
    training sequencer.

    With multiple training sequence objects, a MultiSequence object is returned
    which is a light wrapper around multiple sequences that samples across them

    With validation data, a ValidationMultiSequence is always returned, even
    for 1 dataset, as this data structure is expected in the Validation
    callback.

    Args:
        train_datasets_queues: (list)        TODO
        hparams:               (YAMLHParams) The hyperparameters to init the
                                             sequencers with
        val_dataset_queues:    (list)        TODO

    Returns:
        A training Sequence or MultiSequence objects
        A ValidatonMultiSequence object if no_val=False, otherwise None
    """
    n_classes = hparams.get_group('/build/n_classes')
    train_seqs = [get_batch_sequence(dataset_queue=d,
                                     random_batches=True,
                                     n_classes=n_classes,
                                     augmenters=hparams.get("augmenters"),
                                     **hparams["fit"]) for d in train_datasets_queues]
    val_seq = None
    if val_dataset_queues:
        val_seq = [get_batch_sequence(dataset_queue=d,
                                      n_classes=n_classes,
                                      **hparams['fit']) for d in val_dataset_queues]
    if len(train_seqs) > 1:
        # Wrap sequencers in MultiSequence object which creates batches by sampling
        # across its stores individual sequencers
        train_seq = MultiSequence(train_seqs, hparams['fit']['batch_size'])
    else:
        train_seq = train_seqs[0]
    if val_seq:
        assert len(val_seq) == len(train_seqs)
        val_seq = ValidationMultiSequence(val_seq)
    return train_seq, val_seq


def merge_train_and_val(train, val):
    """
    Takes two SleepStudyDataset objects 'train' and 'val' and merges them by
    adding all stored SleepStudy pairs of 'val' to the list of pairs in 'train'
    Then changes the 'identifier' attribute of the 'train' sequencer to reflect
    the changes and returns the, now merged, 'train' dataset only (in a list).

    Args:
        train: A SleepStudyDataset object
        val:   A SleepStudyDataset object

    Returns:
        A list of 1 SleepStudyDataset object storing data from both 'train'
        and 'val'
    """
    train.add_pairs(val.pairs)
    train._identifier = train.identifier + "_" + val.identifier.split("/")[-1]
    train.log()
    return [train]


def get_samples_per_epoch(train_seq, max_train_samples_per_epoch):
    """
    Returns the number of samples to take from the training sequence objects
    for 1 epoch to be considered finished.

    Specifically, the specified number of training 'samples' in args
    (number of sleep 'epochs'/'segments'/'periods') will be divided by the
    total number of such segments that the model takes as input in each pass
    I.e. if train_samples_per_epoch is set to 100 for a model which considers
    10 epochs at a time, this function will return 100/10 = 10 steps per train
    epoch.

    Note: The (non-standardized) training samples is upper bounded by the
    total number of samples in the dataset.

    Args:
        train_seq:                  (Sequence) The training Sequence or
                                               MultiSequence object
        max_train_samples_per_epoch (int)      Maximum number of samples for
                                               training. The actual number will
                                               be the lesser of this value and
                                               the total number of samples.

    Returns:
        Number of samples to take in training and validation
    """
    try:
        total_periods = train_seq.total_periods
    except (NotLoadedError, TypeError):
        # train_seq.total_period is not available (not all samples loaded or limitation queue). Use estimate.
        n_studies = train_seq.num_pairs
        total_periods = 2000 * n_studies
        logger.warning(f"Property 'total_periods' not available on sequence {train_seq}. "
                       f"Using (over)estimate total periods of {total_periods} based on dataset length of {n_studies}.")
    train_samples_per_epoch = min(total_periods, max_train_samples_per_epoch)
    if train_seq.margin:
        # For sequence models, we only sample a number of batches to cover
        # all data in once (in expectation).
        m = train_seq.margin*2+1
        train_samples_per_epoch = int(train_samples_per_epoch / m)
    return train_samples_per_epoch


def get_lr_at_epoch(epoch, log_dir):
    """
    TODO
    """
    log_path = os.path.join(log_dir, "training.csv")
    if not os.path.exists(log_path):
        print("No training.csv file found at %s. Continuing with default "
              "learning rate found in parameter file." % log_dir)
        return None, None
    df = pd.read_csv(log_path)
    possible_names = ("lr", "LR", "learning_rate", "LearningRate")
    try:
        in_df = [l in df.columns for l in possible_names].index(True)
    except ValueError:
        return None, None
    col_name = possible_names[in_df]
    return float(df[col_name][int(epoch)]), col_name


def clear_csv_after_epoch(epoch, csv_file):
    """
    TODO
    """
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
        except pd.errors.EmptyDataError:
            # Remove the file
            os.remove(csv_file)
            return
        # Remove any trailing runs and remove after 'epoch'
        try:
            df = df[np.flatnonzero(df["epoch"] == 0)[-1]:]
        except IndexError:
            pass
        df = df[:epoch+1]
        # Save again
        with open(csv_file, "w") as out_f:
            out_f.write(df.to_csv(index=False))


def get_last_epoch(csv_file):
    """
    TODO
    """
    epoch = 0
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        epoch = int(df["epoch"].to_numpy()[-1])
    return epoch


def save_final_weights(project_dir, model, file_name):
    """
    Saves the current (normally 'final') weights of 'model' to h5 archive at
    path project_dir/model/'file_name'.

    If a model of the same name exists, it will be overwritten.
    If the directory 'project_dir'/model does not exist, it will be created.

    Args:
        project_dir: (string)         Path to the project directory
        model:       (tf.keras Model) The model instance which weights will be
                                      saved.
        file_name:   (string)         Name of the saved parameter file
    """
    # Save final model weights
    if not os.path.exists("%s/model" % project_dir):
        os.mkdir("%s/model" % project_dir)
    model_path = "{}/model/{}.h5".format(project_dir,
                                         os.path.splitext(file_name)[0])
    logger.info(f"Saving current model to: {model_path}")
    if os.path.exists(model_path):
        os.remove(model_path)
    model.save_weights(model_path)
