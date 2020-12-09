"""
A set of functions for needed for running training in various settings
"""

from mpunet.logging.default_logger import ScreenLogger


def get_train_and_val_datasets(hparams, no_val, train_on_val, logger):
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
        logger:       (Logger)      A Logger object

    Returns:
        A list of (training, validation) pairs of SleepStudyData objects as
        described in 'hparams'.
    """
    if no_val:
        load = ("train_data",)
        if train_on_val:
            raise ValueError("Should not specify --no_val with --train_on_val")
    else:
        load = ("train_data", "val_data")
    from utime.utils.scriptutils import get_splits_from_all_datasets
    datasets = [*get_splits_from_all_datasets(hparams, load, logger)]
    if train_on_val:
        if any([len(ds) != 2 for ds in datasets]):
            raise ValueError("Did not find a validation set for one or more "
                             "pairs in {}".format(datasets))
        logger("[OBS] Merging training and validation sets")
        datasets = [merge_train_and_val(*ds) for ds in datasets]
        no_val = True
    return datasets, no_val


def get_generators(datasets, hparams, no_val):
    """
    Takes a list of (training, validation) dataset pairs and returns a training
    and validation sequence object (see utime.sequences). If no_val=True,
    returns only a training sequencer.

    With multiple training sequence objects, a MultiSequence object is returned
    which is a light wrapper around multiple sequences that samples uniformly
    across them.

    A ValidationMultiSequence is always returned, even for 1
    dataset, as this data structure is expected in the Validation callback.

    Args:
        datasets: (list)        A list of (training, validation) pairs of
                                SleepStudyData objects or (training,) if no_val
                                is True
        hparams:  (YAMLHParams) The hyperparameters to init the sequencers with
        no_val:   (bool)        Indicates that no validation is to be performed
                                and that entries in 'datasets' are of length 1.

    Returns:
        A training Sequence or MultiSequence objects
        A ValidatonMultiSequence object if no_val=False, otherwise None
    """
    from utime.sequences import MultiSequence, ValidationMultiSequence
    train_seqs = [d[0].get_batch_sequence(random_batches=True,
                                          **hparams["fit"]) for d in datasets]
    val_seq = None
    if not no_val:
        random_val_batches = True  # Currently only option supported
        val_seq = [d[1].get_batch_sequence(random_batches=random_val_batches,
                                           **hparams["fit"]) for d in datasets]
    if len(train_seqs) > 1:
        assert len(val_seq) == len(train_seqs)
        train_seq = MultiSequence(train_seqs, hparams['fit']['batch_size'],
                                  logger=train_seqs[0].logger)
    else:
        train_seq = train_seqs[0]
    if val_seq:
        val_seq = ValidationMultiSequence(val_seq, logger=train_seq.logger)
    return train_seq, val_seq


def find_and_set_gpus(gpu_mon, force_GPU, num_GPUs):
    """
    Given a mpunet GPUMonitor object and the parsed command-line
    arguments, either looks for free GPUs and sets them, or sets a forced
    GPU visibility.

    Specifically, if args.force_GPU is set, set the visibility accordingly,
    count the number of GPUs set and return this number.
    If not, use args.num_GPUs currently available GPUs and return args.num_GPUs

    Args:
        gpu_mon:   (GPUMonitor) Initialized GPUMonitor
        force_GPU: (string)     A CUDA_VISIBLE_DEVICES type string to be set
        num_GPUs:  (int)        Number of free/available GPUs to automatically
                                select using 'gpu_mon' when 'force_GPU' is not
                                set.

    Returns:
        (int) The actual number of GPUs now visible
    """
    if not force_GPU:
        gpu_mon.await_and_set_free_GPU(N=num_GPUs, stop_after=True)
    else:
        gpu_mon.set_GPUs = force_GPU
    return gpu_mon.num_currently_visible


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
    train.pairs.extend(val.pairs)
    train._identifier = train.identifier + "_" + val.identifier.split("/")[-1]
    train.log()
    return [train]


def get_samples_per_epoch(train_seq,
                          max_train_samples_per_epoch,
                          val_samples_per_epoch):
    """
    Returns the number of samples to take from the training and validation
    sequence objects for 1 epoch to be considered finished.

    Specifically, the specified number of training and validation 'samples' in
    args (number of sleep 'epochs'/'segments'/'periods') will be divided by the
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
        val_samples_per_epoch       (int)      Number of samples for validation

    Returns:
        Number of samples to take in training and validation
    """
    train_samples_per_epoch = min(train_seq.total_periods,
                                  max_train_samples_per_epoch)
    if train_seq.margin:
        # For sequence models, we only sample a number of batches to cover
        # all data in once (in expectation).
        m = train_seq.margin*2+1
        train_samples_per_epoch = int(train_samples_per_epoch / m)
        val_samples_per_epoch = int(val_samples_per_epoch / m)
    return train_samples_per_epoch, val_samples_per_epoch


def save_final_weights(project_dir, model, file_name, logger=None):
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
        logger:      (Logger)         Optional Logger instance
    """
    import os
    # Save final model weights
    if not os.path.exists("%s/model" % project_dir):
        os.mkdir("%s/model" % project_dir)
    model_path = "{}/model/{}.h5".format(project_dir,
                                         os.path.splitext(file_name)[0])
    logger = logger or ScreenLogger()
    logger("Saving current model to: %s" % model_path)
    if os.path.exists(model_path):
        os.remove(model_path)
    model.save_weights(model_path)
