"""
A set of functions for needed for running training in various settings
"""

import os
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
        A list of training SleepStudyDataset objects
        A list of validation SleepStudyDataset objects, or [] if not val.
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
    if not no_val:
        train_datasets, val_datasets = zip(*datasets)
    else:
        train_datasets = [d[0] for d in datasets]
        val_datasets = []
    return train_datasets, val_datasets


def get_h5_train_and_val_datasets(hparams, no_val, train_on_val, logger):
    """
    TODO

    Args:
        hparams:      (YAMLHParams) A hyperparameter object to load dataset
                                    configurations from.
        no_val:       (bool)        Do not load validation data
        train_on_val: (bool)        Load validation data, but merge it into
                                    the training data. Then return only the
                                    'trainin' (train+val) dataset.
        logger:       (Logger)      A Logger object

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

    from utime.dataset.sleep_study_dataset import SingleH5Dataset
    from utime.utils.scriptutils import get_all_dataset_hparams
    data_hparams = get_all_dataset_hparams(hparams)

    h5_dataset = None
    train_datasets, val_datasets = [], []
    for dataset_id, hparams in data_hparams.items():
        if h5_dataset is None:
            h5_dataset = SingleH5Dataset(hparams['train_data']['data_dir'],
                                         logger=logger)
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
        from utime.utils.scriptutils import select_sample_strip_scale_quality
        select_sample_strip_scale_quality(*ds, hparams=hparams, logger=logger)

    return train_datasets, val_datasets


def get_data_queues(datasets,
                    queue_type,
                    max_loaded_per_dataset,
                    num_access_before_reload,
                    logger,
                    study_loader=None):
    """
    TODO.

    Args:

    Returns:

    """
    from utime.dataset.queue import (StudyLoader, LimitationQueue,
                                     LazyQueue, EagerQueue)
    map_ = {'eager': EagerQueue,
            'lazy': LazyQueue,
            'limitation': LimitationQueue}
    queue_type = map_[queue_type.lower()]
    logger("Using data queue type:", queue_type.__name__)

    if queue_type is LimitationQueue:
        if study_loader is None:
            logger("Creating study loader...")
            # Get loader for limitation queue(s)
            max_loaded = (max_loaded_per_dataset or 0) * len(datasets)
            study_loader = StudyLoader(n_threads=7,
                                       max_queue_size=max_loaded or None,
                                       logger=logger)
    else:
        study_loader = None

    dataset_queues = []
    for dataset in datasets:
        if max_loaded_per_dataset >= len(dataset) and queue_type is LimitationQueue:
            # TODO: Implement load/access_time_random_channel_selector for EagerQueue, see NotImplementedError below.
            logger.warn(f"Replacing queue type '{queue_type.__name__}' for dataset {dataset} with queue type "
                        f"'{EagerQueue.__name__}' (because max_loaded_per_dataset = {max_loaded_per_dataset} "
                        f">= len(dataset) = {len(dataset)})")
            queue_type = EagerQueue
        if queue_type is EagerQueue and \
                (any([getattr(ss, 'load_time_random_channel_selector') or
                      getattr(ss, 'access_time_random_channel_selector') for ss in dataset])):
            raise NotImplementedError(
                "The 'eager' data loading queue currently does not support datasets with "
                "the 'load_time_channel_sampling_groups' or "
                "'access_time_channel_sampling_groups' attributes set. "
                "If you want to train using random channel combinations, either "
                "pre-process the data using the 'ut preprocess' command and then re-run "
                "training using 'ut train --preprocessed', or run training with the "
                "limitation queue loader using the '--train_queue_type "
                "limitation' command."
            )
        dataset_queues.append(queue_type(
            dataset=dataset,
            max_loaded=max_loaded_per_dataset,
            num_access_before_reload=num_access_before_reload,  # TODO
            preload_now=True,
            await_preload=False,
            study_loader=study_loader,
            logger=logger
        ))
    if study_loader:
        study_loader.join()
    return dataset_queues


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
    from utime.sequences import (MultiSequence, ValidationMultiSequence,
                                 get_batch_sequence)

    n_classes = hparams.get_from_anywhere('n_classes')
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
        train_seq = MultiSequence(train_seqs, hparams['fit']['batch_size'],
                                  logger=train_seqs[0].logger)
    else:
        train_seq = train_seqs[0]
    if val_seq:
        assert len(val_seq) == len(train_seqs)
        val_seq = ValidationMultiSequence(val_seq, logger=train_seq.logger)
    return train_seq, val_seq


def find_and_set_gpus(gpu_mon, force_GPU, num_GPUs):
    """
    Given a MultiPlanarUnet GPUMonitor object and the parsed command-line
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
    from utime.errors import NotLoadedError
    try:
        train_samples_per_epoch = min(train_seq.total_periods,
                                      max_train_samples_per_epoch)
    except NotLoadedError:
        train_samples_per_epoch = max_train_samples_per_epoch
    if train_seq.margin:
        # For sequence models, we only sample a number of batches to cover
        # all data in once (in expectation).
        m = train_seq.margin*2+1
        train_samples_per_epoch = int(train_samples_per_epoch / m)
    return train_samples_per_epoch


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
