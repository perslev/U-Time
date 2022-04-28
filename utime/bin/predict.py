"""
Script which predicts on a set of data and saves the results to disk.
Comparable to bin/evaluate.py except ground truth data is not needed as
evaluation is not performed.
Can also be used to predict on (a) individual file(s) outside of the datasets
originally described in the hyperparameter files.
"""

import logging
import os
import numpy as np
from argparse import ArgumentParser
from utime import Defaults
from utime.utils.system import find_and_set_gpus
from utime.bin.evaluate import (predict_on,
                                prepare_output_dir, get_and_load_model,
                                get_and_load_one_shot_model, get_sequencer,
                                get_out_dir)
from psg_utils.io.channels import filter_non_available_channels
from utime.utils.scriptutils import add_logging_file_handler, with_logging_level_wrapper

logger = logging.getLogger(__name__)


def get_argparser():
    """
    Returns an argument parser for this script
    """
    parser = ArgumentParser(description='Predict using a U-Time model.')
    parser.add_argument("--folder_regex", type=str, required=False,
                        help='Regex pattern matching files to predict on. '
                             'If not specified, prediction will be launched '
                             'on the test_data as specified in the '
                             'hyperparameter file.')
    parser.add_argument("--data_per_prediction", type=int, default=None,
                        help='Number of samples that should make up each sleep'
                             ' stage scoring. Defaults to sample_rate*30, '
                             'giving 1 segmentation per 30 seconds of signal. '
                             'Set this to 1 to score every data point in the '
                             'signal.')
    parser.add_argument("--channels", nargs='*', type=str, default=None,
                        help="A list of channels to use instead of those "
                             "specified in the parameter file.")
    parser.add_argument("--majority", action="store_true",
                        help="Output a majority vote across channel groups in addition "
                             "to the individual channels.")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Optional space separated list of datasets of those stored in the hparams "
                             "file that prediction should be performed on. Ignored when --folder_regex is set. "
                             "Default is 'None' in which case all datasets are predicted on.")
    parser.add_argument("--data_split", type=str, default="test_data",
                        help="Which split of data of those stored in the "
                             "hparams file should the prediction be performed "
                             "on. Ignored when --folder_regex is set.")
    parser.add_argument("--out_dir", type=str, default="predictions",
                        help="Output folder to store results")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs to use for this job")
    parser.add_argument("--strip_func", type=str, default=None,
                        help="Use a different strip function from the one "
                             "specified in the hyperparameters file")
    parser.add_argument("--num_test_time_augment", type=int, default=0,
                        help="Number of prediction passes over each sleep "
                             "study with augmentation enabled.")
    parser.add_argument("--one_shot", action="store_true",
                        help="Segment each SleepStudy in one forward-pass "
                             "instead of using (GPU memory-efficient) sliding "
                             "window predictions.")
    parser.add_argument("--save_true", action="store_true",
                        help="Save the true labels matching the predictions "
                             "(will be repeated if --data_per_prediction is "
                             "set to a non-default value)")
    parser.add_argument("--force_gpus", type=str, default="")
    parser.add_argument("--no_argmax", action="store_true",
                        help="Do not argmax prediction volume prior to save.")
    parser.add_argument("--weights_file_name", type=str, required=False,
                        help="Specify the exact name of the weights file "
                             "(located in <project_dir>/model/) to use.")
    parser.add_argument("--continue_", action="store_true", 
                        help="Skip already predicted files.")
    parser.add_argument("--overwrite", action='store_true',
                        help='Overwrite previous results at the output folder and previous log files')
    parser.add_argument("--log_file", type=str, default="prediction_log",
                        help="Relative path (from Defaults.LOG_DIR as specified by ut --log_dir flag) of "
                             "output log file for this script. "
                             "Set to an empty string to not save any logs to file for this run. "
                             "Default is 'prediction_log'")
    return parser


def assert_args(args):
    """ Not yet implemented """
    pass


def set_new_strip_func(dataset_hparams, strip_func):
    if 'strip_func' not in dataset_hparams:
        dataset_hparams['strip_func'] = {}
    dataset_hparams['strip_func'] = {'strip_func': strip_func}


def get_prediction_channel_sets(sleep_study, dataset):
    """
    TODO

    Args:
        sleep_study:
        dataset:

    Returns:

    """
    # If channel_groups are set in dataset.misc, run on all pairs of channels
    channel_groups = dataset.misc.get('channel_groups')
    if channel_groups and hasattr(sleep_study, 'psg_file_path'):
        channel_groups = filter_non_available_channels(
            channel_groups=channel_groups,
            psg_file_path=sleep_study.psg_file_path
        )
        channel_groups = [c.original_names for c in channel_groups]
        # Return all combinations
        from itertools import product
        combinations = product(*channel_groups)
        return [
            ("+".join(c), c) for c in combinations
        ]
    elif channel_groups:
        raise NotImplementedError("Cannot perform channel group predictions "
                                  "on sleep study objects that have no "
                                  "psg_file_path attribute. "
                                  "Not yet implemented.")
    else:
        # Use default select channels
        return [(None, None)]


def get_datasets(hparams, args):
    from utime.utils.scriptutils import (get_dataset_from_regex_pattern,
                                         get_dataset_splits_from_hparams,
                                         get_all_dataset_hparams)
    # Get dictonary of dataset IDs to hparams
    all_dataset_hparams = get_all_dataset_hparams(hparams)

    # Make modifications to the hparams before dataset init if needed
    for dataset_id, dataset_hparams in all_dataset_hparams.items():
        if args.strip_func:
            # Replace the set strip function
            set_new_strip_func(dataset_hparams, args.strip_func)
        # Check if channel sampling groups are set
        channel_groups = dataset_hparams.get('channel_sampling_groups')
        if channel_groups:
            # Add the channel groups to a separate field, handled at pred. time
            # Make sure all available channels are available in the misc attr.
            del dataset_hparams['channel_sampling_groups']
            dataset_hparams['misc'] = {'channel_groups': channel_groups}

    if args.folder_regex:
        # We predict on a single dataset, specified by the folder_regex arg
        # We load the dataset hyperparameters of one of those specified in
        # the stored hyperparameter files and use it as a guide for how to
        # handle this new, undescribed dataset
        dataset_hparams = list(all_dataset_hparams.values())[0]
        datasets = [(get_dataset_from_regex_pattern(args.folder_regex,
                                                    hparams=dataset_hparams),)]
    else:
        # predict on datasets described in the hyperparameter files
        datasets = []
        for dataset_id, dataset_hparams in all_dataset_hparams.items():
            if not args.datasets or dataset_id in args.datasets:
                datasets.append(get_dataset_splits_from_hparams(
                    hparams=dataset_hparams,
                    splits_to_load=(args.data_split,),
                    id=dataset_id
                ))
        if len(datasets) == 0:
            raise RuntimeError(f"Cannot run prediction on 0 datasets. "
                               f"No datasets left with --datasets {args.datasets} and datasets in "
                               f"hparams: {list(all_dataset_hparams.keys())}")
    return datasets


def predict_study(sleep_study_pair, seq, model, model_func, num_test_time_augment=0, no_argmax=False):
    # Predict
    with sleep_study_pair.loaded_in_context():
        y, pred = predict_on(study_pair=sleep_study_pair,
                             seq=seq,
                             model=model,
                             model_func=model_func,
                             n_aug=num_test_time_augment,
                             argmax=False)
    org_pred_shape = pred.shape
    if callable(getattr(pred, "numpy", None)):
        pred = pred.numpy()
    pred, y = pred.reshape(-1, 5), y.reshape(-1, 1)
    if not no_argmax:
        pred = pred.argmax(-1)
    return pred, y, org_pred_shape


def get_save_path(out_dir, file_name, sub_folder_name=None):
    # Get paths
    if sub_folder_name is not None:
        out_dir_pred = os.path.join(out_dir, sub_folder_name)
    else:
        out_dir_pred = out_dir
    out_path = os.path.join(out_dir_pred, file_name)
    return out_path


def save_file(path, arr, argmax):
    path = os.path.abspath(path)
    dir_ = os.path.split(path)[0]
    os.makedirs(dir_, exist_ok=True)
    if argmax:
        arr = arr.argmax(-1)
    logger.info(f"Saving array of shape {arr.shape} to {path}")
    np.save(path, arr)


def get_updated_majority_voted(majority_voted, pred):
    if majority_voted is None:
        majority_voted = pred.copy()
    else:
        majority_voted += pred
    return majority_voted


def run_pred_on_channels(sleep_study_pair, seq, model, model_func, num_test_time_augment=0):
    pred, y, org_pred_shape = predict_study(
        sleep_study_pair=sleep_study_pair,
        seq=seq,
        model=model,
        model_func=model_func,
        num_test_time_augment=num_test_time_augment,
        no_argmax=True
    )
    if len(org_pred_shape) == 3:
        y = np.repeat(y, org_pred_shape[1])
    return pred, y


def run_pred_on_pair(sleep_study_pair, seq, model, model_func, out_dir, channel_sets, args):
    majority_voted = None
    path_mj = get_save_path(out_dir, sleep_study_pair.identifier + "_PRED.npy", "majority")
    path_true = get_save_path(out_dir, sleep_study_pair.identifier + "_TRUE.npy", None)
    for k, (sub_folder_name, channels_to_load) in enumerate(channel_sets):
        # Get prediction out path
        path_pred = get_save_path(out_dir, sleep_study_pair.identifier + "_PRED.npy", sub_folder_name)

        # If not --overwrite set, and path exists, we skip it here
        if os.path.exists(path_pred) and not args.overwrite:
            logger.info(f"Skipping (channels={channels_to_load}) - already exists and --overwrite not set.")
            # Load and increment the majority_voted array before continue
            majority_voted = get_updated_majority_voted(majority_voted, np.load(path_pred))
            continue

        # Load and predict on the set channels
        if channels_to_load:
            logger.info(f" -- Channels: {channels_to_load}")
            sleep_study_pair.select_channels = channels_to_load

        # Get the prediction and true values
        pred, y = run_pred_on_channels(
            sleep_study_pair=sleep_study_pair,
            seq=seq,
            model=model,
            model_func=model_func,
            num_test_time_augment=args.num_test_time_augment
        )
        # Sum the predictions into the majority_voted array
        majority_voted = get_updated_majority_voted(majority_voted, pred)

        if args.save_true and not os.path.exists(path_true):
            # Save true to disk, only save once if multiple channel sets
            # Note that we save the true values to the folder storing
            # results for each channel if multiple channel sets
            save_file(path_true, arr=y, argmax=False)
        # Save prediction
        save_file(path_pred, arr=pred, argmax=not args.no_argmax)
    if args.majority:
        if not os.path.exists(path_mj) or args.overwrite:
            save_file(path_mj, arr=majority_voted, argmax=not args.no_argmax)
        else:
            logger.info("Skipping (channels=MAJORITY) - already exists and --overwrite not set.")


def run_pred(dataset,
             out_dir,
             model,
             model_func,
             hparams,
             args):
    """
    Run prediction on a all entries of a SleepStudyDataset

    Args:
        dataset:     A SleepStudyDataset object storing one or more SleepStudy
                     objects
        out_dir:     Path to directory that will store predictions and
                     evaluation results
        model:       An initialized model used for prediction
        model_func:  A callable that returns an initialized model for pred.
        hparams:     An YAMLHparams object storing all hyperparameters
        args:        Passed command-line arguments
    """
    logger.info(f"\nPREDICTING ON {len(dataset.pairs)} STUDIES")
    seq = get_sequencer(dataset, hparams)

    # Predict on all samples
    for i, sleep_study_pair in enumerate(dataset):
        logger.info(f"[{i+1}/{len(dataset)}] Predicting on SleepStudy: {sleep_study_pair.identifier}")

        # Get list of channel sets to predict on
        channel_sets = get_prediction_channel_sets(sleep_study_pair, dataset)
        if len(channel_sets) > 20:
            logger.info(f"OBS: Many ({len(channel_sets)}) combinations of channels in channel "
                        f"groups. Prediction for this study may take a while.")
        if len(channel_sets) == 0:
            logger.info(f"Found no valid channel sets for study {sleep_study_pair}. Skipping study.")
        else:
            run_pred_on_pair(
                sleep_study_pair=sleep_study_pair,
                seq=seq,
                model=model,
                model_func=model_func,
                out_dir=out_dir,
                channel_sets=channel_sets,
                args=args
            )


def run(args):
    """
    Run the script according to args - Please refer to the argparser.
    """
    assert_args(args)
    logger.info(f"Args dump: \n{vars(args)}")
    # Check project folder is valid
    from utime.utils.scriptutils import assert_project_folder
    project_dir = os.path.abspath(Defaults.PROJECT_DIRECTORY)
    assert_project_folder(project_dir, evaluation=True)

    # Prepare output dir
    if not args.folder_regex:
        out_dir = get_out_dir(args.out_dir, args.data_split)
    else:
        out_dir = args.out_dir
    prepare_output_dir(out_dir, True)

    # Get hyperparameters and init all described datasets
    from utime.hyperparameters import YAMLHParams
    hparams = YAMLHParams(Defaults.get_hparams_path(project_dir))
    hparams["build"]["data_per_prediction"] = args.data_per_prediction
    if args.channels:
        hparams["select_channels"] = args.channels
        hparams["channel_sampling_groups"] = None
        logger.info(f"Evaluating using channels {args.channels}")

    # Get model
    find_and_set_gpus(args.num_gpus, args.force_gpus)
    model, model_func = None, None
    if args.one_shot:
        # Model is initialized for each sleep study later
        def model_func(n_periods):
            return get_and_load_one_shot_model(n_periods, project_dir, hparams, args.weights_file_name)
        model_func = with_logging_level_wrapper(model_func, logging.ERROR)
    else:
        model = get_and_load_model(project_dir, hparams, args.weights_file_name)

    # Run pred on all datasets
    for dataset in get_datasets(hparams, args):
        dataset = dataset[0]
        if "/" in dataset.identifier:
            # Multiple datasets, separate results into sub-folders
            ds_out_dir = os.path.join(out_dir,
                                      dataset.identifier.split("/")[0])
            if not os.path.exists(ds_out_dir):
                os.mkdir(ds_out_dir)
        else:
            ds_out_dir = out_dir
        logger.info(f"[*] Running eval on dataset {dataset}\n"
                    f"    Out dir: {ds_out_dir}")
        run_pred(dataset=dataset,
                 out_dir=ds_out_dir,
                 model=model,
                 model_func=model_func,
                 hparams=hparams,
                 args=args)


def entry_func(args=None):
    # Parse command line arguments
    parser = get_argparser()
    args = parser.parse_args(args)
    add_logging_file_handler(args.log_file, args.overwrite, mode="w" if not args.continue_ else "a")
    run(args)


if __name__ == "__main__":
    entry_func()
