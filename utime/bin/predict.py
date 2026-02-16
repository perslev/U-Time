"""
Script which predicts on a set of data and saves the results to disk.
Comparable to bin/evaluate.py except ground truth data is not needed as
evaluation is not performed.
Can also be used to predict on (a) individual file(s) outside of the datasets
originally described in the hyperparameter files.
"""

import logging
import os
from pathlib import Path
import numpy as np
import traceback
import shutil
import json
from argparse import ArgumentParser
from utime import Defaults
from utime.utils.system import find_and_set_gpus
from utime.bin.evaluate import (predict_on,
                                prepare_output_dir, get_and_load_model,
                                get_and_load_one_shot_model, get_sequencer,
                                get_out_dir)
from utime.hyperparameters import YAMLHParams
from psg_utils.io.channels import filter_non_available_channels
from psg_utils.io.channels.utils import get_channel_group_combinations
from psg_utils.errors import CouldNotLoadError
from psg_utils.io.header import extract_header
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
                             ' stage scoring. Defaults to sample_rate*period_length_sec, '
                             'giving 1 segmentation per period_length_sec seconds of signal. '
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
    parser.add_argument("--filter_settings", type=json.loads, default=None,
                        help="Use a different set of filtering settings from the one "
                             "specified in the hyperparameters file. You must pass a JSON string, e.g.: "
                             "\"{'l_freq': 0.3, 'h_freq': 35}\" for a 0.3-35 band-pass filter.")
    parser.add_argument("--notch_filter_settings", type=json.loads, default=None,
                        help="Use a different set of notch filtering settings from the one "
                             "specified in the hyperparameters file. You must pass a JSON string, e.g.: "
                             "\"{'freqs': 50\" for a 50 Hz notch filter.")
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
    parser.add_argument("--weight_channel", type=str, default=None,
                        help="Name of a channel in the PSG file to extract as weights. "
                             "The channel will be resampled to match prediction sample rate "
                             "and downsampled to match prediction length (one value per period). "
                             "Only valid when --no_argmax is used. Weight array will be saved "
                             "as {study_id}_WEIGHT.npy alongside predictions.")
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
    parser.add_argument("--move_study_to_folder_on_error", type=str, default=None,
                        help="Optional path to a folder to which sleep study subject directories should be "
                             "moved (including all content files) when the study cannot be loaded for 1 or more "
                             "of the requested channel combinations. E.g., used to filter out all studies that need "
                             "further manual investigation for data processing/loading issues.")
    parser.add_argument("--group_classes", type=str, default=None,
                        help="Specify how to group classes by summing probabilities. A comma-separated list of class mappings"
                             " in format 'source1:target1,source2:target2'. For example, '2:1,3:1' will sum probabilities "
                             "of classes 2 and 3 into class 1.")
    
    # Weights & Biases (wandb) arguments
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B logging. Creates a new run for prediction. "
                             "Use --wandb-run-id to resume an existing training run instead.")
    parser.add_argument("--wandb-run-id", type=str, default=None,
                        help="W&B run ID to resume and log prediction results to. "
                             "If provided, resumes the existing run for logging.")
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B project name")
    parser.add_argument("--wandb-group", type=str, default=None,
                        help="W&B group name (groups related prediction runs)")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="W&B run name")
    parser.add_argument("--wandb-tags", nargs='*', type=str, default=None,
                        help="W&B tags for the prediction run (space-separated)")
    parser.add_argument("--wandb-save-artifact", action="store_true",
                        help="If set, save prediction outputs as a W&B artifact (uploads the output directory).")
    parser.add_argument("--wandb-artifact-name", type=str, default=None,
                        help="Optional W&B artifact name (default: '<run_id>-predictions').")
    parser.add_argument("--wandb-artifact-type", type=str, default="predictions",
                        help="W&B artifact type (default: 'predictions').")

    # Optional output structuring arguments
    parser.add_argument(
        "--split_study_suffix_to_subdir",
        action="store_true",
        help="If set, and a study identifier ends with <sep><suffix> (e.g. '123_LEFT'), "
             "predictions are saved to '<out_dir>/<suffix>/' and file names use the base id "
             "(e.g. '123_PRED.npy') instead of the full identifier."
    )
    parser.add_argument(
        "--study_suffixes",
        nargs="*",
        type=str,
        default=["LEFT", "RIGHT"],
        help="Suffix tokens that trigger --split_study_suffix_to_subdir. "
             "Matched against the final token after splitting by --study_suffix_sep. "
             "Default: LEFT RIGHT."
    )
    parser.add_argument(
        "--study_suffix_sep",
        type=str,
        default="_",
        help="Separator used when parsing --study_suffixes from study identifiers. Default: '_'"
    )
    parser.add_argument(
        "--study_suffix_case_sensitive",
        action="store_true",
        help="If set, suffix matching for --study_suffixes is case sensitive. Default is case-insensitive."
    )
    
    return parser


def assert_group_classes(group_classes, n_classes):

    if group_classes is None:
        return

    try:
        group_map = {}
        for pair in group_classes.split(","):
            source, target = map(int, pair.split(":"))

            if source < 0 or source >= n_classes:
                raise ValueError(f"Invalid group_classes. Source class index out of range. "
                                f"Expected range: 0 to {n_classes-1}. Got: {source}")

            group_map[source] = target
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Invalid group_classes format. Expected 'source:target,...' "
                        f"(e.g. '2:1,3:1'). Got: {group_classes}") from e

    if len(group_map) != n_classes:
        raise ValueError(f"Invalid group_classes. Number of classes in group_classes does not match number of classes in the model. "
                        f"Expected: {n_classes}. Got: {len(group_map)}")

    target_classes = set(group_map.values())
    if target_classes - set(range(len(target_classes))):
        raise ValueError(f"Invalid group_classes. Expected target classes to be in range 0 to {len(target_classes)-1}. Got: {target_classes - set(range(len(target_classes)))}")

    logger.info(f"Grouping classes: {group_map}")
    return group_map

def set_new_strip_func(dataset_hparams, strip_func):
    if 'strip_func' not in dataset_hparams:
        dataset_hparams['strip_func'] = {}
    dataset_hparams['strip_func'] = {'strip_func': strip_func}


def set_new_filter_settings(dataset_hparams, filter_settings):
    if 'filter_settings' not in dataset_hparams:
        dataset_hparams['filter_settings'] = {}
    dataset_hparams['filter_settings'] = filter_settings


def set_new_notch_filter_settings(dataset_hparams, notch_filter_settings):
    if 'notch_filter_settings' not in dataset_hparams:
        dataset_hparams['notch_filter_settings'] = {}
    dataset_hparams['notch_filter_settings'] = notch_filter_settings


def _split_study_identifier_suffix(identifier: str,
                                  sep: str,
                                  suffixes,
                                  case_sensitive: bool):
    """
    Split a study identifier like '<base><sep><suffix>' into (base, suffix).
    Returns (identifier, None) if no match.
    """
    if not identifier or not sep or not suffixes:
        return identifier, None
    if sep not in identifier:
        return identifier, None

    base, suffix = identifier.rsplit(sep, 1)
    if not base or not suffix:
        return identifier, None

    if case_sensitive:
        suffix_set = set(suffixes)
        return (base, suffix) if suffix in suffix_set else (identifier, None)
    else:
        suffix_set = {s.lower() for s in suffixes}
        return (base, suffix) if suffix.lower() in suffix_set else (identifier, None)


def get_save_out_dir_and_stem(out_dir: str, identifier: str, args):
    """
    Decide where to save outputs and what filename stem to use.

    Important: We do NOT modify sleep_study_pair.identifier, as that may be used
    internally for data loading (sequencer/dataset keys). This function only
    affects output structure and file naming.
    """
    if not getattr(args, "split_study_suffix_to_subdir", False):
        return out_dir, identifier

    base, suffix = _split_study_identifier_suffix(
        identifier=identifier,
        sep=getattr(args, "study_suffix_sep", "_"),
        suffixes=getattr(args, "study_suffixes", None),
        case_sensitive=getattr(args, "study_suffix_case_sensitive", False)
    )
    if suffix is None:
        return out_dir, identifier
    return os.path.join(out_dir, suffix), base


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
        # Return all combinations except unordered duplicates ([[EEG 1, EEG 2], [EEG 2, EEG 1]] -> [[EEG 1, EEG 2]])
        combinations = get_channel_group_combinations(*channel_groups, remove_unordered_duplicates=True)
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
        if args.filter_settings:
            # Replace the set filter settings
            set_new_filter_settings(dataset_hparams, args.filter_settings)
        if args.notch_filter_settings:
            # Replace set set notch filter settings
            set_new_notch_filter_settings(dataset_hparams, args.notch_filter_settings)
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
        pred = predict_on(study_pair=sleep_study_pair,
                             seq=seq,
                             model=model,
                             model_func=model_func,
                             n_aug=num_test_time_augment,
                             argmax=False)
    if callable(getattr(pred, "numpy", None)):
        pred = pred.numpy()
    pred = pred.reshape(-1, pred.shape[-1])
    if not no_argmax:
        pred = pred.argmax(-1)
    return pred


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


def extract_weight_channel(sleep_study_pair, seq, weight_channel_name, data_per_prediction):
    
    try:
        # Extract header to get channel names
        header = extract_header(sleep_study_pair.psg_file_path, sleep_study_pair.header_file_path)
        channel_names = header.get('channel_names', [])
        
        if weight_channel_name not in channel_names:
            logger.warning(f"Weight channel '{weight_channel_name}' not found in PSG file. "
                          f"Available channels: {channel_names}")
            return None
        
        sleep_study_pair.select_channels = [weight_channel_name]
        seq.n_channels = 1
        weight_data = seq.get_single_study_full_seq(sleep_study_pair.identifier)[0].reshape(-1)
        
        # Pad to whole number of periods with zeros if needed
        n_samples = len(weight_data)
        if n_samples % data_per_prediction != 0:
            padding_needed = data_per_prediction - (n_samples % data_per_prediction)
            weight_data = np.pad(weight_data, (0, padding_needed), mode='constant', constant_values=0)
        
        # Reshape to (n_predictions, data_per_prediction) and take mean (or other aggregation)
        weight_data = weight_data.reshape(-1, data_per_prediction)
        weights = np.mean(weight_data, axis=1)

        logger.debug(f"Weights min: {weights.min()}, max: {weights.max()}")
        logger.info(f"Extracted weight channel '{weight_channel_name}' with shape {weights.shape}")

        return weights
    except Exception as e:
        logger.warning(f"Failed to extract weight channel '{weight_channel_name}': {e}")
        return None


def group_class_probabilities(array, group_map):
    
    num_classes_final = len(set(group_map.values()))
    
    # Initialize output array
    grouped_pred = np.zeros((*array.shape[:-1], num_classes_final), dtype=array.dtype)
    
    # Sum probabilities for mapped classes
    for source_class, target_class in group_map.items():
        grouped_pred[..., target_class] += array[..., source_class]
    
    logger.info(f"Grouped probabilities from {array.shape[-1]} to {num_classes_final} classes.")
    
    return grouped_pred


def group_class_labels(array, group_map):

    grouped_array = array.copy()

    for source_class, target_class in group_map.items():
        grouped_array = np.where(array == source_class, target_class, grouped_array)
    
    return grouped_array


def run_pred_on_pair(sleep_study_pair, seq, model, model_func, out_dir, channel_sets, group_map, args):
    majority_voted = None
    save_out_dir, save_stem = get_save_out_dir_and_stem(out_dir, sleep_study_pair.identifier, args)
    if save_stem != sleep_study_pair.identifier:
        logger.info(f"Saving outputs for '{sleep_study_pair.identifier}' as '{save_stem}' under '{save_out_dir}'")

    path_mj = get_save_path(save_out_dir, save_stem + "_PRED.npy", "majority")
    path_true = get_save_path(save_out_dir, save_stem + "_TRUE.npy", None)
    
    with sleep_study_pair.loaded_in_context():
        true = sleep_study_pair.get_all_hypnogram_periods()
        
        weight_array = None
        if args.weight_channel:
            weight_array = extract_weight_channel(
                sleep_study_pair=sleep_study_pair,
                seq=seq,
                weight_channel_name=args.weight_channel,
                data_per_prediction=args.data_per_prediction
            )
    
    for k, (sub_folder_name, channels_to_load) in enumerate(channel_sets):

        path_pred = get_save_path(save_out_dir, save_stem + "_PRED.npy", sub_folder_name)
        path_weight = get_save_path(save_out_dir, save_stem + "_WEIGHT.npy", sub_folder_name)

        if channels_to_load:
            logger.info(f" -- Channels: {channels_to_load}")
            sleep_study_pair.select_channels = channels_to_load
        seq.n_channels = sleep_study_pair.n_channels

        # Get the prediction and true values
        pred = predict_study(
            sleep_study_pair=sleep_study_pair,
            seq=seq,
            model=model,
            model_func=model_func,
            no_argmax=True
        )
        
        # Group classes if needed (for majority vote consistency)
        if group_map:
            pred = group_class_probabilities(pred, group_map)                

        majority_voted = get_updated_majority_voted(majority_voted, pred)

        if not os.path.exists(path_pred) or args.overwrite:
            save_file(path_pred, arr=pred, argmax=not args.no_argmax)
        else:
            logger.info(f"Prediction file already exists at {path_pred}, skipping (use --overwrite to replace)")
        
        if weight_array is not None:
            if not os.path.exists(path_weight) or args.overwrite:
                logger.info(f"Saving weight channel to {path_weight}")
                save_file(path_weight, arr=weight_array, argmax=False)
            else:
                logger.info(f"Weight file already exists at {path_weight}, skipping (use --overwrite to replace)")
    
    if args.save_true:

        if true is None:
            raise ValueError(f"True values are not available for {sleep_study_pair.identifier}")

        if group_map:
            true = group_class_labels(true, group_map)

        if not os.path.exists(path_true) or args.overwrite:
            logger.info(f"Saving true to {path_true}")
            save_file(path_true, arr=true, argmax=False)
        else:
            logger.info(f"True file already exists at {path_true}, skipping (use --overwrite to replace)")
    
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
             group_map,
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
            try:
                run_pred_on_pair(
                    sleep_study_pair=sleep_study_pair,
                    seq=seq,
                    model=model,
                    model_func=model_func,
                    out_dir=out_dir,
                    channel_sets=channel_sets,
                    group_map=group_map,
                    args=args
                )
            except (CouldNotLoadError, RuntimeError) as e:
                logger.error(f"Error on study {sleep_study_pair}: {str(e)}. "
                             f"Traceback: {traceback.format_exc()}")
                if args.move_study_to_folder_on_error:
                    if not os.path.exists(args.move_study_to_folder_on_error):
                        os.makedirs(args.move_study_to_folder_on_error)
                    logger.info(f"Moving study folder {sleep_study_pair.subject_dir} -> "
                                f"{args.move_study_to_folder_on_error}")
                    shutil.move(sleep_study_pair.subject_dir, args.move_study_to_folder_on_error)
                else:
                    raise e


def run(args):
    """
    Run the script according to args - Please refer to the argparser.
    """
    logger.info(f"Args dump: \n{vars(args)}")
    # Check project folder is valid
    from utime.utils.scriptutils import assert_project_folder
    project_dir = os.path.abspath(Defaults.PROJECT_DIRECTORY)
    assert_project_folder(project_dir, evaluation=True)

    # Initialize Weights & Biases if run_id provided
    from utime.utils.wandb_logger import resume_wandb_run, finish_wandb_run, is_wandb_available
    
    wandb_run = None
    if args.wandb or args.wandb_run_id:
        if not is_wandb_available():
            logger.warning("wandb not installed. Install with: pip install wandb")
        else:
            if args.wandb_run_id:
                # Resume existing run
                wandb_run = resume_wandb_run(args.wandb_run_id, args.wandb_project)
                if wandb_run:
                    logger.info(f"Resumed wandb run: {args.wandb_run_id}")
            else:
                # Create new prediction run
                try:
                    import wandb as wandb_module
                    
                    wandb_run = wandb_module.init(
                        project=args.wandb_project,
                        group=args.wandb_group,
                        name=args.wandb_name,
                        tags=args.wandb_tags,
                        job_type="prediction",
                        config={
                            "data_split": args.data_split,
                            "one_shot": args.one_shot,
                            "folder_regex": args.folder_regex,
                            "majority": args.majority
                        }
                    )
                    logger.info(f"Created new wandb prediction run: {wandb_run.name} (ID: {wandb_run.id})")
                    logger.info(f"View at: {wandb_run.url}")
                except Exception as e:
                    logger.warning(f"Failed to initialize wandb: {e}")

    # Prepare output dir
    if not args.folder_regex:
        out_dir = get_out_dir(args.out_dir, args.data_split)
    else:
        out_dir = args.out_dir
    prepare_output_dir(out_dir, True)

    hparams = YAMLHParams(Defaults.get_hparams_path(project_dir))
    hparams["build"]["data_per_prediction"] = args.data_per_prediction
    if args.channels:
        hparams["select_channels"] = args.channels
        hparams["channel_sampling_groups"] = None
        logger.info(f"Evaluating using channels {args.channels}")

    group_map = assert_group_classes(args.group_classes, hparams["build"]["n_classes"])

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

    # Get datasets once (also used for wandb config below)
    datasets = get_datasets(hparams, args)

    # If wandb enabled, capture a rich config snapshot (args + derived values)
    if wandb_run:
        try:
            import wandb
            dataset_identifiers = []
            for ds_tuple in datasets:
                ds = ds_tuple[0]
                dataset_identifiers.append(getattr(ds, "identifier", str(ds)))

            wandb.config.update(
                {
                    **vars(args),
                    # Derived / convenient fields
                    "argmax": (not args.no_argmax),
                    "project_dir": project_dir,
                    "hparams_path": Defaults.get_hparams_path(project_dir),
                    "output_dir": os.path.abspath(out_dir),
                    "datasets_predicted": dataset_identifiers,
                    "n_datasets_predicted": len(dataset_identifiers),
                    "data_per_prediction": hparams["build"].get("data_per_prediction", args.data_per_prediction),
                    "n_classes": hparams["build"].get("n_classes", None),
                },
                allow_val_change=True
            )
        except Exception as e:
            logger.warning(f"Failed to update wandb config with args/derived values: {e}")

    # Run pred on all datasets
    for dataset in datasets:
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
                 group_map=group_map,
                 args=args)
    
    # Log prediction statistics to wandb if enabled
    if wandb_run:
        try:
            import wandb
            # Count prediction files
            pred_files = list(Path(out_dir).rglob("*PRED.npy"))
            n_predictions = len(pred_files)
            true_files = list(Path(out_dir).rglob("*_TRUE.npy"))
            n_true_files = len(true_files)
            
            # Log statistics
            wandb.log({
                "predict/n_predictions": n_predictions,
                "predict/n_true_files": n_true_files,
                "predict/data_split": args.data_split,
                "predict/output_dir": out_dir
            })
            logger.info(f"Logged {n_predictions} prediction(s) to wandb")

            if getattr(args, "wandb_save_artifact", False):
                artifact_name = args.wandb_artifact_name or f"{wandb_run.id}-predictions"
                artifact = wandb.Artifact(
                    name=artifact_name,
                    type=getattr(args, "wandb_artifact_type", "predictions"),
                    metadata={
                        "data_split": args.data_split,
                        "folder_regex": args.folder_regex,
                        "out_dir": os.path.abspath(out_dir),
                        "n_predictions": n_predictions
                    }
                )
                artifact.add_dir(os.path.abspath(out_dir))
                wandb_run.log_artifact(artifact)
                logger.info(f"Logged wandb artifact '{artifact_name}' from {out_dir}")

            # Save true labels as a dedicated artifact when present
            if n_true_files > 0:
                true_artifact_name = f"{wandb_run.id}-true-labels"
                true_artifact = wandb.Artifact(
                    name=true_artifact_name,
                    type="labels",
                    metadata={
                        "data_split": args.data_split,
                        "folder_regex": args.folder_regex,
                        "out_dir": os.path.abspath(out_dir),
                        "n_true_files": n_true_files
                    }
                )
                for f in true_files:
                    true_artifact.add_file(str(f))
                wandb_run.log_artifact(true_artifact)
                logger.info(f"Logged wandb true-label artifact '{true_artifact_name}' ({n_true_files} file(s))")
            
            finish_wandb_run()
        except Exception as e:
            logger.warning(f"Failed to log prediction results to wandb: {e}")


def entry_func(args=None):
    # Parse command line arguments
    parser = get_argparser()
    args = parser.parse_args(args)
    add_logging_file_handler(args.log_file, args.overwrite, mode="w" if not args.continue_ else "a")
    run(args)


if __name__ == "__main__":
    entry_func()
