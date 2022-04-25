import logging
import os
import numpy as np
from pprint import pformat
from collections import namedtuple
from argparse import ArgumentParser, Namespace
from utime import Defaults
from utime.utils.system import find_and_set_gpus
from utime.bin.evaluate import get_and_load_one_shot_model
from psg_utils.dataset.sleep_study import SleepStudy
from psg_utils.hypnogram.utils import dense_to_sparse
from psg_utils.io.channels import infer_channel_types, VALID_CHANNEL_TYPES
from psg_utils.io.channels import auto_infer_referencing as infer_channel_refs
from utime.utils.scriptutils import add_logging_file_handler

logger = logging.getLogger(__name__)


def get_argparser():
    """
    Returns an argument parser for this script
    """
    parser = ArgumentParser(description='Predict using a U-Time model.')
    parser.add_argument("-f", type=str, required=True,
                        help='Path to file to predict on.')
    parser.add_argument("-o", type=str, required=True,
                        help="Output path for storing predictions. "
                             "Valid extensions are '.hyp' (text file with 1 stage (string) per line), "
                             "'.ids' (init-duration-stage (string) format text file) and '.npy' (numpy array "
                             "of shape [N, 1] storing stages (ints)). "
                             "If any other or no extension is specified, '.npy' is assumed.")
    parser.add_argument("--header_file_name", type=str, default=None,
                        help='Optional header file name. Header must be in the same folder of the input file, see -f.')
    parser.add_argument("--logging_out_path", type=str, default=None,
                        help='Optional path to store prediction log. If not set, <out_folder>/<file_name>.log is used.')
    parser.add_argument("--channels", nargs='+', type=str, default=None,
                        required=True,
                        help="A list of channels to use for prediction. "
                             "To predict on multiple channel groups, pass a string where "
                             "each channel in each channel group is separated by '++' and different groups are "
                             "separated by space or '&&'. E.g. to predict on {EEG1, EOG1} and {EEG2, EOG2}, pass "
                             "'EEG1++EOG1' 'EEG2++EOG2'. Each group will be used for prediction once, and the final "
                             "results will be a majority vote across all. "
                             "You may also specify a list of individual channels and use the --auto_channel_grouping to"
                             " predict on all channel group combinations possible by channel types. "
                             "You may optionally also specify channel types using general channel declarations "
                             "['EEG', 'EOG', 'EMG'] which will be considered when using the --auto_channel_grouping "
                             "flag. Use '<channel_name>==<channel_type>', e.g. 'C3-A2==EEG' 'EOGl==EOG'.")
    parser.add_argument("--auto_channel_grouping", nargs="+", type=str, default=None,
                        help="Attempt to automatically group all channels specified with --channels into channel "
                             "groups by types. Pass a string of format '<type_1> <type_2>' (optional && separaters) "
                             "using the general channel types declarations ['EEG', 'EOG', 'EMG']. "
                             "E.g. to predict on all available channel groups with 1 EEG and 1 EOG channel "
                             "(in that order), pass '--auto_channel_grouping=EEG EOG' and all channels to consider "
                             "with the --channels argument. Channel types may be passed with --channels (see above), "
                             "otherwise, channel types are automatically inferred from the channel names. "
                             "Note that not all models are designed to work with all types, e.g. U-Sleep V1.0 "
                             "does not need EMG inputs and should not be passed.")
    parser.add_argument("--auto_reference_types", nargs='+', type=str, default=None,
                        help="Attempt to automatically reference channels to MASTOID typed channels. Pass channel "
                             "types in ['EEG', 'EOG'] for which this feature should be active. E.g., with "
                             "--channels C3 C4 A1 A2 passed and --auto_reference_types EEG set, the referenced "
                             "channels C3-A2 and C4-A1 will be used instead.")
    parser.add_argument("--strip_func", type=str, default='trim_psg_trailing',
                        help="Strip function to use, default = 'trim_psg_trailing'.")
    parser.add_argument("--model", type=str, default=None,
                        help="Specify a model by string identifier of format <model_name>:<model_version> "
                             "available in the U-Sleep package. OBS: The U-Sleep package must be installed or an "
                             "error is raised. Cannot specify both --model and --project_dir")
    parser.add_argument("--data_per_prediction", type=int, default=None,
                        help='Number of samples that should make up each sleep'
                             ' stage scoring. Defaults to sample_rate*30, '
                             'giving 1 segmentation per 30 seconds of signal. '
                             'Set this to 1 to score every data point in the '
                             'signal.')
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs to use for this job")
    parser.add_argument("--force_gpus", type=str, default="")
    parser.add_argument("--no_argmax", action="store_true",
                        help="Do not argmax prediction volume prior to save.")
    parser.add_argument("--weights_file_name", type=str, required=False,
                        help="Specify the exact name of the weights file "
                             "(located in <project_dir>/model/) to use.")
    parser.add_argument("--overwrite", action='store_true',
                        help='Overwrite existing output files and log files.')
    return parser


def get_processed_args(args):
    """
    Validate and prepare args.
    Returns a new set of args with potential modifications.

    Returns:
         Path to a validated project directory as per --project_dir.
    """
    modified_args = {}
    for key, value in vars(args).items():
        if isinstance(value, list):
            # Allow list-like arguments to be passed either space-separated as normally,
            # or using '&&' delimiters. This is useful e.g. when using Docker.
            split_list = []
            for item in value:
                split_list.extend(map(lambda s: s.strip(), item.split("&&")))
            value = split_list
        modified_args[key] = value
    args = Namespace(**modified_args)
    assert args.num_gpus >= 0, "--num_gpus must be positive or 0."

    if args.model:
        logger.info(f"Using the --model flag. "
                    f"Models (if any) at project directory path {Defaults.PROJECT_DIRECTORY} (if set) "
                    f"will not be considered.")
        try:
            import usleep
        except ImportError as e:
            raise RuntimeError("Cannot use the --model flag when the U-Sleep package is "
                               "not installed.") from e
        model_name, model_version = args.model.split(":")
        project_dir = usleep.get_model_path(model_name, model_version)
    else:
        project_dir = os.path.abspath(Defaults.PROJECT_DIRECTORY)

    # Check project folder is valid
    from utime.utils.scriptutils.scriptutils import assert_project_folder
    assert_project_folder(project_dir, evaluation=True)
    args.project_dir = project_dir

    # Set absolute input file path
    args.f = os.path.abspath(args.f)

    # Check header exists if specified
    if args.header_file_name and not os.path.exists(os.path.join(os.path.split(args.f)[0], args.header_file_name)):
        raise ValueError(f"Could not find header file with name {args.header_file_path} in the "
                         f"folder where input file {args.f} is stored.")

    # Set output file path
    if os.path.isdir(args.o):
        args.o = os.path.join(args.o, os.path.splitext(os.path.split(args.f)[-1])[0] + ".npy")

    # Set logging out path
    default_log_file_path = os.path.splitext(args.o)[0] + ".log"
    if args.logging_out_path is None:
        args.logging_out_path = default_log_file_path
    elif os.path.isdir(args.logging_out_path):
        args.logging_out_path = os.path.join(args.logging_out_path, os.path.split(default_log_file_path)[-1])

    if args.auto_channel_grouping is not None:
        # Check if --auto_channel_grouping has correct format and at least 2 groups
        assert len(args.auto_channel_grouping) > 1, "Should specify at least 2 channel type groups " \
                                                    "with parameter --auto_channel_grouping, " \
                                                    f"e.g. 'EEG' 'EOG', but got {args.auto_channel_grouping}"

    return args


def predict_study(study, model, channel_groups, no_argmax):
    psg = np.expand_dims(study.get_all_periods(), 0)
    pred = None
    for channel_group in channel_groups:
        # Get PSG for particular group
        psg_subset = psg[..., tuple(channel_group.channel_indices)]
        logger.info(f"\n--- Channel names: {channel_group.channel_names}\n"
                    f"--- Channel inds: {channel_group.channel_indices}\n"
                    f"--- Extracted PSG shape: {psg_subset.shape}")
        if pred is None:
            pred = model.predict_on_batch(psg_subset)
        else:
            # Sum into if using multiple channel groups
            pred += model.predict_on_batch(psg_subset)
    if hasattr(pred, "numpy") and callable(pred.numpy):
        pred = pred.numpy()
    pred = pred.reshape(-1, pred.shape[-1])
    if no_argmax:
        return pred
    else:
        return np.expand_dims(pred.argmax(-1), -1)


def save_hyp(path, pred, **kwargs):
    """
    Save predictions as stage strings with 1 stage (segment) per line in a plain text file.
    """
    # Map integer outputs to string stages
    stage_strings = np.vectorize(Defaults.get_class_int_to_stage_string().get)(pred.ravel())
    with open(path, "w") as out_f:
        out_f.write("\n".join(stage_strings))


def save_ids(path, pred, period_length_sec, **kwargs):
    """
    Save predictions as stage strings in init-duration-stage format in a plain text file.
    """
    # Map integer outputs to string stages
    stage_strings = np.vectorize(Defaults.get_class_int_to_stage_string().get)(pred.ravel())
    ids = dense_to_sparse(stage_strings, period_length_sec, allow_trim=True)
    with open(path, "w") as out_f:
        for i, d, s in ids:
            out_f.write(f"{i},{d},{s}\n")


def save_npy(path, pred, **kwargs):
    """
    Save predictions as a numpy file storing a [N, 1] array of integer stages
    """
    np.save(path, pred.reshape(len(pred), -1))


def save_prediction(pred, out_path, period_length_sec, no_argmax):
    dir_, fname = os.path.split(out_path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    basename, ext = os.path.splitext(fname)
    if ext == ".hyp":
        # Save as plain text of 1 stage per line
        out_path = os.path.join(dir_, basename + ext)
        save_func = save_hyp
        assert not no_argmax, "Cannot save to .hyp format with the --no_argmax flag. Please use .npy."
    elif ext == ".ids":
        # Save as plain text in IDS format
        out_path = os.path.join(dir_, basename + ext)
        save_func = save_ids
        assert not no_argmax, "Cannot save to .ids format with the --no_argmax flag. Please use .npy."
    else:
        # Save as npy
        out_path = os.path.join(dir_, basename + ".npy")
        save_func = save_npy
    # Save pred to disk
    logger.info(f"Saving prediction array of shape {pred.shape} to {out_path}")
    logger.info(f"Using save function: {save_func.__name__}")
    save_func(out_path, pred, period_length_sec=period_length_sec)


def split_channel_types(channels):
    """
    TODO

    Args:
        channels: list of channel names

    Returns:
        stripped channels
        channel_types
    """
    stripped, types = [], []
    for channel in channels:
        type_ = None
        if "==" in channel:
            channel, type_ = channel.split("==")
            type_ = type_.strip().upper()
            if type_ not in VALID_CHANNEL_TYPES:
                raise ValueError(f"Invalid channel type '{type_}' specified for channel '{channel}'. "
                                 f"Valid are: {VALID_CHANNEL_TYPES}")
        types.append(type_)
        stripped.append(channel)
    return stripped, types


def unpack_channel_groups(channels):
    """
    TODO
    """
    channels_to_load, channel_groups = [], []
    grouped = map(lambda chan: "++" in chan, channels)
    if all(grouped):
        for channel in channels:
            group = channel.split("++")
            channels_to_load.extend(group)
            channel_groups.append(group)
        # Remove duplicated while preserving order
        from collections import OrderedDict
        channels_to_load = list(OrderedDict.fromkeys(channels_to_load))
    elif not any(grouped):
        channels_to_load = channels
        channel_groups = [channels]
    else:
        raise ValueError("Must specify either a list of channels "
                         "or a list of channel groups, got a mix: {}".format(channels))

    return channels_to_load, channel_groups


def strip_and_infer_channel_types(channels_to_load, channel_groups):
    """
    TODO

    Args:
        channels_to_load:
        channel_groups:

    Returns:

    """
    # Infer and strip potential channel types
    channels_to_load, channel_types = split_channel_types(channels_to_load)
    channel_groups = [split_channel_types(group)[0] for group in channel_groups]

    # Infer channel types, may not be specified by user
    # If user did not specify all channels, use inferred for those missing
    inferred_channel_types = infer_channel_types(channels_to_load)
    for i, (inferred, passed) in enumerate(zip(inferred_channel_types, channel_types)):
        if passed is None:
            channel_types[i] = inferred

    return channels_to_load, channel_groups, channel_types


def get_channel_groups(channels, channel_types, channel_group_spec):
    def upper_stripped(s):
        return s.strip().upper()
    channel_types = list(map(upper_stripped, channel_types))
    channel_group_spec = list(map(upper_stripped, channel_group_spec))
    if any([c not in channel_group_spec for c in channel_types]):
        raise ValueError(f"Cannot get channel groups for spec {channel_group_spec} with channels "
                         f"{channels} and types {channel_types}: One or more types are not in the requested "
                         f"channel group spec.")
    channels_by_group = [[] for _ in range(len(channel_group_spec))]
    for channel, type_ in zip(channels, channel_types):
        channels_by_group[channel_group_spec.index(type_)].append(channel)
    # Return all combinations
    from itertools import product
    return list(product(*channels_by_group))


def get_load_and_group_channels(channels, auto_channel_grouping, auto_reference_types):
    """
    TODO

    Args:
        channels:
        auto_channel_grouping:
        auto_reference_types:

    Returns:

    """
    logger.info(f"Processing input channels: {channels}")
    channels_to_load, channel_groups = unpack_channel_groups(channels)
    channels_to_load, channel_groups, channel_types = strip_and_infer_channel_types(channels_to_load,
                                                                                    channel_groups)
    logger.info(f"\nFound:\n"
                f"-- Load channels: {channels_to_load}\n"
                f"-- Groups: {channel_groups}\n"
                f"-- Types: {channel_types}")
    if isinstance(auto_reference_types, list):
        assert len(channel_groups) == 1, "Cannot use channel groups with --auto_reference_types."
        channels_to_load, channel_types = infer_channel_refs(channel_names=channels_to_load,
                                                             channel_types=channel_types,
                                                             types=auto_reference_types,
                                                             on_already_ref="warn")
        channel_groups = [channels_to_load]
        logger.warning(f"OBS: Auto referencing returned channels: {channels_to_load}")
    if isinstance(auto_channel_grouping, list):
        channel_groups = get_channel_groups(channels_to_load, channel_types, auto_channel_grouping)
        logger.warning(f"OBS: Auto channel grouping returned groups: {channel_groups}")

    # Add channel inds to groups
    channel_set = namedtuple("ChannelSet", ["channel_names", "channel_indices"])
    for i, group in enumerate(channel_groups):
        channel_groups[i] = channel_set(
            channel_names=group,
            channel_indices=[channels_to_load.index(channel) for channel in group]
        )

    return channels_to_load, channel_groups


def get_sleep_study(psg_path,
                    channels,
                    header_file_name=None,
                    auto_channel_grouping=False,
                    auto_reference_types=False,
                    **params):
    """
    Loads a specified sleep study object with no labels
    Sets scaler and quality control function

    Returns:
        A loaded SleepStudy object
    """
    if params.get('batch_wise_scaling'):
        raise NotImplementedError("Batch-wise scaling is currently not "
                                  "supported. Use ut predict/evaluate instead")
    logger.info(f"Evaluating using parameters:\n{pformat(params)}")
    dir_, regex = os.path.split(os.path.abspath(psg_path))
    study = SleepStudy(subject_dir=dir_, psg_regex=regex,
                       header_regex=header_file_name,
                       no_hypnogram=True,
                       period_length_sec=params.get('period_length_sec', 30))

    channels_to_load, channel_groups = get_load_and_group_channels(channels,
                                                                   auto_channel_grouping,
                                                                   auto_reference_types)

    logger.info(f"\nLoading channels: {channels_to_load}\n"
                f"Channel groups: {channel_groups}")
    study.set_strip_func(**params['strip_func'])
    study.select_channels = channels_to_load
    study.sample_rate = params['set_sample_rate']
    study.scaler = params['scaler']
    study.set_quality_control_func(**params['quality_control_func'])
    study.load()
    logger.info(f"\nStudy loaded with shape: {study.get_psg_shape()}\n"
                f"Channels: {study.select_channels} (org names: {study.select_channels.original_names})")
    return study, channel_groups


def run(args, return_prediction=False, dump_args=None):
    """
    Run the script according to args - Please refer to the argparser.
    """
    args = get_processed_args(args)

    # Get a logger
    log_dir, log_file_name = os.path.split(args.logging_out_path)
    add_logging_file_handler(log_file_name, args.overwrite, log_dir=log_dir, mode="w")
    if dump_args:
        logger.info(f"Args dump: {vars(args)}")

    # Get hyperparameters and init all described datasets
    from utime.hyperparameters import YAMLHParams
    hparams = YAMLHParams(Defaults.get_hparams_path(args.project_dir), no_version_control=True)

    # Get the sleep study
    logger.info("Loading and pre-processing PSG file...")
    hparams['prediction_params']['channels'] = args.channels
    hparams['prediction_params']['strip_func']['strip_func_str'] = args.strip_func
    study, channel_groups = get_sleep_study(psg_path=args.f,
                                            header_file_name=args.header_file_name,
                                            auto_channel_grouping=args.auto_channel_grouping,
                                            auto_reference_types=args.auto_reference_types,
                                            **hparams['prediction_params'])

    # Set GPU and get model
    find_and_set_gpus(args.num_gpus, args.force_gpus)
    hparams["build"]["data_per_prediction"] = args.data_per_prediction
    logger.info(f"Predicting with {args.data_per_prediction} data per prediction")
    model = get_and_load_one_shot_model(
        n_periods=study.n_periods,
        project_dir=args.project_dir,
        hparams=hparams,
        weights_file_name=hparams.get('weights_file_name')
    )
    logger.info("Predicting...")
    pred = predict_study(study, model, channel_groups, args.no_argmax)
    logger.info(f"Predicted shape: {pred.shape}")
    if return_prediction:
        return pred
    else:
        save_prediction(pred=pred,
                        out_path=args.o,
                        period_length_sec=hparams.get('period_length_sec', 30),
                        no_argmax=args.no_argmax)


def entry_func(args=None):
    # Parse command line arguments
    parser = get_argparser()
    args = parser.parse_args(args)
    run(args, dump_args=args)


if __name__ == "__main__":
    entry_func()
