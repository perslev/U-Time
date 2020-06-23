"""

"""

import os
import readline
import numpy as np
from argparse import ArgumentParser
from utime.bin.evaluate import (set_gpu_vis,
                                get_and_load_one_shot_model,
                                get_logger)
from utime import defaults
from pprint import pformat

readline.parse_and_bind('tab: complete')


def get_argparser():
    """
    Returns an argument parser for this script
    """
    parser = ArgumentParser(description='Predict using a U-Time model.')
    parser.add_argument("-f", type=str, required=True,
                        help='Regex pattern matching files to predict on. '
                             'If not specified, prediction will be launched '
                             'on the test_data as specified in the '
                             'hyperparameter file.')
    parser.add_argument("-o", type=str, required=True,
                        help="Output path for storing predictions.")
    parser.add_argument("--channels", nargs='*', type=str, default=None,
                        required=True,
                        help="A list of channels to use for prediction.")
    parser.add_argument("--project_dir", type=str, default="./",
                        help='Path to U-Time project folder')
    parser.add_argument("--data_per_prediction", type=int, default=None,
                        help='Number of samples that should make up each sleep'
                             ' stage scoring. Defaults to sample_rate*30, '
                             'giving 1 segmentation per 30 seconds of signal. '
                             'Set this to 1 to score every data point in the '
                             'signal.')
    parser.add_argument("--num_GPUs", type=int, default=1,
                        help="Number of GPUs to use for this job")
    parser.add_argument("--force_GPU", type=str, default="")
    parser.add_argument("--no_argmax", action="store_true",
                        help="Do not argmax prediction volume prior to save.")
    parser.add_argument("--weights_file_name", type=str, required=False,
                        help="Specify the exact name of the weights file "
                             "(located in <project_dir>/model/) to use.")
    return parser


def assert_args(args):
    """ Not yet implemented """
    return


def predict_study(study, model, no_argmax):
    # Predict
    psg = np.expand_dims(study.get_all_periods(), 0)
    pred = model.predict_on_batch(psg)
    pred = pred.numpy().reshape(-1, pred.shape[-1])
    if no_argmax:
        return pred
    else:
        return np.expand_dims(pred.argmax(-1), -1)


def save_prediction(pred, out_path, input_file_path, logger):
    out_path = os.path.abspath(out_path)
    if os.path.isdir(out_path):
        out_path = os.path.join(out_path, os.path.split(input_file_path)[-1])
    dir_, fname = os.path.split(out_path)
    os.makedirs(dir_, exist_ok=True)
    fname = os.path.splitext(fname)[0] + ".npy"

    # Save pred to disk
    out_path = os.path.join(dir_, fname)
    logger("* Saving prediction array of shape {} to {}".format(
        pred.shape, out_path
    ))
    np.save(out_path, pred)


def get_sleep_study(psg_path,
                    logger,
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
    logger("Evaluating using parameters:\n{}".format(pformat(params)))
    from utime.dataset.sleep_study import SleepStudy
    dir_, regex = os.path.split(os.path.abspath(psg_path))
    study = SleepStudy(subject_dir=dir_, psg_regex=regex,
                       no_hypnogram=True,
                       period_length_sec=params.get('period_length_sec'),
                       logger=logger)
    study.select_channels = params['channels']
    study.sample_rate = params['set_sample_rate']
    study.scaler = params['scaler']
    study.set_quality_control_func(**params['quality_control_func'])
    study.load()
    logger("Study loaded with shape: {}".format(study.get_psg_shape()))
    logger("Channels: {} (org names: {})".format(study.select_channels,
                                                 study.select_channels.original_names))
    return study


def run(args):
    """
    Run the script according to args - Please refer to the argparser.
    """
    assert_args(args)
    # Check project folder is valid
    from utime.utils.scriptutils.scriptutils import assert_project_folder
    project_dir = os.path.abspath(args.project_dir)
    assert_project_folder(project_dir, evaluation=True)

    # Get a logger
    logger = get_logger(project_dir, True, name="prediction_log")
    logger("Args dump: \n{}".format(vars(args)))

    # Get hyperparameters and init all described datasets
    from utime.hyperparameters import YAMLHParams
    hparams = YAMLHParams(defaults.get_hparams_path(project_dir), logger,
                          no_version_control=True)

    # Get the sleep study
    logger("Loading and pre-processing PSG file...")
    hparams['prediction_params']['channels'] = args.channels
    study = get_sleep_study(psg_path=args.f,
                            logger=logger,
                            **hparams['prediction_params'])

    # Set GPU and get model
    set_gpu_vis(args.num_GPUs, args.force_GPU, logger)
    hparams["build"]["data_per_prediction"] = args.data_per_prediction
    logger("Predicting with {} data per prediction".format(args.data_per_prediction))
    model = get_and_load_one_shot_model(
        n_periods=study.n_periods,
        project_dir=project_dir,
        hparams=hparams,
        logger=logger,
        weights_file_name=hparams.get_from_anywhere('weight_file_name')
    )

    logger("Predicting...")
    pred = predict_study(study, model, args.no_argmax)
    save_prediction(pred=pred,
                    out_path=args.o,
                    input_file_path=study.psg_file_path,
                    logger=logger)


def entry_func(args=None):
    # Parse command line arguments
    parser = get_argparser()
    run(parser.parse_args(args))


if __name__ == "__main__":
    entry_func()
