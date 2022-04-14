"""
Script for downloading and preprocessing certain datasets described in the
paper. Support for more datasets (hopefully all) will be added over time.
"""

import logging
import os
from sys import exit
from argparse import ArgumentParser
from psg_utils.downloads import (download_dataset,
                                  preprocess_dataset,
                                  DOWNLOAD_FUNCS)
from utime.utils.scriptutils import add_logging_file_handler

logger = logging.getLogger(__name__)

# list of currently supported datasets
DATASETS = list(DOWNLOAD_FUNCS.keys())


def get_argparser():
    """
    Returns an argument parser for this script
    """
    parser = ArgumentParser(description="Download and preprocess a sleep "
                                        "staging (PSG) dataset.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="The name of the dataset to download. "
                             f"Must be one of: {', '.join(DATASETS)}")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Path to a directory that should store the data."
                             " If the directory does not exist, it will be "
                             "created.")
    parser.add_argument("--N_first", type=int, default=None,
                        help="Download only the N first samples "
                             "(default: download all)")
    parser.add_argument("--no_preprocessing", action="store_true",
                        help="Do not apply preprocessing on the downloaded "
                             "data.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing log files (see --log_file). "
                             "Note that fetch data is only overwritten if the file has invalid SHA256 values. "
                             "Otherwise, with valid SHA256, existing files on disk are always skipped. "
                             "I.e., the --overwrite flag does not stored influence data for this script.")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Relative path (from Defaults.LOG_DIR as specified by ut --log_dir flag) of "
                             "output log file for this script. "
                             "Set to an empty string to not save any logs to file for this run. "
                             "Default is None (no log file)")
    return parser


def validate_dataset(dataset):
    """ Asserts that the specified dataset is supported """
    if dataset not in DATASETS:
        datasets_str = '\n- '.join(DATASETS)
        logger.info(f"Dataset {dataset} is invalid, must be one of:\n- {datasets_str}")
        exit(0)


def validate_and_create_out_dir(out_dir):
    """
    Creates directory/directories if not existing
    """
    if not os.path.exists(out_dir):
        logger.info("Creating output directory {}".format(out_dir))
        os.makedirs(out_dir)


def run(args):
    add_logging_file_handler(args.log_file, args.overwrite, mode="w")
    out_dir = os.path.abspath(args.out_dir)
    validate_dataset(args.dataset)
    validate_and_create_out_dir(out_dir)
    download_dataset(args.dataset, out_dir, args.N_first)
    if not args.no_preprocessing:
        preprocess_dataset(args.dataset, out_dir)


def entry_func(args=None):
    # Get the script to execute, parse only first input
    parser = get_argparser()
    run(parser.parse_args(args))


if __name__ == "__main__":
    entry_func()
