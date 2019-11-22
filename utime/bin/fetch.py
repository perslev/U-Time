"""
Script for downloading and preprocessing certain datasets described in the
paper. Support for more datasets (hopefully all) will be added over time.
"""

import os
from sys import exit
from argparse import ArgumentParser
from utime.preprocessing.dataset_preparation import (download_dataset,
                                                     preprocess_dataset,
                                                     DOWNLOAD_FUNCS)


# list of currently supported datasets
DATASETS = list(DOWNLOAD_FUNCS.keys())


def get_argparser():
    """
    Returns an argument parser for this script
    """
    parser = ArgumentParser(description='Download and preprocess a sleep '
                                        'staging (PSG) dataset.')
    parser.add_argument("--dataset", type=str, required=True,
                        help='The name of the dataset to download. '
                             'Must be one of: {}'.format(", ".join(DATASETS)))
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
    return parser


def validate_dataset(dataset):
    """ Asserts that the specified dataset is supported """
    if dataset not in DATASETS:
        print("Dataset {} is invalid, must be one of:\n- {}".format(
            dataset, "\n- ".join(DATASETS)
        ))
        exit(0)


def validate_and_create_out_dir(out_dir):
    """
    Creates directory/directories if not existing
    """
    if not os.path.exists(out_dir):
        print("Creating output directory {}".format(out_dir))
        os.makedirs(out_dir)


def run(args):
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
