"""
Small utility script that groups files stored in a directory into sub-dirs
according to shared prefixes in their respective file names.
"""

import logging
import os
import shutil
from argparse import ArgumentParser
from utime.bin.cv_split import pair_by_names
from psg_utils.dataset.utils import filter_by_regex
from utime.utils.scriptutils import add_logging_file_handler

logger = logging.getLogger(__name__)


def get_argparser():
    """
    Returns an argument parser for this script
    """
    parser = ArgumentParser(description='Group files into sub-folders '
                                        'according to file name similarities')
    parser.add_argument("--data_dir", type=str, default="./",
                        help='The directory in which files to be grouped are '
                             'stored')
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Optional directory in which new sub-folders "
                             "will be created (defaults ot --data_dir)")
    parser.add_argument("--file_regex", type=str, default='.*', required=False,
                        help='A regex pattern (note: not glob) that should '
                             'match the all files that are to be considered '
                             'for grouping.')
    parser.add_argument("--common_prefix_length", type=int, required=False,
                        help="Consider only the first N characters of the "
                             "filenames when grouping.")
    parser.add_argument("--overwrite", action='store_true',
                        help='Overwrite existing log files.')
    parser.add_argument("--log_file", type=str, default=None,
                        help="Relative path (from Defaults.LOG_DIR as specified by ut --log_dir flag) of "
                             "output log file for this script. "
                             "Set to an empty string to not save any logs to file for this run. "
                             "Default is None (no log file)")
    return parser


def move_files(pairs, out_dir, subject_dir_name):
    """
    Move files in tuple/list 'pairs' to 'out_dir' under a sub-directory of name
    'subject_dir_name'.

    OBS: files are moved. Uses shutil.move.

    Args:
        pairs:             Tuple/list of files to move
        out_dir:           Path pointing to folder that should store the
                           subject dir
                           which in turn stores the individual files
        subject_dir_name:  Name of sub-dir to store individual files.
    """
    subject_dir = os.path.join(out_dir, subject_dir_name)
    if not os.path.exists(subject_dir):
        os.mkdir(subject_dir)
    for f in pairs:
        out_p = os.path.join(subject_dir, os.path.split(f)[-1])
        shutil.move(f, out_p)


def run(args):
    """ Run script with the specified args. See argparser for details. """
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.exists(out_dir):
        raise OSError(f"'out_dir' {out_dir} does not exist")

    # Get all files
    data_dir = os.path.abspath(args.data_dir)
    files = filter_by_regex(os.listdir(data_dir), args.file_regex)
    pairs = pair_by_names(files, args.common_prefix_length)
    logger.info(f"Found {len(files)} files\n"
                f"Found {len(pairs)} pairs\n"
                f"Moving to: {out_dir}")
    if input("Move? (y/N) ").lower() == "y":
        print("Moving...")
        for p in pairs:
            if args.common_prefix_length is None:
                subject_dir_name = os.path.splitext(os.path.split(p[0])[-1])[0]
            else:
                subject_dir_name = os.path.split(p[0])[-1][:args.common_prefix_length]
            move_files(p, out_dir, subject_dir_name)


def entry_func(args=None):
    # Get the script to execute, parse only first input
    parser = get_argparser()
    parser.parse_args(args)
    add_logging_file_handler(args.log_file, args.overwrite, mode="w")
    run(args)


if __name__ == "__main__":
    entry_func()
