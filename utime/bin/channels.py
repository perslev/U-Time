"""
Small script that prints the channels found in one or more PSG files
matching a glob pattern.
"""

import os
import logging
from argparse import ArgumentParser
from glob import glob
from psg_utils.io.header import extract_header
from psg_utils.dataset import SleepStudy
from utime.utils.scriptutils import add_logging_file_handler

logger = logging.getLogger(__name__)


def get_argparser():
    parser = ArgumentParser(description='Print the channels of files '
                                        'matching a glob pattern.')
    parser.add_argument("--subject_dir_pattern", type=str, required=True)
    parser.add_argument("--psg_regex", type=str, required=False)
    parser.add_argument("--overwrite", action='store_true',
                        help='Overwrite existing log files.')
    parser.add_argument("--log_file", type=str, default=None,
                        help="Relative path (from Defaults.LOG_DIR as specified by ut --log_dir flag) of "
                             "output log file for this script. "
                             "Set to an empty string to not save any logs to file for this run. "
                             "Default is None (no log file)")
    return parser


def run(args):
    files = glob(args.subject_dir_pattern)
    if len(files) == 0:
        logger.info(f"No subject dirs match pattern {args.subject_dir_pattern}")
    else:
        logger.info("Channels:")
        for subject_dir in files:
            psg_regex = args.psg_regex or None
            if not psg_regex and os.path.isfile(subject_dir):
                subject_dir, psg_regex = os.path.split(subject_dir)
            ss = SleepStudy(subject_dir=subject_dir,
                            psg_regex=psg_regex,
                            no_hypnogram=True,
                            period_length_sec=30)
            header = extract_header(ss.psg_file_path)
            logger.info(header['channel_names'], header['sample_rate'], " Hz")


def entry_func(args=None):
    # Get the script to execute, parse only first input
    parser = get_argparser()
    args = parser.parse_args(args)
    add_logging_file_handler(args.log_file, args.overwrite, mode="w")
    run(args)


if __name__ == "__main__":
    entry_func()
