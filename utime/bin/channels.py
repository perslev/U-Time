"""
Small script that prints the channels found in one or more PSG files
matching a glob pattern.
"""

import logging
from argparse import ArgumentParser
from glob import glob
from psg_utils.io.header import extract_header
from psg_utils.io.channels import infer_channel_types
from utime.utils.scriptutils import add_logging_file_handler

logger = logging.getLogger(__name__)


def get_argparser():
    parser = ArgumentParser(description='Print the channels of one or more files')
    parser.add_argument("-f", type=str, required=True, help='Path or glob-like statement to one or more files.')
    parser.add_argument("--overwrite", action='store_true',
                        help='Overwrite existing log files.')
    parser.add_argument("--select_types", type=str, nargs="+", default=None,
                        help='A list of channel types (e.g., EEG EOG) to select and '
                             'print to screen as an escaped list (e.g., may output: "EEG Fpz-Cz" "EOG horizontal").')
    parser.add_argument("--log_file", type=str, default=None,
                        help="Relative path (from Defaults.LOG_DIR as specified by ut --log_dir flag) of "
                             "output log file for this script. "
                             "Set to an empty string to not save any logs to file for this run. "
                             "Default is None (no log file)")
    return parser


def run(args):
    files = glob(args.f)
    if len(files) == 0:
        logger.info(f"No subject dirs match pattern {args.subject_dir_pattern}")
    else:
        select_types = [type_.upper().strip() for type_ in (args.select_types or [])]
        for file_ in files:
            logger.info(f"File: {file_}")
            header = extract_header(file_)
            inferred_types = infer_channel_types(header['channel_names'])
            selected = []
            for name, type_ in zip(header['channel_names'], inferred_types):
                if type_ in select_types:
                    selected.append(f"'{name}=={type_}'")
                logger.info(f"{name} ({type_}?)")
            if selected:
                logger.info(f"Selected: {' '.join(selected)}")


def entry_func(args=None):
    # Get the script to execute, parse only first input
    parser = get_argparser()
    args = parser.parse_args(args)
    add_logging_file_handler(args.log_file, args.overwrite, mode="w")
    run(args)


if __name__ == "__main__":
    entry_func()
