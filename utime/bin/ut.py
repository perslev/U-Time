"""
Entry script redirecting all command line arguments to a specified script
from the utime.bin folder.

Usage:
ut [--help] script [script args]... [--seed]
"""

import logging
import argparse
import os
import sys
import importlib
import pkgutil
import psg_utils
from utime import bin, Defaults
from utime.version import __version__

logger = logging.getLogger(__name__)


def get_parser():
    mods = pkgutil.iter_modules(bin.__path__)

    ids = f"U-Time ({__version__})"
    sep = "-" * len(ids)
    usage = (f"ut [--help] script [script args]... [--seed]\n\n"
             f"{ids}\n"
             f"{sep}\n"
             f"Available scripts:\n")

    choices = []
    file_name = os.path.split(os.path.abspath(__file__))[-1]
    for m in mods:
        if isinstance(m, tuple):
            name, ispkg = m[1], m[2]
        else:
            name, ispkg = m.name, m.ispkg
        if name == file_name[:-3] or ispkg:
            continue
        usage += "- " + name + "\n"
        choices.append(name)

    # Top level parser
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("script", help="Name of the ut script to run.",
                        choices=choices)
    parser.add_argument("--project_dir", type=str, default="./",
                        help='Path to U-Time project folder. '
                             'Default is "./" (current working directory)')
    parser.add_argument("--log_dir", default="logs", type=str,
                        help="Path to directory that should store logs.")
    parser.add_argument("--log_level", default="INFO", type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'WARN', 'ERROR', 'FATAL', 'CRITICAL'],
                        help="Set the logging level for this script. Default 'INFO'.")
    parser.add_argument("--seed", default=None, type=int,
                        help="Run this script with numpy, random and tensorflow RNGs seeded "
                             "from integer --seed.")
    return parser


def split_help_from_args(args):
    other_args, help_args = [], []
    for arg in args:
        if arg == "-h" or arg == "--help":
            help_args.append("--help")
        else:
            other_args.append(arg)
    return other_args, help_args


def entry_func():
    # Get the script to execute, parse only first input
    args, help_agrs = split_help_from_args(sys.argv[1:])
    parsed, script_args = get_parser().parse_known_args(args or help_agrs)

    # Set logging output dir on Defaults object
    # OBS we do not create the folder yet.
    # Handled in add_logging_file_handler of individual script where overwriting is also checked.
    Defaults.LOG_DIR = os.path.abspath(parsed.log_dir)

    # Init both the utime and psg_utils package-level loggers to share formatter and handlers
    Defaults.init_package_level_loggers(parsed.log_level, package_names=(Defaults.PACKAGE_NAME,
                                                                         psg_utils.__name__))
    logger.info(f"Entry script args dump: {vars(parsed)}")

    # Set project directory for the script
    Defaults.set_project_directory(parsed.project_dir)

    # Set Tensorflow logging level to ERROR or higher.
    # This omits a range of (usually....) unimportant warning message.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    script = parsed.script
    if parsed.seed is not None:
        Defaults.set_global_seed(parsed.seed)

    # Import the script
    mod = importlib.import_module("utime.bin." + script)

    # Call entry function with remaining arguments
    mod.entry_func(script_args + help_agrs)


if __name__ == "__main__":
    entry_func()
