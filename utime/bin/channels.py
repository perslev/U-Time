"""
Small script that prints the channels found in one or more PSG files
matching a glob pattern.
"""

from argparse import ArgumentParser
from glob import glob


def get_argparser():
    parser = ArgumentParser(description='Print the channels of files '
                                        'matching a glob pattern.')
    parser.add_argument("--subject_dir_pattern", type=str, required=True)
    return parser


def run(subject_dir_pattern):
    files = glob(subject_dir_pattern)
    if len(files) == 0:
        print("No subject dirs match pattern {}".format(subject_dir_pattern))
    from utime.dataset import SleepStudy
    print("Channels:")
    for f in files:
        print(SleepStudy(subject_dir=f,
                         no_hypnogram=True,
                         period_length_sec=30).load().select_channels)


def entry_func(args=None):
    # Get the script to execute, parse only first input
    parser = get_argparser()
    args = vars(parser.parse_args(args))
    run(**args)


if __name__ == "__main__":
    entry_func()
