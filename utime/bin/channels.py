"""
Small script that prints the channels found in one or more PSG files
matching a glob pattern.
"""

from argparse import ArgumentParser
from glob import glob
from utime.io.file_loaders import read_psg_header
import os


def get_argparser():
    parser = ArgumentParser(description='Print the channels of files '
                                        'matching a glob pattern.')
    parser.add_argument("--subject_dir_pattern", type=str, required=True)
    parser.add_argument("--psg_regex", type=str, required=False)
    return parser


def run(subject_dir_pattern, psg_regex):
    files = glob(subject_dir_pattern)
    if len(files) == 0:
        print("No subject dirs match pattern {}".format(subject_dir_pattern))
    else:
        from utime.dataset import SleepStudy
        print("Channels:")
        for subject_dir in files:
            psg_regex = psg_regex or None
            if not psg_regex and os.path.isfile(subject_dir):
                subject_dir, psg_regex = os.path.split(subject_dir)
            ss = SleepStudy(subject_dir=subject_dir,
                            psg_regex=psg_regex,
                            no_hypnogram=True,
                            period_length_sec=30)
            header = read_psg_header(ss.psg_file_path)
            print(header['channel_names'],
                  header['sample_rate'], " Hz")


def entry_func(args=None):
    # Get the script to execute, parse only first input
    parser = get_argparser()
    args = vars(parser.parse_args(args))
    run(**args)


if __name__ == "__main__":
    entry_func()
