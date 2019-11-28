"""
Small utility script that extracts a set of channels from a set of PSG files
and saves them to a folder in .h5 files with minimally required header info
attached as h5 attributes (sample rate etc.).

The PSG file must be loadable using:
utime.io.high_level_file_loaders import load_psg
"""

from argparse import ArgumentParser
from glob import glob
import os
from utime.errors import ChannelNotFoundError
from utime.io.channels import ChannelMontageTuple
from MultiPlanarUNet.logging import Logger


def get_argparser():
    """
    Returns an argument parser for this script
    """
    parser = ArgumentParser(description='Extract a set of channels from a set '
                                        'of PSG files, various formats '
                                        'supported. The extracted data will be'
                                        ' saved to .h5 files with minimal '
                                        'header information attributes.')
    parser.add_argument("--file_regex", type=str,
                        help='A glob statement matching all files to extract '
                             'from')
    parser.add_argument("--out_dir", type=str,
                        help="Directory in which extracted files will be "
                             "stored")
    parser.add_argument("--channels", nargs="+", type=str,
                        help="Space-separated list of channels to extract")
    parser.add_argument('--ignore_reference_channels', action='store_true',
                        help='Match only against the channel names and not'
                             ' their (potential) reference channel name.')
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files of identical name")
    return parser


def run(file_regex, out_dir, channels, overwrite):
    files = glob(file_regex)
    out_dir = os.path.abspath(out_dir)
    n_files = len(files)
    logger = Logger(out_dir,
                    active_file='extraction_log',
                    overwrite_existing=overwrite)
    logger("Found {} files matching glob statement".format(n_files))
    if n_files == 0:
        return
    channels = ChannelMontageTuple(channels, relax=True)
    logger("Extracting channels {}".format(channels))
    logger("Saving .h5 files to '{}'".format(out_dir))

    from utime.io.high_level_file_loaders import load_psg
    from utime.utils.scriptutils import to_h5_file
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i, file_ in enumerate(files):
        name = os.path.splitext(os.path.split(file_)[-1])[0]
        print("  {}/{} Processing {}".format(i+1, n_files, name),
              flush=True, end="\r")
        out_dir_subject = os.path.join(out_dir, name)
        if not os.path.exists(out_dir_subject):
            os.mkdir(out_dir_subject)
        out = os.path.join(out_dir_subject, name + ".h5")
        if os.path.exists(out):
            if not overwrite:
                continue
            os.remove(out)
        try:
            psg, header = load_psg(file_, load_channels=channels)
        except ChannelNotFoundError as e:
            logger("\n-----\nCHANNEL ERROR ON FILE {}".format(file_))
            logger(str(e) + "\n-----")
            os.rmdir(out_dir_subject)
            continue
        to_h5_file(out, psg, **header)


def entry_func(args=None):
    # Get the script to execute, parse only first input
    parser = get_argparser()
    args = vars(parser.parse_args(args))
    run(**args)


if __name__ == "__main__":
    entry_func()
