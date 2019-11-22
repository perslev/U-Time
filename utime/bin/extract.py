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
                        help="Comma sepparated list of channels to extract")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files of identical name")
    return parser


def run(file_regex, out_dir, channels, overwrite):
    files = glob(file_regex)
    out_dir = os.path.abspath(out_dir)
    n_files = len(files)
    print("Found {} files matching glob statement".format(n_files))
    if n_files == 0:
        return
    print("Extracting channels {}".format(channels))
    print("Saving .h5 files to '{}'".format(out_dir))

    from utime.io.high_level_file_loaders import load_psg
    from utime.utils.scriptutils import to_h5_file
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i, file_ in enumerate(files):
        name = os.path.splitext(os.path.split(file_)[-1])[0]
        print("  {}/{} Processing {}".format(i+1, n_files, name),
              flush=True, end="\r")
        out = os.path.join(out_dir, name + ".h5")
        if os.path.exists(out):
            if not overwrite:
                continue
            os.remove(out)
        psg, header = load_psg(file_, load_channels=channels)
        to_h5_file(out, psg, **header)


def entry_func(args=None):
    # Get the script to execute, parse only first input
    parser = get_argparser()
    args = vars(parser.parse_args(args))
    run(**args)


if __name__ == "__main__":
    entry_func()
