"""
Small utility script that extracts a set of channels from a set of PSG files
and saves them to a folder in .h5 files with minimally required header info
attached as h5 attributes (sample rate etc.).

The PSG file must be loadable using:
utime.io.high_level_file_loaders import load_psg
"""

import os
import numpy as np
import pickle
from argparse import ArgumentParser
from glob import glob
from sleeputils.errors import ChannelNotFoundError
from sleeputils.io.channels import ChannelMontageTuple, ChannelMontageCreator
from sleeputils.io.header import extract_header
from sleeputils.io.high_level_file_loaders import load_psg
from sleeputils.io import to_h5_file
from sleeputils.preprocessing.psg_sampling import set_psg_sample_rate
from mpunet.logging import Logger


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
                        help="Space-separated list of CHAN1-CHAN2 format of"
                             "referenced channel montages to extract. A "
                             "montage will be created if the referenced "
                             "channel is not already available in the file. If"
                             " the channel does not already exist and if "
                             "CHAN1 or CHAN2 is not available, an error is "
                             "raised.")
    parser.add_argument("--rename_channels", nargs="+", type=str,
                        help="Space-separated list of channel names to save"
                             " as instead of the originally extracted names. "
                             "Must match in length --channels.")
    parser.add_argument('--resample', type=int, default=None,
                        help='Re-sample the selected channels before storage.')
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files of identical name")
    parser.add_argument("--continue_", action="store_true",
                        help="Skip already existing files.")
    parser.add_argument("--use_dir_names", action="store_true",
                        help='Each PSG file will be saved as '
                             '<parent directory>.h5 instead of <file_name>.h5')
    parser.add_argument("--trim_leading_seconds_dict", type=str, default=None,
                        help="Path to .pickle dictionary of format ['filename' (without extension): float 'seconds'] "
                             "where 'seconds' is the number of seconds to trim from the start of file 'filename' before "
                             "saving to newly extracted file. Note that 'filename' is the dictionary name if the "
                             "--use_dir_names flag is also set.")
    return parser


def filter_channels(renamed_channels, selected_original_channels,
                    original_channels):
    inds_selected = [i for i, chan in enumerate(original_channels)
                     if chan in selected_original_channels]
    return [chan for i, chan in enumerate(renamed_channels)
            if i in inds_selected]


def _extract(file_,
             out_path,
             channels,
             renamed_channels,
             trim_leading_sec,
             logger,
             args):
    channels_in_file = extract_header(file_)["channel_names"]
    chan_creator = ChannelMontageCreator(existing_channels=channels_in_file,
                                         channels_required=channels,
                                         allow_missing=True)
    logger("[*] Channels in file: " + ", ".join(chan_creator.existing_channels.names))
    logger("[*] Output channels: " + ", ".join(chan_creator.output_channels.names))
    logger("[*] Channels to load: " + ", ".join(chan_creator.channels_to_load.names))
    try:
        psg, header = load_psg(file_,
                               load_channels=chan_creator.channels_to_load,
                               check_num_channels=False)
    except ChannelNotFoundError as e:
        logger("\n-----\nCHANNEL ERROR ON FILE {}".format(file_))
        logger(str(e) + "\n-----")
        os.rmdir(os.path.split(out_path)[0])
        return

    # create montages
    psg, channels = chan_creator.create_montages(psg)
    header['channel_names'] = channels
    logger("[*] Original PSG shape: {}".format(psg.shape))

    # Trim if needed
    if trim_leading_sec and not np.isnan(trim_leading_sec):
        n_trim = trim_leading_sec * header['sample_rate']
        if not np.isclose(n_trim, int(n_trim)):
            logger.warn(f"The number of trim seconds {trim_leading_sec} does not give a "
                        f"whole number of datapoints to remove ({n_trim}) given sample rate "
                        f"{header['sample_rate']}. Skipping file!")
            return
        n_trim = int(n_trim)
        psg = psg[n_trim:]
        logger(f"[*] PSG shape after trimming (leading, seconds={trim_leading_sec}, sr={header['sample_rate']}, "
               f"N trim={n_trim}): {psg.shape}")

    if psg.shape[0] % header['sample_rate']:
        logger(f"--- OBS: Length {len(psg)} not divisible by sample rate! "
               f"Trimming N items from end {len(psg) % header['sample_rate']}.")
        psg = psg[:-(psg.shape[0] % header['sample_rate'])]
        logger("--- New PSG shape: {}".format(psg.shape))

    # Resample
    if args.resample:
        psg = set_psg_sample_rate(psg,
                                  new_sample_rate=args.resample,
                                  old_sample_rate=header['sample_rate'])
        header['sample_rate'] = args.resample
        logger("[*] PSG shape after re-sampling: {}".format(psg.shape))

    # Rename channels
    if renamed_channels:
        org_names = header['channel_names'].original_names
        header['channel_names'] = filter_channels(renamed_channels,
                                                  org_names,
                                                  args.channels)
    else:
        header['channel_names'] = header['channel_names'].original_names
    logger("[*] Extracted {} channels: {}".format(psg.shape[1],
                                                  header['channel_names']))
    to_h5_file(out_path, psg, **header)


def extract(files, out_dir, channels, renamed_channels, trim_leading_seconds_dict, logger, args):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i, file_ in enumerate(files):
        if args.use_dir_names:
            name = os.path.split(os.path.split(file_)[0])[-1]
        else:
            name = os.path.splitext(os.path.split(file_)[-1])[0]
        logger("------------------")
        logger("[*] {}/{} Processing {}".format(i + 1, len(files), name))
        out_dir_subject = os.path.join(out_dir, name)
        if not os.path.exists(out_dir_subject):
            os.mkdir(out_dir_subject)
        out_path = os.path.join(out_dir_subject, name + ".h5")
        if os.path.exists(out_path):
            if args.continue_:
                logger("-- Skipping (already exists, overwrite=False)")
                continue
            elif not args.overwrite:
                raise OSError(f"File already exists at '{out_path}' abd neither --overwrite nor --continue was set.")
            else:
                os.remove(out_path)
        trim_leading_secs = 0.0
        if trim_leading_seconds_dict:
            if name not in trim_leading_seconds_dict:
                raise ValueError(f"The filename '{name}' does not exist as a key in dictioanry "
                                 f"'trim_leading_seconds_dict'. All files to extract must be represented in the trim "
                                 f"dictionary. If you do not want to trim this particular file, enter the filename "
                                 f"anyway into the dictionary with key:value pair ({name}: 0.0) to effectively apply "
                                 f"no trimming.")
            trim_leading_secs = trim_leading_seconds_dict[name]
        _extract(
            file_=file_,
            out_path=out_path,
            channels=channels,
            renamed_channels=renamed_channels,
            trim_leading_sec=trim_leading_secs,
            logger=logger,
            args=args
        )


def get_trim_dict(path):
    if not os.path.exists(path):
        raise OSError(f"No trim dict exists at path {path}")
    with open(path, "rb") as in_f:
        trim_dict = pickle.load(in_f)
    if not isinstance(trim_dict, dict):
        raise ValueError(f"The trim dict at path {path} with has an unexpected type ({type(trim_dict)}) "
                         f"- expected {dict}")
    return trim_dict


def run(args):
    files = glob(args.file_regex)
    out_dir = os.path.abspath(args.out_dir)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if args.overwrite and args.continue_:
        raise RuntimeError("Only one of the flags '--continue' and '--overwrite' may be used.")
    logger = Logger(out_dir,
                    active_file='extraction_log',
                    overwrite_existing=args.overwrite,
                    append_existing=args.continue_,
                    print_calling_method=False)
    logger("Args dump: {}".format(vars(args)))
    logger("Found {} files matching glob statement".format(len(files)))
    if len(files) == 0:
        return
    channels = ChannelMontageTuple(args.channels, relax=True)
    renamed_channels = args.rename_channels
    if renamed_channels and (len(renamed_channels) != len(channels)):
        raise ValueError("--rename_channels argument must have the same number"
                         " of elements as --channels. Got {} and {}.".format(
            len(channels), len(renamed_channels)
        ))

    logger("Extracting channels {}".format(channels.names))
    if renamed_channels:
        logger("Saving channels under names {}".format(renamed_channels))
    logger("Saving .h5 files to '{}'".format(out_dir))
    logger("Re-sampling: {}".format(args.resample))
    trim_leading_seconds_dict = None
    if args.trim_leading_seconds_dict:
        logger(f"Using (trailing) trim dict at path '{args.trim_leading_seconds_dict}'")
        trim_leading_seconds_dict = get_trim_dict(args.trim_leading_seconds_dict)
    logger("-*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*-")
    extract(
        files=files,
        out_dir=out_dir,
        channels=channels,
        renamed_channels=renamed_channels,
        trim_leading_seconds_dict=trim_leading_seconds_dict,
        logger=logger,
        args=args
    )


def entry_func(args=None):
    # Get the script to execute, parse only first input
    parser = get_argparser()
    args = parser.parse_args(args)
    run(args)


if __name__ == "__main__":
    entry_func()
