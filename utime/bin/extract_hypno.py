import logging
import os
import numpy as np
from argparse import ArgumentParser
from glob import glob
from pathlib import Path
from psg_utils.io.hypnogram import extract_ids_from_hyp_file
from psg_utils.hypnogram.utils import fill_hyp_gaps
from utime.utils.scriptutils import add_logging_file_handler

logger = logging.getLogger(__name__)


def get_argparser():
    parser = ArgumentParser(description='Extract hypnograms from various'
                                        ' file formats.')
    parser.add_argument("--file_regex", type=str,
                        help='A glob statement matching all files to extract '
                             'from')
    parser.add_argument("--out_dir", type=str,
                        help="Directory in which extracted files will be "
                             "stored")
    parser.add_argument("--fill_blanks", type=str, default=None,
                        help="A stage string value to insert into the hypnogram when gaps "
                             "occour, e.g. 'UNKNOWN' or 'Not Scored', etc.")
    parser.add_argument("--extract_func", type=str, default=None,
                        help="Name of hyp extraction function. If not specified, the file extension defines the "
                             "function to use.")
    parser.add_argument("--remove_offset", action="store_true",
                        help="Remove potential offsets so that the first sleep stage always starts at init sec 0.")
    parser.add_argument("--correct_zero_durations", type=int, default=None, help="Optionally change any stage with duration "
                                                                                 "0 seconds to some other duration. E.g., --correct_zero_durations 30 will set those events to 30 seconds.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files of identical name")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files of identical name and log files")
    parser.add_argument("--log_file", type=str, default="hyp_extraction_log",
                        help="Relative path (from Defaults.LOG_DIR as specified by ut --log_dir flag) of "
                             "output log file for this script. "
                             "Set to an empty string to not save any logs to file for this run. "
                             "Default is 'hyp_extraction_log'")
    return parser


def to_ids(start, durs, stage, out):
    with open(out, "w") as out_f:
        for i, d, s in zip(start, durs, stage):
            out_f.write("{},{},{}\n".format(int(i), int(d), s))


def remove_offset(inits):
    offset = inits[0]
    for i in range(len(inits)):
        new_init = inits[i] - offset
        rounded_new_init = np.round(new_init)
        if new_init - rounded_new_init > 1e-6:
            raise ValueError(f"Unexpectedly large difference of {new_init - rounded_new_init} between new_init of "
                             f"{new_init} and round(new_init) of {rounded_new_init} when "
                             "removing offset. The implementation expects inits to land on whole-seconds, not "
                             "fractions.")
        inits[i] = new_init
    return inits


def run(args):
    logger.info(f"Args dump: {vars(args)}")
    files = glob(args.file_regex)
    out_dir = Path(args.out_dir).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    n_files = len(files)
    logger.info(f"Found {n_files} files matching glob statement")
    if n_files == 0:
        return
    logger.info(f"Saving .ids files to '{out_dir}'")
    if n_files == 0:
        return

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for i, file_ in enumerate(files):
        file_name = os.path.split(file_)[-1].split(".", 1)[0]
        folder_name = os.path.split(os.path.split(file_)[0])[-1]
        out_dir_subject = os.path.join(out_dir, folder_name)
        out = os.path.join(out_dir_subject, file_name + ".ids")
        logger.info(f"{i+1}/{n_files} Processing {file_name}\n"
                    f"-- In path    {file_}\n"
                    f"-- Out path   {out}")
        if not os.path.exists(out_dir_subject):
            os.mkdir(out_dir_subject)
        if os.path.exists(out):
            if not args.overwrite:
                continue
            os.remove(out)
        inits, durs, stages = extract_ids_from_hyp_file(file_,
                                                        period_length_sec=30,
                                                        extract_func=args.extract_func,
                                                        replace_zero_durations=args.correct_zero_durations)
        if args.remove_offset:
            try:
                inits = remove_offset(inits)
            except ValueError:
                import shutil
                shutil.move(file_, "missing_labels")
        if args.fill_blanks:
            inits, durs, stages = fill_hyp_gaps(inits, durs, stages, args.fill_blanks)
        to_ids(inits, durs, stages, out)


def entry_func(args=None):
    # Get the script to execute, parse only first input
    parser = get_argparser()
    args = parser.parse_args(args)
    add_logging_file_handler(args.log_file, args.overwrite, mode="w")
    run(args)


if __name__ == "__main__":
    entry_func()
