"""
Script for outputting summary statistics over metrics in one or more CSV files.
Assumes that input CSV files store metrics computed for each subject (in rows)
and each class (in columns).

Useful for aggregating evaluations across splits.

For confusion matrix computations and computation of global (cross subject)
scores, see utime.bin.cm
"""

import logging
import os
import sys
import pandas as pd
from glob import glob
from argparse import ArgumentParser
from utime.utils.scriptutils import add_logging_file_handler

logger = logging.getLogger(__name__)


def get_argparser():
    """
    Returns an argument parser for this script
    """
    parser = ArgumentParser(description='Summary over U-Time project'
                                        ' evaluations csv file(s).')
    parser.add_argument("--csv_pattern", type=str,
                        default="split_*/predictions/test_dataset/*dice*csv",
                        help="Glob pattern used to match evaluation files.")
    parser.add_argument("--drop_rows", nargs='*',
                        default="Grand mean",
                        help="A series of row names to drop from each csv "
                             "file before merging. For instance, specify "
                             "{--drop_rows Grand mean some_row} to drop "
                             "rows 'Grand mean' and 'some_row' "
                             "(defaults to 'Grand mean')")
    parser.add_argument("--drop_cols", nargs='*',
                        default="mean",
                        help="Same as --drop_rows, but for column names. "
                             "(defaults to 'mean')")
    parser.add_argument("--print_all", action="store_true",
                        help="Print in addition the entire, merged data frame "
                             "from which mean scores are computed.")
    parser.add_argument("--round", type=int, default=4,
                        help="Round float numbers. (defaults to 4)")
    parser.add_argument("--overwrite", action='store_true',
                        help='Overwrite existing log files.')
    parser.add_argument("--log_file", type=str, default=None,
                        help="Relative path (from Defaults.LOG_DIR as specified by ut --log_dir flag) of "
                             "output log file for this script. "
                             "Set to an empty string to not save any logs to file for this run. "
                             "Default is None (no log file)")
    return parser


def print_reduced_mean(df, print_all=False, round_=4):
    """
    Takes a DataFrame 'df' of per-subject (rows) of a metric on all classes
    (columns), computes summary statistics over the subjects and prints the
    results to screen. With print_all=True, the per-subject scores are also
    printed.

    Args:
        df:         (DataFrame) Dataframe of shape NxK (N subjects, K classes)
                                of metric values.
        print_all:  (bool)      Print the 'df' to screen as well as the summary
                                statics over 'df'
        round_:     (int)       Rounding precision for printing.
    """
    def make_df(df, axis):
        """ Create summary statistics DataFrame """
        mean = df.mean(axis=axis)
        std = df.std(axis=axis)
        min_ = df.min(axis=axis)
        max_ = df.max(axis=axis)
        return pd.DataFrame({"mean": mean,
                             "std": std,
                             "min": min_,
                             "max": max_}, index=mean.index)

    df = make_df(df, axis=0)
    logger.info("\nSUMMARY RESULT\n" +
                "--------------\n" +
                ("\nMerged evaluation files:\n" +
                 f"{df.round(round_)}\n" if print_all else "") +
                "\nMean over axis 0 (rows):\n" +
                f"{df.round(round_)}\n" +
                f"Mean of means: {round(df['mean'].mean(), round_)}")


def parse_and_add(file_, results, drop_rows, drop_cols):
    """
    Load a CSV file, drop rows and/or columns as specified and merge the data
    with the pandas.DataFrame 'results'.

    Concatenates the DataFrames over axis 0 (on rows)

    Args:
        file_:      (string)    Path to a CSV file to load and add
        results:    (DataFrame) DataFrame storing other data (or empty)
        drop_rows:  (list)      A list of string row names to drop from each
                                csv file before merging.
        drop_cols:  (list)      A list of string col names to drop from each
                                csv file before merging.

    Returns:
        A DataFrame that stores all data from 'results' merged with data from
        CSV file 'file_'.
    """
    df = pd.read_csv(file_, index_col=0)
    try:
        df = df.drop(index=drop_rows, columns=drop_cols)
    except KeyError:
        from sys import exit
        logger.error("[PARSE ERROR] Invalid row or column in {drop_rows} or {drop_cols} respectively.\n"
                     "One or more of these were not found in file:\n{file_}\n\n"
                     "This file has the following:\n"
                     "Rows:    {list(df.index)}\n"
                     "Columns: {list(df.columns)}")
        exit(1)
    if len(results) == 0:
        return df
    o = pd.concat((df, results), axis=0, sort=True)
    return o


def parse_results(csv_files, drop_rows, drop_cols, print_all, round_):
    """
    Load, merge and print metrics from one or more CSV files.

    Args:
        csv_files:    (list) A list of paths to .csv files storing per-subject
                             metrics.
        drop_rows:    (list) A list of string row names to drop from each csv
                             file before merging.
        drop_cols:    (list) A list of string col names to drop from each csv
                             file before merging.
        print_all:    (bool) Print the entire, merged data frame from which
                             mean scores are computed.
        round_:       (int)  Rounding precision
    """
    results = pd.DataFrame()
    for file_ in csv_files:
        results = parse_and_add(file_=file_,
                                results=results,
                                drop_rows=drop_rows,
                                drop_cols=drop_cols)
    print_reduced_mean(results,
                       print_all=print_all,
                       round_=round_)


def run(args):
    """ Run this script with passed args - see argparser for details """
    # Get folder/folders - 3 levels possible
    logger.info("... Looking for files matching pattern")
    pattern = args.csv_pattern
    csv_files = glob(pattern, recursive=False)
    logger.info(f"Found {len(csv_files)} files matching pattern '{pattern}'")
    if not csv_files:
        sys.exit(0)
    csv_files.sort()
    logger.info("\n".join(map(os.path.abspath, csv_files)))
    in_ = input("\nCorrect? (Y/n) ")
    if in_.lower() not in ("n", "no"):
        parse_results(csv_files=csv_files,
                      drop_rows=args.drop_rows,
                      drop_cols=args.drop_cols,
                      print_all=args.print_all,
                      round_=args.round)


def entry_func(args=None):
    parser = get_argparser()
    args = parser.parse_args(args)
    add_logging_file_handler(args.log_file, args.overwrite, mode="w")
    run(args)


if __name__ == "__main__":
    entry_func()
