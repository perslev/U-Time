"""
Script that prepares a folder of data for cross-validation experiments by
randomly splitting the dataset into partitions and storing links to the
relevant files in sub-folders for each split.
"""

from glob import glob
import os
import numpy as np
import random
from MultiPlanarUNet.utils import create_folders
import argparse


# These values are normally overwritten from the command-line, see argparser
_DEFAULT_TEST_FRACTION = 0.20
_DEFAULT_VAL_FRACTION = 0.20


def get_argparser():
    """
    Returns an argument parser for this script
    """
    parser = argparse.ArgumentParser(description="Prepare a data folder for a"
                                                 "CV experiment setup.")
    parser.add_argument("--data_dir", type=str,
                        help="Path to data directory")
    parser.add_argument("--subject_dir_pattern", type=str,
                        help="Glob-like pattern used to select subject folders")
    parser.add_argument("--CV", type=int, default=5,
                        help="Number of splits (default=5)")
    parser.add_argument("--out_dir", type=str, default="views",
                        help="Directory to store CV subfolders "
                             "(default=views")
    parser.add_argument("--copy", action="store_true",
                        help="Copy files to CV-subfolders instead of "
                             "symlinking (not recommended)")
    parser.add_argument("--file_list", action="store_true",
                        help="Create text files with paths pointing to the "
                             "files needed under each split instead of "
                             "symlink/copying. This is usefull on systems "
                             "were symlink is not supported, but the dataset "
                             "size is too large to store in copies. "
                             "NOTE: Only one of --copy and --file_list "
                             "flags must be set.")
    parser.add_argument("--validation_fraction", type=float,
                        default=_DEFAULT_VAL_FRACTION,
                        help="Fraction of OVERALL data size used for "
                             "validation in each split. In a 5-CV setting with"
                             " N=100 and val_frac=0.20, each split will have "
                             "N_train=60, N_val=20 and N_test=20 "
                             "subjects/records")
    parser.add_argument("--test_fraction", type=float,
                        default=_DEFAULT_TEST_FRACTION,
                        help="Fraction of data size used for test if CV=1.")
    parser.add_argument("--common_prefix_length", type=int, required=False,
                        help="If specified, files of identical naming in the"
                             " first 'common_prefix_length' letters will be"
                             " considered a single entry. This is useful for"
                             " splitting multiple studies on the same subject"
                             " as together.")
    return parser


def assert_dir_structure(data_dir, out_dir):
    """ Asserts that the data_dir exists and the out_dir does not """
    if not os.path.exists(data_dir):
        raise OSError("Invalid data directory '%s'. Does not exist." % data_dir)
    if os.path.exists(out_dir):
        raise OSError("Output directory at '%s' already exists." % out_dir)


def create_view_folders(out_dir, n_splits):
    """
    Helper function that creates a set of 'split_0', 'split_1'..., folders
    within a directory 'out_dir'. If n_splits == 1, only creates the out_dir.
    """
    if not os.path.exists(out_dir):
        print("Creating directory at %s" % out_dir)
        os.makedirs(out_dir)
    if n_splits > 1:
        for i in range(n_splits):
            split_dir = os.path.join(out_dir, "split_%i" % i)
            print("Creating directory at %s" % split_dir)
            os.mkdir(split_dir)


def add_files(file_paths, out_folder, link_func=os.symlink):
    """
    Add all files pointed to by paths in list 'file_paths' to folder
    'out_folder' using the linking/copy function 'link_func'.

    Args:
        file_paths: A list of file paths
        out_folder: A path to a directory that should store the linked files
        link_func:  A function to apply on relative file paths in 'file_paths'
                    and absolute paths in 'file_paths'.
                    Typically one of os.symlink, os.copy or
                    _add_to_file_list_fallback.
    """
    for file_path in file_paths:
        if not isinstance(file_path, (list, tuple, np.ndarray)):
            file_path = (file_path,)
        for path in file_path:
            file_name = os.path.split(str(path))[-1]
            rel_path = os.path.relpath(path, out_folder)
            link_func(rel_path, out_folder + "/%s" % file_name)


def _add_to_file_list_fallback(rel_file_path,
                               file_path,
                               fname="LIST_OF_FILES.txt"):

    """
    On some system symlinks are not supported, if --files_list flag is set,
    uses this function to add each absolute file path to a list at the final
    subfolder that is supposed to store symlinks or actual files (--copy)

    At run-time, these files must be loaded by reading the path from this
    file instead.

    Args:
        rel_file_path: (string) Relative path pointing to the file from the
                                current working directory.
        file_path:     (string) Absolute path to the file
        fname:         (string) Filename of the file to store the paths
    """
    # Get folder where list of files should be stored
    folder = os.path.split(file_path)[0]

    # Get absolute path to file
    # We change dir to get the correct abs path from the relative
    os.chdir(folder)
    abs_file_path = os.path.abspath(rel_file_path)

    # Get path to the list of files
    list_file_path = os.path.join(folder, fname)

    with open(list_file_path, "a") as out_f:
        out_f.write(abs_file_path + "\n")


def pair_by_names(files, common_prefix_length=None):
    """
    Takes a list of file names and returns a list of tuples of file names in
    the list that share 'common_prefix_length' of identical leading characters

    That is, a list of files ['FILE_1_1', 'FILE_1_2', 'FILE_2_1'] and
    common_prefix_length 6 will result in:

       [ ('FILE_1_1', 'FILE_1_2') , ('FILE_2_1',) ]

    Args:
        files:                (list) A list of filenames
        common_prefix_length: (int)  A number of leading characters to match

    Returns:
        A list of tuples of paired filenames
    """
    from collections import defaultdict
    if common_prefix_length is not None:
        names = [os.path.split(i)[-1][:common_prefix_length] for i in files]
    else:
        names = [os.path.splitext(os.path.split(i)[-1])[0] for i in files]
    inds = defaultdict(list)
    for i, item in enumerate(names):
        inds[item].append(i)
    pairs = inds.values()
    return [tuple(np.array(files)[i]) for i in pairs]


def get_split_sizes(subject_dirs, n_splits, args, desc):
    """
    Returns the number of samples to include in the training, validation and
    testing sub-sets in each split given the parsed arguments (see argparser)

    Also prints the results to screen.

    Args:
        subject_dirs: (list)   List of all subject dirs in the dataset
        n_splits:     (int)    Number of splits to perform
        args:         (tuple)  Arguments passed, see argparser.
        desc:         (string) A string describing whether each entity in
                               'subject_dirs' is a 'subject' or 'record'.

    Returns:
        3 ints, number of training-, validation- and testing samples for each
        split.
    """
    n_total = len(subject_dirs)
    if n_splits > 1:
        n_test = int(np.ceil(n_total / n_splits))
    else:
        n_test = int(np.ceil(n_total * args.test_fraction))
    n_val = int(np.ceil(n_total * args.validation_fraction))
    if n_val + n_test >= n_total:
        raise ValueError("Too large test/validation_fraction - "
                         "No training samples left!")
    n_train = n_total - n_test - n_val
    print("-----")
    print("Total {}:".ljust(40).format(desc), n_total)
    print("(Approx.) Train {} pr. split:".ljust(40).format(desc), n_train)
    print("Validation {} pr. split:".ljust(40).format(desc), n_val)
    print("(Approx.) Test {} pr. split:".ljust(40).format(desc), n_test)
    return n_train, n_val, n_test


def run_on_split(split_path, test_split, train_val_data, n_val, args):
    """
    Add the train/val/test files of a single split to the split directories

    Depending on the arguments parsed (--copy, --file_list etc.) either copies,
    symlinks or creates a LIST_OF_FILES.txt file of absolute paths in each
    split sub-directory.

    Args:
        split_path:      (string) Path to the split directory
        test_split:      (list)   List of paths pointing to split test data
        train_val_data:  (list)   List of paths pointing to the remaining data
        n_val:           (int)    Number of samples in 'train_val_data' to use
                                  for validation - rest is used for training.
        args:            (tuple)  Parsed arguments, see argparser.
    """
    # Define train, val and test sub-dirs
    train_path = os.path.join(split_path, "train")
    val_path = os.path.join(split_path, "val") if n_val else None
    test_path = os.path.join(split_path, "test")

    # Create folders if not existing
    create_folders([train_path, val_path, test_path])

    # Copy or symlink?
    if args.copy:
        from shutil import copyfile
        move_func = copyfile
    elif args.file_list:
        move_func = _add_to_file_list_fallback
    else:
        move_func = os.symlink

    # Extract validation data from the remaining
    random.shuffle(train_val_data)
    validation = train_val_data[:n_val]
    training = train_val_data[n_val:]

    # Add training
    add_files(training, train_path, move_func)
    # Add test data
    add_files(test_split, test_path, move_func)
    if n_val:
        # Add validation
        add_files(validation, val_path, move_func)


def run(args):
    """
    Run the script according to 'args' - Please refer to the argparser.
    """
    data_dir = os.path.abspath(args.data_dir)
    n_splits = int(args.CV)
    if n_splits > 1:
        out_dir = os.path.join(data_dir, args.out_dir, "%i_CV" % n_splits)
    else:
        out_dir = os.path.join(data_dir, args.out_dir, "fixed_split")

    if n_splits == 1 and not args.test_fraction:
        raise ValueError("Must specify --test_fraction with --CV=1.")
    if args.copy and args.file_list:
        raise ValueError("Only one of --copy and --file_list "
                         "flags must be set.")
    if (args.test_fraction != _DEFAULT_TEST_FRACTION) and (args.CV > 1):
        raise ValueError("Should not set --test_fraction with --CV > 1")

    # Assert suitable folders
    assert_dir_structure(data_dir, out_dir)

    # Create sub-folders
    create_view_folders(out_dir, n_splits)

    # Get subject dirs
    subject_dirs = glob(os.path.join(data_dir, args.subject_dir_pattern))

    desc = "records"
    if args.common_prefix_length:
        print("OBS: Pairing files based on first "
              "{} characters".format(args.common_prefix_length))
        subject_dirs = pair_by_names(subject_dirs, args.common_prefix_length)
        desc = "subjects"

    if n_splits > len(subject_dirs):
        raise ValueError("CV ({}) cannot be larger than number of "
                         "subjects ({})".format(n_splits, len(subject_dirs)))

    # Get train/val/test sizes
    n_train, n_val, n_test = get_split_sizes(subject_dirs, n_splits, args, desc)

    # Shuffle and split the files into CV parts
    random.shuffle(subject_dirs)
    splits = np.array_split(subject_dirs, n_splits)

    # Symlink / copy files
    for split_index, test_split in enumerate(splits):
        print("  Split %i/%i" % (split_index + 1, n_splits),
              end="\r", flush=True)

        # Set root path to split folder
        if n_splits > 1:
            split_path = os.path.join(out_dir, "split_%i" % split_index)
        else:
            split_path = out_dir
            # Here we kind of hacky force the following code to work with CV=1
            # Define a test set and overwrite the current split
            # add the data, as splits was never split with n_splits=1
            test_split = splits[0][:n_test]

            # Overwrite the splits variable to a length 2 array with the
            # remaining data which will be used as val+train. The loop still
            # refers to the old split and thus will only execute once
            splits = [test_split, splits[0][n_test:]]

        # make flat list of remaining splits (train+val)
        train_val_data = [x for ind, x in enumerate(splits) if ind != split_index]
        train_val_data = [item for sublist in train_val_data for item in sublist]

        # Add/copy/symlink the files to the split directories
        run_on_split(split_path=split_path,
                     test_split=test_split,
                     train_val_data=train_val_data,
                     n_val=n_val,
                     args=args)


def entry_func(args=None):
    parser = get_argparser()
    run(parser.parse_args(args))


if __name__ == "__main__":
    entry_func()
