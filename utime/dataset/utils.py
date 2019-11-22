"""
A set of helper functions for finding subject folders and PSG/HYP file pairs
within such.

These functions are used in particular by the utime.dataset SleepStudy and
SleepStudyDataset objects.
"""

import os
import re


def unpack_file_list(file_list_path):
    """
    Utility function typically used to read the filenames stored in a
    LIST_OF_FILES.txt file (created by the utime.bin.cv_split script in some
    circumstances).

    Simply reads all lines of the specified file and returns them as a list,
    ignoring empty lines.
    """
    with open(file_list_path, "r") as in_f:
        files = list(map(lambda x: x.strip('\n'),
                         filter(None, in_f.readlines())))
    return files


def filter_by_regex(items, regex):
    """
    Helper function that matches regex 'regex' to all elements of a list
    'items' and returns a boolean array of matches of len == len(items).
    """
    return list(filter(lambda x: re.match(regex, x), items))


def match_n_in_folder(folder_path, regex, num_expected_matches=1):
    """
    Given a path to a folder 'folder_path' and a regex 'regex' return the
    'num_expected_matches' number of filenames within the folder that matches
    the regex. If a number != num_expected_matches of filenames match, raises
    a ValueError.

    The regex is always prepended with '.*?' (match a minimum number of needed
    preceding characters).

    Args:
        folder_path:            A path to a directory with files to match to
        regex:                  A regex pattern (string) to match files against
        num_expected_matches:   The number of expected filenames that matches
                                the passed regex within the folder.

    Returns:
        num_expected_matches filenames that match regex within folder_path
    """
    content = os.listdir(folder_path)
    regex = re.compile(".*?" + regex.lstrip(".*?"))
    matches = filter_by_regex(content, regex)
    if len(matches) != num_expected_matches:
        raise ValueError("Found invalid number of valid matches ({}; {}) for "
                         "files in folder {} with content {} and "
                         "regex {} (expected exactly {})."
                         "".format(len(matches), matches, folder_path,
                                   os.listdir(folder_path), regex,
                                   num_expected_matches))
    else:
        return matches


def infer_hyp_file(subject_dir,
                   tries=("hypnogram", "hypno", "hyp", "stage", "label")):
    """
    Attempts to infer which of the files stored in 'subject_dir' corresponds
    to a hypnogram/sleep stages/labels file using simple file-name matching
    rules. Specifically, a filename is considered a potential label file if
    it (as lower-case) contains any of the strings in the tries-tuple.

    Args:
        subject_dir: A path to a directory that stores a hypnogram/labels file
        tries:       A tuple of strings that are matched to sub-strings of the
                     filenames of subject_dir entries.
    Returns:
        If only 1 match is found, returns the filename that matches
    Raises:
        if != 1 matches raises a RuntimeError.
    """
    def match(item, name):
        return name in item.lower()
    content = os.listdir(subject_dir)
    for tag in tries:
        matches = [f for f in content if match(f, tag)]
        if len(matches) == 1:
            return matches[0]
    raise RuntimeError("Could not uniqely infer hypnogram file from "
                       "subject_dir {} with content {} and key-word "
                       "tries {}".format(subject_dir, content, tries))


def infer_psg_file(subject_dir, excludes, file_types):
    """
    Attempts to infer which of the files stored in 'subject_dir' corresponds
    to a PSG/data file using simple file-extension matching rules.
    Specifically, a filename is considered a potential PSG/data file if its
    extension matches any listed in 'file_types' and the the filename is not
    listed in 'excludes'.

    Args:
        subject_dir: A path to a directory that stores a hypnogram/labels file
        excludes:    A list of filenames to ignore/not consider
        file_types:  A list of file extensions to match files with
    Returns:
        If only 1 match is found, returns the filename that matches
    Raises:
        if != 1 matches raises a RuntimeError.
    """
    content = os.listdir(subject_dir)
    exts = [os.path.splitext(s)[-1] for s in content]
    all_matches = []
    for t in file_types:
        regex = ".*{}".format(t.strip("."))
        matches = [content[i] for i, ext in enumerate(exts) if
                   re.match(regex, ext) and content[i] not in excludes]
        all_matches.extend(matches)
    if len(all_matches) > 1:
        raise RuntimeError("Found multiple potential PSG files: "
                           "{}".format(all_matches))
    if len(all_matches) == 0:
        raise RuntimeError("Could not detect any potential PSG files at "
                           "subject directory {}. This is most likely because "
                           "both PSG and HYP data is stored in a single .edf "
                           "file, which is currently not "
                           "supported. Otherwise, the file may have an "
                           "unexpected extension type; Please raise an issue "
                           "on GitHub".format(subject_dir))
    return all_matches[0]


def find_hyp_file(subject_dir,
                  hyp_regex=None,
                  tries=("hypnogram", "hypno", "hyp", "stages")):
    """
    Find a HYP/sleep stages/labels file in a folder 'subject_dir' using either
    a passed regex 'hyp_regex' that should match uniquely to 1 file in the
    folder, or attempt to detect the file using simple filename similarities.

    See 'infer_hyp_file' for details on the latter approach.

    Args:
        subject_dir: A path to a directory that stores a HYP/labels file
        hyp_regex:   Optional regex used to match select the HYP/labels file
        tries:       A tuple of strings that are matches against filenames to
                     infer a possible HYP file automatically (w.o. hyp_regex).

    Returns:
        The filename of a single HYP/labels file under folder 'subject_dir'
    """
    if hyp_regex:
        return match_n_in_folder(subject_dir, hyp_regex, 1)[0]
    else:
        # Find a hyp file
        return infer_hyp_file(subject_dir,
                              tries=tries)


def find_psg_file(subject_dir,
                  psg_regex=False,
                  excludes=None,
                  file_types=("edf", "np", "npz", "mat", "fif",
                              "pickle", "h5", "dat")):
    """
    Find a PSG/data file in a folder 'subject_dir' using either a passed regex
    'psg_regex' that should match uniquely to 1 file in the folder,
    or chose the file based simply on its filename extension.

    See 'infer_psg_file' for details on the latter approach.

    Args:
        subject_dir: A path to a directory that stores a PSG/data file
        psg_regex:   Optional regex used to match select the PSG/data file
        excludes:    A list of filenames to ignore/not consider
        file_types:  A list of file extensions to match files with

    Returns:
        The filename of a single HYP/labels file under folder 'subject_dir'
    """
    if psg_regex:
        return match_n_in_folder(subject_dir, psg_regex, 1)[0]
    else:
        return infer_psg_file(subject_dir,
                              excludes=excludes,
                              file_types=file_types)


def find_psg_and_hyp(subject_dir,
                     psg_regex=None,
                     hyp_regex=None,
                     no_hypnogram=False):
    """
    Wrapper around the find_psg and find_hyp functions.
    Detects the PSG and HYP (unless no_hypnogram=True) file(s) within folder
    'subject_dir'  and returns their absolute paths.
    Asserts that the found PSG and HYP files are not the same, otherwise raises
    a RuntimeError.

    Args:
        subject_dir:  A path to a directory that stores a PSG/data file
        psg_regex:    Optional regex used to match select the PSG/data file
        hyp_regex:    Optional regex used to match select the HYP/labels file
        no_hypnogram: Do not look for a hypnogram/sleep stage/label file

    Returns:
        If not no_hypnogram, returns two strings, absaloute paths pointing to
        the PSG and HYP files.
        If no_hypnogram, returns string (PSG path) and None
    """
    if not no_hypnogram:
        hyp_file = find_hyp_file(subject_dir, hyp_regex)
    else:
        hyp_file = ""
    psg_file = find_psg_file(subject_dir,
                             psg_regex=psg_regex,
                             excludes=(hyp_file,))
    if hyp_file == psg_file:
        raise RuntimeError("OBS: A file was found to be both a valid PSG and "
                           "hyp file according to the regex {} and {} "
                           "matching {} and {} respectively.".format(
            psg_regex, hyp_regex, psg_file, hyp_file
        ))
    psg_file = os.path.join(subject_dir, psg_file)
    if not no_hypnogram:
        hyp_file = os.path.join(subject_dir, hyp_file)
    return psg_file, hyp_file or None


def find_subject_folders(data_dir, folder_regex=None):
    """
    Returns all non-hidden sub-folders within folder 'data_dir' that matches
    'folder_regex', if specified, otherwise all non-hidden sub-folders.

    If a folder_regex is not specified, a file named 'LIST_OF_FILES.txt'
    exactly will also be considered. This file is created by the
    utime.bin.cv_split script in some circumstances, and stores paths to
    subject_dirs.

    If the LIST_OF_FILES.txt file is found (using either method) the file will
    be unpacked (/read, see unpack_file_list) and its content extended with
    other subject_dirs found.

    Args:
        data_dir:       A path to a directory storing one or more subject
                        sub-directories.
        folder_regex:   An optional regex used to specifically match folders
                        within data_dir to return.

    Returns:
        A list of paths to subject directories.
    """
    folder_content = os.listdir(data_dir)
    if folder_regex:
        # Take only folders under data_dir matching folder_regex
        matches = filter_by_regex(folder_content, folder_regex)
        subject_dirs = [os.path.join(data_dir, m) for m in matches]
        f = lambda p: os.path.isdir(p) and not os.path.split(p)[-1].startswith(".")
        subject_dirs = list(filter(f, subject_dirs))
    else:
        # Take all (non-hidden) folders
        d = os.listdir(data_dir)
        fp = lambda p: (os.path.isdir(p) and os.path.split(p)[-1][0] != ".")
        subject_dirs = list(filter(fp, [os.path.join(data_dir, s) for s in d]))

    # Include and unpack potential LIST_OF_FILES documents
    subject_dirs.extend([os.path.join(data_dir, p) for p in folder_content
                         if p == "LIST_OF_FILES.txt"])
    subject_dirs_unpacked = []
    for i, dir_ in enumerate(subject_dirs):
        if "LIST_OF_FILES" in dir_:
            subject_dirs_unpacked.extend(unpack_file_list(dir_))
        else:
            subject_dirs_unpacked.append(dir_)
    return subject_dirs_unpacked
