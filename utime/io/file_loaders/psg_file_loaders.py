"""
A collection of lower level data loader for various data files such as EDF,
EDF+, CSV, mat, numpy...
"""
import os
import warnings
from utime.utils import mne_no_log_context
from utime.io.extractors.header_extractors import extract_header


def read_edf_header(file_path, **kwargs):
    """
    Header reader function for .edf extension files.
    Redirects to mne.io.read_raw_edf.

    Returns:
        A dictionary of header information
    """
    from mne.io import read_raw_edf
    with mne_no_log_context(), warnings.catch_warnings(record=True) as warns:
        warnings.filterwarnings('default')
        raw_edf = read_raw_edf(file_path, preload=False,
                               stim_channel=None, verbose=False)
    header = extract_header(raw_edf)
    if warns:
        duplicates = str(warns[0]).split("{")[-1].split("}")[0].replace("'", "").split(",")
        duplicates = [d.strip() for d in duplicates]
    else:
        duplicates = []
    header['duplicates'] = duplicates
    return header


def read_wfdb_header(file_path, **kwargs):
    """
    Header reader function for .dat and .mat WFDB extension files.
    Redirects to the wfdb.io.rdheader function.

    Returns:
        A dictionary of header information
    """
    from wfdb.io import rdheader
    return extract_header(rdheader(record_name=os.path.splitext(file_path)[0]))


def open_dcsm_pickle(pickle_path, **kwargs):
    """
    Header reader function for .picle extension files.
    Used only for the DCSM (private) dataset.

    Returns:
        A dictionary of header information
    """
    class DCSMDict(dict): pass  # We define a recognizably named type
    import pickle
    with open(pickle_path, "rb") as in_f:
        signal_pairs = pickle.load(in_f)
    return extract_header(DCSMDict(signal_pairs))


def read_h5_header(h5_path, **kwargs):
    """
    Header reader function for .h5 extension files.

    Returns:
        A dictionary of header information
    """
    import h5py
    with h5py.File(h5_path, "r") as psg_obj:
        return extract_header(psg_obj)


_EXT_TO_LOADER = {
    "edf": read_edf_header,
    "mat": read_wfdb_header,
    "dat": read_wfdb_header,
    "pickle": open_dcsm_pickle,
    "h5": read_h5_header
}


def read_psg_header(file_path):
    """
    Loads the header from a PSG-type file at path 'file_path'.
    """
    fname = os.path.split(os.path.abspath(file_path))[-1]
    _, ext = os.path.splitext(fname)
    load_func = _EXT_TO_LOADER[ext[1:]]
    header = load_func(file_path)
    # Add file location data
    file_path, file_name = os.path.split(file_path)
    header['data_dir'] = file_path
    header["file_name"] = file_name
    return header
