"""
Functions for loading various types of hypnogram/sleep stage/labels file
formats from disk. Returns raw data either as a numpy array or
StartDurationStageFormat tuples.

Functions in utime.io.extractors.hyp_extractors will convert these data types
into utime.hypnogram objects which are used for all downstream operations.
"""

import os
from utime.hypnogram.formats import StartDurationStageFormat


def load_hyp_edf(file_path):
    """
    Loader for hypnogram stored in EDF files in the EDF Annotations channel.
    Uses BaseEDFReader from .dhedreader to extract the data as
    Start-Duration-Stage lists. Returns data of type StartDurationStageFormat.

    See utime.hypnogram.formats.StartDurationStageFormat

    Returns:
        A StartDurationStageFormat object
    """
    from .dhedreader import BaseEDFReader
    with open(file_path, "rb") as in_f:
        # Get raw header
        base_edf = BaseEDFReader(in_f)
        base_edf.read_header()
        ann = tuple(zip(*tuple(base_edf.records())[0][-1]))
    return StartDurationStageFormat(ann)


def load_start_dur_stage(file_path):
    """
    Loader for CSV-like files that store hypnogram information in the
    Start-Duration-Stage format.
    See utime.hypnogram.formats.StartDurationStageFormat

    Returns:
        A StartDurationStageFormat object
    """
    import pandas as pd
    df = pd.read_csv(file_path, header=None)
    return StartDurationStageFormat(zip(*df.to_numpy()))


def load_np(file_path):
    """
    Loader for hypnograms stored in numpy arrays (npz, npy).

    Returns:
        np.ndarray of sleep stages
    """
    import numpy as np
    arr = np.load(file_path)
    if not isinstance(arr, np.ndarray):
        # npz
        keys = list(arr.keys())
        assert len(keys) == 1
        arr = arr[keys[0]]
    return arr


_EXT_TO_LOADER = {
    "edf": load_hyp_edf,
    "sds": load_start_dur_stage,
    "ids": load_start_dur_stage,
    "npz": load_np,
    "npy": load_np
}


def load_hyp_file(file_path):
    """ Load a hypnogram from a file at 'file_path' """
    extension = os.path.splitext(file_path)[-1].lower()[1:]
    return _EXT_TO_LOADER[extension](file_path)
