"""
Functions for loading various types of hypnogram/sleep stage/labels file
formats from disk. Returns raw data either as a numpy array or
StartDurationStageFormat tuples.

Functions in utime.io.extractors.hyp_extractors will convert these data types
into utime.hypnogram objects which are used for all downstream operations.
"""

import os
from utime.hypnogram.formats import StartDurationStageFormat
from utime.hypnogram.utils import (sparse_hypnogram_from_ids_format,
                                   sparse_hypnogram_from_array)


def extract_from_edf(file_path, period_length_sec, annotation_dict, **kwargs):
    """
    Loader for hypnogram stored in EDF files in the EDF Annotations channel.
    Uses BaseEDFReader from .dhedreader to extract the data as
    Start-Duration-Stage lists. Returns data of type StartDurationStageFormat.

    See utime.hypnogram.formats.StartDurationStageFormat

    Returns:
        A SparseHypnogram object, annotation dict
    """
    from .dhedreader import BaseEDFReader
    with open(file_path, "rb") as in_f:
        # Get raw header
        base_edf = BaseEDFReader(in_f)
        base_edf.read_header()
        ann = tuple(zip(*tuple(base_edf.records())[0][-1]))
    return sparse_hypnogram_from_ids_format(
        ids_tuple=StartDurationStageFormat(ann),
        period_length_sec=period_length_sec,
        ann_to_class=annotation_dict
    )


def extract_from_start_dur_stage(file_path, period_length_sec, annotation_dict, **kwargs):
    """
    Loader for CSV-like files that store hypnogram information in the
    Start-Duration-Stage format.
    See utime.hypnogram.formats.StartDurationStageFormat

    Returns:
        A SparseHypnogram object, annotation dict
    """
    import pandas as pd
    df = pd.read_csv(file_path, header=None)
    return sparse_hypnogram_from_ids_format(
            ids_tuple=StartDurationStageFormat(zip(*df.to_numpy())),
            period_length_sec=period_length_sec,
            ann_to_class=annotation_dict
        )


def extract_from_np(file_path, period_length_sec, annotation_dict, sample_rate):
    """
    Loader for hypnograms stored in numpy arrays (npz, npy).

    Returns:
        A SparseHypnogram object, annotation dict
    """
    import numpy as np
    arr = np.load(file_path)
    if not isinstance(arr, np.ndarray):
        # npz
        keys = list(arr.keys())
        assert len(keys) == 1
        arr = arr[keys[0]]
    return sparse_hypnogram_from_array(
        array=arr,
        period_length_sec=period_length_sec,
        ann_to_class=annotation_dict,
        sample_rate=sample_rate
    )


_EXT_TO_LOADER = {
    "edf": extract_from_edf,
    "sds": extract_from_start_dur_stage,
    "ids": extract_from_start_dur_stage,
    "npz": extract_from_np,
    "npy": extract_from_np
}


def extract_hyp_data(file_path, period_length_sec, annotation_dict, sample_rate):
    """
    Load a hypnogram from a file at 'file_path'

    Returns:
        A SparseHypnogram object, annotation dict
    """
    extension = os.path.splitext(file_path)[-1].lower()[1:]
    return _EXT_TO_LOADER[extension](file_path, period_length_sec, annotation_dict, sample_rate)
