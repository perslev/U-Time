"""
Functions for loading various types of hypnogram/sleep stage/labels file
formats from disk. Returns raw data either as a numpy array or
StartDurationStageFormat tuples.

Functions in utime.io.extractors.hyp_extractors will convert these data types
into utime.hypnogram objects which are used for all downstream operations.
"""

import os
from utime.hypnogram.formats import StartDurationStageFormat
from utime.hypnogram.utils import sparse_hypnogram_from_ids_format, ndarray_to_ids_format


def extract_from_edf(file_path, **kwargs):
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


def extract_from_start_dur_stage(file_path, **kwargs):
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


def extract_from_xml(file_path, **kwargs):
    """
    Extracts hypnograms from NSRR XML formatted annotation files.

    Returns:
        A StartDurationStageFormat object
    """
    import xml.etree.ElementTree as ET
    events = ET.parse(file_path).findall('ScoredEvents')
    assert len(events) == 1
    stage_dict = {
        "Wake|0": "W",
        "Stage 1 sleep|1": "N1",
        "Stage 2 sleep|2": "N2",
        "Stage 3 sleep|3": "N3",
        "Stage 4 sleep|4": "N3",
        "REM sleep|5": "REM",
        "Movement|6": "UNKNOWN",
        "Unscored|9": "UNKNOWN"
    }
    starts, durs, stages = [], [], []
    for event in events[0]:
        if not event[0].text == "Stages|Stages":
            continue
        stage = stage_dict[event[1].text]
        start = int(float(event[2].text))
        dur = int(float(event[3].text))
        starts.append(start)
        durs.append(dur)
        stages.append(stage)
    return StartDurationStageFormat((starts, durs, stages))


def extract_from_np(file_path, period_length_sec, sample_rate):
    """
    Loader for hypnograms stored in numpy arrays (npz, npy).

    Returns:
        A StartDurationStageFormat object
    """
    import numpy as np
    arr = np.load(file_path)
    if not isinstance(arr, np.ndarray):
        # npz
        keys = list(arr.keys())
        assert len(keys) == 1
        arr = arr[keys[0]]
    return ndarray_to_ids_format(
        array=arr,
        period_length_sec=period_length_sec,
        sample_rate=sample_rate
    )


_EXT_TO_LOADER = {
    "edf": extract_from_edf,
    "sds": extract_from_start_dur_stage,
    "ids": extract_from_start_dur_stage,
    "xml": extract_from_xml,
    "npz": extract_from_np,
    "npy": extract_from_np
}


def extract_ids_from_hyp_file(file_path, period_length_sec=None, sample_rate=None):
    """
    Entry function for extracing start-duration-stage format data from variable input files

    Args:
        file_path: str path to hypnogram file
        period_length_sec: integer or None - only used for loading ndarray data. If None, ndarray must be
                           dense (not signal-dense)
        sample_rate: integer or None - - only used for loading ndarray data. If None, ndarray must be
                     dense (not signal-dense)

    Returns:
        A StartDurationStageFormat object
    """
    extension = os.path.splitext(file_path)[-1].lower()[1:]
    return _EXT_TO_LOADER[extension](file_path=file_path,
                                     period_length_sec=period_length_sec,
                                     sample_rate=sample_rate)


def extract_hyp_data(file_path, period_length_sec, annotation_dict, sample_rate):
    """
    Load a hypnogram from a file at 'file_path'

    Returns:
        A SparseHypnogram object, annotation dict
    """
    ids_tuple = extract_ids_from_hyp_file(file_path, period_length_sec, sample_rate)
    return sparse_hypnogram_from_ids_format(
        ids_tuple=ids_tuple,
        period_length_sec=period_length_sec,
        ann_to_class=annotation_dict
    )
