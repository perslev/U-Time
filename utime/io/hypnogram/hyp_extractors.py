"""
Functions for loading various types of hypnogram/sleep stage/labels file
formats from disk. Returns raw data either as a numpy array or
StartDurationStageFormat tuples.

Functions in utime.io.extractors.hyp_extractors will convert these data types
into utime.hypnogram objects which are used for all downstream operations.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utime import Defaults
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


def extract_from_stg_txt(file_path, period_length_sec, sample_rate):
    import pandas as pd
    import numpy as np
    df = pd.read_csv(file_path, delimiter="\t")
    epoch, stages = df['Epoch'].values, df['User-Defined Stage'].values
    map_ = np.vectorize(
        {0: Defaults.AWAKE[0],
         1: Defaults.NON_REM_STAGE_1[0],
         2: Defaults.NON_REM_STAGE_2[0],
         3: Defaults.NON_REM_STAGE_3[0],
         4: Defaults.NON_REM_STAGE_3[0],
         5: Defaults.REM[0],
         7: Defaults.UNKNOWN[0]}.get
    )
    # Map integer stages to default string values
    stages = map_(stages)
    # Insert UNKNOWN stages if there are gaps in 'epoch' list of epoch inds
    stages_proccessed = []
    for epoch_ind, stage in zip(epoch, stages):
        # Note: epoch_ind is 1 indexed, i.e. the first epoch has epoch ind '1' in file.
        n_missing = epoch_ind - (len(stages_proccessed) + 1)
        stages_proccessed.extend([Defaults.UNKNOWN[0]] * n_missing)
        stages_proccessed.append(stage)
    return ndarray_to_ids_format(
        array=stages,
        period_length_sec=period_length_sec,
        sample_rate=sample_rate
    )


def absolute_time_stages_to_relative_ids(start_times, stages_dense, start_date,
                                         period_length_sec, event_date_fmt,
                                         event_length="fixed"):
    if event_length not in ('fixed', 'fill'):
        raise ValueError("Parameter 'event_length' must be one of ['fixed', 'fill'], got '{}'".format(event_length))
    # Filter Nones/False/empty from stages
    start_times, stages_dense = zip(*list(filter(lambda s: s[1], zip(start_times, stages_dense))))
    inits, durs, stages = [], [], []
    for event_index, (time, event) in enumerate(zip(start_times, stages_dense)):
        event_time = datetime.strptime(time, event_date_fmt)
        event_time = event_time.replace(year=start_date.year,
                                        month=start_date.month,
                                        day=start_date.day,
                                        tzinfo=start_date.tzinfo)
        diff_sec = (event_time - start_date).total_seconds()
        while diff_sec < 0:
            # Event time went to (one of) next day(s)
            event_time = event_time + timedelta(days=1)
            diff_sec = (event_time - start_date).total_seconds()
            assert np.isclose(diff_sec, int(diff_sec)), "Only implemented for whole second event times."
            diff_sec = int(diff_sec)
        if stages and (event == stages[-1] and (inits[-1] + durs[-1]) == diff_sec):
            # Continued stage, update last entry
            durs[-1] += period_length_sec
        else:
            # New event
            inits.append(diff_sec)
            durs.append(period_length_sec)
            stages.append(event)
    if event_length == "fill":
        # Update durations to span from event index i to event index i+1
        for i in range(len(inits)-1):
            durs[i] = inits[i+1] - inits[i]
    return StartDurationStageFormat((inits, durs, stages))


def extract_from_wsc_allscore(file_path, period_length_sec, sample_rate, event_date_fmt='%H:%M:%S.%f'):
    df = pd.read_csv(file_path, delimiter="\t", header=None, encoding="latin1")
    times, events = map(list, (df[0].values, df[1].values))
    start_time = datetime.strptime(times[events.index("START RECORDING")], event_date_fmt)
    map_ = np.vectorize(
        {"STAGE - W": Defaults.AWAKE[0],
         "STAGE - N1": Defaults.NON_REM_STAGE_1[0],
         "STAGE - N2": Defaults.NON_REM_STAGE_2[0],
         "STAGE - N3": Defaults.NON_REM_STAGE_3[0],
         "STAGE - R": Defaults.REM[0]}.get
    )
    stages = map_(list(map(lambda s: str(s).strip(), events)))
    return absolute_time_stages_to_relative_ids(
        start_times=times,
        stages_dense=stages,
        start_date=start_time,
        event_length="fill",
        period_length_sec=period_length_sec,
        event_date_fmt=event_date_fmt
    )


def extract_from_stages_csv(file_path, period_length_sec, sample_rate, event_date_fmt='%H:%M:%S'):
    import mne
    # Load the file start time
    edf_path = file_path.replace(".csv", ".edf")
    if not os.path.exists(edf_path):
        raise OSError("The hyp loader 'extract_from_stages_csv' requires an EDF file at path '{}' when processing "
                      "hypnogram in file '{}' in order to infer correct event start times relative to the EDF file. "
                      "However, no EDF file exists at the path.".format(edf_path, file_path))
    start_date = mne.io.read_raw_edf(edf_path, preload=False).info['meas_date']
    if not start_date:
        raise ValueError("Recording has no start time in EDF file at path '{}'. Cannot infer relative event start "
                         "times in event file '{}'".format(edf_path, file_path))
    df = pd.read_csv(file_path, names=['Start Time', 'Duration (seconds)', 'Event'])
    map_ = np.vectorize(
        {"Wake": Defaults.AWAKE[0],
         "Stage1": Defaults.NON_REM_STAGE_1[0],
         "Stage2": Defaults.NON_REM_STAGE_2[0],
         "Stage3": Defaults.NON_REM_STAGE_3[0],
         "REM": Defaults.REM[0]}.get
    )
    stages = map_(list(map(lambda s: str(s).strip(), df['Event'].values)))
    return absolute_time_stages_to_relative_ids(
        start_times=df['Start Time'],
        stages_dense=stages,
        start_date=start_date,
        event_length="fixed",
        period_length_sec=period_length_sec,
        event_date_fmt=event_date_fmt
    )


_EXTRACT_FUNCS = {
    "edf": extract_from_edf,
    "sds": extract_from_start_dur_stage,
    "ids": extract_from_start_dur_stage,
    "xml": extract_from_xml,
    "npz": extract_from_np,
    "npy": extract_from_np,
    "stg.txt": extract_from_stg_txt,
    "wsc_allscore": extract_from_wsc_allscore,
    "STAGES": extract_from_stages_csv
}


def extract_ids_from_hyp_file(file_path, period_length_sec=None, sample_rate=None, extract_func=None):
    """
    Entry function for extracing start-duration-stage format data from variable input files

    Args:
        file_path: str path to hypnogram file
        period_length_sec: integer or None - only used for loading ndarray data. If None, ndarray must be
                           dense (not signal-dense)
        sample_rate: integer or None - - only used for loading ndarray data. If None, ndarray must be
                     dense (not signal-dense)
        extract_func: callable, str or None: Callable or string identifier for callable as registered in _EXTRACT_FUNCS.
                                             If None, the file extension is used as string identifier, e.g. 'file.ids'
                                             will be loaded by the callable in _EXTRACT_FUNCS['ids'].

    Returns:
        A StartDurationStageFormat object
    """
    if extract_func is None:
        extract_func = os.path.split(file_path)[-1].split('.', 1)[-1].lower()
    if not callable(extract_func):
        extract_func = _EXTRACT_FUNCS[extract_func]
    return extract_func(file_path=file_path,
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
