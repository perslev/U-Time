import numpy as np
import os
from pandas import DataFrame
from utime.utils import mne_no_log_context
from utime.errors import ChannelNotFoundError


def extract_from_mne_raw(edf_obj, **kwargs):
    """
    Function for loading the actual data from a RawEDF and Raw object.
    Note: Channels have already been selected at RawEDF/Raw init time.
    We simply return the data here.

    Returns:
        pandas.DataFrame
    """
    with mne_no_log_context():  # prevents unimportant logging info
        df = edf_obj.to_data_frame()
    return df


def extract_from_wfdb(record_obj, **kwargs):
    """
    Function for loading the actual data from a WFDB Record object.
    Note: Channels have already been selected at Record init time.
    We simply return the data here.

    Returns:
        ndarray
    """
    return record_obj.p_signal


def load_dcsm_dict(chnl_dict, load_channels, data_dir, **kwargs):
    """
    Function for loading the actual data from a DCSMDict PSG object.
    If specified, loads only channels from the 'load_channels' list

    Args:
        ...
        data_dir: A path to the subject directory storing the actual data.
                  This is needed as the DCSM picle object only stores the name
                  of the channel to load within the subject directory.

    Returns:
        pandas.DataFrame object
    """
    if not load_channels:
        load_channels = list(chnl_dict.keys())
    data = {}
    for chnl in load_channels:
        path = os.path.join(data_dir, chnl_dict[chnl][0])
        dtype = os.path.splitext(path)[-1][1:]
        try:
            data[chnl] = np.fromfile(path, dtype=np.dtype(dtype))
        except FileNotFoundError as e:
            raise ChannelNotFoundError from e
    return DataFrame(data=data)


def load_h5_file(h5_file, load_channels, **kwargs):
    """
    Function for loading the actual data from a h5py.File PSG object.
    If specified, loads only channels from the 'load_channels' list

    Returns:
        pandas.DataFrame object
    """
    if not load_channels:
        load_channels = list(h5_file["channels"].keys())
    data = {}
    for chnl in load_channels:
        try:
            data[chnl] = h5_file["channels"][chnl]
        except KeyError as e:
            raise ChannelNotFoundError from e
    return DataFrame(data=data)


_OBJ_TYPE_TO_DATA_LOADER = {
    "RawEDF": extract_from_mne_raw,
    "Raw": extract_from_mne_raw,
    "Record": extract_from_wfdb,
    "DCSMDict": load_dcsm_dict,
    "File": load_h5_file
}


def extract_psg_data(psg_obj, load_channels, **kwargs):
    """
    Extract final ndarray data from the PSG object
    """
    loader = _OBJ_TYPE_TO_DATA_LOADER[type(psg_obj).__name__]
    psg_data = loader(psg_obj, load_channels=load_channels, **kwargs)

    # Convert to float32 ndarray
    psg_data = np.array(psg_data, dtype=np.float32)

    if load_channels and psg_data.shape[1] != len(load_channels):
            raise ChannelNotFoundError("Unexpected channel loading error. "
                                       "Should have loaded {} channels ({}), "
                                       "but the PSG array has shape {}. There "
                                       "might be an error in the code. Please"
                                       " rais an issue on GitHub.".format(
                len(load_channels), load_channels, psg_data.shape
            ))
    return psg_data
