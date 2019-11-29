import numpy as np
import os
from pandas import DataFrame
from utime.utils import mne_no_log_context
from utime.errors import ChannelNotFoundError


def extract_from_edf(psg_file_path, exclude_channels, **kwargs):
    """
    TODO

    Returns:
        pandas.DataFrame
    """
    from mne.io import read_raw_edf
    with mne_no_log_context():
        return read_raw_edf(psg_file_path, preload=False,
                            stim_channel=None, verbose=False,
                            exclude=exclude_channels).to_data_frame()


def extract_from_wfdb(wfdb_file_path, include_channels, **kwargs):
    """
    TODO

    Returns:
        ndarray
    """
    from wfdb.io import rdrecord
    return rdrecord(record_name=os.path.splitext(wfdb_file_path)[0],
                    channels=include_channels).p_signal


def extract_from_pickle(pickle_file_path, header, include_channels, **kwargs):
    """
    TODO

    Args:
        ...
        data_dir: A path to the subject directory storing the actual data.
                  This is needed as the DCSM picle object only stores the name
                  of the channel to load within the subject directory.

    Returns:
        pandas.DataFrame object
    """
    import pickle
    with open(pickle_file_path, "rb") as in_f:
        chnl_dict = pickle.load(in_f)
    data_dir = header['data_dir']
    data = {}
    for chnl in include_channels:
        path = os.path.join(data_dir, chnl_dict[chnl][0])
        dtype = os.path.splitext(path)[-1][1:]
        data[chnl] = np.fromfile(path, dtype=np.dtype(dtype))
    return DataFrame(data=data)


def extract_from_h5(h5_file_path, include_channels, **kwargs):
    """
    TODO

    Returns:
        pandas.DataFrame object
    """
    data = {}
    import h5py
    with h5py.File(h5_file_path, "r") as h5_file:
        for chnl in include_channels:
            data[chnl] = h5_file["channels"][chnl]
    return DataFrame(data=data)


_EXT_TO_LOADER = {
    "edf": extract_from_edf,
    "mat": extract_from_wfdb,
    "dat": extract_from_wfdb,
    "pickle": extract_from_pickle,
    "h5": extract_from_h5
}


def extract_psg_data(psg_file_path, header, include_channels, exclude_channels):
    """
    Extract final ndarray data from the PSG object
    """
    fname = os.path.split(os.path.abspath(psg_file_path))[-1]
    _, ext = os.path.splitext(fname)
    load_func = _EXT_TO_LOADER[ext[1:]]
    psg_data = load_func(psg_file_path,
                         header=header,
                         include_channels=include_channels,
                         exclude_channels=exclude_channels)

    # Convert to float32 ndarray
    psg_data = np.array(psg_data, dtype=np.float32)

    if include_channels and psg_data.shape[1] != len(include_channels):
            raise ChannelNotFoundError("Unexpected channel loading error. "
                                       "Should have loaded {} channels ({}), "
                                       "but the PSG array has shape {}. There "
                                       "might be an error in the code. Please"
                                       " rais an issue on GitHub.".format(
                len(include_channels), include_channels, psg_data.shape
            ))
    return psg_data
