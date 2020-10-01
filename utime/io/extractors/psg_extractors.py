import numpy as np
import os
import h5py
from utime.utils import mne_no_log_context
from utime.errors import ChannelNotFoundError


def extract_from_edf(psg_file_path, header, include_channels, exclude_channels, **kwargs):
    """
    TODO

    Returns:
        ndarray, shape NxC
    """
    from mne.io import read_raw_edf
    # Channels longer than 16 characters are truncated to length 16.
    # Check no duplicates in include list
    exclude_channels_short = [s[:16] for s in exclude_channels]
    include_channels_short = [s[:16] for s in include_channels]
    if len(set(include_channels_short)) != len(include_channels_short):
        raise ValueError(f"Cannot load file {psg_file_path} with include_channels "
                         f"{include_channels} as duplicate channel names occur. "
                         f"This may be because two or more channel names of length > 16 "
                         f"were expected, but those were shortened to their maximum length of 16 "
                         f"by the EDF(+) format. Shortened channel names are: {include_channels_short}.")
    if set(exclude_channels_short) & set(include_channels_short):
        raise ValueError(f"Cannot load file {psg_file_path} with include_channels "
                         f"{include_channels} and exclude channels {exclude_channels} "
                         f"as one or more channel names occur in the exclude set that are also in the "
                         f"include set. "
                         f"This may be because two or more channel names of length > 16 "
                         f"were truncated to the same channel name of the EDF format maximum length of 16 "
                         f"Shortened channel names are (include): {include_channels_short} and "
                         f"(exclude): {exclude_channels_short}.")
    with mne_no_log_context():
        edf = read_raw_edf(psg_file_path, preload=False,
                           stim_channel=None, verbose=False,
                           exclude=exclude_channels_short)
        # Update header with actually used sample rate
        header["sample_rate"] = float(edf.info['sfreq'])
        return edf.get_data().T


def extract_from_wfdb(wfdb_file_path, include_channels, header, **kwargs):
    """
    TODO

    Returns:
        ndarray, shape NxC
    """
    from wfdb.io import rdrecord
    rec = rdrecord(record_name=os.path.splitext(wfdb_file_path)[0],
                   channel_names=include_channels)
    header["sample_rate"] = float(rec.fs)
    return rec.p_signal


def extract_from_pickle(pickle_file_path, header, include_channels, **kwargs):
    """
    TODO

    Args:
        ...
        data_dir: A path to the subject directory storing the actual data.
                  This is needed as the DCSM picle object only stores the name
                  of the channel to load within the subject directory.

    Returns:
        ndarray
    """
    import pickle
    with open(pickle_file_path, "rb") as in_f:
        chnl_dict = pickle.load(in_f)
    data_dir = header['data_dir']
    data = []
    for chnl in include_channels:
        path = os.path.join(data_dir, chnl_dict[chnl][0])
        dtype = os.path.splitext(path)[-1][1:]
        data.append(np.fromfile(path, dtype=np.dtype(dtype)))
    return np.array(data).T


def extract_from_h5(h5_file_path, include_channels, **kwargs):
    """
    TODO

    Returns:
        ndarray
    """
    data = []
    with h5py.File(h5_file_path, "r") as h5_file:
        for chnl in include_channels:
            data.append(np.array(h5_file["channels"][chnl]))
    return np.array(data).T


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
    psg_data = np.asarray(psg_data)

    if include_channels and psg_data.shape[1] != len(include_channels):
        raise ChannelNotFoundError("Unexpected channel loading error. "
                                   "Should have loaded {} channels ({}), "
                                   "but the PSG array has shape {}. There "
                                   "might be an error in the code. Please"
                                   " rais an issue on GitHub.".format(
            len(include_channels), include_channels, psg_data.shape
        ))
    return psg_data
