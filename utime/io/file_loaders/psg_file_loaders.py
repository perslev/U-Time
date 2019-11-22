"""
A collection of lower level data loader for various data files such as EDF,
EDF+, CSV, mat, numpy...
"""
import os
import numpy as np
from utime.utils import mne_no_log_context
from utime.errors import ChannelNotFoundError


def _get_excludes(channel_names, include_names):
    """
    Utility function that computes the channels to exclude from a list of all
    channels and a list of channels to include.
    If any channel in include_names is not in channel_names, raises a
    ChannelNotFoundError.
    """
    chan_error = ChannelNotFoundError("List of channels to include {} does not"
                                      " match the channels in the file ({})"
                                      "".format(include_names, channel_names))
    channel_names_set = set(channel_names)
    include_names_set = set(include_names)
    exclude_set = channel_names_set - include_names_set
    if len(exclude_set) != (len(channel_names_set) - len(include_names_set)):
        raise chan_error
    return list(exclude_set)


def load_edf(file_path, load_channels, **kwargs):
    """
    File loader function for .edf extension files.
    Redirects to mne.io.read_raw_edf.
    If load_channels is specified, read_raw_edf is called twice:
      1st time used to load the available channel names
      2nd time used to load with the 'exclude' parameter set - excluding all
        channel names in the file not in 'load_channels'.

    Returns:
        mne.io.RawEDF object
    """
    from mne.io import read_raw_edf
    edf = read_raw_edf(file_path, preload=False,
                       stim_channel=None, verbose=False)
    if not load_channels:
        return edf
    else:
        exclude = _get_excludes(edf.info["ch_names"], load_channels)
        return read_raw_edf(file_path, preload=False, exclude=exclude,
                            stim_channel=None, verbose=False)


def load_wfdb(file_path, load_channels, **kwargs):
    """
    File loader function for .dat and .mat WFDB extension files.
    Redirects to the wfdb.io.rdrecord function.

    Returns:
        wfdb.io.record.Record object
    """
    from wfdb.io import rdrecord
    rec = rdrecord(record_name=os.path.splitext(file_path)[0],
                   channel_names=load_channels)
    chans = rec.sig_name
    load_channels = load_channels or chans
    if not chans or not (np.array(load_channels) == np.array(chans)).all():
        from utime.errors import ChannelNotFoundError
        raise ChannelNotFoundError("Could not load channels {} "
                                   "from file {}. "
                                   "Actually loaded: {}".format(load_channels,
                                                                file_path,
                                                                chans))
    return rec


def load_fif(file_path, **kwargs):
    """
    File loader function for .fif and .raw.fif extension files.
    Redirects to the mne.io.read_raw_fif function.

    Loads all channels, non-needed channels are dropped in the extract function.
    See utime.io.extractors.psg_extractors

    Returns:
        mne.io.Raw fif data array
    """
    from mne.io import read_raw_fif
    with mne_no_log_context():
        raw = read_raw_fif(file_path)
    return raw


def load_dcsm_pickle(pickle_path, **kwargs):
    """
    File loader function for .picle extension files.
    Used only for the DCSM (private) dataset.
    A dictionary representation of the pickled data is returned, but actual
    data is loaded in the extract function
    (see utime.io.extractors.psg_extractors).

    Returns:
        A DCSMDict dictionary
    """
    class DCSMDict(dict): pass  # We define a recognizably named type
    import pickle
    with open(pickle_path, "rb") as in_f:
        signal_pairs = pickle.load(in_f)
    return DCSMDict(signal_pairs)


def load_h5(h5_path, **kwargs):
    """
    File loader function for .h5 extension files.
    A h5py.File object is returned, but actual data is loaded in the extract
    function (see utime.io.extractors.psg_extractors).

    Returns:
        An open (read) h5py.File object
    """
    import h5py
    return h5py.File(h5_path, "r")


_EXT_TO_LOADER = {
    "edf": load_edf,
    "mat": load_wfdb,
    "dat": load_wfdb,
    "raw.fif": load_fif,
    "fif": load_fif,
    "pickle": load_dcsm_pickle,
    "h5": load_h5
}

_ALLOWED_CHAN_SYNONYMS = (("A1", "M1"), ("A2", "M2"),
                          ("ROC", "E2"), ("LOC", "E1"))


def load_psg_file(file_path, load_channels=None):
    """
    Loads a PSG-type file at path 'file_path'.
    load_channels is passed to the loader, which may allow some leaders to
    selectively load only data that is needed from disk.
    However, in some cases, the load_channels argument has no effect at this
    time, as loading of data from disk may occur only later in the
    'utime.io.extractors.extract_psg_data' function.
    """
    fname = os.path.split(os.path.abspath(file_path))[-1]
    _, ext = os.path.splitext(fname)
    load_func = _EXT_TO_LOADER[ext[1:]]
    return load_func(file_path, load_channels=load_channels)
