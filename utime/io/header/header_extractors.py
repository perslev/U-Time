"""
A collection of functions for extracting a header of the following format:

{
    'n_channels': int,
    'channel_names': list of strings,
    'sample_rate': int
    'date': datetime or None
    'length_sec': float
}
"""
import os
import warnings
from utime.utils import mne_no_log_context
from utime.io.header.header_standardizers import (_standardized_edf_header,
                                                  _standardized_wfdb_header,
                                                  _standardized_h5_header)


def extract_edf_header(file_path):
    """
    Header reader function for .edf extension files.
    Redirects to mne.io.read_raw_edf.

    Returns:
        A dictionary of header information
    """
    from mne.io import read_raw_edf
    with mne_no_log_context(), warnings.catch_warnings(record=True) as _:
        warnings.filterwarnings('default')
        raw_edf = read_raw_edf(file_path, preload=False,
                               stim_channel=None, verbose=False)
    header = _standardized_edf_header(raw_edf)

    # Manually read channel names as-are in file without renaming, truncations etc. that
    # may be applied by MNE (as of v0.21) to ensure we exclude using the proper names.
    with open(file_path, "rb") as in_f:
        in_f.seek(252)
        n_channels = int(in_f.read(4).decode().rstrip('\x00'))
        channel_names = [in_f.read(16).strip().decode('latin-1') for _ in range(n_channels)]
        # Ignore EDF TAL channels, as is done in MNE too (v0.21)
        channel_names = [chan for chan in channel_names if "EDF Annotatio" not in chan]
        assert len(channel_names) == header["n_channels"], "Manually loaded number of channels " \
                                                           "does not match number read by MNE"
        header["channel_names"] = channel_names
    return header


def extract_wfdb_header(file_path):
    """
    Header reader function for .dat and .mat WFDB extension files.
    Redirects to the wfdb.io.rdheader function.

    Returns:
        A dictionary of header information
    """
    from wfdb.io import rdheader
    header = rdheader(record_name=os.path.splitext(file_path)[0])
    return _standardized_wfdb_header(header)


def extract_h5_header(h5_path, try_channel_dir_names=("channels", "signals", "psg")):
    """
    Header reader function for .h5 extension files.

    Returns:
        A dictionary of header information
    """
    import h5py
    with h5py.File(h5_path, "r") as psg_obj:
        for channel_dir in try_channel_dir_names:
            try:
                return _standardized_h5_header(psg_obj, channel_dir)
            except KeyError:
                continue
        raise KeyError(f"Could not read header from H5 archive '{os.path.basename(h5_path)}'. "
                       f"The archive does not contain any group named one of "
                       f"'{try_channel_dir_names}'. Individual channel H5 datasets must descend from a "
                       f"root group with name in this particular list of possible values (e.g. a valid dataset would "
                       f"be stored at /channels/eeg/C3-M2, where /C3-M2 would not be valid.")


_EXT_TO_LOADER = {
    "edf": extract_edf_header,
    "mat": extract_wfdb_header,
    "dat": extract_wfdb_header,
    "h5": extract_h5_header,
    "hdf5": extract_h5_header
}


def extract_header(file_path):
    """
    Loads the header from a PSG-type file at path 'file_path'.

    Returns:
        dictionary of header information
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
