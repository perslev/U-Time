"""
A set of functions for extracting header information from PSG objects
Typically only used internally in from unet.io.header.header_extractors

Each function takes some PSG or header-like object and returns a dictionary with at least
the following keys:

{
    'n_channels': int,
    'channel_names': list of strings,
    'sample_rate': int
    'date': datetime or None
    'length_sec': float
}

"""

from datetime import datetime
import numpy as np


def _standardized_edf_header(raw_edf):
    """
    Header extraction function for RawEDF and Raw objects.
    Reads the number of channels, channel names and sample rate properties
    If existing, reads the date information as well.

    Returns:
        Header information as dict
    """
    # Each tuple below follows the format:
    # 1) output name, 2) edf_obj name, 3) function to apply to the read
    # value, 4) whether a missing value should raise an error.
    header_map = [("n_channels", "nchan", int, True),
                  ("channel_names", "ch_names", list, True),
                  ("sample_rate", "sfreq", int, True),
                  ("date", "meas_date", datetime.utcfromtimestamp, False)]
    if isinstance(raw_edf.info["meas_date"], (tuple, list)):
        assert raw_edf.info["meas_date"][1] == 0
        raw_edf.info["meas_date"] = raw_edf.info["meas_date"][0]
    header = {}
    for renamed, org, transform, raise_err in header_map:
        value = raw_edf.info.get(org)
        try:
            value = transform(value)
        except Exception as e:
            if raise_err:
                raise ValueError("Missing or invalid value in EDF for key {} "
                                 "- got {}".format(org, value)) from e
        header[renamed] = value
    # Add record length in seconds
    header["length_sec"] = len(raw_edf) / header['sample_rate']
    return header


def _read_wfdb_record_header(record_obj, **kwargs):
    """
    Header extraction function for WFDB Record objects.
    Reads the number of channels, channel names and sample rate properties
    If existing, reads the date information as well.

    Returns:
        Header information as dict
    """
    # Each tuple below follows the format:
    # 1) output name, 2) record_obj name, 3) function to apply to the read
    # value, 4) whether a missing value should raise an error.
    header_map = [("n_channels", "n_sig", int, True),
                  ("channel_names", "sig_name", list, True),
                  ("sample_rate", "fs", int, True),
                  ("date", "base_date", datetime.utcfromtimestamp, False)]
    header = {}
    for renamed, org, transform, raise_err in header_map:
        value = getattr(record_obj, org, None)
        try:
            value = transform(value)
        except Exception as e:
            if raise_err:
                raise ValueError("Missing or invalid value in WFDB for key {} "
                                 "- got {}".format(org, value)) from e
        header[renamed] = value
    return header


def _read_dcsm_dict_header(dict, **kwargs):
    """
    Header extraction function for DCSMDict objects.
    The DCSMDict dictionary stores key-values pairs:
       channel_name : (file_path, sample_rate)
    Here, we extract the channel names and sample rates.
    No date information is carried.

    Returns:
        A dictionary with header elements
    """
    sample_rates = np.array([v[1] for v in dict.values()])
    if np.any(sample_rates != sample_rates[0]):
        raise NotImplementedError("Cannot deal with non-identical sample rates"
                                  " across channels.")
    return {
        "n_channels": len(dict),
        "channel_names": list(dict.keys()),
        "sample_rate": sample_rates[0],
        "date": None
    }


def _read_h5_file(h5_file, **kwargs):
    """
    Header extraction function for h5py.File objects.
    The object must:
      - Have an attribute 'sample_rate'
      - Have a group named 'channels' which stores the data for all channels as
        Dataset entries under the group
    Can have:
      - An attribute 'date' which gives a date string or unix timestamp integer

    Returns:
        A dictionary with header elements
    """
    d = h5_file.attrs.get("date")
    if not isinstance(d, str) and (isinstance(d, int) or np.issubdtype(d, np.integer)):
        d = datetime.fromtimestamp(d)
    return {
        "n_channels": len(h5_file["channels"]),
        "channel_names": list(h5_file["channels"].keys()),
        "sample_rate": h5_file.attrs["sample_rate"],
        "date": d
    }
