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
    'length': int
}

Note: length gives the number of samples, divide by sample_rate to get length_sec
"""

from datetime import datetime
import numpy as np


def _assert_header(header):
    """
    Checks that a standardized header:
        1) contains the right field names
        2) each value has an expected type
        3) the 'length' value is greater than 0
    Args:
        header: dict
    Returns: dict
    """
    field_requirements = [
        ("n_channels", [int]),
        ("channel_names", [list]),
        ("sample_rate", [int]),
        ("date", [datetime, None]),
        ("length", [int])
    ]
    for field, valid_types in field_requirements:
        if field not in header:
            raise ValueError(f"Missing value '{field}' from header '{header}'. "
                             "This could be an error in the code implementation. "
                             "Please raise this issue on GitHub.")
        type_ = type(header[field])
        if type_ not in valid_types:
            raise TypeError(f"Field {field} of type {type_} was not expected, expected one of {valid_types}")
    if header['length'] <= 0:
        raise ValueError(f"Expected key 'length' to be a non-zero integer, "
                         f"but header {header} has value {header['length']}")
    return header


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
    header["length"] = len(raw_edf)
    return _assert_header(header)


def _standardized_wfdb_header(wfdb_record):
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
                  ("date", "base_date", datetime.utcfromtimestamp, False),
                  ("length", "sig_len", int, True)]
    header = {}
    for renamed, org, transform, raise_err in header_map:
        value = getattr(wfdb_record, org, None)
        try:
            value = transform(value)
        except Exception as e:
            if raise_err:
                raise ValueError("Missing or invalid value in WFDB for key {} "
                                 "- got {}".format(org, value)) from e
        header[renamed] = value
    return _assert_header(header)


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
    header = {
        "n_channels": len(dict),
        "channel_names": list(dict.keys()),
        "sample_rate": sample_rates[0],
        "date": None
    }
    return _assert_header(header)


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
    header = {
        "n_channels": len(h5_file["channels"]),
        "channel_names": list(h5_file["channels"].keys()),
        "sample_rate": h5_file.attrs["sample_rate"],
        "date": d
    }
    return _assert_header(header)
