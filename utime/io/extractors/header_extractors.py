"""
A set of functions for extracting header information from PSG objects output
from unet.io.extractors.psg_extractors.
"""

from datetime import datetime
import numpy as np


def read_mne_raw_edf_header(edf_obj, **kwargs):
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
                  ("sample_rate", "sfreq", float, True),
                  ("date", "meas_date", datetime.utcfromtimestamp, False)]
    if isinstance(edf_obj.info["meas_date"], (tuple, list)):
        assert edf_obj.info["meas_date"][1] == 0
        edf_obj.info["meas_date"] = edf_obj.info["meas_date"][0]
    header = {}
    for renamed, org, transform, raise_err in header_map:
        value = edf_obj.info.get(org)
        try:
            value = transform(value)
        except Exception as e:
            if raise_err:
                raise ValueError("Missing or invalid value in EDF for key {} "
                                 "- got {}".format(org, value)) from e
        header[renamed] = value
    return header


def read_wfdb_record_header(record_obj, **kwargs):
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
                  ("sample_rate", "fs", float, True),
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


def read_dcsm_dict_header(dict, **kwargs):
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


def read_h5_file(h5_file, **kwargs):
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
    if isinstance(d, int) or np.issubdtype(d, np.integer):
        d = datetime.fromtimestamp(d)
    return {
        "n_channels": len(h5_file["channels"]),
        "channel_names": list(h5_file["channels"].keys()),
        "sample_rate": h5_file.attrs["sample_rate"],
        "date": d
    }


_DATA_TYPE_NAME_TO_HEADER_LOADER = {
    "RawEDF": read_mne_raw_edf_header,
    "Record": read_wfdb_record_header,
    "Raw": read_mne_raw_edf_header,
    "DCSMDict": read_dcsm_dict_header,
    "File": read_h5_file
}


def extract_header(psg_obj):
    """
    Extraxt header information from a 'PSG object' as returned by functions in
    utime.io.extractors.psg_extractors.

    Returns:
        A header (dictionary) of information on the PSG sample
    """
    # Find the appropriate function to extract information from the given
    # PSG object type
    data_type_name = type(psg_obj).__name__
    header_load_func = _DATA_TYPE_NAME_TO_HEADER_LOADER[data_type_name]

    # Construct the header using the inferred function
    header = header_load_func(psg_obj)

    # Make sure sample rate is int
    old_sr = header["sample_rate"]
    int_sr = int(old_sr)
    if int_sr != old_sr:
        raise ValueError("Sample rate is a float value. Expected integer.")
    header["sample_rate"] = int_sr

    return header
