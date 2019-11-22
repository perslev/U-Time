"""
A collection of high level of data loader functions for PSG and hypnogram files

This file should contain only the following functions:

- load_psg(file_path, *args, **kwargs) --> psg_array, psg_header
- load_hypnogram(file_path, *args, **kwargs) --> hypnogram, annotation dict

"""

from utime.io.file_loaders import load_psg_file, load_hyp_file
from utime.io.extractors import (extract_psg_data,
                                 extract_header,
                                 extract_hyp_data)
import os


def load_psg(psg_file_path, load_channels=None):
    """
    Returns a numpy object of shape NxC (N data points, C channels) and a
    dictionary of header information as given by 'extract_header'.

    Args:
        psg_file_path: Path to PSG file
        load_channels: List of channels to read from the file, defaults to all

    Returns:
        A numpy array of shape NxC (N samples, C channels)
        A dictionary of header information
    """
    # Load the PSG file - depending on file type this may not load any actual
    # data from disk yet, but rather return an object representing the file,
    # from which actual data is loaded in 'extract_psg_data'.
    psg_obj = load_psg_file(psg_file_path, load_channels=load_channels)

    # Extract header information, most importantly the sample rate
    header = extract_header(psg_obj)
    if load_channels:
        header['channel_names'] = load_channels

    # Actually load data from disk, if not done already in load_psg_file
    # Select the relevant channels if not done already in load_psg_file
    psg_data = extract_psg_data(psg_obj, load_channels,
                                data_dir=os.path.split(psg_file_path)[0])
    return psg_data, header


def load_hypnogram(file_path, period_length_sec, annotation_dict, sample_rate):
    """
    Returns a utime.hypnogram SparseHypnogram object representation of the
    hypnogram / sleep stages / labels data at path 'file_path'.

    Args:
        file_path:          A string path pointing to the file to load
        period_length_sec:  The sleep staging 'epoch' length in seconds
        annotation_dict:    A dictionary mapping labels as stored in
                            'file_path' to integer label values. Can be None,
                            in which case a default or automatically inferred
                            annotation_dict will be used.
        sample_rate:        The sample of the original signal - used in rare
                            cases to convert a 'signal dense' hypnogram
                            (see utime.hypnogram.utils).

    Returns:
        A SparseHypnogram object
        A dictionary annotation_dict. Will be identical to the passed
        annotation_dict unless None was passed for annotation_dict, in which
        case the returned annotation_dict will be the automatically inferred
    """
    hyp_obj = load_hyp_file(file_path)
    hyp, annotation_dict = extract_hyp_data(hyp_obj=hyp_obj,
                                            period_length_sec=period_length_sec,
                                            annotation_dict=annotation_dict,
                                            sample_rate=sample_rate)
    return hyp, annotation_dict
