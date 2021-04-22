"""
A collection of high level of data loader functions for PSG and hypnogram files

This file should contain only the following functions:

- load_psg(file_path, *args, **kwargs) --> psg_array, psg_header
- load_hypnogram(file_path, *args, **kwargs) --> hypnogram, annotation dict
"""

import h5py
from utime.io.psg import extract_psg_data
from utime.io.hypnogram import extract_hyp_data
from utime.io.header import extract_header
from utime.io.channels.utils import get_org_include_exclude_channel_montages
from utime.errors import ChannelNotFoundError


def load_psg(psg_file_path,
             load_channels=None,
             ignore_reference_channels=False,
             load_time_channel_selector=None,
             check_num_channels=True):
    """
    Returns a numpy object of shape NxC (N data points, C channels) and a
    dictionary of header information as given by 'extract_header'.

    Args:
        psg_file_path: Path to PSG file
        load_channels: A list of channel name strings or a ChannelMontageTuple
                       storing ChannelMontage objects representing all channels
                       to load.
        ignore_reference_channels: TODO
        load_time_channel_selector: TODO
        check_num_channels: TODO

    Returns:
        A numpy array of shape NxC (N samples, C channels)
        A dictionary of header information
    """
    # Load the header of a PSG file. Stores e.g. channel names and sample rates
    header = extract_header(psg_file_path=psg_file_path)

    if load_time_channel_selector:
        # Randomly select from the available channels in groups according to
        # passed RandomChannelSelector object
        if load_channels is not None:
            raise ValueError("Must not specify the 'load_channels' argument "
                             "with the 'load_time_channel_selector' argument.")
        try:
            load_channels = load_time_channel_selector.sample(
                available_channels=header["channel_names"]
            )
        except ChannelNotFoundError as e:
            raise ChannelNotFoundError(
                "The PSG file at path {} is missing channels according to one "
                "or multiple of the specified channel sampling groups. "
                "File has: {}, requested groups: {}"
                "".format(psg_file_path, header['channel_names'],
                          load_time_channel_selector.channel_groups)) from e

    # Work out which channels to include and exclude during loading
    org_channels, include_channels, exclude_channels, montage_creator = \
        get_org_include_exclude_channel_montages(
            load_channels=load_channels,
            header=header,
            ignore_reference_channels=ignore_reference_channels,
            check_num_channels=check_num_channels,
            allow_auto_referencing=True,
            check_duplicates=True
        )
    header["channel_names"] = org_channels  # Now a ChannelMontageTuple object

    # Actually load data from disk, if not done already in open_psg_file
    # Select the relevant channels if not done already in open_psg_file
    psg_data = extract_psg_data(psg_file_path=psg_file_path,
                                header=header,
                                include_channels=include_channels.original_names,
                                exclude_channels=exclude_channels.original_names)
    if montage_creator:
        psg_data, include_channels = montage_creator.create_montages(psg_data)

    # Update header with actually kept channels
    header["channel_names"] = include_channels
    header["n_channels"] = len(include_channels)

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
    hyp, annotation_dict = extract_hyp_data(file_path=file_path,
                                            period_length_sec=period_length_sec,
                                            annotation_dict=annotation_dict,
                                            sample_rate=sample_rate)
    return hyp, annotation_dict


def open_h5_archive(h5_file_path,
                    load_channels=None,
                    ignore_reference_channels=False,
                    check_num_channels=True,
                    dataset_name='channels'):
    # Open archive
    h5_obj = h5py.File(h5_file_path, "r")

    # Get channels in file
    header = {'channel_names': list(h5_obj[dataset_name].keys())}

    # Work out which channels to include and exclude during loading
    org_channels, include_channels, _ = \
        get_org_include_exclude_channel_montages(
            load_channels=load_channels,
            header=header,
            ignore_reference_channels=ignore_reference_channels,
            check_num_channels=check_num_channels
        )
    data = {}
    for chnl in include_channels:
        data[chnl] = h5_obj[dataset_name][chnl.original_name]
    return h5_obj, data, include_channels
