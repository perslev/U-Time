"""
A collection of high level of data loader functions for PSG and hypnogram files

This file should contain only the following functions:

- load_psg(file_path, *args, **kwargs) --> psg_array, psg_header
- load_hypnogram(file_path, *args, **kwargs) --> hypnogram, annotation dict
"""

from utime.io.file_loaders import read_psg_header, read_hyp_file
from utime.io.channels import ChannelMontageTuple
from utime.io.extractors import extract_psg_data, extract_hyp_data
from utime.errors import ChannelNotFoundError


def get_org_include_exclude_channel_montages(load_channels, header,
                                             ignore_reference_channels=False,
                                             check_num_channels=True):
    """
    TODO

    Args:
        load_channels:
        header:
        ignore_reference_channels:
        check_num_channels:

    Returns:

    """
    channels_in_file = ChannelMontageTuple(header['channel_names'], relax=True)
    if load_channels:
        if not isinstance(load_channels, ChannelMontageTuple):
            load_channels = ChannelMontageTuple(load_channels, relax=True)
        if ignore_reference_channels:
            include_channels = load_channels.match_ignore_reference(channels_in_file,
                                                                    take_target=True)
        else:
            include_channels = load_channels.match(channels_in_file,
                                                   take_target=True)
        if check_num_channels and len(include_channels) != len(load_channels):
            raise ChannelNotFoundError(
                "Could not load {} channels ({}) from file with {} channels "
                "({}). Found the follow {} matches: {}".format(
                    len(load_channels), load_channels.original_names,
                    len(channels_in_file), channels_in_file.original_names,
                    len(include_channels), include_channels.original_names
                )
            )
    else:
        include_channels = channels_in_file
    exclude_channels = [c for c in channels_in_file if c not in include_channels]
    exclude_channels = ChannelMontageTuple(exclude_channels)
    return channels_in_file, include_channels, exclude_channels


def load_psg(psg_file_path,
             load_channels=None,
             ignore_reference_channels=False,
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
        check_num_channels: TODO

    Returns:
        A numpy array of shape NxC (N samples, C channels)
        A dictionary of header information
    """
    # Load the header of a PSG file. Stores e.g. channel names and sample rates
    header = read_psg_header(psg_file_path)

    org_channels, include_channels, exclude_channels = \
        get_org_include_exclude_channel_montages(
            load_channels=load_channels,
            header=header,
            ignore_reference_channels=ignore_reference_channels,
            check_num_channels=check_num_channels
        )
    header["channel_names"] = include_channels
    header["n_channels"] = len(include_channels)

    # Actually load data from disk, if not done already in open_psg_file
    # Select the relevant channels if not done already in open_psg_file
    psg_data = extract_psg_data(psg_file_path, header,
                                include_channels=include_channels.original_names,
                                exclude_channels=exclude_channels.original_names)
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
    hyp_obj = read_hyp_file(file_path)
    hyp, annotation_dict = extract_hyp_data(hyp_obj=hyp_obj,
                                            period_length_sec=period_length_sec,
                                            annotation_dict=annotation_dict,
                                            sample_rate=sample_rate)
    return hyp, annotation_dict
