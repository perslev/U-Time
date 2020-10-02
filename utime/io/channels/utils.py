from utime.io.channels import ChannelMontageTuple
from utime.errors import ChannelNotFoundError


def get_org_include_exclude_channel_montages(load_channels, header,
                                             ignore_reference_channels=False,
                                             check_num_channels=True,
                                             check_duplicates=True):
    """
    TODO

    Args:
        load_channels:
        header:
        ignore_reference_channels:
        check_num_channels:
        check_duplicates:

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
    if check_duplicates:
        for channel in include_channels:
            if channels_in_file.count(channel) > 1:
                raise ValueError(f"Cannot load channel with name \"{channel.original_name}\" as this channel "
                                 f"name occurs multiple times in the file: {channels_in_file.original_names}). "
                                 f"This is most likely a result of longer channel names which were truncated to "
                                 f"identical, shorter names due to limitations in the used file format. "
                                 f"E.g. EDF files may only store channel names at most 16 characters in length. "
                                 f"Please rename your channels and try again.")
    exclude_channels = [c for c in channels_in_file if c not in include_channels]
    exclude_channels = ChannelMontageTuple(exclude_channels)
    return channels_in_file, include_channels, exclude_channels
