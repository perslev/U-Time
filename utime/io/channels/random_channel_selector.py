import numpy as np
from utime.io.channels import ChannelMontageTuple
from utime.io.file_loaders import read_psg_header


def filter_non_available_channels(channel_groups,
                                  psg_file_path=None,
                                  available_channels=None):
    """
    TODO

    Args:
        psg_file_path:
        channel_groups:
        available_channels:

    Returns:

    """
    if psg_file_path is None and available_channels is None:
        raise ValueError("Must specify either 'psg_file_path' or "
                         "'available_channels'.")
    if available_channels is None:
        available_channels = read_psg_header(psg_file_path)["channel_names"]
    if not isinstance(available_channels, ChannelMontageTuple):
        available_channels = ChannelMontageTuple(available_channels)
    return [
        available_channels.match(c_grp) for c_grp in channel_groups
    ]


class RandomChannelSelector:
    """
    TODO
    """
    def __init__(self, *channel_groups):
        """
        TODO

        Args:
            *channel_groups:
        """
        self.channel_groups = []
        for channel_group in channel_groups:
            if not isinstance(channel_group, ChannelMontageTuple):
                channel_group = ChannelMontageTuple(channel_group, relax=True)
            self.channel_groups.append(channel_group)

    def sample(self, psg_file_path=None, available_channels=None):
        """
        TODO

        Args:
            psg_file_path:
            available_channels:

        Returns:

        """
        if psg_file_path or available_channels:
            channel_groups = filter_non_available_channels(self.channel_groups,
                                                           psg_file_path,
                                                           available_channels)
        else:
            channel_groups = self.channel_groups
        return [
            c[np.random.randint(0, len(c))[0]] for c in channel_groups
        ]
