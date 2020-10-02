import numpy as np
from utime.io.channels import ChannelMontageTuple
from utime.io.header import extract_header
from utime.errors import ChannelNotFoundError


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
        available_channels = extract_header(psg_file_path)["channel_names"]
    if not isinstance(available_channels, ChannelMontageTuple):
        available_channels = ChannelMontageTuple(available_channels)
    for i, channel_group in enumerate(channel_groups):
        # Cast channel group to channel montage tuple if not already
        if not isinstance(channel_group, ChannelMontageTuple):
            channel_groups[i] = ChannelMontageTuple(channel_group)
    return [
        available_channels.match(c_grp) for c_grp in channel_groups
    ]


class RandomChannelSelector:
    """
    TODO
    """
    def __init__(self, *channel_groups, channel_group_ids=None):
        """
        TODO

        Args:
            channel_groups: (tuple of lists)
            channel_ids:    (list, tuple, None)
        """
        self.channel_groups = []
        self.channel_group_ids = []
        if channel_group_ids and len(channel_group_ids) != len(channel_groups):
            raise ValueError("Length of (optional) 'channel_group_ids' "
                             "argument must match length of 'channel_groups' "
                             "argument. Got lengths {} and {}."
                             "".format(len(channel_group_ids),
                                       len(channel_groups)))
        for i, (channel_group, channel_id) in enumerate(zip(
            channel_groups, channel_group_ids or [None] * len(channel_groups)
        )):
            if not isinstance(channel_group, ChannelMontageTuple):
                channel_group = ChannelMontageTuple(channel_group, relax=True)
            self.channel_groups.append(channel_group)
            self.channel_group_ids.append(channel_id or f"group_{i}")

    @property
    def n_output_channels(self):
        return len(self.channel_groups)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "RandomChannelSelector({})".format(
            ", ".join(
                f"{id_}: {group.names}" for id_, group in
                zip(self.channel_group_ids,
                    self.channel_groups)
            )
        )

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
        out = []
        for c in channel_groups:
            if c:
                out.append(c[np.random.randint(0, len(c))])
            else:
                raise ChannelNotFoundError(
                    "No channel was found in this file for one or more of the "
                    "requested channel groups."
                )
        return out
