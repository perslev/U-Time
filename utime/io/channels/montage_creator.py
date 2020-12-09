import numpy as np
from utime.io.channels import ChannelMontageTuple


def get_existing_separated_channels(channel, existing_channels):
    """
    TODO

    Args:
        channel:
        existing_channels:

    Returns:

    """
    return [c for c in channel.separate() if c in existing_channels]


class ChannelMontageCreator:
    """
    TODO
    """
    def __init__(self,
                 existing_channels,
                 channels_required,
                 allow_missing=False):
        """
        TODO
        """
        if not isinstance(existing_channels, ChannelMontageTuple):
            existing_channels = ChannelMontageTuple(existing_channels, True)
        if not isinstance(channels_required, ChannelMontageTuple):
            channels_required = ChannelMontageTuple(channels_required, True)
        self.channels_required = channels_required
        self.existing_channels = existing_channels

        channels_to_load = []
        output_channels = []
        for channel in self.channels_required:
            if channel in self.existing_channels:
                channels_to_load.append(channel)
                if channel not in output_channels:
                    output_channels.append(channel)
            else:
                sep = get_existing_separated_channels(channel,
                                                      self.existing_channels)
                if len(sep) == 2:
                    channels_to_load += sep
                    if channel not in output_channels:
                        output_channels.append(channel)
                elif not allow_missing:
                    raise ValueError(
                        "One or both of the channels in the requested montage "
                        "{} ({}) does not exist in the file with channels {} "
                        "(allow_missing=False)".format(channel,
                                                       channel.separate(),
                                                       self.existing_channels))
                else:
                    # Skip the channel
                    pass
        self.channels_to_load = ChannelMontageTuple(
            list(set(channels_to_load)), True
        )
        self.output_channels = ChannelMontageTuple(
            output_channels, True
        )

    @staticmethod
    def _create_montage(channel_data, channels, montage_to_create):
        """
        TODO

        Args:
            channel_data:
            channels:
            montage_to_create:

        Returns:

        """
        main_chan, ref_chan = montage_to_create.separate()
        main_chan_data = channel_data[:, channels.index(main_chan)]
        ref_chan_data = channel_data[:, channels.index(ref_chan)]
        return main_chan_data - ref_chan_data

    def create_montages(self, channel_data):
        """
        TODO

        Args:
            channel_data:
            channels:
            allow_missing:

        Returns:

        """
        channels = self.channels_to_load
        if channel_data.shape[1] != len(channels):
            raise ValueError("Input 'channel_data' of shape {} does not "
                             "match channel number in 'self.channels_to_load'"
                             "({} channels, {}). Make sure you loaded exactly"
                             " the channels specified in "
                             "self.channels_to_load."
                             "".format(channel_data.shape,
                                       len(channels),
                                       channels))
        new_channel_data = []
        for montage in self.output_channels:
            try:
                data = channel_data[:, channels.index(montage)]
            except ValueError:
                data = self._create_montage(channel_data, channels, montage)
            new_channel_data.append(data)
        new_channel_data = np.array(new_channel_data).T
        return new_channel_data, self.output_channels
