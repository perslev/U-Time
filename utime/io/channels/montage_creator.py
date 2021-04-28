import numpy as np
import re
from utime.io.channels import ChannelMontageTuple, ChannelMontage, infer_channel_types, assert_channel_types


def auto_infer_referencing(channel_names, channel_types=None, types=("EEG",)):
    """
    Attempts to automatically infer referencing of all channels of type in "types".
    Referencing is done to MASTOID classified channels on opposite hemisphere.
    Channels must contain the appropriate MASTOIDS, otherwise a value error is raised.

    TODO: Simplify / Split into sub functions
    """
    assert_channel_types(types, ("EEG", "EOG"))
    channel_names = ChannelMontageTuple(channel_names, relax=True)
    inferred_types = infer_channel_types(channel_names)
    channel_types = list(channel_types or inferred_types)
    if "MASTOID" not in channel_types:
        # If passed by user, try to overwrite possibly "EEG" labelled passed MASTOID channels
        # If no mastoid was inferred, this does nothing and an error is raised below
        for i, (c1, c2) in enumerate(zip(channel_types, inferred_types)):
            if c2 == "MASTOID":
                channel_types[i] = c2
    mastoids = ChannelMontageTuple(("M1", "M2")).match(channel_names, take_target=True)
    other_channels = [(c, type_) for c, type_ in zip(channel_names, channel_types) if type_ != "MASTOID"]
    if not mastoids:
        raise ValueError(f"Could not automatically infer referencing for channels {channel_names}. "
                         f"Inferred types {channel_types} does not contain 1 or more MASTOID typed channels.")
    referenced, referenced_types = [], []
    for channel, type_ in other_channels:
        referenced_types.append(type_)
        if type_ not in types:
            referenced.append(channel.original_name)
        elif channel.reference is not None:
            raise ValueError(f"Could not infer referencing for channel {channel}, which seems to already be "
                             f"referenced to {channel.reference}.")
        else:
            numbers = list(map(int, re.findall(r"\d", channel.channel)))
            check_string = channel.channel.replace(type_, "")
            perhaps_left = bool(re.match(r'(_|\b)L($|_)|(_|\b)LEFT($|_)', check_string, re.IGNORECASE))
            perhaps_right = bool(re.match(r'(_|\b)R($|_)|(_|\b)RIGHT($|_)', check_string, re.IGNORECASE))
            if not numbers and (perhaps_left or perhaps_right):
                assert perhaps_left != perhaps_right
                numbers = [1 if perhaps_left else 2]
            if not numbers or len(numbers) > 1:
                raise ValueError(f"Could not automatically infer referencing for channel {channel} (type={type_}). "
                                 f"The channel name {channel.channel} should contain 1 digit, e.g. as in 'EEG C3', "
                                 f"but found {len(numbers)} ({numbers})")
            needed_mastoid = ChannelMontage("M" + str(1 + (numbers[0] % 2)))
            if needed_mastoid in mastoids:
                ref_mastoid = mastoids[mastoids.index(needed_mastoid)]
                referenced.append(f"{channel.original_name}-{ref_mastoid.original_name}")
            else:
                raise ValueError(f"Could not automatically infer referencing for channel {channel} (type={type_}). "
                                 f"The channel should be referenced to mastoid {needed_mastoid}, "
                                 f"but a match to such was not found among possible: {mastoids}")
    return referenced, referenced_types


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
