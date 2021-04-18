import re
from mne.channels import make_standard_montage
from utime.io.channels import ChannelMontageTuple, ChannelMontage

EEG_CHANNELS = make_standard_montage("standard_1020").ch_names
EEG_REGEX = re.compile("(EEG|{})".format("|".join(EEG_CHANNELS)), re.IGNORECASE)
EOG_REGEX = re.compile(r"(EOG|E\d|ROC|LOC)", re.IGNORECASE)
EMG_REGEX = re.compile(r"(EMG)", re.IGNORECASE)


def is_eeg(channel_name):
    if not isinstance(channel_name, ChannelMontage):
        channel_name = ChannelMontage(channel_name, relax=True)
    return bool(re.match(EEG_REGEX, channel_name.channel))


def is_eog(channel_name):
    if not isinstance(channel_name, ChannelMontage):
        channel_name = ChannelMontage(channel_name, relax=True)
    return bool(re.match(EOG_REGEX, channel_name.channel))


def is_emg(channel_name):
    if not isinstance(channel_name, ChannelMontage):
        channel_name = ChannelMontage(channel_name, relax=True)
    return bool(re.match(EMG_REGEX, channel_name.channel))


def infer_channel_types(channels):
    """
    Attempts to classify a list of string channel names into one of 5 categories:

    EEG, EOG, EMG, OTHER, AMBIGUOUS

    Args:
        channels: List like of strings

    Returns:
        List of strings of suggested channel types
        len(output) = len(channels)
    """
    channels = ChannelMontageTuple(channels, relax=True)
    types = []
    for channel in channels:
        possible_eeg = is_eeg(channel)
        possible_eog = is_eog(channel)
        possible_emg = is_emg(channel)
        n_matches = bool(possible_eeg) + bool(possible_eog) + bool(possible_emg)
        if n_matches > 1:
            match = "AMBIGUOUS"
        elif n_matches == 0:
            match = "OTHER"
        else:
            # 1 match
            if possible_eeg:
                match = "EEG"
            elif possible_eog:
                match = "EOG"
            else:
                match = "EMG"
        types.append(match)
    return types
