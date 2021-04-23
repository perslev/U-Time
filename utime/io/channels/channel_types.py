import re
from mne.channels import make_standard_montage
from utime.io.channels import ChannelMontageTuple, ChannelMontage

EEG_CHANNELS = list(filter(lambda chan: chan not in ("M1", "M2", "A1", "A2"),
                           make_standard_montage("standard_1020").ch_names))
EEG_REGEX = re.compile(r"(\bEEG|\b{})".format(r"|\b".join(EEG_CHANNELS)), re.IGNORECASE)
EOG_REGEX = re.compile(r"(\bEOG|\bE\d|\bROC|\bLOC)", re.IGNORECASE)
EMG_REGEX = re.compile(r"(\bEMG)", re.IGNORECASE)
MASTOID_REGEX = re.compile(r"(\bA1|\bA2|\bM1|\bM2)", re.IGNORECASE)


def is_eeg(channel_name):
    if not isinstance(channel_name, ChannelMontage):
        channel_name = ChannelMontage(channel_name, relax=True)
    return bool(re.search(EEG_REGEX, channel_name.channel))


def is_eog(channel_name):
    if not isinstance(channel_name, ChannelMontage):
        channel_name = ChannelMontage(channel_name, relax=True)
    return bool(re.search(EOG_REGEX, channel_name.channel))


def is_emg(channel_name):
    if not isinstance(channel_name, ChannelMontage):
        channel_name = ChannelMontage(channel_name, relax=True)
    return bool(re.search(EMG_REGEX, channel_name.channel))


def is_mastoid(channel_name):
    if not isinstance(channel_name, ChannelMontage):
        channel_name = ChannelMontage(channel_name, relax=True)
    return bool(re.search(MASTOID_REGEX, channel_name.channel))


def infer_channel_types(channels):
    """
    Attempts to classify a list of string channel names into one of 6 categories:

    EEG, EOG, EMG, MASTOID, OTHER, AMBIGUOUS

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
        possible_mastoid = is_mastoid(channel)
        n_matches = bool(possible_eeg) + bool(possible_eog) + bool(possible_emg) + bool(possible_mastoid)
        if n_matches > 1:
            eeg_and_mastoid = (possible_eeg and possible_mastoid)
            if eeg_and_mastoid and is_eeg(channel.channel.replace("EEG", "")):
                # Matches both EEG and MASTOID.
                # Removing "EEG" from channel name still matches EEG.
                # ---> This excludes strings like 'EEG M1', see below.
                # Most likely a referenced EEG written as e.g. 'C3A1'
                match = "EEG"
            elif eeg_and_mastoid:
                # Could be something like 'EEG M1'
                match = "MASTOID"
            else:
                # We cannot be sure what is going on
                match = "AMBIGUOUS"
        elif n_matches == 0:
            # Something else
            match = "OTHER"
        else:
            # 1 match
            if possible_eeg:
                match = "EEG"
            elif possible_eog:
                match = "EOG"
            elif possible_emg:
                match = "EMG"
            else:
                match = "MASTOID"
        types.append(match)
    return types
