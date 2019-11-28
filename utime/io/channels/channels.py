from mne.channels import read_layout
from numpy import ndarray

_ALLOWED_CHAN_SYNONYMS = {"A1": "M1",
                          "A2": "M2",
                          "ROC": "E2",
                          "LOC": "E1"}

_ALLOWED_CHANNELS_LIST = read_layout("biosemi").names + \
                         list(_ALLOWED_CHAN_SYNONYMS.keys()) + \
                         list(_ALLOWED_CHAN_SYNONYMS.values())
ALLOWED_CHANNELS = {c: None for c in _ALLOWED_CHANNELS_LIST}


def get_standardized_channel_name(name):
    name = name.upper().strip(" -/")
    if name in _ALLOWED_CHAN_SYNONYMS:
        name = _ALLOWED_CHAN_SYNONYMS[name]
    return name


def split_by_valid_chan(channel_str):
    channel_str_standard = channel_str.upper().strip(" -/")
    for chan in ALLOWED_CHANNELS:
        if channel_str_standard.startswith(chan.upper()):
            return channel_str_standard.replace(chan, "{}-".format(chan))
    return channel_str


def infer_channels(channel_str, relax=False):
    split_channel_str = channel_str.upper().strip(" -/").replace(" ", "-").split("-")
    for chan in split_channel_str:
        if not relax and chan not in ALLOWED_CHANNELS:
            new_chan = split_by_valid_chan(chan)
            if new_chan != chan and len(split_channel_str) == 1:
                return infer_channels(new_chan)
            else:
                raise ValueError("Channel '{}' (inferred from '{}') does "
                                 "not seem to be valid. Valid channels are: "
                                 "{}".format(chan, channel_str,
                                             _ALLOWED_CHANNELS_LIST))
    if len(split_channel_str) == 1:
        return split_channel_str[0], None
    elif len(split_channel_str) == 2:
        return split_channel_str
    else:
        raise ValueError("Could not infer 1 or 2 (chan + ref) channels from "
                         "channel string '{}'".format(channel_str))


class ChannelMontage:
    def __init__(self, channel_name, relax=False):
        self._original_name = channel_name
        channel, reference = infer_channels(channel_name, relax)
        self._channel = get_standardized_channel_name(channel)
        self._reference = reference
        if self.reference is not None:
            self._reference = get_standardized_channel_name(reference)

    @property
    def original_name(self):
        return self._original_name

    @property
    def channel(self):
        return self._channel

    @property
    def reference(self):
        return self._reference

    def __str__(self):
        return "{}-{}".format(self.channel, self.reference)

    def __repr__(self):
        return "ChannelMontage({}-{})".format(self.channel, self.reference)

    def match_reference(self, channel_montage):
        return self.reference == channel_montage.reference

    def match_channel(self, channel_montage):
        return self.channel == channel_montage.channel

    def match(self, channel_montage):
        return self.match_channel(channel_montage) and \
               self.match_reference(channel_montage)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, channel_montage):
        if not isinstance(channel_montage, ChannelMontage):
            raise TypeError("Cannot compare {} object '{}' to '{}' of "
                            "type {}".format(self.__class__.__name__,
                                             repr(self), channel_montage,
                                             type(channel_montage)))
        return self.match(channel_montage)


class ChannelMontageTuple(tuple):
    def __new__(cls, channels, relax=False):
        if not isinstance(channels, (list, tuple, ndarray)):
            raise TypeError("Input to {} should be a list, tuple or "
                            "ndarray of channel names or ChannelMontage "
                            "objects. Got {} "
                            "(type {}).".format(cls.__name__,
                                                channels, type(channels)))
        montage_objs = []
        for c in channels:
            if isinstance(c, ChannelMontage):
                montage_objs.append(c)
            elif isinstance(c, (str, list, tuple)):
                montage_objs.append(ChannelMontage(c, relax=relax))
            else:
                raise TypeError("ChannelMontageSet does not accept input '{}' "
                                "(type {}) from passed argument "
                                "'{}'. Should pass a list of ChannelMontage "
                                "objects, channel name iterables or channel "
                                "name strings.".format(c, type(c), channels))
        return super(ChannelMontageTuple, cls).__new__(cls, montage_objs)

    @property
    def original_names(self):
        return [montage.original_name for montage in self]

    def _match(self, other_list, match_func_name):
        matches = []
        for montage in self:
            for target in other_list:
                if getattr(montage, match_func_name)(target):
                    matches.append(montage)
                    break
        return ChannelMontageTuple(matches)

    def match(self, channel_montage_set):
        return self._match(channel_montage_set, 'match')

    def match_ignore_reference(self, channel_montage_set):
        return self._match(channel_montage_set, 'match_channel')
