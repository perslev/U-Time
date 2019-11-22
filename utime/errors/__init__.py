"""
Small collection of custom objects.
Sometimes used for their custom names only.
"""


class CouldNotLoadError(ResourceWarning):
    def __init__(self, *args, study_id=None):
        super(CouldNotLoadError, self).__init__(*args)
        self.study_id = study_id


class ChannelNotFoundError(CouldNotLoadError):
    def __init__(self, *args, **kwargs):
        super(ChannelNotFoundError, self).__init__(*args, **kwargs)


class NotLoadedError(ResourceWarning): pass


class StripError(RuntimeError): pass


class MarginError(ValueError):
    def __init__(self, *args, shift=None, **kwargs):
        super(MarginError, self).__init__(*args, **kwargs)
        self.shift = shift
