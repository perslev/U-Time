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


class H5ChannelRootError(KeyError): pass


class H5VariableAttributesError(ValueError): pass


class MissingHeaderFieldError(KeyError): pass


class HeaderFieldTypeError(TypeError): pass


class LengthZeroSignalError(ValueError): pass


class NotLoadedError(ResourceWarning): pass


class StripError(RuntimeError): pass


class MarginError(ValueError):
    def __init__(self, *args, shift=None, **kwargs):
        super(MarginError, self).__init__(*args, **kwargs)
        self.shift = shift


# A list of errors that may be raised during loading of a PSG header with error messages
# safe to be displayed to front-end users
IO_HEADER_ERRORS = [
    H5ChannelRootError,
    H5VariableAttributesError,
    MissingHeaderFieldError,
    HeaderFieldTypeError,
    LengthZeroSignalError
]
