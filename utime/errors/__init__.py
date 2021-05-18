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


class VariableSampleRateError(ValueError): pass


class MissingHeaderFieldError(KeyError): pass


class HeaderFieldTypeError(TypeError): pass


class LengthZeroSignalError(ValueError): pass


class DuplicateChannelError(ValueError): pass


class NotLoadedError(ResourceWarning): pass


class StripError(RuntimeError): pass


class MarginError(ValueError):
    def __init__(self, *args, shift=None, **kwargs):
        super(MarginError, self).__init__(*args, **kwargs)
        self.shift = shift


""" Warnings """
class HeaderWarning(UserWarning): pass


class FloatSampleRateWarning(HeaderWarning): pass


class DuplicateChannelWarning(HeaderWarning): pass
