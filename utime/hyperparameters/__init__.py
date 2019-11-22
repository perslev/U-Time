import utime
from MultiPlanarUNet.hyperparameters import YAMLHParams as _YAMLHParams


class YAMLHParams(_YAMLHParams):
    """
    Wraps the YAMLHParams class from MultiPlanarUNet, passing 'utime' as the
    package for correct version controlling.
    """
    def __init__(self, *args, **kwargs):
        kwargs["package"] = utime.__name__
        super(YAMLHParams, self).__init__(
            *args, **kwargs
        )
