__version__ = "0.0.1"


class _Defaults:
    """
    Stores and potentially updates default values for sleep stages etc.
    This class should not be initiated directly, as this is automatically
    done when importing the U-Time package.
    """
    def __init__(self):
        # Standardized string representation for 5 typical sleep stages
        self.AWAKE = ["W", 0]
        self.NON_REM_STAGE_1 = ["N1", 1]
        self.NON_REM_STAGE_2 = ["N2", 2]
        self.NON_REM_STAGE_3 = ["N3", 3]
        self.REM = ["REM", 4]
        self.UNKNOWN = ["UNKNOWN", 5]

        # Visualization defaults
        self.STAGE_COLORS = ["darkblue", "darkred",
                             "darkgreen", "darkcyan",
                             "darkorange", "black"]

        # Default segmentation length in seconds
        self.PERIOD_LENGTH_SEC = 30

    @property
    def vectorized_stage_colors(self):
        import numpy as np
        map_ = {i: col for i, col in enumerate(self.STAGE_COLORS)}
        return np.vectorize(map_.get)

    @property
    def stage_lists(self):
        return [self.AWAKE, self.NON_REM_STAGE_1, self.NON_REM_STAGE_2,
                self.NON_REM_STAGE_3, self.REM, self.UNKNOWN]

    @property
    def stage_string_to_class_int(self):
        # Dictionary mapping from the standardized string rep to integer
        # representation
        return {s[0]: s[1] for s in self.stage_lists}

    @property
    def class_int_to_stage_string(self):
        # Dictionary mapping from integer representation to standardized
        # string rep
        return {s[1]: s[0] for s in self.stage_lists}

    def get_default_period_length(self, logger=None):
        from MultiPlanarUNet.logging import ScreenLogger
        l = logger or ScreenLogger()
        l.warn("Using default period length of {} seconds."
               "".format(self.PERIOD_LENGTH_SEC))
        return self.PERIOD_LENGTH_SEC


defaults = _Defaults()
