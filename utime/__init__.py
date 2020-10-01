__version__ = "0.0.2"
import os


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
        self.OUT_OF_BOUNDS = ["OUT_OF_BOUNDS", 6]

        # Visualization defaults
        self.STAGE_COLORS = ["darkblue", "darkred",
                             "darkgreen", "darkcyan",
                             "darkorange", "black"]

        # Default segmentation length in seconds
        self.PERIOD_LENGTH_SEC = 30

        # Default hyperparameters path (relative to project dir)
        self.hparams_dir = 'hyperparameters'
        self.hparams_name = 'hparams.yaml'
        self.pre_processed_hparams_name = 'pre_proc_hparams.yaml'
        self.dataset_conf_dir = "dataset_configurations"
        self.pre_processed_data_conf_dir = "preprocessed"

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
        from mpunet.logging import ScreenLogger
        l = logger or ScreenLogger()
        l.warn("Using default period length of {} seconds."
               "".format(self.PERIOD_LENGTH_SEC))
        return self.PERIOD_LENGTH_SEC

    def get_hparams_dir(self, project_dir):
        return os.path.join(project_dir, self.hparams_dir)

    def get_hparams_path(self, project_dir):
        return os.path.join(project_dir, self.hparams_dir, self.hparams_name)

    def get_pre_processed_hparams_path(self, project_dir):
        return os.path.join(project_dir, self.hparams_dir,
                            self.pre_processed_hparams_name)

    def get_dataset_configurations_dir(self, project_dir):
        return os.path.join(self.get_hparams_dir(project_dir),
                            self.dataset_conf_dir)

    def get_pre_processed_data_configurations_dir(self, project_dir):
        return os.path.join(self.get_dataset_configurations_dir(project_dir),
                            self.pre_processed_data_conf_dir)


defaults = _Defaults()
