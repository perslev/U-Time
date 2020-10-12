import os
import numpy as np


class _Defaults:
    """
    Stores and potentially updates default values for sleep stages etc.
    """
    # Standardized string representation for 5 typical sleep stages
    AWAKE = ["W", 0]
    NON_REM_STAGE_1 = ["N1", 1]
    NON_REM_STAGE_2 = ["N2", 2]
    NON_REM_STAGE_3 = ["N3", 3]
    REM = ["REM", 4]
    UNKNOWN = ["UNKNOWN", 5]
    OUT_OF_BOUNDS = ["OUT_OF_BOUNDS", 6]

    # Visualization defaults
    STAGE_COLORS = ["darkblue", "darkred",
                    "darkgreen", "darkcyan",
                    "darkorange", "black"]

    # Default segmentation length in seconds
    PERIOD_LENGTH_SEC = 30

    # Default hyperparameters path (relative to project dir)
    HPARAMS_DIR = 'hyperparameters'
    HPARAMS_NAME = 'hparams.yaml'
    PRE_PROCESSED_HPARAMS_NAME = 'pre_proc_hparams.yaml'
    DATASET_CONF_DIR = "dataset_configurations"
    PRE_PROCESSED_DATA_CONF_DIR = "preprocessed"

    # Default dtypes
    PSG_DTYPE = np.float32
    HYP_DTYPE = np.uint8

    # Global RNG seed
    GLOBAL_SEED = None

    @classmethod
    def set_global_seed(cls, seed):
        import tensorflow as tf
        import numpy as np
        import random
        cls.GLOBAL_SEED = int(seed)
        print("Seeding TensorFlow, numpy and random modules with seed: {}".format(cls.GLOBAL_SEED))
        tf.random.set_seed(cls.GLOBAL_SEED)
        np.random.seed(cls.GLOBAL_SEED)
        random.seed(cls.GLOBAL_SEED)

    @classmethod
    def get_vectorized_stage_colors(cls):
        import numpy as np
        map_ = {i: col for i, col in enumerate(cls.STAGE_COLORS)}
        return np.vectorize(map_.get)

    @classmethod
    def get_stage_lists(cls):
        return [cls.AWAKE, cls.NON_REM_STAGE_1, cls.NON_REM_STAGE_2,
                cls.NON_REM_STAGE_3, cls.REM, cls.UNKNOWN]

    @classmethod
    def get_stage_string_to_class_int(cls):
        # Dictionary mapping from the standardized string rep to integer
        # representation
        return {s[0]: s[1] for s in cls.get_stage_lists()}

    @classmethod
    def get_class_int_to_stage_string(cls):
        # Dictionary mapping from integer representation to standardized
        # string rep
        return {s[1]: s[0] for s in cls.get_stage_lists()}

    @classmethod
    def get_default_period_length(cls, logger=None):
        from mpunet.logging import ScreenLogger
        l = logger or ScreenLogger()
        l.warn("Using default period length of {} seconds."
               "".format(cls.PERIOD_LENGTH_SEC))
        return cls.PERIOD_LENGTH_SEC

    @classmethod
    def get_hparams_dir(cls, project_dir):
        return os.path.join(project_dir, cls.HPARAMS_DIR)

    @classmethod
    def get_hparams_path(cls, project_dir):
        return os.path.join(project_dir, cls.HPARAMS_DIR, cls.HPARAMS_NAME)

    @classmethod
    def get_pre_processed_hparams_path(cls, project_dir):
        return os.path.join(project_dir, cls.HPARAMS_DIR,
                            cls.PRE_PROCESSED_HPARAMS_NAME)

    @classmethod
    def get_dataset_configurations_dir(cls, project_dir):
        return os.path.join(cls.get_hparams_dir(project_dir),
                            cls.DATASET_CONF_DIR)

    @classmethod
    def get_pre_processed_data_configurations_dir(cls, project_dir):
        return os.path.join(cls.get_dataset_configurations_dir(project_dir),
                            cls.PRE_PROCESSED_DATA_CONF_DIR)
