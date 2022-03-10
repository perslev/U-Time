import logging
import os
from sleeputils import Defaults

logger = logging.getLogger(__name__)


class _Defaults(Defaults):
    """
    Stores and potentially updates default values for sleep stages etc.
    """
    # Default hyperparameters path (relative to project dir)
    HPARAMS_DIR = 'hyperparameters'
    HPARAMS_NAME = 'hparams.yaml'
    PRE_PROCESSED_HPARAMS_NAME = 'pre_proc_hparams.yaml'
    DATASET_CONF_DIR = "dataset_configurations"
    PRE_PROCESSED_DATA_CONF_DIR = "preprocessed"

    # Global RNG seed
    GLOBAL_SEED = None

    @classmethod
    def set_global_seed(cls, seed):
        import tensorflow as tf
        import numpy as np
        import random
        cls.GLOBAL_SEED = int(seed)
        logger.info(f"Seeding TensorFlow, numpy and random modules with seed: {cls.GLOBAL_SEED}")
        tf.random.set_seed(cls.GLOBAL_SEED)
        np.random.seed(cls.GLOBAL_SEED)
        random.seed(cls.GLOBAL_SEED)

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
