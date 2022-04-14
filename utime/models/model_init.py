"""
Functions for initializing models from build hyperparameters and loading of
parameters.
"""

import logging
import os
import tensorflow as tf
from utime import models
from utime.models.utils import get_best_model, get_last_model
from utime.utils.scriptutils.train import (get_last_epoch, get_lr_at_epoch,
                                           clear_csv_after_epoch)

logger = logging.getLogger(__name__)


def init_model(build_hparams, clear_previous=False):
    """
    From a set of hyperparameters 'build_hparams' (dict) initializes the
    model specified under build_hparams['model_class_name'].

    Typically, this function is not called directly, but used by the
    higher-level 'initialize_model' function.

    Args:
        build_hparams:  A dictionary of model build hyperparameters
        clear_previous: Clear previous tf sessions

    Returns:
        A tf.keras Model instance
    """
    if clear_previous:
        tf.keras.backend.clear_session()
    # Build new model of the specified type
    cls_name = build_hparams["model_class_name"]
    logger.info(f"Creating new model of type '{cls_name}'")
    return models.__dict__[cls_name](**build_hparams)


def load_from_file(model, file_path, by_name=True):
    """
    Load parameters from file 'file_path' into model 'model'.

    Args:
        model:      A tf.keras Model instance
        file_path:  A path to a parameter file (h5 format typically)
        by_name:    Load parameters by layer names instead of order (default).
    """
    model.load_weights(file_path, by_name=by_name)
    logger.info(f"Loading parameters from: {file_path}")


def init_and_load_model(hparams, weights_file, clear_previous=False, by_name=True):
    """
    Initializes a model according to hparams. Then sets its parameters from
    the parameters in h5 file 'weights_file'.

    Args:
        hparams:        A YAMLHparams object of hyperparameters
        weights_file:   A path to a h5 parameter file to load
        clear_previous: Clear previous keras session before initializing new model graph.
        by_name:        Load parameters by layer names instead of order (default).

    Returns:
        A tf.keras Model instance
    """
    model = init_model(build_hparams=hparams["build"], clear_previous=clear_previous)
    load_from_file(model, weights_file, by_name=by_name)
    return model


def init_and_load_best_model(hparams, model_dir, clear_previous=False, by_name=True):
    """
    Initializes a model according to hparams. Then finds the best model in
    model_dir and loads it (see utime.utils.get_best_model).

    Args:
        hparams:        A YAMLHparams object of hyperparameters
        model_dir:      A path to the directory that stores model param files
        clear_previous: Clear previous keras session before initializing new model graph.
        by_name:        Load parameters by layer names instead of order (default).

    Returns:
        A tf.keras Model instance
        The file name of the parameter file that was loaded
    """
    model = init_model(hparams["build"], clear_previous=clear_previous)
    model_path = get_best_model(model_dir)
    load_from_file(model, model_path, by_name=by_name)
    model_file_name = os.path.split(model_path)[-1]
    return model, model_file_name


def init_and_load_latest_model(hparams, model_dir, clear_previous=False, by_name=True):
    """
    Initializes a model according to hparams. Then finds the latest model in
    model_dir and loads it (see utime.utils.get_latest_model).

    Args:
        hparams:        A YAMLHparams object of hyperparameters
        model_dir:      A path to the directory that stores model param files
        clear_previous: Clear previous keras session before initializing new model graph.
        by_name:        Load parameters by layer names instead of order (default).

    Returns:
        A tf.keras Model instance
        The file name of the parameter file that was loaded
        The epoch of training that the file corresponds to
    """
    model = init_model(hparams["build"], clear_previous=clear_previous)
    model_path, epoch = get_last_model(model_dir)
    if model_path is None:
        raise OSError("Did not find any model files in "
                      "directory {}".format(model_dir))
    load_from_file(model, model_path, by_name=by_name)
    model_file_name = os.path.split(model_path)[-1]
    return model, model_file_name, epoch


def prepare_for_continued_training(hparams, project_dir):
    """
    Prepares the hyperparameter set and project directory for continued
    training.

    Will find the latest model (highest epoch number) of parameter files in
    the 'model' subdir of 'project_dir' and base the continued training on this
    file. If no file is found, training will start from scratch as
    normally (note: no error is raised, but None is returned instead of a path
    to a parameter file).

    The hparams['fit']['init_epoch'] parameter will be set to match the found
    parameter file or to 0 if no file was found. Note that if init_epoch is set
    to 0 all rows in the training.csv file will be deleted.

    The hparams['fit']['optimizer_kwargs']['learning_rate'] parameter will
    be set according to the value stored in the project_dir/logs/training.csv
    file at the corresponding epoch (left default if no init_epoch was found)

    Args:
        hparams:      (YAMLHParams) The hyperparameters to use for training
        project_dir:  (string)      The path to the current project directory

    Returns:
        A path to the model weight files to use for continued training.
        Will be None if no model files were found
    """
    model_path, epoch = get_last_model(os.path.join(project_dir, "model"))
    if model_path:
        model_name = os.path.split(model_path)[-1]
    else:
        model_name = None
    csv_path = os.path.join(project_dir, "logs", "training.csv")
    if epoch == 0:
        epoch = get_last_epoch(csv_path)
    else:
        if epoch is None:
            epoch = 0
        clear_csv_after_epoch(epoch, csv_path)
    hparams["fit"]["init_epoch"] = epoch + 1
    # Get the LR at the continued epoch
    lr, name = get_lr_at_epoch(epoch, os.path.join(project_dir, "logs"))
    if lr:
        hparams["fit"]["optimizer_kwargs"][name] = lr
    logger.info(f"[NOTICE] Training continues from:\n"
                f"Model: {model_name or '<No model found - Starting for scratch!>'}\n"
                f"Epoch: {epoch}\n"
                f"LR:    {lr}")
    return model_path
