import logging
import os
import re
import glob
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

logger = logging.getLogger(__name__)


def standardize_batch_shape(batch_shape):
    batch_shape = list(batch_shape)
    if len(batch_shape) == 3:
        return batch_shape[1:]
    elif len(batch_shape) == 2:
        return batch_shape
    elif len(batch_shape) == 1:
        return batch_shape + [1]
    elif len(batch_shape) == 4:
        return batch_shape[2:]
    else:
        raise ValueError(f"Passed 'batch_shape' could not be standardized to"
                         f" a length 2 list of format "
                         f"[input_dim, input_channels]. Got: {batch_shape}")


def get_best_model(model_dir):
    if len(os.listdir(model_dir)) == 0:
        raise OSError("Model dir {} is empty.".format(model_dir))
    # look for models, order: val_dice, val_loss, dice, loss, model_weights
    patterns = [
        ("@epoch*val_dice*", np.argmax),
        ("@epoch*val_loss*", np.argmin),
        ("@epoch*dice*", np.argmax),
        ("@epoch*loss*", np.argmin)
    ]
    for pattern, select_func in patterns:
        models = glob.glob(os.path.join(model_dir, pattern))
        if models:
            scores = []
            for m in models:
                scores.append(float(re.findall(r"(\d+[.]\d+)", m)[0]))
            return os.path.abspath(models[select_func(np.array(scores))])
    m = os.path.abspath(os.path.join(model_dir, "model_weights.h5"))
    if not os.path.exists(m):
        raise OSError("Did not find any model files matching the patterns {} "
                      "and did not find a model_weights.h5 file."
                      "".format(patterns))
    return m


def get_last_model(model_dir):
    models = glob.glob(os.path.join(model_dir, "@epoch*"))
    epochs = []
    for m in models:
        epochs.append(int(re.findall(r"@epoch_(\d+)_", m)[0]))
    if epochs:
        last = np.argmax(epochs)
        return os.path.abspath(models[last]), int(epochs[int(last)])
    else:
        generic_path = os.path.join(model_dir, "model_weights.h5")
        if os.path.exists(generic_path):
            # Return epoch 0 as we dont know where else to start
            # This may be changed elsewhere in the code based on the
            # training data CSV file
            return generic_path, 0
        else:
            # Start from scratch, or handle as see fit at call point
            return None, None


def save_frozen_model(keras_model, out_folder, pb_file_name):
    assert len(keras_model.inputs) == 1, 'Only implemented for models wit 1 input'
    assert len(keras_model.outputs) == 1, 'Only implemented for models wit 1 output'
    model = tf.function(keras_model, input_signature=[tf.TensorSpec(keras_model.inputs[0].shape,
                                                                    keras_model.inputs[0].dtype)])
    frozen_func = convert_variables_to_constants_v2(model.get_concrete_function())
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=out_folder,
                      name=pb_file_name,
                      as_text=False)
